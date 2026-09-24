"""Shared adapter contracts and explicit table/field mapping helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
import io
from pathlib import Path
import numpy as np

from ..composition import Composition, Species
from ..errors import DataError, FormatError
from ..fields import Axis, Field
from ..model import Dataset, Report, Series, Snapshot


class Adapter(ABC):
    format = ""
    description = ""
    mappings = {}

    @abstractmethod
    def detect(self, source):
        """Conservative detection; ambiguous matches are rejected by the registry."""

    @abstractmethod
    def read(self, source, **options):
        """Return a Dataset with a conversion Report in dataset.report."""

    def write(self, target, selection, **options):
        raise FormatError(f"{self.format} does not support export")


def numeric_table(path, *, skiprows=0):
    text = Path(path).read_text().replace("D+", "E+").replace("D-", "E-").replace("d+", "e+").replace("d-", "e-")
    try:
        return np.loadtxt(io.StringIO(text), skiprows=skiprows, ndmin=2)
    except ValueError as error:
        raise FormatError(f"Cannot parse numeric table {path}: {error}") from error


def mapped_field(native, values, spec, path, *, valid=None):
    spec = dict(spec)
    location = spec.pop("location", "cell")
    name, unit = spec.pop("name"), spec.pop("unit")
    transform = spec.pop("transform", None)
    values = np.asarray(values)
    source = {"name": native, "location": str(path), "transformations": []}
    if transform in ("log10", "ln"):
        values = 10.**values if transform == "log10" else np.exp(values)
        source["transformations"].append({"operation": "decode_log", "base": transform})
    elif transform is not None:
        raise FormatError(f"Unknown source encoding {transform}")
    dims = spec.pop("dimensions", () if location == "global" else ("time",) if location == "sample" else (location,))
    return Field(name, values, unit, dimensions=dims, location=location, valid=valid,
                 source=source, **spec)


def composition_field(values, labels, path, *, location="cell", basis="mass_fraction", complete=False, groups=(), valid=None):
    species = [label if isinstance(label, Species) else Species.parse(label, groups=groups) for label in labels]
    dim = "time" if location == "sample" else location
    field = Field("abundances", values, "1", dimensions=(dim, "species"), location=location,
                  sampling="average" if location == "cell" else "point", valid=valid,
                  source={"location": str(path), "source_labels": [s.source_label for s in species]})
    return Composition(field, species, basis=basis, complete=complete)


def finish(snapshots, report, *, name="hydrodynamics", time_reference="simulation_start", metadata=None):
    """Split restart branches, never silently dropping duplicate or decreasing times."""
    branches, current = [], []
    for snapshot in snapshots:
        if current and (snapshot.time is None or current[-1].time is None or snapshot.time <= current[-1].time):
            branches.append(current)
            current = []
        current.append(snapshot)
    if current:
        branches.append(current)
    if len(branches) > 1:
        report.transformations.append({"operation": "split_restart_branches", "count": len(branches)})
    series = {name if len(branches) == 1 else f"{name}_branch_{i:03d}": Series(branch, time_reference=time_reference)
              for i, branch in enumerate(branches)}
    dataset = Dataset(series=series, report=report, metadata=metadata)
    validation = dataset.validate("scientific")
    validation.raise_for_errors()
    report.issues.extend(validation.issues)
    return dataset


def history(path, *, columns=None, field_map, time_column="time", time_unit="s", time_reference="simulation_start", name="history"):
    lines = Path(path).read_text().splitlines()
    skip = 0
    if columns is None:
        for i, line in enumerate(lines):
            words = line.strip().lstrip("#").split()
            if time_column in words:
                columns, skip = words, i+1
                break
        if columns is None:
            raise FormatError("History requires a named header or explicit columns")
    data = numeric_table(path, skiprows=skip)
    if data.shape[1] != len(columns) or time_column not in columns:
        raise FormatError("History columns do not match the table")
    time = Field("time", data[:, columns.index(time_column)], time_unit, location="sample", dimensions=("time",)).read()
    report, snapshots = Report(), []
    mappings = {key: dict(value) for key, value in field_map.items()}
    report.unmapped_fields = [c for c in columns if c != time_column and c not in mappings]
    for row, instant in zip(data, time):
        fields = [mapped_field(c, row[columns.index(c)], {**spec, "location": "global", "dimensions": ()}, path)
                  for c, spec in mappings.items() if c in columns]
        snapshots.append(Snapshot(fields, time=instant, source_id=str(path)))
    return finish(snapshots, report, name=name, time_reference=time_reference)


def require_snapshot(selection):
    if not isinstance(selection, Snapshot) or selection.grid is None:
        raise DataError("This exporter requires a gridded Snapshot selection")
    selection.grid.validate()
    return selection


def checked_composition(snapshot, name="composition", *, isotopes=False):
    c = snapshot.compositions[name]
    f = c.mass_fractions()
    x = f.require_valid()
    if f.dimensions != ("cell", "species") or not c.complete or np.any(x < 0) or not np.allclose(x.sum(axis=1), 1., rtol=1e-6, atol=1e-10):
        raise DataError("Export requires complete, nonnegative, normalized cell compositions; repair explicitly first")
    if isotopes and any(s.kind != "isotope" for s in c.species):
        raise DataError("This native format requires isotope identities")
    return c, x


def prepare_target(path, *, directory=False, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if directory:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
