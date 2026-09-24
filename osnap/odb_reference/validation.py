"""Structural and scientific validation without changing source data."""
import numpy as np

from .model import Report
from .errors import DataError


def _check_field(f, report, path):
    values, valid = f.read(), f.read_validity()
    if np.any(valid & ~np.isfinite(values)):
        report.add("Unmasked non-finite values", path=path)
    if f.name in ("density", "temperature", "pressure") and np.any(valid & (values < 0)):
        report.add("Negative physical values", path=path)
    if f.name == "electron_fraction" and np.any(valid & ((values < 0) | (values > 1))):
        report.add("Electron fraction outside [0, 1]", path=path)


def _check_composition(c, report, path, supplied=None):
    x = c.mass_fractions()
    values, valid = x.read(), x.read_validity()
    if np.any(valid & ((values < 0) | ~np.isfinite(values))):
        report.add("Invalid abundance values", path=path)
    rows = np.all(valid, axis=-1)
    if c.complete and np.any(rows & ~np.isclose(values.sum(axis=-1), 1, rtol=1e-6, atol=1e-10)):
        report.add("Complete composition does not sum to one", path=path)
    if not c.complete:
        report.add("Composition is explicitly incomplete; normalization and Ye cannot be assumed", path=path)
    if supplied is not None and c.complete and all(s.kind == "isotope" for s in c.species):
        ye = c.electron_fraction()
        if ye.shape == supplied.shape:
            mask = ye.read_validity() & supplied.read_validity()
            if np.any(mask & ~np.isclose(ye.read(), supplied.read(), rtol=1e-6, atol=1e-10)):
                report.add("Source electron fraction differs from composition-derived Ye", path=path)


def validate_dataset(dataset, level):
    if level not in ("structure", "scientific"):
        raise DataError("Validation level must be 'structure' or 'scientific'")
    report = Report()
    for name, series in dataset.series.items():
        if not name or "/" in name:
            report.add("Invalid series name", path=name, severity="error")
        times = []
        for index, snapshot in enumerate(series.snapshots):
            path = f"series/{name}/{index}"
            try:
                snapshot.validate_structure()
                if snapshot.grid:
                    snapshot.grid.validate()
                times.append(snapshot.time)
                if level == "scientific":
                    for f in snapshot.fields.values():
                        _check_field(f, report, path+"/"+f.name)
                    supplied = snapshot.fields.get(("cell", "electron_fraction"))
                    for label, c in snapshot.compositions.items():
                        _check_composition(c, report, path+"/"+label, supplied)
                    if snapshot.grid and snapshot.grid.mass_faces is not None and ("cell", "density") in snapshot.fields:
                        rho = snapshot.field("density", location="cell")
                        if rho.sampling == "average":
                            native = np.diff(snapshot.grid.mass_faces.read())
                            inferred = rho.read()*snapshot.grid.volumes()
                            if not np.allclose(native, inferred, rtol=1e-6, atol=0):
                                report.add("Source shell masses differ from density-derived masses", path=path)
            except (DataError, KeyError, TypeError) as error:
                report.add(str(error), path=path, severity="error")
        if len(times) > 1 and (any(t is None for t in times) or np.any(np.diff(times) <= 0)):
            report.add("Series times must be known and strictly increasing; separate restart branches", path=name, severity="error")
    for name, tracer_set in dataset.tracers.items():
        if not name or "/" in name:
            report.add("Invalid tracer-set name", path=name, severity="error")
        for identity, t in tracer_set.trajectories.items():
            path = f"tracers/{name}/{identity}"
            try:
                if identity != t.tracer_id:
                    raise DataError("Tracer mapping key differs from persistent ID")
                times = t.time.read()
                if not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
                    raise DataError("Tracer times must be finite, nonempty and strictly increasing")
                if t.represented_mass is not None and float(t.represented_mass.require_valid()) < 0:
                    raise DataError("Represented mass cannot be negative")
                if level == "scientific":
                    for f in t.fields.values():
                        _check_field(f, report, path+"/"+f.name)
                    for label, c in t.compositions.items():
                        _check_composition(c, report, path+"/"+label, t.fields.get(("sample", "electron_fraction")))
            except (DataError, KeyError, TypeError) as error:
                report.add(str(error), path=path, severity="error")
    return report
