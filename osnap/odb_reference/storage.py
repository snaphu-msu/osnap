"""Version 1.0 HDF5 persistence. No pickle and no implicit full-array loading."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .composition import Composition, Species
from .errors import ClosedDatasetError, FormatError
from .fields import Axis, DiskArray, Field
from .model import Dataset, RadialGrid, Report, Series, Snapshot, TracerSet, Trajectory
from .units import UnitRegistry

SCHEMA_VERSION = "1.0"
WRITER_VERSION = "1.0.0"
TEXT = h5py.string_dtype("utf-8")


def _json(group, name, value):
    group.create_dataset(name, data=json.dumps(value, allow_nan=False), dtype=TEXT)


def _load_json(group, name, default=None):
    return json.loads(group[name].asstr()[()]) if name in group else default


def _field_write(parent, name, field):
    g = parent.create_group(name)
    meta = {key: getattr(field, key) for key in ("name", "unit", "dimensions", "location", "sampling", "description", "axes", "source", "metadata")}
    # Common discovery properties are attributes as well as complete JSON metadata.
    g.attrs.update(name=field.name, unit=field.unit, location=field.location,
                   sampling=field.sampling, description=field.description)
    _json(g, "metadata_json", meta)
    options = dict(chunks=True, compression="gzip", compression_opts=4, shuffle=True) if field.shape and all(field.shape) else {}
    values = g.create_dataset("values", shape=field.shape, dtype=field.dtype, **options)
    mask = g.create_dataset("valid", shape=field.shape, dtype=bool, **options) if field._valid is not None else None
    selections = values.iter_chunks() if values.chunks else [None]
    for selection in selections:
        index = () if selection is None else selection
        values[index] = field.read(selection)
        if mask is not None:
            mask[index] = field.read_validity(selection)


def _field_read(g, owner, registry):
    meta = _load_json(g, "metadata_json")
    if meta is None or "values" not in g:
        raise FormatError(f"Missing field metadata or values in {g.name}")
    return Field(values=DiskArray(g["values"], owner),
                 valid=DiskArray(g["valid"], owner) if "valid" in g else None,
                 registry=registry, **meta)


def _axis_write(parent, name, axis):
    g = parent.create_group(name)
    _json(g, "metadata_json", dict(name=axis.name, unit=axis.unit, description=axis.description, metadata=axis.metadata))
    values = axis.read()
    g.create_dataset("values", data=values.astype(object) if values.dtype.kind in "USO" else values,
                     dtype=TEXT if values.dtype.kind in "USO" else values.dtype)
    g.attrs["unit"] = axis.unit
    if axis.bounds:
        _field_write(g, "bounds", axis.bounds)


def _axis_read(g, owner, registry):
    return Axis(values=DiskArray(g["values"], owner), registry=registry,
                bounds=_field_read(g["bounds"], owner, registry) if "bounds" in g else None,
                **_load_json(g, "metadata_json"))


def _composition_write(parent, name, composition, root, tables):
    g = parent.create_group(name)
    table = [s.to_dict() for s in composition.species]
    signature = json.dumps(table, sort_keys=True)
    if signature not in tables:
        table_id = f"n{len(tables):06d}"
        tables[signature] = table_id
        _json(root["species_tables"].create_group(table_id), "species_json", table)
    _json(g, "metadata_json", dict(basis=composition.basis, complete=composition.complete,
        interpretation=composition.interpretation, metadata=composition.metadata,
        species_table=tables[signature]))
    _field_write(g, "abundances", composition.abundances)


def _composition_read(g, owner, registry):
    meta = _load_json(g, "metadata_json")
    table = owner["species_tables"][meta.pop("species_table")]
    species = [Species(**s) for s in _load_json(table, "species_json")]
    return Composition(_field_read(g["abundances"], owner, registry), species, **meta)


def _snapshot_write(g, snapshot, root, tables):
    _json(g, "metadata_json", dict(time=snapshot.time, metadata=snapshot.metadata,
                                   provenance=snapshot.provenance, source_id=snapshot.source_id))
    if snapshot.grid:
        grid = snapshot.grid
        gg = g.create_group("grid")
        _json(gg, "metadata_json", dict(frame=grid.frame, metadata=grid.metadata))
        _field_write(gg, "radius_faces", grid.radius_faces)
        _field_write(gg, "radius_cells", grid.radius_cells)
        if grid.mass_faces is not None:
            _field_write(gg, "mass_faces", grid.mass_faces)
    axes = g.create_group("axes")
    for name, axis in snapshot.axes.items():
        _axis_write(axes, name, axis)
    fields = g.create_group("fields")
    for (location, name), field in snapshot.fields.items():
        _field_write(fields.require_group(location), name, field)
    comps = g.create_group("compositions")
    for name, c in snapshot.compositions.items():
        _composition_write(comps, name, c, root, tables)


def _snapshot_read(g, owner, registry):
    grid = None
    if "grid" in g:
        gg = g["grid"]
        grid = RadialGrid(_field_read(gg["radius_faces"], owner, registry),
            radius_cells=_field_read(gg["radius_cells"], owner, registry),
            mass_faces=_field_read(gg["mass_faces"], owner, registry) if "mass_faces" in gg else None,
            registry=registry, **_load_json(gg, "metadata_json"))
    fields = [_field_read(f, owner, registry) for location in g["fields"].values() for f in location.values()]
    return Snapshot(fields, grid=grid,
        axes={name: _axis_read(axis, owner, registry) for name, axis in g["axes"].items()},
        compositions={name: _composition_read(c, owner, registry) for name, c in g["compositions"].items()},
        **_load_json(g, "metadata_json"))


def _trajectory_write(g, t, root, tables):
    _json(g, "metadata_json", dict(tracer_id=t.tracer_id, metadata=t.metadata,
        provenance=t.provenance, time_reference=t.time_reference))
    _axis_write(g, "time", t.time)
    fields, properties = g.create_group("fields"), g.create_group("properties")
    for f in t.fields.values():
        _field_write(fields.require_group(f.location), f.name, f)
    for f in t.properties.values():
        _field_write(properties, f.name, f)
    if t.represented_mass is not None:
        _field_write(g, "represented_mass", t.represented_mass)
    comps = g.create_group("compositions")
    for name, c in t.compositions.items():
        _composition_write(comps, name, c, root, tables)


def _trajectory_read(g, owner, registry):
    fields = [_field_read(f, owner, registry) for location in g["fields"].values() for f in location.values()]
    return Trajectory(time=_axis_read(g["time"], owner, registry), fields=fields,
        properties=[_field_read(f, owner, registry) for f in g["properties"].values()],
        represented_mass=_field_read(g["represented_mass"], owner, registry) if "represented_mass" in g else None,
        compositions={name: _composition_read(c, owner, registry) for name, c in g["compositions"].items()},
        **_load_json(g, "metadata_json"))


class _LazySnapshots(Sequence):
    def __init__(self, group, index, registry):
        self.group, self.index, self.registry = group, index, registry

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if not self.group.id.valid:
            raise ClosedDatasetError("The OSNAP HDF5 file has been closed")
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        return _snapshot_read(self.group[self.index[index]["storage_id"]], self.group.file, self.registry)


class _LazyTracers(Mapping):
    def __init__(self, group, index, registry):
        self.group, self.registry = group, registry
        self.index = {row["tracer_id"]: row["storage_id"] for row in index}
        if len(self.index) != len(index):
            raise FormatError("Duplicate persistent tracer IDs in HDF5 index")

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    def __getitem__(self, key):
        if not self.group.id.valid:
            raise ClosedDatasetError("The OSNAP HDF5 file has been closed")
        return _trajectory_read(self.group[self.index[key]], self.group.file, self.registry)


def _registries(dataset):
    """Gather custom definitions from all fields without loading their values."""
    result = UnitRegistry(dataset.registry.to_dict())
    def add(registry):
        for name, definition in registry.to_dict().items():
            result.register(name, **definition)
    for series in dataset.series.values():
        for s in series.snapshots:
            if s.grid:
                add(s.grid.registry)
            for a in s.axes.values():
                add(a.registry)
            for f in s.fields.values():
                add(f.registry)
            for c in s.compositions.values():
                add(c.abundances.registry)
    for ts in dataset.tracers.values():
        for t in ts.trajectories.values():
            add(t.time.registry)
            for f in [*t.fields.values(), *t.properties.values(), *([t.represented_mass] if t.represented_mass else [])]:
                add(f.registry)
            for c in t.compositions.values():
                add(c.abundances.registry)
    return result


def write_dataset(dataset, path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    dataset.validate().raise_for_errors()
    registry = _registries(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs.update(format="OSNAP", schema_version=SCHEMA_VERSION, writer_version=WRITER_VERSION,
                              dataset_id=dataset.dataset_id, created_utc=datetime.now(timezone.utc).isoformat())
            _json(root, "metadata_json", dataset.metadata)
            _json(root, "report_json", dataset.report.to_dict())
            units = root.create_group("units")
            units.attrs["version"] = registry.version
            _json(units, "definitions_json", registry.to_dict())
            root.create_group("species_tables")
            provenance = root.create_group("provenance")
            for i, record in enumerate(dataset.provenance):
                _json(provenance, f"p{i:08d}", record)
            tables = {}
            series_group = root.create_group("series")
            for name, series in dataset.series.items():
                g = series_group.create_group(name)
                _json(g, "metadata_json", dict(time_reference=series.time_reference, metadata=series.metadata))
                snapshots = g.create_group("snapshots")
                index = []
                for i, snapshot in enumerate(series.snapshots):
                    sid = f"s{i:08d}"
                    index.append(dict(storage_id=sid, time=snapshot.time, source_id=snapshot.source_id))
                    _snapshot_write(snapshots.create_group(sid), snapshot, root, tables)
                _json(g, "index", index)
                g["index"].attrs["time_unit"] = "s"
            tracer_group = root.create_group("tracers")
            for name, ts in dataset.tracers.items():
                g = tracer_group.create_group(name)
                _json(g, "metadata_json", ts.metadata)
                trajectories = g.create_group("trajectories")
                index = []
                for i, t in enumerate(ts.trajectories.values()):
                    tid = f"t{i:08d}"
                    index.append(dict(storage_id=tid, tracer_id=t.tracer_id))
                    _trajectory_write(trajectories.create_group(tid), t, root, tables)
                _json(g, "index", index)
            root.flush()
        with read_dataset(temporary) as check:
            check.validate().raise_for_errors()
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication; a racing creator cannot be overwritten.
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_dataset(path):
    root = h5py.File(path, "r")
    try:
        if root.attrs.get("format") != "OSNAP" or root.attrs.get("schema_version") != SCHEMA_VERSION:
            raise FormatError("Unsupported OSNAP format/schema version")
        if root.attrs.get("required_features", ""):
            raise FormatError("This file requires unsupported OSNAP features")
        registry = UnitRegistry(_load_json(root["units"], "definitions_json"))
        if root["units"].attrs.get("version") != registry.version:
            raise FormatError("Unsupported unit-definition version")
        series = {}
        for name, g in root["series"].items():
            index = _load_json(g, "index")
            ids = [row["storage_id"] for row in index]
            if len(ids) != len(set(ids)) or any(sid not in g["snapshots"] for sid in ids):
                raise FormatError("Invalid snapshot index")
            series[name] = Series(_LazySnapshots(g["snapshots"], index, registry), **_load_json(g, "metadata_json"))
        tracers = {name: TracerSet(_LazyTracers(g["trajectories"], _load_json(g, "index"), registry),
                                 metadata=_load_json(g, "metadata_json")) for name, g in root["tracers"].items()}
        dataset = Dataset(series=series, tracers=tracers, registry=registry,
            metadata=_load_json(root, "metadata_json"),
            provenance=[json.loads(g.asstr()[()]) for g in root["provenance"].values()],
            dataset_id=root.attrs["dataset_id"], report=Report(**_load_json(root, "report_json", {})))
        dataset._file = root
        return dataset
    except Exception:
        root.close()
        raise
