"""Containers for spherical snapshots, nonspatial products and tracer histories."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
import uuid
import numpy as np

from .errors import DataError
from .fields import Axis, Field, field_map, lookup, metadata_copy
from .composition import Composition
from .units import DEFAULT_UNITS


@dataclass
class Report:
    """Machine-readable validation or conversion report, with no implicit repair."""
    issues: list = dataclass_field(default_factory=list)
    transformations: list = dataclass_field(default_factory=list)
    assumptions: list = dataclass_field(default_factory=list)
    unmapped_fields: list = dataclass_field(default_factory=list)
    information_loss: list = dataclass_field(default_factory=list)

    @property
    def ok(self):
        return not any(i["severity"] == "error" for i in self.issues)

    def add(self, message, *, path="", severity="warning"):
        self.issues.append({"path": path, "severity": severity, "message": message})

    def raise_for_errors(self):
        if not self.ok:
            raise DataError("; ".join(i["path"] + ": " + i["message"] for i in self.issues if i["severity"] == "error"))
        return self

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class RadialGrid:
    """Spherical shell boundaries. Coordinate Fields can be disk-backed."""
    def __init__(self, radius_faces, *, radius_cells=None, mass_faces=None,
                 frame="eulerian", metadata=None, registry=None):
        self.registry = registry or DEFAULT_UNITS
        self.frame, self.metadata = frame, metadata_copy(metadata)
        if frame not in ("eulerian", "lagrangian"):
            raise DataError("Grid frame must be eulerian or lagrangian")
        self.radius_faces = self._coordinate(radius_faces, "radius", "cm", "face")
        if len(self.radius_faces.shape) != 1 or self.radius_faces.shape[0] < 2:
            raise DataError("A radial grid requires at least two faces")
        self.ncells = self.radius_faces.shape[0] - 1
        self._radius_cells = None if radius_cells is None else self._coordinate(radius_cells, "radius", "cm", "cell")
        if radius_cells is None:
            self.metadata.setdefault("cell_sampling", "radial_midpoint")
        self.mass_faces = None if mass_faces is None else self._coordinate(mass_faces, "enclosed_mass", "g", "face")
        if self._radius_cells is not None and self._radius_cells.shape != (self.ncells,):
            raise DataError("Wrong number of cell sampling radii")
        if self.mass_faces is not None and self.mass_faces.shape != self.radius_faces.shape:
            raise DataError("Mass and radial face counts differ")
        # Disk coordinates are checked only when validation/numerical work requests them.
        from .fields import DiskArray
        if not isinstance(self.radius_faces._values, DiskArray):
            self.validate()

    def _coordinate(self, value, name, unit, location):
        result = value if isinstance(value, Field) else Field(name, value, unit,
            dimensions=(location,), location=location, registry=self.registry)
        result.registry.factor(result.unit, unit)
        if result.location != location:
            raise DataError(f"Grid {name} must be located at {location}")
        return result

    @property
    def radius_cells(self):
        if self._radius_cells is not None:
            return self._radius_cells
        edges = self.radius_faces.read()
        return Field("radius", (edges[:-1] + edges[1:])/2, "cm", dimensions=("cell",),
                     location="cell", source={"operation": "radial_midpoint"}, registry=self.registry)

    def volumes(self):
        r = self.radius_faces.require_valid()
        # Factored difference is more accurate for thin shells than subtracting cubes.
        return 4*np.pi/3 * np.diff(r) * (r[1:]**2 + r[1:]*r[:-1] + r[:-1]**2)

    def validate(self):
        r = self.radius_faces.require_valid()
        if np.any(r < 0) or np.any(np.diff(r) <= 0):
            raise DataError("Radial faces must be nonnegative and strictly increasing")
        c = self.radius_cells.require_valid()
        if np.any(c < r[:-1]) or np.any(c > r[1:]) or np.any(np.diff(c) <= 0):
            raise DataError("Cell sampling radii must increase and lie within their shells")
        if self.mass_faces is not None:
            m = self.mass_faces.require_valid()
            if np.any(m < 0) or np.any(np.diff(m) < 0):
                raise DataError("Enclosed-mass faces must be nonnegative and nondecreasing")


class Snapshot:
    def __init__(self, fields=(), *, grid=None, axes=None, compositions=None, time=None,
                 metadata=None, provenance=None, source_id=None):
        self.fields = field_map(fields)
        self.grid, self.axes = grid, dict(axes or {})
        self.compositions = dict(compositions or {})
        self.time = None if time is None else float(time)
        if self.time is not None and not np.isfinite(self.time):
            raise DataError("Snapshot time must be finite or None")
        self.metadata, self.provenance = metadata_copy(metadata), metadata_copy(provenance or [])
        self.source_id = source_id
        self.validate_structure()

    def field(self, name, *, location=None):
        return lookup(self.fields, name, location)

    def validate_structure(self):
        sizes = {key: axis.shape[0] for key, axis in self.axes.items()}
        if self.grid:
            sizes.update(cell=self.grid.ncells, face=self.grid.ncells+1)
        for f in self.fields.values():
            self._validate_field(f, sizes)
        for composition in self.compositions.values():
            self._validate_field(composition.abundances, {**sizes, "species": len(composition.species)})

    @staticmethod
    def _validate_field(f, sizes):
        for dim, length in zip(f.dimensions, f.shape):
            reference = f.axes[dim]
            if reference not in sizes or sizes[reference] != length:
                raise DataError(f"Field {f.name}: dimension {dim!r} references missing/incompatible axis {reference!r}")

    def with_field(self, field):
        values = {**self.fields, field.key: field}
        return Snapshot(values, grid=self.grid, axes=self.axes, compositions=self.compositions,
                        time=self.time, metadata=self.metadata, source_id=self.source_id,
                        provenance=self.provenance + [{"operation": "with_field", "field": field.name}])

    def derive(self, name, **parameters):
        from .derived import derive
        return derive(self, name, **parameters)

    def remap(self, target_edges, *, coordinate, mass_source, field_policies, **options):
        from .numerics import remap
        return remap(self, target_edges, coordinate=coordinate, mass_source=mass_source,
                     field_policies=field_policies, **options)

    def interpolate_field(self, name, *, location=None, target_location, boundary=None):
        from .numerics import center_field
        return center_field(self, self.field(name, location=location), target_location, boundary)

    def reduce(self, name, *, location=None, operation="sum", weights=None):
        f = self.field(name, location=location)
        values = f.require_valid()
        if values.ndim != 1:
            raise DataError("Reductions currently require a one-dimensional field")
        if operation not in ("sum", "average"):
            raise DataError("Reduction operation must be sum or average")
        if operation == "average" and weights is None:
            raise DataError("Averages require explicit weights")
        unit = f.unit
        if weights is not None:
            w = weights.require_valid()
            if w.shape != values.shape or np.any(w < 0) or w.sum() <= 0:
                raise DataError("Weights must match the field, be nonnegative and have positive sum")
            result = np.sum(values*w)
            if operation == "average":
                result /= w.sum()
            else:
                unit = f"({unit})*({weights.unit})"
        else:
            result = values.sum()
        return Field(f"{name}_{operation}", result, unit, registry=f.registry,
                     source={"operation": operation, "field": name, "weights": None if weights is None else weights.name})

    def select_cells(self, start, stop, *, mass_source=None):
        if self.grid is None or not 0 <= start < stop <= self.grid.ncells:
            raise DataError("Select a nonempty, contiguous range of existing shells")
        grid = self.grid
        if mass_source is None:
            if grid.mass_faces is None:
                raise DataError("Selection requires mass_source='density' to report excluded mass")
            mass_source = "mass_faces"
        masses = self.derive("cell_mass", source=mass_source).read()
        mfaces = grid.mass_faces.subset(slice(start, stop+1)) if grid.mass_faces is not None else None
        new_grid = RadialGrid(grid.radius_faces.subset(slice(start, stop+1)),
            radius_cells=grid.radius_cells.subset(slice(start, stop)), mass_faces=mfaces,
            frame=grid.frame, metadata=grid.metadata, registry=grid.registry)
        def subset(f):
            if f.location in ("cell", "face"):
                return f.subset(slice(start, stop + (f.location == "face")))
            return f
        compositions = {k: Composition(subset(c.abundances), c.species, basis=c.basis,
                        complete=c.complete, interpretation=c.interpretation, metadata=c.metadata)
                        for k, c in self.compositions.items()}
        return Snapshot([subset(f) for f in self.fields.values()], grid=new_grid, axes=self.axes,
            compositions=compositions, time=self.time, metadata=self.metadata, source_id=self.source_id,
            provenance=self.provenance + [{"operation": "select_cells", "start": start, "stop": stop,
                "excluded_mass": {"value": float(masses[:start].sum()+masses[stop:].sum()), "unit": "g"}}])


class Series:
    def __init__(self, snapshots=(), *, time_reference="simulation_start", metadata=None):
        if not time_reference:
            raise DataError("A series requires a declared time reference")
        self.snapshots = snapshots if isinstance(snapshots, Sequence) else tuple(snapshots)
        self.time_reference, self.metadata = time_reference, metadata_copy(metadata)

    def snapshot(self, index):
        return self.snapshots[index]

    def __len__(self):
        return len(self.snapshots)

    def interpolate(self, time, *, target_grid, coordinate, field_policies, mass_source, time_reference=None):
        from .numerics import interpolate_series
        if time_reference is not None and time_reference != self.time_reference:
            raise DataError("Time references differ; explicitly shift the series first")
        return interpolate_series(self, time, target_grid=target_grid, coordinate=coordinate,
                                  field_policies=field_policies, mass_source=mass_source)

    def shifted_time(self, offset, *, time_reference):
        if not np.isfinite(offset):
            raise DataError("Time offset must be finite seconds")
        snapshots = []
        for s in self.snapshots:
            if s.time is None:
                raise DataError("Cannot shift an unknown time")
            snapshots.append(Snapshot(s.fields, grid=s.grid, axes=s.axes, compositions=s.compositions,
                time=s.time+offset, metadata=s.metadata, source_id=s.source_id,
                provenance=s.provenance + [{"operation": "time_offset", "seconds": float(offset),
                                           "from": self.time_reference, "to": time_reference}]))
        return Series(snapshots, time_reference=time_reference, metadata=self.metadata)


class Trajectory:
    def __init__(self, tracer_id, time, fields=(), *, compositions=None, represented_mass=None,
                 properties=None, metadata=None, provenance=None, time_reference="simulation_start"):
        if isinstance(tracer_id, np.integer):
            tracer_id = int(tracer_id)
        if not isinstance(tracer_id, (int, str)) or isinstance(tracer_id, bool):
            raise DataError("Tracer IDs must be strings or integers")
        self.tracer_id = tracer_id
        self.time = time if isinstance(time, Axis) else Axis("time", time, "s")
        self.time.registry.factor(self.time.unit, "s")
        self.fields, self.compositions = field_map(fields), dict(compositions or {})
        self.represented_mass = represented_mass
        if represented_mass is not None:
            if not isinstance(represented_mass, Field) or represented_mass.shape:
                raise DataError("Represented mass must be a scalar Field with mass units")
            represented_mass.registry.factor(represented_mass.unit, "g")
        self.properties = field_map(properties)
        self.metadata, self.provenance = metadata_copy(metadata), metadata_copy(provenance or [])
        self.time_reference = time_reference
        if not time_reference:
            raise DataError("Trajectories require a time reference")
        sizes = {"time": self.time.shape[0]}
        for f in self.fields.values():
            Snapshot._validate_field(f, sizes)
        for c in self.compositions.values():
            Snapshot._validate_field(c.abundances, {**sizes, "species": len(c.species)})
        if any(f.shape for f in self.properties.values()):
            raise DataError("Tracer properties must be scalar Fields")

    def field(self, name, *, location=None):
        return lookup(self.fields, name, location)

    def interpolate(self, times, *, fields):
        from .numerics import linear
        target = np.asarray(times, dtype=np.float64)
        if target.ndim != 1 or not len(target) or np.any(np.diff(target) <= 0):
            raise DataError("Target tracer times must be a nonempty increasing vector")
        chosen, compositions = [], {}
        for name in fields:
            if name in self.compositions:
                c = self.compositions[name]
                values = linear(self.time.read(), c.abundances.require_valid(), target)
                compositions[name] = Composition(c.abundances.replaced(values), c.species,
                    basis=c.basis, complete=c.complete, interpretation=c.interpretation, metadata=c.metadata)
            else:
                f = self.field(name)
                chosen.append(f.replaced(linear(self.time.read(), f.require_valid(), target)))
        return Trajectory(self.tracer_id, target, chosen, compositions=compositions,
            represented_mass=self.represented_mass, properties=self.properties, metadata=self.metadata,
            time_reference=self.time_reference, provenance=self.provenance + [{"operation": "time_interpolation"}])


class TracerSet:
    def __init__(self, trajectories=(), *, metadata=None):
        if isinstance(trajectories, Mapping):
            self.trajectories = trajectories
        else:
            trajectories = list(trajectories)
            self.trajectories = {t.tracer_id: t for t in trajectories}
            if len(self.trajectories) != len(trajectories):
                raise DataError("Duplicate tracer IDs")
        self.metadata = metadata_copy(metadata)

    def trajectory(self, tracer_id):
        return self.trajectories[tracer_id]

    def species_yields(self, *, composition="composition", sample=-1):
        totals = {}
        for t in self.trajectories.values():
            if t.represented_mass is None:
                raise DataError(f"Tracer {t.tracer_id} has no represented mass")
            mass = float(t.represented_mass.require_valid())
            if mass < 0:
                raise DataError("Represented mass cannot be negative")
            c = t.compositions[composition]
            x = c.mass_fractions().require_valid()[sample]
            if np.any(x < 0):
                raise DataError("Yield integration requires nonnegative abundances")
            for species, value in zip(c.species, x):
                totals[species.id] = totals.get(species.id, 0.0) + mass*value
        return {name: Field(name, value, "g", source={"operation": "tracer_yield"}) for name, value in totals.items()}


class Dataset:
    def __init__(self, *, series=None, tracers=None, metadata=None, provenance=None,
                 registry=None, dataset_id=None, report=None):
        self.series, self.tracers = dict(series or {}), dict(tracers or {})
        self.metadata, self.provenance = metadata_copy(metadata), metadata_copy(provenance or [])
        self.registry = registry or DEFAULT_UNITS
        self.dataset_id = dataset_id or str(uuid.uuid4())
        self.report = report or Report()
        self._file = None

    @classmethod
    def from_native(cls, source, *, format, **adapter_options):
        from .adapters import read_native
        return read_native(source, format=format, **adapter_options)

    @classmethod
    def open(cls, path):
        from .storage import read_dataset
        return read_dataset(path)

    def write(self, path, *, overwrite=False):
        from .storage import write_dataset
        return write_dataset(self, path, overwrite=overwrite)

    def export_native(self, target, *, format, selection, **options):
        from .adapters import export_native
        return export_native(self, target, format=format, selection=selection, **options)

    def validate(self, level="structure"):
        from .validation import validate_dataset
        return validate_dataset(self, level)

    def close(self):
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
