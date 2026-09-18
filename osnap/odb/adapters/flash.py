"""FLASH 4 per-variable HDF5 profiles and STIR's associated histories."""
from pathlib import Path
import h5py
import numpy as np

from ..errors import FormatError, DataError
from ..fields import Field
from ..model import RadialGrid, Snapshot, Report
from .common import Adapter, composition_field, finish, mapped_field, history, require_snapshot, prepare_target


def parameters(handle, name):
    if name not in handle:
        return {}
    data = handle[name][()]
    if not data.dtype.names or len(data.dtype.names) < 2:
        raise FormatError(f"Unsupported FLASH parameter layout: {name}")
    first, second = data.dtype.names[:2]
    result = {}
    for row in data:
        key = row[first].decode().strip() if isinstance(row[first], bytes) else str(row[first]).strip()
        value = row[second]
        if isinstance(value, bytes):
            value = value.decode().strip()
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


class FLASHAdapter(Adapter):
    format = "flash"
    description = "FLASH 4 HDF5, spherical 1D, per-variable arrays"
    mappings = {
        "dens": dict(name="density", unit="g/cm^3", sampling="average"),
        "temp": dict(name="temperature", unit="K", sampling="average"),
        "pres": dict(name="pressure", unit="dyn/cm^2", sampling="average"),
        "velx": dict(name="radial_velocity", unit="cm/s", sampling="point"),
        "ye": dict(name="electron_fraction", unit="1", sampling="average"),
        "gpot": dict(name="gravitational_potential", unit="erg/g", sampling="average"),
        "ener": dict(name="specific_energy", unit="erg/g", sampling="average",
                     metadata={"energy_terms": "native FLASH ener; solver-dependent", "zero_point": "source-defined"}),
    }

    def detect(self, source):
        try:
            if not h5py.is_hdf5(source):
                return False
            with h5py.File(source, "r") as f:
                return all(n in f for n in ("bounding box", "node type", "unknown names"))
        except (OSError, TypeError):
            return False

    def read(self, source, *, geometry=None, guard_cells=0, field_map=None, complete_composition=False,
             time_reference="simulation_start", species_names=None):
        paths = source if isinstance(source, (list, tuple)) else [source]
        report, snapshots = Report(), []
        if int(guard_cells) != guard_cells or guard_cells < 0:
            raise FormatError("guard_cells must be an explicit nonnegative integer")
        mappings = {**self.mappings, **(field_map or {})}
        for path in paths:
            with h5py.File(path, "r") as f:
                strings = parameters(f, "string runtime parameters")
                geom = geometry or strings.get("geometry")
                if geom != "spherical":
                    raise FormatError("Only spherical 1D FLASH is supported; supply geometry if absent")
                integers = {**parameters(f, "integer runtime parameters"), **parameters(f, "integer scalars")}
                if integers.get("dimensionality", 1) != 1 or integers.get("nyb", 1) != 1 or integers.get("nzb", 1) != 1:
                    raise FormatError("Multidimensional FLASH data are not supported")
                if geometry:
                    report.assumptions.append({"geometry": geometry, "source": str(path)})
                leaf = np.flatnonzero(np.asarray(f["node type"]).reshape(-1) == 1)
                if not len(leaf):
                    raise FormatError("FLASH file has no leaf blocks")
                boxes = np.asarray(f["bounding box"])
                if boxes.ndim != 3 or boxes.shape[-1] != 2:
                    raise FormatError("Unsupported FLASH bounding-box layout")
                names = []
                for item in np.asarray(f["unknown names"]):
                    raw = b"".join(np.asarray(item).reshape(-1).tolist()) if np.asarray(item).dtype.kind == "S" else str(item).encode()
                    names.append(raw.decode().strip())
                actual = {key.strip(): key for key in f.keys()}
                prototype = next((f[actual[n]] for n in names if n in actual), None)
                if prototype is None or prototype.ndim != 4 or prototype.shape[1:3] != (1, 1):
                    raise FormatError("Expected per-variable arrays shaped (blocks, 1, 1, cells)")
                n = prototype.shape[-1]-2*guard_cells
                if n <= 0 or ("nxb" in integers and integers["nxb"] != n):
                    raise FormatError("Guard-cell count does not match native nxb")
                leaf = leaf[np.argsort(boxes[leaf, 0, 0], kind="stable")]
                chunks = [np.linspace(boxes[i,0,0], boxes[i,0,1], n+1) for i in leaf]
                for a, b in zip(chunks, chunks[1:]):
                    if not np.isclose(a[-1], b[0], rtol=1e-12, atol=0):
                        raise FormatError("FLASH leaf blocks overlap or leave radial gaps")
                radius = np.concatenate([c[:-1] for c in chunks]+[chunks[-1][-1:]])
                grid = RadialGrid(radius, metadata={"native_blocks": leaf.tolist(), "source": str(path)})
                fields, abundances, labels = [], [], []
                cell_slice = slice(guard_cells, prototype.shape[-1]-guard_cells)
                for name in names:
                    if name not in actual:
                        raise FormatError(f"Declared FLASH variable {name!r} is missing")
                    native = f[actual[name]]
                    if native.shape != prototype.shape:
                        raise FormatError(f"Unsupported placement/shape for FLASH variable {name}")
                    values = np.concatenate([native[i,0,0,cell_slice] for i in leaf])
                    if name in mappings:
                        spec = mappings[name]
                        if spec.get("location", "cell") != "cell":
                            raise FormatError("This FLASH dialect stores cell values; face arrays require a separate dialect")
                        fields.append(mapped_field(name, values, spec, path))
                    else:
                        try:
                            from ..composition import Species
                            s = Species.parse(name)
                            is_species = s.kind == "isotope" and (species_names is None or name in species_names)
                        except DataError:
                            is_species = False
                        if is_species:
                            labels.append(name); abundances.append(values)
                        else:
                            report.unmapped_fields.append(f"{path}:{name}")
                compositions = {}
                if labels:
                    compositions["composition"] = composition_field(np.stack(abundances, axis=1), labels, path, complete=complete_composition)
                time = parameters(f, "real scalars").get("time")
                snapshots.append(Snapshot(fields, grid=grid, compositions=compositions, time=time,
                    source_id=str(path), metadata={"source_code": self.format}, provenance=[{"operation": "flatten_leaf_blocks", "guard_cells": guard_cells}]))
                report.transformations.append({"operation": "flatten_leaf_blocks", "source": str(path), "guard_cells": guard_cells})
        return finish(snapshots, report, time_reference=time_reference)

    def write(self, target, selection, *, columns=None, overwrite=False):
        s = require_snapshot(selection)
        columns = columns or {"temp": "temperature", "dens": "density", "velx": "radial_velocity", "ye": "electron_fraction", "sumy": "sum_y"}
        values = [s.grid.radius_cells.read()]
        for native, name in columns.items():
            f = s.field(name, location="cell")
            if f.shape != (s.grid.ncells,):
                raise DataError("FLASH initialization columns must be one value per shell")
            values.append(f.require_valid())
        target = prepare_target(target, overwrite=overwrite)
        with target.open("w") as handle:
            handle.write("# OSNAP spherical progenitor; radius in cm, fields in CGS\n")
            handle.write(f"number of variables = {len(columns)}\n"+"\n".join(columns)+"\n")
            np.savetxt(handle, np.column_stack(values), fmt="%.17e")
        return Report(transformations=[{"operation": "export_progenitor", "target": str(target)}],
                      assumptions=["Cell values are passed at native sampling radii without relocation"])


class STIRAdapter(FLASHAdapter):
    format = "stir"
    description = "STIR FLASH-based HDF5 profiles and named/explicit-column ASCII diagnostic histories"

    def read(self, source, *, kind="profile", columns=None, field_map=None, **options):
        if kind == "history":
            if field_map is None:
                raise FormatError("STIR histories require an explicit field_map with units")
            return history(source, columns=columns, field_map=field_map, **options)
        if kind != "profile":
            raise FormatError("STIR kind must be profile or history")
        return super().read(source, field_map=field_map, **options)
