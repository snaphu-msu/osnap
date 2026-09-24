"""SkyNet abundance histories and the explicitly versioned OSNAP driver bundle."""
from pathlib import Path
import json
import h5py
import numpy as np

from ..composition import Species
from ..errors import DataError, FormatError
from ..fields import Field
from ..model import Dataset, Report, Trajectory, TracerSet
from .common import Adapter, composition_field, mapped_field, prepare_target


class SkyNetAdapter(Adapter):
    format = "skynet"
    description = "SkyNet HDF5 A/Z/Y histories; OSNAP SkyNet-driver bundle v1"

    def detect(self, source):
        try:
            with h5py.File(source, "r") as f:
                return all(name in f for name in ("A", "Z", "Y"))
        except (OSError, TypeError):
            return False

    def read(self, source, *, tracer_ids=None, represented_masses=None, time_reference="simulation_start",
             dataset_names=None, temperature_unit=None, complete_composition=True):
        paths = source if isinstance(source, (list, tuple)) else [source]
        if tracer_ids is not None and len(tracer_ids) != len(paths):
            raise DataError("Supply one tracer ID per SkyNet file")
        if represented_masses is not None and len(represented_masses) != len(paths):
            raise DataError("Supply one represented mass per SkyNet file")
        report, trajectories = Report(), []
        for i, path in enumerate(paths):
            with h5py.File(path, "r") as f:
                a, z, y = np.asarray(f["A"]), np.asarray(f["Z"]), np.asarray(f["Y"])
                if a.ndim != 1 or z.shape != a.shape or y.ndim != 2 or y.shape[1] != len(a):
                    raise FormatError("Expected A/Z vectors and Y(time, species)")
                if np.any(a != a.astype(int)) or np.any(z != z.astype(int)):
                    raise FormatError("SkyNet A and Z must be integers")
                species = [Species.isotope(Z, A) for Z, A in zip(z, a)]
                aliases = {"time": ("Time", "time", "t"), "temperature": ("Temperature", "temperature", "T"),
                           "density": ("Density", "density", "rho"), "electron_fraction": ("Ye", "ye")}
                names = dict(dataset_names or {})
                for canonical, candidates in aliases.items():
                    if canonical not in names:
                        found = [n for n in candidates if n in f]
                        if len(found) > 1:
                            raise FormatError(f"Ambiguous SkyNet {canonical}; specify dataset_names")
                        if found:
                            names[canonical] = found[0]
                if "time" not in names:
                    raise FormatError("SkyNet history lacks a recognized time vector")
                td = f[names["time"]]
                time = Field("time", np.asarray(td), _unit(td, "s"), location="sample", dimensions=("time",)).read()
                if y.shape[0] != len(time):
                    raise FormatError("SkyNet time and abundance lengths differ")
                fields = []
                for name in ("temperature", "density", "electron_fraction"):
                    if name not in names:
                        continue
                    native = f[names[name]]
                    default = {"temperature": temperature_unit, "density": "g/cm^3", "electron_fraction": "1"}[name]
                    unit = _unit(native, default)
                    if unit is None:
                        raise FormatError("SkyNet temperature units are absent; specify temperature_unit (usually GK)")
                    fields.append(mapped_field(names[name], np.asarray(native), dict(name=name, unit=unit, location="sample"), path))
                identity = tracer_ids[i] if tracer_ids is not None else str(Path(path))
                if tracer_ids is None:
                    report.assumptions.append({"tracer_id_from_source_path": identity})
                mass = None if represented_masses is None else Field("represented_mass", represented_masses[i], "g")
                c = composition_field(y, species, path, location="sample", basis="number_per_baryon", complete=complete_composition)
                trajectories.append(Trajectory(identity, time, fields, compositions={"composition": c},
                    represented_mass=mass, time_reference=time_reference, metadata={"source": str(path)}))
                report.unmapped_fields.extend(f"{path}:{key}" for key in f.keys() if key not in {*names.values(), "A", "Z", "Y"})
        dataset = Dataset(tracers={"nucleosynthesis": TracerSet(trajectories)}, report=report)
        validation = dataset.validate("scientific")
        validation.raise_for_errors()
        report.issues.extend(validation.issues)
        return dataset

    def write(self, target, selection, *, composition="composition", overwrite=False):
        if not isinstance(selection, Trajectory):
            raise DataError("SkyNet export requires a single Trajectory selection")
        t = selection
        time = t.time.read(unit="s")
        if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise DataError("SkyNet driver needs at least two strictly increasing times")
        temp, rho = t.field("temperature"), t.field("density")
        temp.require_valid(); rho.require_valid()
        temperature, density = temp.read(unit="GK"), rho.read(unit="g/cm^3")
        if np.any(temperature <= 0) or np.any(density <= 0):
            raise DataError("SkyNet temperature and density must be positive")
        c = t.compositions[composition]
        x = c.mass_fractions().require_valid()[0]
        if not c.complete or any(s.kind != "isotope" for s in c.species) or np.any(x < 0) or not np.isclose(x.sum(), 1, rtol=1e-6, atol=1e-10):
            raise DataError("SkyNet initialization requires a complete, normalized isotope network")
        y = x/np.array([s.A for s in c.species])
        ye = t.field("electron_fraction").require_valid()
        if not np.isclose(ye[0], np.dot(y, [s.Z for s in c.species]), rtol=1e-6, atol=1e-10):
            raise DataError("Initial Ye disagrees with initial abundances; resolve explicitly before export")
        target = prepare_target(target, directory=True, overwrite=overwrite)
        np.savetxt(target/"trajectory.dat", np.column_stack((time, temperature, density, ye)),
                   header="time_s temperature_GK density_g_cm3 electron_fraction", fmt="%.17e")
        np.savetxt(target/"initial_abundances.dat", np.column_stack(([s.Z for s in c.species], [s.A for s in c.species], y)),
                   header="Z A Y_number_per_baryon", fmt=["%d", "%d", "%.17e"])
        manifest = {"format": "OSNAP-SkyNet-driver", "version": "1.0", "tracer_id": t.tracer_id,
                    "time_reference": t.time_reference, "species": [s.to_dict() for s in c.species],
                    "trajectory": "trajectory.dat", "initial_abundances": "initial_abundances.dat"}
        (target/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return Report(transformations=[{"operation": "skynet_driver_bundle", "target": str(target)}],
                      information_loss=["Only the first composition sample initializes the driver"])


def _unit(dataset, default):
    unit = dataset.attrs.get("unit", dataset.attrs.get("units", default))
    return unit.decode() if isinstance(unit, bytes) else unit


def read_driver_bundle(path):
    """Return driver arrays without importing SkyNet or running a network."""
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    if manifest.get("format") != "OSNAP-SkyNet-driver" or manifest.get("version") != "1.0":
        raise FormatError("Unsupported SkyNet driver bundle")
    trajectory = np.loadtxt(path/"trajectory.dat", ndmin=2)
    initial = np.loadtxt(path/"initial_abundances.dat", ndmin=2)
    if trajectory.shape[1] != 4 or initial.shape[1] != 3:
        raise FormatError("Malformed SkyNet driver arrays")
    return {"time": trajectory[:,0], "temperature_GK": trajectory[:,1], "density": trajectory[:,2],
            "electron_fraction": trajectory[:,3], "Z": initial[:,0].astype(int), "A": initial[:,1].astype(int),
            "initial_Y": initial[:,2], "manifest": manifest}
