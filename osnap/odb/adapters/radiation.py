"""SNEC text products and TARDIS simple_ascii models/spectra."""
from pathlib import Path
import re
import numpy as np

from ..composition import Species
from ..errors import DataError, FormatError
from ..fields import Axis, Field
from ..model import RadialGrid, Snapshot, Report
from ..units import DEFAULT_UNITS
from .common import Adapter, numeric_table, composition_field, finish, mapped_field, history, require_snapshot, checked_composition, prepare_target
from .stellar import _outer_grid


class TARDISAdapter(Adapter):
    format = "tardis"
    description = "simple_ascii density/elemental abundance inputs and ASCII spectra"

    def detect(self, source):
        try:
            words = Path(source).read_text().splitlines()[0].split()
            return len(words) == 2 and words[1] in ("day", "s") and float(words[0]) > 0
        except (OSError, UnicodeError, ValueError, IndexError, TypeError):
            return False

    def read(self, source, *, kind="model", abundance_path=None, abundance_format="labelled",
             epoch=None, spectral_unit=None, wavelength_unit="Angstrom", spectral_name="luminosity_density",
             complete_composition=True):
        if kind == "spectrum":
            if spectral_unit is None:
                raise FormatError("ASCII spectra require explicit spectral_unit; flux and luminosity are not interchangeable")
            table = numeric_table(source)
            if table.shape[1] != 2:
                raise FormatError("Expected wavelength and spectral-density columns")
            axis = Axis("wavelength", table[:,0], wavelength_unit)
            field = Field(spectral_name, table[:,1], spectral_unit, location="sample", dimensions=("wavelength",),
                          source={"location": str(source)}, metadata={"spectral_coordinate": "wavelength"})
            return finish([Snapshot([field], axes={"wavelength": axis}, time=epoch, source_id=str(source))],
                          Report(), name="spectra", time_reference="explosion")
        if kind != "model":
            raise FormatError("TARDIS kind must be model or spectrum")
        first = Path(source).read_text().splitlines()[0].split()
        if len(first) != 2:
            raise FormatError("TARDIS model starts with '<epoch> <unit>'")
        time = float(first[0])*DEFAULT_UNITS.factor(first[1], "s")
        if time <= 0:
            raise FormatError("TARDIS expansion epoch must be positive")
        data = numeric_table(source, skiprows=1)
        if data.shape[1] != 3 or not np.array_equal(data[:,0], np.arange(len(data))):
            raise FormatError("TARDIS density rows need sequential index, boundary velocity, density")
        velocity = data[:,1]*1e5
        grid = RadialGrid(velocity*time, frame="lagrangian", metadata={"expansion": "homologous", "epoch_s": time})
        fields = [Field("radial_velocity", velocity, "cm/s", location="face", dimensions=("face",)),
                  Field("density", data[1:,2], "g/cm^3", location="cell", dimensions=("cell",), sampling="average")]
        compositions = {}
        if abundance_path:
            if abundance_format == "labelled":
                labels = Path(abundance_path).read_text().splitlines()[0].lstrip("#").split()
                x = numeric_table(abundance_path, skiprows=1)
            elif abundance_format == "indexed":
                abundance = numeric_table(abundance_path)
                if not np.array_equal(abundance[:,0], np.arange(len(abundance))):
                    raise FormatError("TARDIS abundance shell indices must start at zero")
                x = abundance[:,1:]
                labels = [Species(f"element:{z}", str(z), "element", Z=z) for z in range(1,x.shape[1]+1)]
            else:
                raise FormatError("Specify abundance_format='labelled' or 'indexed'")
            if x.shape != (grid.ncells, len(labels)):
                raise FormatError("TARDIS abundance rows correspond to shells, not boundary rows")
            compositions["composition"] = composition_field(x, labels, abundance_path, complete=complete_composition)
        report = Report(transformations=[{"operation": "boundary_row", "ignored_density_row": 0}],
                        assumptions=[{"operation": "homologous_radius", "epoch_s": time}])
        return finish([Snapshot(fields, grid=grid, compositions=compositions, time=time,
                               metadata={"expansion": "homologous", "time_reference": "explosion"}, source_id=str(source))],
                      report, name="ejecta", time_reference="explosion")

    def write(self, target, selection, *, epoch, homologous=False, composition="composition", overwrite=False,
              abundance_format="indexed"):
        s = require_snapshot(selection)
        if not np.isfinite(epoch) or epoch <= 0:
            raise DataError("TARDIS export requires positive epoch in seconds since explosion")
        if not homologous and s.metadata.get("expansion") != "homologous":
            raise DataError("Declare homologous=True or supply a model marked as homologous")
        expected = s.grid.radius_faces.read()/epoch
        try:
            velocity = s.field("radial_velocity", location="face").require_valid()
            if not np.allclose(velocity, expected, rtol=1e-6, atol=0):
                raise DataError("Native velocities/radii disagree with the supplied homologous epoch")
        except KeyError:
            if not homologous:
                raise DataError("Constructing boundary velocities requires homologous=True") from None
            velocity = expected
        rho = s.field("density", location="cell").require_valid()
        c, _ = checked_composition(s, composition)
        elemental = c.elemental_sums()
        if any(sp.Z is None or sp.Z == 0 or sp.Z > 30 for sp in elemental.species):
            raise DataError("The indexed TARDIS dialect supports elements Z=1..30 and no free neutrons")
        if abundance_format not in ("indexed", "labelled"):
            raise FormatError("Unsupported TARDIS abundance format")
        fractions = elemental.abundances.require_valid()
        if abundance_format == "indexed":
            abundance = np.zeros((s.grid.ncells,30))
            for i, sp in enumerate(elemental.species):
                abundance[:,sp.Z-1] = fractions[:,i]
        else:
            abundance = fractions
        target = prepare_target(target, directory=True, overwrite=overwrite)
        with (target/"density.dat").open("w") as f:
            f.write(f"{epoch:.17e} s\n")
            np.savetxt(f, np.column_stack((np.arange(len(velocity)), velocity/1e5, np.r_[0.,rho])), fmt=["%d","%.17e","%.17e"])
        if abundance_format == "indexed":
            np.savetxt(target/"abundances.dat", np.column_stack((np.arange(s.grid.ncells), abundance)), fmt=["%d"]+["%.17e"]*30)
        else:
            with (target/"abundances.dat").open("w") as f:
                f.write(" ".join(sp.source_label for sp in elemental.species)+"\n")
                np.savetxt(f, abundance, fmt="%.17e")
        return Report(transformations=[{"operation": "tardis_export", "epoch_s": epoch, "abundance_format": abundance_format}],
                      assumptions=["Homologous expansion", "Unrepresented elements are zero in the declared complete network"],
                      information_loss=["Isotopic abundances aggregated to elements"] if any(sp.kind=="isotope" for sp in c.species) else [])


def _xg(path):
    blocks, rows, time = [], [], None
    for line in Path(path).read_text().splitlines():
        if "Time" in line and "=" in line:
            if time is not None:
                blocks.append((time, np.array(rows)))
            time, rows = float(line.split("=",1)[1].strip().strip('"').replace("D","E")), []
        elif line.strip():
            if time is None:
                raise FormatError("SNEC .xg data must follow a Time header")
            row = [float(x.replace("D","E")) for x in line.split()]
            if len(row) != 2:
                raise FormatError("SNEC .xg blocks require two columns")
            rows.append(row)
    if time is not None:
        blocks.append((time,np.array(rows)))
    if not blocks or any(len(row) < 2 for _,row in blocks):
        raise FormatError("Empty SNEC profile blocks")
    return blocks


class SNECAdapter(Adapter):
    format = "snec"
    description = "SNEC .short/.iso.dat inputs, .xg histories, scalar and magnitude time series"
    mappings = {
        "rho": dict(name="density", unit="g/cm^3", sampling="average"),
        "temp": dict(name="temperature", unit="K", sampling="average"),
        "ye": dict(name="electron_fraction", unit="1", sampling="average"),
        "press": dict(name="pressure", unit="dyn/cm^2", sampling="average"),
        "vel": dict(name="radial_velocity", unit="cm/s", location="face"),
        "lum": dict(name="luminosity", unit="erg/s", location="face"),
        "eps": dict(name="specific_internal_energy", unit="erg/g", sampling="average"),
    }

    def detect(self, source):
        try:
            path = Path(source)
            return (path.is_dir() and (path/"radius.xg").exists()) or path.suffix == ".short"
        except TypeError:
            return False

    def read(self, source, *, kind="model", composition_path=None, inner_radius=None, inner_mass=None,
             full_star=False, coordinate_location=None, complete_composition=True, fields=None,
             columns=None, field_map=None, time_reference="simulation_start", magnitude_system=None):
        if kind == "history":
            if field_map is None:
                known = {"lum_observed.dat": ("luminosity","erg/s"), "T_eff.dat": ("effective_temperature","K"),
                         "mass_photo.dat": ("photospheric_mass","g"), "rad_photo.dat": ("photospheric_radius","cm"),
                         "vel_photo.dat": ("photospheric_velocity","cm/s")}
                if Path(source).name not in known:
                    raise FormatError("Supply columns and field_map for this SNEC diagnostic")
                name, unit = known[Path(source).name]
                columns, field_map = ["time",name], {name:dict(name=name,unit=unit)}
            return history(source, columns=columns, field_map=field_map, time_reference=time_reference, name="light_curve")
        if kind == "magnitudes":
            if not columns or columns[:2] != ["time", "temperature"] or not magnitude_system:
                raise FormatError("Supply ordered columns=['time','temperature', ...bands] and magnitude_system")
            data = numeric_table(source)
            if data.shape[1] != len(columns):
                raise FormatError("Magnitude band labels do not match columns")
            snapshots = [Snapshot([
                Field("magnitude", row[2:], "mag", dimensions=("band",), location="sample", metadata={"magnitude_system": magnitude_system}),
                Field("effective_temperature", row[1], "K")], axes={"band": Axis("band", columns[2:])}, time=row[0]) for row in data]
            return finish(snapshots, Report(), name="light_curve", time_reference=time_reference)
        if kind == "profiles":
            directory = Path(source)
            radius = _xg(directory/"radius.xg")
            names = fields or [name for name in self.mappings if (directory/f"{name}.xg").exists()]
            maps = {**self.mappings, **(field_map or {})}
            if any(name not in maps for name in names):
                raise FormatError("Unknown SNEC profile units/centering; supply field_map")
            tables = {name:_xg(directory/f"{name}.xg") for name in names}
            if any(len(table) != len(radius) for table in tables.values()):
                raise FormatError("SNEC profile files have different time samples")
            snapshots = []
            for i,(time,coordinates) in enumerate(radius):
                g = RadialGrid(coordinates[:,1], mass_faces=coordinates[:,0], frame="lagrangian")
                data_fields = []
                for name,table in tables.items():
                    instant, data = table[i]
                    if instant != time or data.shape != coordinates.shape or not np.allclose(data[:,0],coordinates[:,0],rtol=1e-12,atol=0):
                        raise FormatError("SNEC profile times/mass coordinates are not aligned")
                    spec = maps[name]
                    values = data[:,1] if spec.get("location") == "face" else data[:-1,1]
                    data_fields.append(mapped_field(name, values, spec, directory/f"{name}.xg"))
                snapshots.append(Snapshot(data_fields, grid=g, time=time, source_id=str(directory)))
            report = Report(transformations=[{"operation":"remove_outer_ghost_cell", "fields":[n for n in names if maps[n].get("location")!="face"]}],
                unmapped_fields=[str(p) for p in directory.glob("*.xg") if p.stem not in {*names,"radius"}])
            return finish(snapshots, report, time_reference=time_reference)
        if kind != "model" or coordinate_location != "outer_face":
            raise FormatError("SNEC model import requires coordinate_location='outer_face' for the supported progenitor table convention")
        lines = Path(source).read_text().splitlines()
        n = int(lines[0])
        data = numeric_table(source,skiprows=1)
        if data.shape != (n,8) or not np.array_equal(data[:,0],np.arange(1,n+1)):
            raise FormatError("Expected .short rows: index mass radius temperature density velocity Ye Omega")
        g = _outer_grid(data[:,2],data[:,1],inner_radius=inner_radius,inner_mass=inner_mass,full_star=full_star)
        fields_out = [mapped_field(native,data[:,column],dict(name=name,unit=unit,sampling="average" if name in ("density","temperature","electron_fraction") else "point"),source)
                      for column,native,name,unit in [(3,"T","temperature","K"),(4,"rho","density","g/cm^3"),
                      (5,"velocity","radial_velocity","cm/s"),(6,"Ye","electron_fraction","1"),(7,"Omega","angular_velocity","1/s")]]
        compositions = {}
        if composition_path:
            clines = Path(composition_path).read_text().splitlines()
            nc, ns = map(int,clines[0].split())
            a, z = np.array(clines[1].split(),dtype=float), np.array(clines[2].split(),dtype=float)
            comp = numeric_table(composition_path,skiprows=3)
            if nc != n or len(a) != ns or len(z) != ns or comp.shape != (n,ns+2) or not np.allclose(comp[:,:2],data[:,1:3],rtol=1e-12,atol=0):
                raise FormatError("SNEC composition/structure coordinates do not match; remap explicitly")
            species = [Species.isotope(Z,A) for Z,A in zip(z,a)]
            compositions["composition"] = composition_field(comp[:,2:],species,composition_path,complete=complete_composition)
        return finish([Snapshot(fields_out,grid=g,compositions=compositions,source_id=str(source))],
            Report(assumptions=[{"coordinate_location":"outer_face","full_star":full_star}]),name="progenitor",time_reference=time_reference)

    def write(self,target,selection,*,composition="composition",coordinate_location,overwrite=False):
        s = require_snapshot(selection)
        if coordinate_location != "outer_face":
            raise FormatError("Supported SNEC export convention is coordinate_location='outer_face'")
        if s.grid.mass_faces is None:
            raise DataError("SNEC export requires enclosed-mass faces")
        c,x = checked_composition(s,composition,isotopes=True)
        cols = [np.arange(1,s.grid.ncells+1), s.grid.mass_faces.read()[1:], s.grid.radius_faces.read()[1:]]
        for name in ("temperature","density","radial_velocity","electron_fraction","angular_velocity"):
            cols.append(s.field(name,location="cell").require_valid())
        target = prepare_target(target,directory=True,overwrite=overwrite)
        with (target/"model.short").open("w") as f:
            f.write(f"{s.grid.ncells}\n")
            np.savetxt(f,np.column_stack(cols),fmt=["%d"]+["%.17e"]*7)
        with (target/"composition.iso.dat").open("w") as f:
            f.write(f"{s.grid.ncells} {len(c.species)}\n")
            f.write(" ".join(str(sp.A) for sp in c.species)+"\n")
            f.write(" ".join(str(sp.Z) for sp in c.species)+"\n")
            np.savetxt(f,np.column_stack((cols[1],cols[2],x)),fmt="%.17e")
        return Report(transformations=[{"operation":"snec_export","target":str(target)}],
                      assumptions=["Progenitor table uses outer-shell coordinates; SNEC performs its own input mapping"])
