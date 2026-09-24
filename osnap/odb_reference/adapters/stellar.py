"""Bounded MESA text and KEPLER presupernova text adapters."""
from pathlib import Path
import io
import re
import shlex
import numpy as np

from ..composition import Species
from ..errors import DataError, FormatError
from ..fields import Field
from ..model import RadialGrid, Snapshot, Report
from ..units import DEFAULT_UNITS
from .common import Adapter, composition_field, finish, mapped_field, numeric_table, require_snapshot, checked_composition, prepare_target


def _number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def mesa_table(path):
    lines = Path(path).read_text().splitlines()
    numbered = []
    for i, line in enumerate(lines[:-1]):
        tokens = line.split()
        if tokens and all(t.isdigit() for t in tokens) and [int(t) for t in tokens] == list(range(1, len(tokens)+1)):
            numbered.append(i)
    if len(numbered) < 2:
        raise FormatError("Expected MESA header and data column-number rows")
    header_row, table_row = numbered[:2]
    names = lines[header_row+1].split()
    values = shlex.split(lines[header_row+2])
    if len(names) != len(values):
        raise FormatError("MESA header names and values differ")
    header = {}
    for name, value in zip(names, values):
        try:
            header[name] = _number(value)
        except ValueError:
            header[name] = value
    columns = lines[table_row+1].split()
    data = numeric_table(path, skiprows=table_row+2)
    if data.shape[1] != len(columns):
        raise FormatError("MESA data columns do not match header")
    return header, {name: data[:,i] for i, name in enumerate(columns)}


def mesa_model(path):
    lines = Path(path).read_text().splitlines()
    flag_row = next((i for i, line in enumerate(lines) if "-- model for mesa/star" in line), None)
    table_row = next((i for i, line in enumerate(lines) if line.split()[:3] == ["lnd", "lnT", "lnR"]), None)
    if flag_row is None or table_row is None:
        raise FormatError("Not a supported CGS MESA saved-model text file")
    flags = int(lines[flag_row].split()[0])
    allowed = (1<<2)|(1<<3)|(1<<5)|(1<<7)|(1<<9)
    if flags & ~allowed:
        raise FormatError("Unsupported MESA model flags (rotation/RSP/obsolete structure)")
    header = {}
    for line in lines[flag_row+1:table_row]:
        tokens = shlex.split(line.split("!",1)[0])
        if len(tokens) >= 2:
            try:
                header[tokens[0]] = _number(tokens[1])
            except ValueError:
                header[tokens[0]] = tokens[1]
            if len(tokens) >= 4 and tokens[2] == "log_rel_run_E_err":
                header[tokens[2]] = _number(tokens[3])
    required = ("n_shells", "species", "net_name", "M/Msun", "version_number")
    if any(k not in header for k in required):
        raise FormatError("Incomplete MESA saved-model header")
    columns = lines[table_row].split()
    count = int(header["n_shells"])
    text = "\n".join(lines[table_row+1:table_row+1+count]).replace("D", "E")
    values = np.loadtxt(io.StringIO(text), ndmin=2)
    if values.shape != (count, len(columns)+1) or not np.array_equal(values[:,0], np.arange(1,count+1)):
        raise FormatError("MESA model zone indices/columns do not match header")
    expected = ["lnd", "lnT", "lnR", "L", "dq"]
    for bit, name in ((3,"v"),(9,"u"),(7,"alpha_RTI"),(5,"mlt_vc")):
        if flags & (1<<bit):
            expected.append(name)
    if columns[:-int(header["species"])] != expected:
        raise FormatError("MESA model flags and hydro columns disagree")
    return header, {name: values[:,i+1] for i,name in enumerate(columns)}, lines, flag_row, table_row


def _outer_grid(radius, mass, *, inner_radius, inner_mass, full_star, metadata=None):
    if inner_radius is None:
        if not full_star:
            raise FormatError("Specify inner_radius or full_star=True; the inner boundary cannot be inferred")
        inner_radius = 0.
    if mass is not None and inner_mass is None:
        if not full_star:
            raise FormatError("Specify inner_mass for an excised stellar profile")
        inner_mass = 0.
    return RadialGrid(np.r_[inner_radius, radius], mass_faces=None if mass is None else np.r_[inner_mass, mass],
                      frame="lagrangian", metadata=metadata)


class MESAAdapter(Adapter):
    format = "mesa"
    description = "MESA numbered-header profiles/history and CGS nonrotating saved models"
    mappings = {
        "logRho": dict(name="density", unit="g/cm^3", transform="log10", sampling="average"),
        "density": dict(name="density", unit="g/cm^3", sampling="average"),
        "logT": dict(name="temperature", unit="K", transform="log10", sampling="average"),
        "temperature": dict(name="temperature", unit="K", sampling="average"),
        "pressure": dict(name="pressure", unit="dyn/cm^2", sampling="average"),
        "ye": dict(name="electron_fraction", unit="1", sampling="average"),
        "velocity": dict(name="radial_velocity", unit="cm/s", location="face"),
        "luminosity": dict(name="luminosity", unit="L_sun", location="face"),
        "conv_vel": dict(name="convective_velocity", unit="cm/s"),
    }

    def detect(self, source):
        try:
            lines = Path(source).read_text().splitlines()[:10]
            return any("model for mesa/star" in line or "version_number" in line for line in lines)
        except (OSError, UnicodeError, TypeError):
            return False

    def read(self, source, *, kind="profile", inner_radius=None, inner_mass=None, full_star=False,
             inner_fields=None, field_map=None, complete_composition=False, time_reference="stellar_age"):
        paths = source if isinstance(source, (list, tuple)) else [source]
        report, snapshots = Report(), []
        for path in paths:
            is_model = kind == "model" or "model for mesa/star" in Path(path).read_text()[:4096]
            if is_model:
                header, columns, *_ = mesa_model(path)
            else:
                header, columns = mesa_table(path)
            if kind == "history":
                if "star_age" not in columns:
                    raise FormatError("MESA history requires star_age")
                maps = {"star_mass": dict(name="stellar_mass", unit="M_sun"),
                        "log_L": dict(name="luminosity", unit="L_sun", transform="log10"),
                        "log_Teff": dict(name="effective_temperature", unit="K", transform="log10"), **(field_map or {})}
                report.unmapped_fields.extend(c for c in columns if c not in maps and c != "star_age")
                for i, age in enumerate(columns["star_age"]):
                    fields = [mapped_field(c, v[i], {**maps[c], "location": "global"}, path) for c,v in columns.items() if c in maps]
                    snapshots.append(Snapshot(fields, time=age*DEFAULT_UNITS.factor("yr", "s"), source_id=str(path), metadata={"mesa_header": header}))
                continue
            if is_model:
                outer_r = np.exp(columns["lnR"])[::-1]
                total = header["M/Msun"]*DEFAULT_UNITS.factor("M_sun", "g")
                envelope = header.get("xmstar", total)
                m0 = max(0., total-envelope)
                outer_m = m0 + envelope*np.cumsum(columns["dq"][::-1])
                grid = _outer_grid(outer_r, outer_m, inner_radius=header.get("R_center", 0.), inner_mass=m0, full_star=False)
                maps = {"lnd": dict(name="density", unit="g/cm^3", transform="ln", sampling="average"),
                        "lnT": dict(name="temperature", unit="K", transform="ln", sampling="average"),
                        "L": dict(name="luminosity", unit="erg/s", location="face"),
                        "v": dict(name="radial_velocity", unit="cm/s", location="face"),
                        "u": dict(name="radial_velocity", unit="cm/s"),
                        "alpha_RTI": dict(name="alpha_RTI", unit="1"),
                        "mlt_vc": dict(name="convective_velocity", unit="cm/s"), **(field_map or {})}
                bounds = {"v": header.get("v_center", 0.), "L": header.get("L_center", 0.), **(inner_fields or {})}
                species_labels = list(columns)[-int(header["species"]):]
                complete = True
                ignored = {"lnR", "dq"}
            else:
                if "radius_cm" in columns:
                    outer_r = columns["radius_cm"][::-1]
                elif "radius" in columns:
                    outer_r = columns["radius"][::-1]*DEFAULT_UNITS.factor("R_sun", "cm")
                elif "logR" in columns:
                    outer_r = 10.**columns["logR"][::-1]*DEFAULT_UNITS.factor("R_sun", "cm")
                else:
                    raise FormatError("MESA profile has no recognized outer-face radius")
                outer_m = columns["mass"][::-1]*DEFAULT_UNITS.factor("M_sun", "g") if "mass" in columns else None
                grid = _outer_grid(outer_r, outer_m, inner_radius=inner_radius, inner_mass=inner_mass, full_star=full_star)
                maps = {**self.mappings, **(field_map or {})}
                # Prefer linear columns when both log and linear representations exist.
                if "density" in columns:
                    maps.pop("logRho", None)
                if "temperature" in columns:
                    maps.pop("logT", None)
                bounds, complete, species_labels = inner_fields or {}, complete_composition, []
                ignored = {"zone", "mass", "radius", "radius_cm", "logR"}
                for label in columns:
                    if label not in maps and label not in ignored:
                        try:
                            if Species.parse(label).kind == "isotope":
                                species_labels.append(label)
                        except DataError:
                            pass
            fields = []
            for native, spec in maps.items():
                if native not in columns:
                    continue
                values = columns[native][::-1]
                valid = None
                if spec.get("location") == "face":
                    value = bounds.get(native, np.nan)
                    values = np.r_[value, values]
                    valid = np.r_[np.isfinite(value), np.ones(len(values)-1, dtype=bool)]
                    if not valid[0]:
                        report.issues.append({"severity": "warning", "path": str(path), "message": f"Unknown inner-face {native}; retained as invalid"})
                fields.append(mapped_field(native, values, spec, path, valid=valid))
            compositions = {}
            if species_labels:
                compositions["composition"] = composition_field(np.column_stack([columns[c][::-1] for c in species_labels]), species_labels, path, complete=complete)
            report.unmapped_fields.extend(f"{path}:{c}" for c in columns if c not in maps and c not in species_labels and c not in ignored)
            report.transformations.append({"operation": "reverse_zones", "source": str(path)})
            if full_star:
                report.assumptions.append({"full_star": True, "source": str(path)})
            age = header.get("star_age")
            snapshots.append(Snapshot(fields, grid=grid, compositions=compositions,
                time=None if age is None else age*DEFAULT_UNITS.factor("yr", "s"), source_id=str(path),
                metadata={"mesa_header": header, "time_reference": time_reference}))
        return finish(snapshots, report, name="history" if kind == "history" else "progenitor", time_reference=time_reference)

    def write(self, target, selection, *, template, header_values, composition="composition", overwrite=False):
        s = require_snapshot(selection)
        original, columns, lines, flag_row, table_row = mesa_model(template)
        c, x = checked_composition(s, composition, isotopes=True)
        labels = list(columns)[-int(original["species"]):]
        desired = [Species.parse(label).id for label in labels]
        if set(desired) != {sp.id for sp in c.species}:
            raise DataError("MESA template and composition networks must match exactly")
        g = s.grid
        if g.mass_faces is None:
            raise DataError("MESA model export requires enclosed-mass faces")
        m = g.mass_faces.require_valid()
        envelope = m[-1]-m[0]
        if envelope <= 0:
            raise DataError("MESA model must have positive envelope mass")
        required_headers = {"star_age", "Teff", "power_nuc_burn", "power_h_burn", "power_he_burn",
                            "power_z_burn", "power_photo", "total_energy", "cumulative_energy_error",
                            "cumulative_error/total_energy", "log_rel_run_E_err"} & set(original)
        if not required_headers <= set(header_values):
            raise DataError(f"Supply physical header_values in native units: {sorted(required_headers-set(header_values))}")
        updates = {**header_values, "M/Msun": m[-1]/DEFAULT_UNITS.factor("M_sun","g"),
                   "n_shells": g.ncells, "species": len(labels), "xmstar": envelope,
                   "R_center": g.radius_faces.read(0)}
        arrays = {}
        for name in columns:
            if name in labels:
                arrays[name] = x[:, [sp.id for sp in c.species].index(Species.parse(name).id)]
            elif name in ("lnd", "lnT"):
                field = s.field("density" if name == "lnd" else "temperature", location="cell")
                values = field.require_valid()
                if np.any(values <= 0):
                    raise DataError("MESA logarithmic structure values must be positive")
                arrays[name] = np.log(values)
            elif name == "lnR":
                arrays[name] = np.log(g.radius_faces.read()[1:])
            elif name == "dq":
                arrays[name] = np.diff(m)/envelope
            elif name in ("L", "v"):
                f = s.field("luminosity" if name == "L" else "radial_velocity", location="face")
                values = f.require_valid()
                arrays[name] = values[1:]
                updates["L_center" if name == "L" else "v_center"] = values[0]
            else:
                canonical = {"u": "radial_velocity", "mlt_vc": "convective_velocity"}.get(name, name)
                arrays[name] = s.field(canonical, location="cell").require_valid()
        if any(v.shape != (g.ncells,) for v in arrays.values()):
            raise DataError("MESA output columns must contain one value per zone")
        flags = int(lines[flag_row].split()[0]) & ~(1<<2)
        output = lines[:flag_row] + [str(flags)+" -- model for mesa/star; CGS; exported by OSNAP", ""]
        merged = {**original, **updates}
        for name, value in merged.items():
            if name == "log_rel_run_E_err":
                continue
            if name == "cumulative_error/total_energy":
                output.append(f"{name:>32}  {float(value):.17e}  log_rel_run_E_err  {float(merged['log_rel_run_E_err']):.17e}")
            elif isinstance(value, str):
                output.append(f"{name:>32}  '{value}'")
            elif name in ("n_shells", "species", "model_number", "num_retries"):
                output.append(f"{name:>32}  {int(value)}")
            else:
                if not np.isfinite(value):
                    raise DataError(f"Non-finite MESA header value {name}")
                output.append(f"{name:>32}  {float(value):.17e}")
        output.extend(["", " ".join(columns)])
        for i in range(g.ncells):
            output.append(str(i+1)+" "+" ".join(f"{arrays[c][-1-i]:.17e}" for c in columns))
        target = prepare_target(target, overwrite=overwrite)
        target.write_text("\n".join(output)+"\n")
        return Report(transformations=[{"operation": "mesa_model_export", "target": str(target), "template": str(template)}],
                      information_loss=["Previous-model timestep state removed; export starts a new model generation"])


class KEPLERAdapter(Adapter):
    format = "kepler"
    description = "Sukhbold/Heger presupernova ASCII tables with named columns"

    def detect(self, source):
        try:
            return "NETWORK" in Path(source).read_text()[:4096]
        except (OSError, UnicodeError, TypeError):
            return False

    def read(self, source, *, inner_radius=None, inner_mass=None, full_star=False,
             complete_composition=False, field_map=None, columns=None, species_names=None):
        lines = Path(source).read_text().splitlines()
        header = next((line for line in lines if "NETWORK" in line.upper()), None)
        if columns is None:
            if header is None:
                raise FormatError("KEPLER requires a NETWORK header or explicit column names")
            columns = header.lstrip("#").split()
            if columns[0].lower() in ("grid", "zone", "cell"):
                pass
            else:
                raise FormatError("KEPLER header must identify the zone-index column")
        rows = []
        for line in lines:
            tokens = line.split()
            if tokens and tokens[0].rstrip(":").isdigit():
                if len(tokens) != len(columns):
                    raise FormatError("KEPLER row width differs from header; no implicit abundance padding")
                rows.append([np.nan if t == "---" else _number(t.rstrip(":")) for t in tokens])
        if not rows:
            raise FormatError("No KEPLER profile rows")
        values = np.array(rows)
        data = {c.lower(): values[:,i] for i,c in enumerate(columns)}
        def col(*names):
            for name in names:
                if name in data:
                    return data[name]
            raise FormatError(f"KEPLER table lacks columns {names}")
        radius, mass = col("radius", "r"), col("mass", "m")
        grid = _outer_grid(radius, mass, inner_radius=inner_radius, inner_mass=inner_mass, full_star=full_star)
        maps = {"density": dict(name="density", unit="g/cm^3", sampling="average"),
                "temp": dict(name="temperature", unit="K", sampling="average"),
                "temperature": dict(name="temperature", unit="K", sampling="average"),
                "ye": dict(name="electron_fraction", unit="1", sampling="average"),
                "velocity": dict(name="radial_velocity", unit="cm/s", location="face"), **(field_map or {})}
        fields = []
        for name, spec in maps.items():
            if name in data:
                v = data[name]
                if spec.get("location") == "face":
                    v = np.r_[np.nan, v]
                fields.append(mapped_field(name, v, spec, source, valid=np.isfinite(v)))
        upper = [c.upper() for c in columns]
        labels = species_names or (columns[upper.index("NETWORK")+1:] if "NETWORK" in upper else [])
        compositions = {}
        if labels:
            abundance = np.column_stack([data[c.lower()] for c in labels])
            compositions["composition"] = composition_field(abundance, labels, source, groups=("fe",),
                complete=complete_composition, valid=np.isfinite(abundance))
        report = Report(unmapped_fields=[c for c in columns if c.lower() not in maps and c not in labels and c.lower() not in ("mass", "radius", "grid", "zone", "network")])
        if full_star:
            report.assumptions.append({"full_star": True})
        return finish([Snapshot(fields, grid=grid, compositions=compositions, source_id=str(source))], report, name="progenitor", time_reference="stellar_age")
