"""
Functions for converting STIR profiles and their progenitor into a MESA-readable .mod file.
"""

from .load_data import *
from .config import *
from .plotting import *
from .stitching import *
import numpy as np
import json
import gzip
import os
import csv
try:
    import yaml
except Exception:
    yaml = None
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def save_to_json(data, file_path):
    """
    Save the provided dictionary to a JSON file in a human-readable format.

    - Pretty prints with indentation
    - Converts NumPy types/arrays to native Python types
    - Creates parent directories if missing
    - If file_path ends with '.gz', writes gzip-compressed JSON text
    """

    def to_plain_python(obj, seen=None):
        # Convert to JSON-serializable plain Python types, with cycle detection
        if seen is None:
            seen = set()

        # Primitives (fast path)
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # NumPy scalars
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        obj_id = id(obj)
        if obj_id in seen:
            return "<<circular>>"
        seen.add(obj_id)

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return [to_plain_python(x, seen) for x in obj.tolist()]

        # Pandas objects (optional)
        if pd is not None:
            if isinstance(obj, pd.DataFrame):
                # Convert to dict of lists for compactness and readability
                return {str(k): to_plain_python(v, seen) for k, v in obj.to_dict(orient='list').items()}
            if isinstance(obj, pd.Series):
                return to_plain_python(obj.tolist(), seen)
            if isinstance(obj, pd.Index):
                return to_plain_python(obj.tolist(), seen)

        # Mappings
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # Ensure keys are strings
                key_str = k if isinstance(k, str) else str(k)
                out[key_str] = to_plain_python(v, seen)
            return out

        # Sequences
        if isinstance(obj, (list, tuple)):
            return [to_plain_python(x, seen) for x in obj]

        # Sets -> sorted list of stringified elements for stability
        if isinstance(obj, set):
            return sorted([to_plain_python(x, seen) for x in obj], key=lambda x: str(x))

        # Fallback: use string representation
        return str(obj)

    parent_dir = os.path.dirname(file_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    # If data is a JSON string, parse it first to avoid double-encoding
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            plain_data = to_plain_python(parsed)
            json_text = json.dumps(plain_data, indent=2, ensure_ascii=False)
        except Exception:
            # Not valid JSON; write raw text as-is
            json_text = data
    else:
        plain_data = to_plain_python(data)
        json_text = json.dumps(plain_data, indent=2, ensure_ascii=False)

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            f.write(json_text)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_text)

    print(f"Successfully wrote JSON to '{file_path}'")


def save_to_tsv(data, file_path):
    """
    Save tabular data to a TSV file.

    Accepts:
    - pandas.DataFrame
    - dict[str, list] (columns)
    - { "rows": list[dict] } (row records)
    - { "isotopes": list[str], "radius": list[float], "abundances": list[list[float]] }

    Creates parent directories. If path ends with '.gz', writes gzip-compressed TSV.
    """

    def to_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if pd is not None and isinstance(value, (pd.Series, pd.Index)):
            return value.tolist()
        return value

    # Build headers and rows from various accepted input shapes
    headers = []
    rows = []

    if pd is not None and isinstance(data, pd.DataFrame):
        headers = [str(c) for c in data.columns.tolist()]
        rows = [[x if not isinstance(x, (np.floating, np.integer, np.bool_)) else (float(x) if isinstance(x, (np.floating,)) else (int(x) if isinstance(x, (np.integer,)) else bool(x))) for x in r] for r in data.itertuples(index=False, name=None)]

    elif isinstance(data, dict) and "isotopes" in data and "abundances" in data and "radius" in data:
        isotopes = [str(i) for i in to_list(data["isotopes"])]
        radius = to_list(data["radius"])
        abundances = to_list(data["abundances"])  # list of rows
        headers = ["radius"] + isotopes
        rows = []
        for r_val, row_vals in zip(radius, abundances):
            row_clean = [r_val] + [v for v in row_vals]
            rows.append(row_clean)

    elif isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        # Union of keys to form headers in stable order (first row's order preferred)
        first_keys = list(data["rows"][0].keys()) if data["rows"] else []
        key_set = {k for row in data["rows"] for k in row.keys()}
        headers = [str(k) for k in first_keys] + [str(k) for k in key_set if k not in first_keys]
        for row in data["rows"]:
            rows.append([row.get(h, None) for h in headers])

    elif isinstance(data, dict):
        # Assume dict of columns
        columns = {str(k): to_list(v) for k, v in data.items()}
        headers = list(columns.keys())
        lengths = [len(v) for v in columns.values() if isinstance(v, list)]
        if lengths and all(l == lengths[0] for l in lengths):
            n = lengths[0]
            for i in range(n):
                rows.append([columns[h][i] if isinstance(columns[h], list) else columns[h] for h in headers])
        else:
            # Not columnar; write key\tvalue pairs
            headers = ["key", "value"]
            rows = [[k, json.dumps(v, ensure_ascii=False)] for k, v in data.items()]

    else:
        # Fallback: write a single column representation
        headers = ["value"]
        rows = [[str(data)]]

    parent_dir = os.path.dirname(file_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    def open_out(path):
        if path.endswith('.gz'):
            return gzip.open(path, 'wt', encoding='utf-8', newline='')
        return open(path, 'w', encoding='utf-8', newline='')

    with open_out(file_path) as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    print(f"Successfully wrote TSV to '{file_path}'")


def save_stitched_data(df, file_path, metadata=None, pad=2):
    """
    Save a pandas DataFrame as a fixed-width, human-readable table with optional metadata.

    - Writes a comment header with metadata (YAML if available, else JSON)
    - Uses pandas formatting to align columns; floats use double-precision scientific notation
    - Supports gzip when file path ends with '.gz'
    """

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Double-precision float formatter
    float_fmt = lambda x: f"{np.float64(x):.16e}" if pd.notna(x) else "nan"

    # Build metadata header
    header_lines = []
    if metadata:
        header_lines.append("# --- metadata ---")
        if yaml is not None:
            meta_text = yaml.safe_dump(metadata, sort_keys=False).rstrip().splitlines()
        else:
            meta_text = json.dumps(metadata, indent=2, ensure_ascii=False).splitlines()
        header_lines.extend([f"# {line}" if line else "#" for line in meta_text])
        header_lines.append("# --- end metadata ---")

    # Render DataFrame to fixed-width text with aligned columns
    table_text = df.to_string(
        index=False,
        float_format=float_fmt,
        col_space=pad
    )

    parent_dir = os.path.dirname(file_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    def open_out(path):
        if path.endswith('.gz'):
            return gzip.open(path, 'wt', encoding='utf-8', newline='')
        return open(path, 'w', encoding='utf-8', newline='')

    with open_out(file_path) as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write(table_text + "\n")

    print(f"Saved stitched data to '{file_path}'")


def convert_to_mesa(model_name, stir_alpha, progenitor_source = "MESA", stir_portion = 0.8, plotted_profiles = ["DEFAULT"]):
    """
    Loads in a MESA progenitor and STIR profiles, making various modifications to make 
    them compatible, then creates a MESA-readable .mod file.

    Parameters:
        model_name (str) :   
            The name of the model (what comes before .mod or .data in the MESA progenitors).
        stir_alpha (float) : 
            The alpha value used for the STIR simulations.
        stir_portion (float) :
            What fraction of the STIR domain to include. A value of 0.8 will use progenitor data for the last 20% of the STIR domain.
        plotted_profiles (numpy array (str)) : 
            The names of each profile/variable you want to see plotted. Use ["COMPOSITION"] to plot only composition. Default of ["DEFAULT"] will plot enclosed mass, radius, density, temperature, velocity, total specific energy, and pressure.
            The profiles available for plotting are: enclosed_mass, density, temp, r, L, dq, v, mlt_vc, ener, pressure, and any nuclear network composition
    """

    if progenitor_source == "MESA":
        prog = load_mesa_progenitor(model_name)
    elif progenitor_source == "Kepler":
        prog = load_kepler_progenitor(model_name)
        
    stir = load_stir_profiles(f"{model_name}_a{stir_alpha}", prog["nuclear_network"])
    data = combine_data(stir, prog, stir_portion)
    write_mesa_model(data, prog, f"{model_name}_a{stir_alpha}")
    
    # Plot the desired profiles
    if "DEFAULT" in plotted_profiles: plotted_profiles = default_plotted_profiles
    if "COMPOSITION" in plotted_profiles: plotted_profiles = prog["nuclear_network"]
    for profile in plotted_profiles:
        plot_profile(data, profile)


def write_mesa_model(data, prog, model_name):
    '''Writes the star's data into MESA input files.''' 
    
    # Easy way to convert values into the correct format for MESA model files
    def format_float(num):
        return f"{num:.16e}".replace('e', 'D')
    
    def format_int(num):
        return ' ' * (25 - int(np.floor(np.log10(num)))) + str(num)

    # Excise all zones within the PNS radius, since MESA does not want them
    data["profiles"] = data["profiles"].drop(np.arange(data["pns_masscut_index"] + 1))
    
    # Calculate the remaining energy of the star without the PNS
    data["total_energy"] = data["profiles"]["total_specific_energy"].values * data["profiles"]['density'].values * data["profiles"]['cell_volume'].values

    # Prepares the data for MESA output by adding missing columns
    data["profiles"] = data["profiles"].assign(lnR = np.log(data["profiles"]['r'].values), lnd = np.log(data["profiles"]['density'].values), lnT = np.log(data["profiles"]['temp'].values))
    data["profiles"] = data["profiles"].assign(mlt_vc = np.zeros(data["profiles"].shape[0]))
    
    # MESA needs the surface luminosity which is the same as in progenitor, but all other values should just be a copy of that value
    data["profiles"] = data["profiles"].assign(L = np.ones(data["profiles"].shape[0]) * prog['profiles']["L"].values[-1])
    
    # Header Info
    avg_core_density = data["pns_masscut"] * M_sun / (4/3 * np.pi * data["pns_radius"]**3) 
    file_header = prog["header_start"] + f"""
                  version_number   {prog["version_number"]}
                          M/Msun      {format_float(data["total_mass"] / M_sun)}
                    model_number      {format_int(prog["model_number"])}
                        star_age      {format_float(prog["star_age"])}
                       initial_z      {format_float(prog["initial_z"])}
                        n_shells      {format_int(data["profiles"].shape[0])}
                        net_name   {prog["net_name"]}
                         species      {format_int(prog["species"])}
                          xmstar      {format_float(data["xmstar"])}  ! above core (g).  core mass: Msun, grams:      {format_float(data["pns_masscut"])}    {format_float(data["pns_masscut"] * M_sun)}
                        R_center      {format_float(data["pns_radius"])}  ! radius of core (cm).  R/Rsun, avg core density (g/cm^3):      {format_float(data["pns_radius"] / R_sun)}    {format_float(avg_core_density)}
                            Teff      {format_float(prog["Teff"])}
                  power_nuc_burn      {format_float(prog["power_nuc_burn"])}
                    power_h_burn      {format_float(prog["power_h_burn"])}
                   power_he_burn      {format_float(prog["power_he_burn"])}
                    power_z_burn      {format_float(prog["power_z_burn"])}
                     power_photo      {format_float(prog["power_photo"])}
                    total_energy      {format_float(np.sum(data["total_energy"]))}
         cumulative_energy_error      {format_float(prog["cumulative_energy_error"])}
   cumulative_error/total_energy      {format_float(prog["cumulative_error/total_energy"])}  log_rel_run_E_err      {format_float(prog["log_rel_run_E_err"])}
                     num_retries                               0

""" + prog["table_header"]

    # Data that will be written in table form in the stir_output.mod file
    # Also reverses the order of rows in the table so that the first cell is the outer radius and last cell is the center
    mesa_columns = np.concat((['lnd', 'lnT', 'lnR', 'L', 'dq', velocity_name, 'mlt_vc'], prog["nuclear_network"]))
    mesa_input = data["profiles"][mesa_columns].iloc[::-1].reset_index(drop=True)

    # Add one line for each cell, consisting of all it's properties
    new_lines = []
    for line_index in range(data["profiles"].shape[0]):

        # Writes the cell/line index 
        spaces = 4 - int(np.floor(np.log10(line_index + 1)))
        new_line = ' ' * spaces + str(line_index + 1)

        # Writes each of the properties
        for column_name in mesa_input.columns:
            spaces = 5 if mesa_input.at[line_index, column_name] >= 0 else 4
            new_line += spaces * ' ' + format_float(mesa_input.at[line_index, column_name])

        new_lines.append(new_line)

    # Footer containing info about the previous model
    file_footer = f"""
    
        previous model

               previous n_shells      {format_int(prog['profiles'].shape[0])}
           previous mass (grams)      {format_float(prog["M/Msun"] * M_sun)}
              timestep (seconds)      {format_float(prog["timestep"])} 
               dt_next (seconds)      {format_float(prog["dt_next"])}

"""

    # Write all of the above to the stir_output.mod file
    output_path = f"{output_directory}/{model_name}{output_suffix}.mod"
    with open(f'{output_path}', 'w') as file:
        file.writelines(file_header + '\n'.join(new_lines) + file_footer)
        print(f"Successfully created/updated '{output_path}'")