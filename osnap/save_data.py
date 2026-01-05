"""
Functions for saving processed data or converting to formats readable by other programs.
"""

from .load_data import *
from .config import *
from .plotting import *
from .stitching import *
import numpy as np
import gzip
import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def save_fixed_width(df, save_path, padding = 2):
    """
    Save a pandas DataFrame as a fixed-width, human-readable table with optional metadata. Supports gzip when file path ends with '.gz'
    """

    # Double-precision float formatter
    float_fmt = lambda x: f"{np.float64(x):.16e}" if pd.notna(x) else "nan"

    # Render DataFrame to fixed-width text with aligned columns
    table_text = df.to_string(
        index=False,
        float_format=float_fmt,
        col_space=padding
    )

    parent_dir = os.path.dirname(save_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    def open_out(path):
        if path.endswith('.gz'):
            return gzip.open(path, 'wt', encoding='utf-8', newline='')
        return open(path, 'w', encoding='utf-8', newline='')

    with open_out(save_path) as f:
        f.write(table_text + "\n")

    print(f"Saved data to '{save_path}'")


# TODO: Make this less messy, right now you need to specify the progenitor twice if you're using a MESA progenitor
def write_mesa_model(data, prog, model_name, mesa_template = None):
    '''Writes the star's data into MESA input files.
    
    parameters
    ----------
    data : dict
        The data to write to the MESA model file. Typically created using the load_data and stitching modules.
    mesa_template : dict
        The MESA template to use for the model file. Use load_data.load_mesa_progenitor() to load a MESA model \
            and pass that in here. If the progenitor was a MESA model, that would simply be your progenitor. Otherwise, \
            you'll need load a MESA model with the correct and header and column setup.
    model_name : str
        The name of the model file.
    '''

    # Easy way to convert values into the correct format for MESA model files
    def format_float(num):
        return f"{num:.16e}".replace('e', 'D')
    
    def format_int(num):
        return ' ' * (25 - int(np.floor(np.log10(num)))) + str(num)

    if mesa_template is None:
        mesa_template = prog

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
    file_header = mesa_template["header_start"] + f"""
                  version_number   {mesa_template["version_number"]}
                          M/Msun      {format_float(data["total_mass"] / M_sun)}
                    model_number      {format_int(mesa_template["model_number"])}
                        star_age      {format_float(mesa_template["star_age"])}
                       initial_z      {format_float(mesa_template["initial_z"])}
                        n_shells      {format_int(data["profiles"].shape[0])}
                        net_name   {mesa_template["net_name"]}
                         species      {format_int(mesa_template["species"])}
                          xmstar      {format_float(data["xmstar"])}  ! above core (g).  core mass: Msun, grams:      {format_float(data["pns_masscut"])}    {format_float(data["pns_masscut"] * M_sun)}
                        R_center      {format_float(data["pns_radius"])}  ! radius of core (cm).  R/Rsun, avg core density (g/cm^3):      {format_float(data["pns_radius"] / R_sun)}    {format_float(avg_core_density)}
                            Teff      {format_float(mesa_template["Teff"])}
                  power_nuc_burn      {format_float(mesa_template["power_nuc_burn"])}
                    power_h_burn      {format_float(mesa_template["power_h_burn"])}
                   power_he_burn      {format_float(mesa_template["power_he_burn"])}
                    power_z_burn      {format_float(mesa_template["power_z_burn"])}
                     power_photo      {format_float(mesa_template["power_photo"])}
                    total_energy      {format_float(np.sum(data["total_energy"]))}
         cumulative_energy_error      {format_float(mesa_template["cumulative_energy_error"])}
   cumulative_error/total_energy      {format_float(mesa_template["cumulative_error/total_energy"])}  log_rel_run_E_err      {format_float(mesa_template["log_rel_run_E_err"])}
                     num_retries                               0

""" + mesa_template["table_header"]

    # Data that will be written in table form in the stir_output.mod file
    # Also reverses the order of rows in the table so that the first cell is the outer radius and last cell is the center
    mesa_columns = np.concat((['lnd', 'lnT', 'lnR', 'L', 'dq', velocity_name, 'alpha_RTI', 'mlt_vc'], mesa_template["nuclear_network"]))
    data["profiles"]["alpha_RTI"] = np.zeros(data["profiles"].shape[0])
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

               previous n_shells      {format_int(data['profiles'].shape[0])}
           previous mass (grams)      {format_float(mesa_template["M/Msun"] * M_sun)}
              timestep (seconds)      {format_float(mesa_template["timestep"])} 
               dt_next (seconds)      {format_float(mesa_template["dt_next"])}

"""

    # Write all of the above to the stir_output.mod file
    output_path = f"{mesa_export_directory}/{model_name}{output_suffix}.mod"
    with open(f'{output_path}', 'w') as file:
        file.writelines(file_header + '\n'.join(new_lines) + file_footer)
        print(f"Successfully created/updated '{output_path}'")