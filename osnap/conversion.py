"""
Functions for converting STIR profiles and their progenitor into a MESA-readable .mod file.
"""

from .load_data import *
from .config import *
from .plotting import *
from .stitching import *
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def convert(model_name, stir_alpha, stir_portion = 0.8, plotted_profiles = ["DEFAULT"]):
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

    prog = load_mesa_progenitor(model_name)
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