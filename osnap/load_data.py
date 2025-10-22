"""
Functions for loading and preparing data from STIR profiles, MESA progenitors, and Kepler progenitors.
"""

from scipy.interpolate import RegularGridInterpolator
from progs.progs import ProgModel
from .config import *
import numpy as np
import pandas as pd
import h5py
import yt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_mesa_progenitor(model_name):
    '''Loads a MESA progenitor.'''

    prog = {}
    profile_path = f"{progenitor_directory}/MESA/{model_name}{progenitor_suffix}.data"
    model_path = f"{progenitor_directory}/MESA/{model_name}{progenitor_suffix}.mod"

    # Reads from the .mod file to get necessary header and footer information
    with open(model_path, "r") as file:

        # Copies the first few lines of the header to ensure bit flags and units are preserved
        lines = file.readlines()
        prog["header_start"] = lines[0] + lines[1] + lines[2]

        # Find the table header and from it grab the nuclear network
        for line in lines:
            if "                lnd" in line:
                prog["table_header"] = line
                break
        prog["nuclear_network"] = prog["table_header"].split()[7:]

        # Load each header and footer variable individually
        for i in np.concat((range(4, 20), range(-3, -1))):
            line_data = lines[i].split()
            value = line_data[-1]
            if "D+" in value or "D-" in value: prog[line_data[0]] = float(value.replace("D", "e")) 
            elif value.isdigit(): prog[line_data[0]] = int(value)
            else: prog[line_data[0]] = value

        # These are on the same line so they have to be handled separately
        prog["cumulative_error/total_energy"] = float(lines[20].split()[1].replace("D", "e"))
        prog["log_rel_run_E_err"] = float(lines[20].split()[3].replace("D", "e"))

    # Load profiles of each variable
    with open(profile_path, "r") as file:
        lines = file.readlines()

        # Find the columns that contain the necessary data
        needed_profiles = np.concat((['mass', 'logRho', 'temperature', 'radius_cm', 'luminosity', 
                                    'logdq', 'velocity', 'conv_vel', 'energy', 'pressure'], prog["nuclear_network"]))
        input_column_names = lines[5].split()
        column_indices = [input_column_names.index(col) for col in needed_profiles if col in input_column_names]

        # Create a 2D array containing the numerical data
        structured_data = [list(map(float, np.array(line.split())[column_indices])) for line in lines[6:]]

        # If any columns are missing from the progenitor, fill them with a very small number. 
        # This is a failsafe, but shouldn't be necessary since the composition we need is pulled from the .mod file.
        for i, col in enumerate(needed_profiles):
            if col not in input_column_names:
                print(f"Column {col} missing from progenitor data. Filling with very small number for all cells.")
                for j in range(len(structured_data)): 
                    structured_data[j].insert(i, 1e-99)

        # Write the numerical data into a pandas dataframe, renaming columns for compatibility and ease of use
        output_column_names = np.concat((['enclosed_mass', 'density', 'temp', 'r', 'L', 'dq', velocity_name, 
                                        'mlt_vc', 'ener', 'pressure'], prog["nuclear_network"]))
        prog["profiles"] = pd.DataFrame(structured_data, columns=output_column_names)
        
        # Invert order of zones since loaded MESA data will have first zone as outer radius
        prog["profiles"] = prog["profiles"].iloc[::-1].reset_index(drop=True)

        # Convert data to the correct scale for compatibility with STIR output
        prog["profiles"]["density"] = (10 ** prog["profiles"]['density'])
        prog["profiles"]["dq"] = 10 ** prog["profiles"]['dq']

        # Calculates the progenitor volume and gravitational potential
        prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog["profiles"]['r'].values**3)))
        prog["profiles"] = prog["profiles"].assign(cell_volume = prog_volume) 
        prog["profiles"] = prog["profiles"].assign(gpot = -G * prog["profiles"]['enclosed_mass'].values / prog["profiles"]['r'].values)

    return prog


def load_kepler_progenitor(source, mass):

        prog = {}
        
        prog["profiles"] = ProgModel(str(mass), source, data_dir = progenitor_directory).profile
        prog["profiles"].rename(columns={"radius_edge": "r", "velx_edge": "v", "temperature": "temp", "neutrons": "neut", "luminosity": "L", "energy": "ener"}, inplace=True)
        prog["nuclear_network"] = ['neut', 'h1', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48', 'fe52', 'fe54', 'ni56', 'fe56', 'fe']

        # TODO: Temporary measure until I figure out what to do about missing compositions
        prog["profiles"] = prog["profiles"].assign(prot = np.zeros(prog["profiles"].shape[0]), 
                                                  co56 = np.zeros(prog["profiles"].shape[0]), 
                                                  cr60 = np.zeros(prog["profiles"].shape[0]))

        # Calculates the progenitor volume, enclosed mass, and gravitational potential
        prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog["profiles"]['r'].values**3)))
        prog["profiles"] = prog["profiles"].assign(cell_volume = prog_volume) 
        prog["profiles"] = prog["profiles"].assign(enclosed_mass = np.cumsum(prog["profiles"]['cell_volume'].values * prog["profiles"]['density'].values) / M_sun)
        prog["profiles"] = prog["profiles"].assign(gpot = -G * prog["profiles"]['enclosed_mass'].values / prog["profiles"]['r'].values)

        # TODO: Remove after testing
        #print(prog["profiles"].columns.to_list())

        return prog


def load_stir_profiles(model_name, nuclear_network, post_proc_nuc = None, verbose = True):
    '''Reads in data from a STIR checkpoint (or plot) file, making modifications and returning it as a dataframe.'''

    if verbose: print("Loading STIR checkpoint file and converting to dataframe.")
    
    # Load the STIR checking/plot data into a dataframe
    profile_path = f"{stir_profiles_directory}/{model_name}{stir_profiles_suffix}"
    stir_ds = yt.load(profile_path)
    stir_data = stir_ds.all_data()
    needed_profiles = [("gas", "density"), ("flash", "temp"), ("gas", "r"), ("flash", "velx"), ("gas", "pressure"),# ("flash", "ye  "),
                ("flash", "cell_volume"), ("flash", "ener"), ("flash", "gpot")]
    stir = stir_data.to_dataframe(needed_profiles)

    if verbose: print("Pulling composition from STIR data.")
    
    # Calculate the enclosed mass for every zone
    enclosed_mass = np.cumsum(stir['cell_volume'].values * stir['density'].values) / M_sun
    stir = stir.assign(enclosed_mass = enclosed_mass)
    
    # Try to pull composition from STIR output. If anything is missing, fill it with 1e-99
    # Collect all new columns first to avoid DataFrame fragmentation
    new_columns = {}
    for nuc in nuclear_network: 
        if ('flash', f'{nuc}') in stir_ds.field_list: 
            new_columns[nuc] = stir_data[nuc].v
        else: 
            if verbose: print(f"{nuc} missing from the STIR profiles. Filling with 1e-99.")
            new_columns[nuc] = np.ones(stir.shape[0]) * 1e-99
    
    # Add all new columns at once to avoid fragmentation
    if new_columns:
        stir = pd.concat([stir, pd.DataFrame(new_columns)], axis=1)
    
    # If nucleosynthesis was post-processed and is not yet part of the stir checkpoint file, add it here
    if not(post_proc_nuc is None):
        updated_nuc = pd.read_csv(post_proc_nuc)
        new_cols = {}
        for nuc in updated_nuc.columns[1:]:
            values = np.interp(stir['enclosed_mass'].values, updated_nuc['mass'].values, updated_nuc[nuc].values, left=1e-99, right=1e-99)
            if nuc in stir.columns: stir[nuc] = values
            else: 
                new_cols[nuc] = values
                nuclear_network.append(nuc)
        stir = pd.concat((stir, pd.DataFrame(new_cols)), axis=1)
        if verbose: 
            print("Updating STIR data with post-processed nucleosynthesis data.")
            print(updated_nuc.columns[1:])

    # Renormalize the composition mass fractions since they need to sum to exactly 1 with double precision
    for i in range(stir.shape[0]):
        mass_fraction_sum = stir.loc[i, nuclear_network].sum()
        stir.loc[i, nuclear_network] /= mass_fraction_sum

    if verbose: print("Adjusting and calculating additional values.")

    # To better match MESA outputs, shift radius to cell edge and rename velocity
    stir['r'] = shift_to_cell_edge(stir['r'].values) 
    stir.rename(columns={"velx": velocity_name}, inplace=True)

    # If using cell edge velocity, shift the STIR velocities to the cell edge as well
    if cell_edge_velocity: 
        stir[velocity_name] = shift_to_cell_edge(stir[velocity_name].values)
    stir["ener"] = calculte_total_specific_energy(stir_data)

    return stir

def calculte_total_specific_energy(stir_data):

    # Load the equation of state used by STIR
    EOS = h5py.File(eos_file_path, 'r')
    mif_logenergy = RegularGridInterpolator((EOS['ye'], EOS['logtemp'], EOS['logrho']), 
                                            EOS['logenergy'][:,:,:], bounds_error=False)

    # Use the STIR EOS to calculate the total specific energy
    lye = stir_data['ye  ']
    llogtemp = np.log10(stir_data['temp'] * 8.61733326e-11)
    llogrho = np.log10(stir_data['dens'])
    energy = 10.0 ** mif_logenergy(np.array([lye, llogtemp, llogrho]).T)
    llogtemp = llogtemp * 0.0 - 2.0
    energy0 = 10.0 ** mif_logenergy(np.array([lye, llogtemp, llogrho]).T)

    return (0.5 * stir_data['velx'] ** 2 + (energy - energy0) * yt.units.erg / yt.units.g).v


def shift_to_cell_edge(profile):
    """
    Takes values which are cell centered and shifts them to the edge of the cell.

    Parameters:
        profile (numpy array) :
            e.g. mass density

    Returns:
        shifted_data (numpy array) :
    """

    shifted_data = np.zeros_like(profile)
    shifted_data[:-1] = profile[:-1] + 0.5 * (profile[1:] - profile[:-1])
    shifted_data[-1] = profile[-1] + 0.5 * (profile[-1] - profile[-2])
    return shifted_data