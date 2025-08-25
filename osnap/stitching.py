"""
Functions for combining progenitor and STIR data.
"""

from .load_data import *
from .plotting import *
from .config import *
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def combine_data(stir, prog, stir_portion):
    """
    Combines data from the STIR domain with the progenitor data outside that domain.
    """

    # Determines the end of the STIR domain and start of the progenitor domain
    stir_domain = stir["r"].values[stir["r"].values <= np.max(stir["r"].values) * stir_portion]
    data = { "stir_domain_end": np.argmax(stir_domain) }
    prog_domain = len(prog['profiles'].loc[prog['profiles']['enclosed_mass'].values > np.max(stir['enclosed_mass'].values[:data["stir_domain_end"]])])

    # Combines STIR and progenitor data, simply placing the progenitor data at the end of the STIR domain
    data["profiles"] = pd.DataFrame(index = pd.RangeIndex(data["stir_domain_end"] + prog_domain), columns = stir.columns, dtype=float)
    for col in stir.columns:
        
        # If the column doesn't exist in progenitor data, notify us and skip it
        if not(col in prog['profiles']):
            print(f"Column {col} missing from progenitor data.")
            continue
        
        # Fill in data for each domain
        data["profiles"][col].values[:data["stir_domain_end"]] = stir[col].values[:data["stir_domain_end"]]
        data["profiles"][col].values[data["stir_domain_end"]:] = prog['profiles'][col].values[-prog_domain:]

    # Determine the total specific energy in each zone
    data["profiles"] = data["profiles"].assign(total_specific_energy = data["profiles"]['ener'].values + data["profiles"]['gpot'].values)

    # Set the PNS mass as the enclosed mass within which all cells have a negative total specific energy
    data["pns_masscut_index"] = np.min(np.where(data["profiles"]['total_specific_energy'].values >= 0)) - 1
    data["pns_masscut"] = data["profiles"]['enclosed_mass'].values[data["pns_masscut_index"]]
    data["pns_radius"] = data["profiles"]['r'].values[data["pns_masscut_index"]]

    # Find the mass of the star, and of the star outside the PNS
    data["total_mass"] = np.sum(data["profiles"]['density'] * data["profiles"]['cell_volume'])
    data["xmstar"] = data["total_mass"] - data["pns_masscut"] * M_sun
    data["profiles"]["dq"] = data["profiles"]['density'] * data["profiles"]['cell_volume'] / data["xmstar"]

    # Excise all zones within the PNS radius, since MESA does not want them
    data["pre_masscut_profiles"] = data["profiles"].copy()
    data["profiles"] = data["profiles"].drop(np.arange(data["pns_masscut_index"] + 1))
    
    # Calculate the remaining energy of the star without the PNS
    data["total_energy"] = data["profiles"]["total_specific_energy"].values * data["profiles"]['density'].values * data["profiles"]['cell_volume'].values

    # Prepares the data for MESA output by adding missing columns
    data["profiles"] = data["profiles"].assign(lnR = np.log(data["profiles"]['r'].values), lnd = np.log(data["profiles"]['density'].values), lnT = np.log(data["profiles"]['temp'].values))
    data["profiles"] = data["profiles"].assign(mlt_vc = np.zeros(data["profiles"].shape[0]))
    
    # MESA needs the surface luminosity which is the same as in progenitor, but all other values should just be a copy of that value
    data["profiles"] = data["profiles"].assign(L = np.ones(data["profiles"].shape[0]) * prog['profiles']["L"].values[-1])

    return data


## TODO: Likely can remove this function after ensuring Kepler progenitors can be used with the combine_data function
# def stitch_kepler(model_name, prog_source, prog_mass, stir_alpha, stir_portion = 0.8, plotted_profiles = ["DEFAULT"]):
#     """
#     Loads in a Kepler progenitor and STIR profiles, making various modifications to make 
#     them compatible, then stitches them together and plots the results.

#     Parameters:
#         model_name (str) :   
#             The name of the model (what comes before .mod or .data in the MESA progenitors).
#         stir_alpha (float) : 
#             The alpha value used for the STIR simulations.
#         stir_portion (float) :
#             What fraction of the STIR domain to include. A value of 0.8 will use progenitor data for the last 20% of the STIR domain.
#         plotted_profiles (numpy array (str)) : 
#             The names of each profile/variable you want to see plotted. Use ["COMPOSITION"] to plot only composition. Default of ["DEFAULT"] will plot enclosed mass, radius, density, temperature, velocity, total specific energy, and pressure.
#             The profiles available for plotting are: enclosed_mass, density, temp, r, L, dq, v, mlt_vc, ener, pressure, and any nuclear network composition
#     """

#     prog = load_kepler_progenitor(prog_source, prog_mass)
#     stir = load_stir_profiles(f"{model_name}_a{stir_alpha}", prog["nuclear_network"])
#     data = combine_data(stir, prog, stir_portion)
    
#     # Plot the desired profiles
#     if "DEFAULT" in plotted_profiles: plotted_profiles = default_plotted_profiles
#     if "COMPOSITION" in plotted_profiles: plotted_profiles = prog["nuclear_network"]
#     for profile in plotted_profiles:
#         plot_profile(data, profile)
    