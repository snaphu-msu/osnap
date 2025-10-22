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
    #print(stir["xe136"].values)
    data["profiles"] = pd.DataFrame(index = pd.RangeIndex(data["stir_domain_end"] + prog_domain), columns = stir.columns, dtype=float)
    #print(data["profiles"]["xe136"].values)
    missing_columns = []
    for col in stir.columns:
        
        # If the column doesn't exist in progenitor data, notify us and skip it
        if not(col in prog['profiles']):
            missing_columns.append(col)
            continue
        
        # Fill in data for each domain
        data["profiles"][col].values[:data["stir_domain_end"]] = stir[col].values[:data["stir_domain_end"]]
        data["profiles"][col].values[data["stir_domain_end"]:] = prog['profiles'][col].values[-prog_domain:]

    print("Columns in STIR data but missing from progenitor data:", missing_columns)

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