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


def combine_data(stir, prog, stir_portion, verbose = False):
    """
    Combines data from the STIR domain with the progenitor data outside that domain.
    """

    # Determines the end of the STIR domain and start of the progenitor domain
    stir_domain = stir["r"].values[stir["r"].values <= np.max(stir["r"].values) * stir_portion]
    data = { "stir_domain_end": np.argmax(stir_domain) }
    prog_domain = len(prog['profiles'].loc[prog['profiles']['enclosed_mass'].values > np.max(stir['enclosed_mass'].values[:data["stir_domain_end"]])])

    # Combines STIR and progenitor data, simply placing the progenitor data at the end of the STIR domain
    data["profiles"] = pd.DataFrame(index = pd.RangeIndex(data["stir_domain_end"] + prog_domain), columns = stir.columns, dtype=float)
    missing_columns = []
    for col in stir.columns:

        # Fill in data for the STIR domain
        data["profiles"][col].values[:data["stir_domain_end"]] = stir[col].values[:data["stir_domain_end"]]
        
        # If the profile exists in the progenitor data, fill in the progenitor data
        if col in prog['profiles']:
            data["profiles"][col].values[data["stir_domain_end"]:] = prog['profiles'][col].values[-prog_domain:]
        else:
            # TODO: Should I fill this in with zeros or extremely small numbers or leave it as NaN?
            missing_columns.append(col)

    if verbose:
        print("Columns in STIR data but missing from progenitor data:", missing_columns)

    # Determine the total specific energy in each zone
    mass_col_index = data["profiles"].columns.get_loc('enclosed_mass')
    data["profiles"].insert(mass_col_index + 1, 'total_specific_energy', data["profiles"]['ener'].values + data["profiles"]['gpot'].values)

    # Set the PNS mass as the enclosed mass within which all cells have a negative total specific energy
    data["pns_masscut_index"] = np.min(np.where(data["profiles"]['total_specific_energy'].values >= 0)) - 1
    data["pns_masscut"] = data["profiles"]['enclosed_mass'].values[data["pns_masscut_index"]]
    data["pns_radius"] = data["profiles"]['r'].values[data["pns_masscut_index"]]

    # Find the mass of the star, and of the star outside the PNS
    data["total_mass"] = np.sum(data["profiles"]['density'] * data["profiles"]['cell_volume'])
    data["xmstar"] = data["total_mass"] - data["pns_masscut"] * M_sun
    data["profiles"].insert(mass_col_index + 2, 'dq', data["profiles"]['density'] * data["profiles"]['cell_volume'] / data["xmstar"])

    # Calculate the compactness parameters for various target masses
    data["compactness_1.75"] = calculate_compactness(1.75, prog["profiles"]['enclosed_mass'].values, prog["profiles"]['r'].values)
    data["compactness_2.0"] = calculate_compactness(2.0, prog["profiles"]['enclosed_mass'].values, prog["profiles"]['r'].values)
    data["compactness_2.5"] = calculate_compactness(2.5, prog["profiles"]['enclosed_mass'].values, prog["profiles"]['r'].values)

    return data

def calculate_compactness(target_mass, mass_profile, radius_profile):
    """
    Calculates the compactness parameter at a target mass for a given mass and radius profile.
    Interpolates the radius for when data does not exist exactly at the target mass.
    """
    mass_index = np.argmin(np.where(mass_profile >= target_mass))
    upper_mass, lower_mass = mass_profile[mass_index], mass_profile[mass_index - 1]
    upper_radius, lower_radius = radius_profile[mass_index], radius_profile[mass_index - 1]
    radius = lower_radius + (upper_radius - lower_radius) * (target_mass - lower_mass) / (upper_mass - lower_mass)
    return target_mass / (radius / 1e8)
