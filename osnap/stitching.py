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


def combine_data(stir, prog, stir_portion, nuclear_network = None, post_proc_nuc = None, verbose = False):
    """
    Combines data from the STIR domain with the progenitor data outside that domain.
    """

    # Determines the end of the STIR domain and start of the progenitor domain
    stir_domain = stir["r"].values[stir["r"].values <= np.max(stir["r"].values) * stir_portion]
    data = { "stir_domain_end": np.argmax(stir_domain) }
    prog_domain = len(prog['profiles'].loc[prog['profiles']['enclosed_mass'].values > np.max(stir['enclosed_mass'].values[:data["stir_domain_end"]])])
    
    # TODO: Make sure this is good with Sean
    prog_cell_volume = (4 / 3) * np.pi * (np.array(prog['profiles']['r'])[1:] ** 3 - np.array(prog['profiles']['r'])[:-1] ** 3)
    prog['profiles']["cell_volume"] = np.concat(([0], prog_cell_volume))

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
            missing_columns.append(col)
            data["profiles"][col].values[data["stir_domain_end"]:] = 0
            print(f"Column {col} is missing from progenitor data, filling with 0.")
            
    # TODO: Finish making it so that progenitor composition is used where STIR had no data
    if nuclear_network is not None:
        for isotope in nuclear_network:
            if isotope in prog["profiles"].columns:
                values = np.interp(data["profiles"]['enclosed_mass'].values, 
                                prog['profiles']["enclosed_mass"], 
                                prog['profiles'][isotope].values, 
                                left=0, right=0)
                data["profiles"][isotope] = values
                
          
    # If nucleosynthesis was post-processed and is not yet part of the stir checkpoint file, add it here
    if post_proc_nuc is not None:

        composition = xr.load_dataset(post_proc_nuc)
        
        # Only replace values in zones for which we have post-processed data
        mask = (data["profiles"]['enclosed_mass'].values >= composition.coords["mass"].values[0]) & (data["profiles"]['enclosed_mass'].values <= composition.coords["mass"].values[-1]) 

        new_cols = {}
        for isotope_index in range(len(composition.coords["isotope"])):
            isotope = composition.coords["isotope"].values[isotope_index]
            values = np.interp(data["profiles"]['enclosed_mass'].values, 
                               composition.coords["mass"], 
                               composition["X"].values[:, isotope_index, -1])
            
            if isotope in data["profiles"].columns:
                data["profiles"][isotope][mask] = values[mask]
            else: 
                new_cols[isotope] = values
                nuclear_network.append(isotope)
                
        data["profiles"] = pd.concat((data["profiles"], pd.DataFrame(new_cols)), axis=1)
        if verbose: 
            print("Updating stitched data with post-processed nucleosynthesis data.")
            print(composition.coords["isotope"])

    # Renormalize the composition mass fractions since they need to sum to exactly 1
    for i in range(data["profiles"].shape[0]):
        mass_fraction_sum = data["profiles"].loc[i, nuclear_network].sum()
        if mass_fraction_sum > 0:
            data["profiles"].loc[i, nuclear_network] /= mass_fraction_sum

    if verbose:
        print("Columns in STIR data but missing from progenitor data:", missing_columns)
    
    # Determine the total specific energy in each zone
    mass_col_index = data["profiles"].columns.get_loc('enclosed_mass')
    data["profiles"].insert(mass_col_index + 1, 'total_specific_energy', data["profiles"]['ener'].values + data["profiles"]['gpot'].values)

    # print("ener", np.isnan(data["profiles"]["ener"].values).any())
    # print("gpot", np.isnan(data["profiles"]["gpot"].values).any())
    # print("ener + gpot", np.isnan(data["profiles"]['total_specific_energy'].values).any())
    # print(data["profiles"]['total_specific_energy'].values)
    # print(len(data["profiles"]['total_specific_energy'].values >= 0))
    # print(np.where(data["profiles"]['total_specific_energy'].values >= 0))
    
    # Set the PNS mass as the enclosed mass within which all cells have a negative total specific energy
    data["pns_masscut_index"] = np.min(np.where(data["profiles"]['total_specific_energy'].values >= 0)) - 1
    data["pns_masscut"] = data["profiles"]['enclosed_mass'].values[data["pns_masscut_index"]]
    data["pns_radius"] = data["profiles"]['r'].values[data["pns_masscut_index"]]
    
    print(f"PNS mass cut: {data['pns_masscut']} M_sun, PNS radius: {data['pns_radius']} cm")
    
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
