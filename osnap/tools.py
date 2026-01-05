"""
Functions for quickly running portions of OSNAP, such as converting to MESA format.
"""

from .load_data import *
from .config import *
from .plotting import *
from .stitching import *
from .save_data import *

def convert_to_mesa(model_name, stir_alpha, progenitor_source = "MESA", stir_portion = 0.8, plotted_profiles = ["DEFAULT"]):
    """
    Loads in a progenitor and STIR profiles, making various modifications to make 
    them compatible, then creates a MESA-readable .mod file.

    Parameters:
        model_name (str) :   
            The name of the model (what comes before .mod or .data in the MESA progenitors).
        progenitor_source (str) :
            The source of the progenitor data. Supported formats are "MESA" or "Kepler".
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
    stitched_data = combine_data(stir, prog, stir_portion)
    write_mesa_model(stitched_data, prog, f"{model_name}_a{stir_alpha}")
    
    # Plot the desired profiles
    if "DEFAULT" in plotted_profiles: plotted_profiles = default_plotted_profiles
    if "COMPOSITION" in plotted_profiles: plotted_profiles = prog["nuclear_network"]
    for profile in plotted_profiles:
        plot_profile(stitched_data, profile)