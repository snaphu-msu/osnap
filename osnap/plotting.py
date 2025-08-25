"""
Functions for plotting profiles of data at various stages of the conversion process.
"""

import matplotlib.pyplot as plt
from .config import *
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_profile(data, profile, zoom_width = 80):
    """
    Plots both the full star and a zoomed in region around the point at which STIR and the progenitor are stitched together.
    
    Parameters:
        data (dict) :
            The combined STIR and MESA data containing paramaters useful drawing domains.
        profile (numpy array) :
            The profile to be plotted.
        zoom_width (int) :
            The full width of the zoomed in region around the stitch point.
    """

    # Most plots should use enclosed mass as the x-axis, except the enclosed mass plot
    if profile == "enclosed_mass":
        xaxis = np.arange(data["pre_masscut_profiles"].shape[0])
        xlabel = "zone"
    else:
        #xaxis = np.log10(data["pre_masscut_profiles"]["r"].values)
        xaxis = data["pre_masscut_profiles"]["enclosed_mass"].values
        #xlabel = "log(radius)"
        xlabel = "enclosed_mass (M_sun)"

    ylog = profile in log_plots
    ylabel = f"log({profile})" if ylog else profile
    yaxis = data["pre_masscut_profiles"][profile].values
    if ylog: yaxis = np.log10(yaxis)

    # Plot the entire curve of stitched data
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(xaxis, yaxis)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(f"Full Profile")
    axs[0].axvspan(xaxis[0], xaxis[data["pns_masscut_index"]], color="red", alpha=0.2, ec=None, label = "PNS")
    axs[0].axvspan(xaxis[data["pns_masscut_index"]], xaxis[data["stir_domain_end"]], color="blue", alpha=0.2, ec=None, label = "STIR")
    axs[0].axvspan(xaxis[data["stir_domain_end"]], xaxis[-1], color="green", alpha=0.2, ec=None, label = "MESA")
    axs[0].legend()

    # Plot again but zoomed in on the stitch region
    zoom_left = max(0, data["stir_domain_end"] - zoom_width//2)
    zoom_right = min(data["pre_masscut_profiles"].shape[0], data["stir_domain_end"] + zoom_width//2)
    axs[1].plot(xaxis[zoom_left : zoom_right], yaxis[zoom_left : zoom_right])
    axs[1].axvspan(xaxis[zoom_left], xaxis[data["stir_domain_end"]], color="blue", alpha=0.2, ec=None)
    axs[1].axvspan(xaxis[data["stir_domain_end"]], xaxis[zoom_right], color="green", alpha=0.2, ec=None)
    axs[1].set_title(f"Interface Region")
    
    plt.show()