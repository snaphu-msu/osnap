"""
Functions for plotting profiles of data at various stages of the conversion process.
"""

import matplotlib.pyplot as plt
from .config import *
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_profile(data, profile, save_path = None, force_log = False):
    """
    Plots both the full star and a zoomed in region around the point at which STIR and the progenitor are stitched together.
    
    Parameters:
        data (dict) :
            The combined STIR and progenitor data.
        profile (numpy array) :
            The profile to be plotted.
        zoom_width (int) :
            The full width of the zoomed in region around the stitch point.
    """

    # Most plots should use enclosed mass as the x-axis, except the enclosed mass plot
    if profile == "enclosed_mass":
        xaxis = np.arange(data["profiles"].shape[0])
        xlabel = "zone"
    else:
        xaxis = data["profiles"]["enclosed_mass"].values
        xlabel = "enclosed_mass (M_sun)"

    ylog = profile in log_plots or force_log
    ylabel = f"log({profile})" if ylog else profile
    yaxis = data["profiles"][profile].values
    if ylog: yaxis = np.log10(yaxis)

    def plot_data(ax, left_lim, right_lim, title):
        ax.plot(xaxis, yaxis)
        ax.axvspan(xaxis[0], xaxis[data["pns_masscut_index"]], color="red", alpha=0.2, ec=None, label = "PNS")
        ax.axvspan(xaxis[data["pns_masscut_index"]], xaxis[data["stir_domain_end"]], color="blue", alpha=0.2, ec=None, label = "STIR")
        ax.axvspan(xaxis[data["stir_domain_end"]], xaxis[-1], color="green", alpha=0.2, ec=None, label = "Progenitor")
        ax.set_xlim(left_lim, right_lim)
        ax.set_title(title)

    # Plot the entire curve of stitched data
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    plot_data(axs[0], xaxis[0], xaxis[-1], "Full Profile")
    axs[0].set_ylabel(ylabel)
    axs[0].legend()

    # Plot again but zoomed in on the PNS-STIR connection
    stir_width = xaxis[data["stir_domain_end"]] - xaxis[data["pns_masscut_index"]]
    zoom_left = max(0, xaxis[data["pns_masscut_index"]] - stir_width / 2)
    zoom_right = min(np.max(xaxis), xaxis[data["stir_domain_end"]] + stir_width / 2)
    plot_data(axs[1], zoom_left, zoom_right, "Zoomed Region")

    # Plot again but zoomed in on the STIR-Progenitor region
    zoom_left = max(0, xaxis[data["stir_domain_end"]] - stir_width / 2)
    zoom_right = min(np.max(xaxis), xaxis[data["stir_domain_end"]] + stir_width / 2)
    plot_data(axs[2], zoom_left, zoom_right, "Progenitor Interface")
    axs[2].set_xlabel(xlabel)

    fig.tight_layout()
    
    if save_path is not None: 
        fig.savefig(save_path)
    else: 
        plt.show()