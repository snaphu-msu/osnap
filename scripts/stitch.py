from osnap import *
import os.path
import flashbang as fb
import nucleosynth.nucleo as sky
import xarray as xr
import numpy as np
import progs.progs as progs
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

# Hide the yt output
import yt
yt.set_log_level(50)

if __name__ == "__main__":

    force_reload_tracers = False
    num_tracers = 100
    zams_mass = 20.0
    alpha = 1.25

    run_date = "14may19"
    base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{alpha}/run_{zams_mass}"
    model_name = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}"

    progenitor = load_data.load_kepler_progenitor("sukhbold_2016", zams_mass)
    stir_Data = load_data.load_stir_profiles(
        "stir2_14may19_s20.0_alpha1.25", 
        progenitor["nuclear_network"], 
        "~/_main/projects/osnap/data/nucleosynthesis/14may19_m20.0_a1.25_tracers100.csv",
        verbose=False)


    stitched = stitching.combine_data(stir_Data, progenitor, stir_portion=0.9)
    print(stitched["profiles"]["xe136"].values)
    #save_data.save_to_json(stitched["profiles"], f"../data/nucleosynthesis/stitched_{model_name}.json")
    save_data.save_to_fixed_table(stitched["profiles"], f"../data/nucleosynthesis/stitched_{model_name}.txt")
    #element = str(sys.argv[1])
    #plotting.plot_profile(stitched, element, save_path=f"plots/enclosed_mass_{element}.png", force_log = True)
    #print(f"Total {element} Mass", np.sum(stitched["profiles"]["cell_volume"] * stitched["profiles"]["density"] * stitched["profiles"][element]) / 1.989E33)

    # ASK: If I save the final stitched data, what format should I save it in?
    # TODO: Save final data in 