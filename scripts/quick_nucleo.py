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
    num_tracers = 100 # TODO: Find ideal tracer count
    zams_mass = 20.0
    alpha = 1.25

    # Generate paths to the different data files
    run_date = "14may19"
    base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{alpha}/run_{zams_mass}"
    model_name = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}"

    # Get some necessary data from the STIR checkpoint and dat files
    _, shock_radius, explosion_energy = np.loadtxt(base_path + "/" + model_name + ".dat", unpack=True, usecols=(0, 11, 9))
    last_checkpoint = base_path + "/output/" + sorted([f for f in os.listdir(base_path + "/output") if "chk" in f])[-1]
    stir_data = yt.load(last_checkpoint).all_data()

    # Do not continue if the model did not explode
    if explosion_energy[-1] <= 0:
        raise ValueError("Model did not explode. Nucleosynthesis cannot be calculated.")

    # Set the minimum tracer mass as the PNS mass, which is determined as the first unbound mass element
    total_specific_energy = load_data.calculte_total_specific_energy(stir_data) + stir_data['flash', 'gpot'].value
    enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
    pns_masscut_index = np.min(np.where(total_specific_energy >= 0)) - 1
    min_tracer_mass = enclosed_mass[pns_masscut_index]
    print("Min Tracer Mass:", min_tracer_mass)

    # Set the maximum tracer mass as the outer edge of the shock at the final time
    enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
    max_tracer_mass = enclosed_mass[np.argmin(np.abs(stir_data['gas', 'r'].value - shock_radius[-1]))]
    print("Max Tracer Mass:", max_tracer_mass)

    ### 
    # TODO:
    # 1. Find which models explode.
    # 2. Find ideal tracer count. Use high number as most accurate ground truth.
    #    Run for varying num_tracers values, recording the abundances of each element for each value.
    #    Find point where error (M_truth - M_test) / M_truth is below a threshold
    #    Log-log plot of error vs tracer count
    ###

    start_time = time.time()

    print("Loading the progenitor")
    progenitor = load_data.load_kepler_progenitor("sukhbold_2016", zams_mass)

    print("Loading tracer data")
    tracers = fb.load_save.get_tracers(
        run = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}", 
        model = f"run_{zams_mass}",
        model_set = f"run_{run_date}_a{alpha}",
        mass_grid = np.linspace(min_tracer_mass, max_tracer_mass, num_tracers), 
        reload = force_reload_tracers,
        config = "stir"
    )

    print("Beginning nucleosynthesis calculations")
    output = sky.do_nucleosynthesis(
        model_path = base_path, 
        stir_model = model_name, 
        progenitor = progenitor["model"],
        domain_radius = 1e9,
        tracers = tracers,
        output_path = f"./skynet_output/{run_date}_a{alpha}_run_{zams_mass}",
        verbose = 1
    )

    # Save the traer data to a csv file
    if not os.path.exists('../data/nucleosynthesis'):
        os.makedirs('../data/nucleosynthesis')
    output.to_csv(f"../data/nucleosynthesis/{run_date}_m{zams_mass}_a{alpha}_n{num_tracers}.csv")

    # Print the time taken to run the nucleosynthesis
    time_elapsed = time.time() - start_time
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    hour_string = f"{hours:.0f}" if hours >= 10 else f"0{hours}"
    min_string = f"{minutes:.0f}" if minutes >= 10 else f"0{minutes}"
    sec_string = f"0{seconds:.2f}" if seconds >= 10 else f"0{seconds:.2f}"
    time_elapsed_str = f"{hour_string}:{min_string}:{sec_string}"
    print("Nucleosynthesis complete!")
    print(f"Time Taken: {time_elapsed_str}", f"(per tracer: {(time_elapsed / num_tracers):.2f}s)")

    # Load the stir output, overwriting isotope abundances with post-processed nucleosynthesis data
    print("Loading STIR profiles w/ added nucleosynthesis data")
    stir_data = load_data.load_stir_profiles(
        model_name, 
        progenitor["nuclear_network"], 
        f"../data/nucleosynthesis/{run_date}_m{zams_mass}_a{alpha}_n{num_tracers}.csv",
        verbose=False)


    # Add the progenitor data outside the stir domain onto the stir output
    print("Stitching progenitor onto STIR profiles")
    stitched = stitching.combine_data(stir_data, progenitor, stir_portion=0.9)

    # Save the final stitched data into it's own file.
    print("Saving final stitched data")
    save_data.save_stitched_data(stitched["profiles"], f"../data/nucleosynthesis/stitched_{model_name}")
