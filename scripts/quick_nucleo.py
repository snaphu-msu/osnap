from osnap import *
import os
import os.path
import flashbang as fb
import nucleosynth.nucleo as nuc
import numpy as np
import time
import xarray as xr
import argparse
import matplotlib.pyplot as plt

# Hide the yt output
import yt
yt.set_log_level(50)

def run_model(zams_mass, alpha, num_tracers, rerun_tracers = False):

    rerun_stitching = True

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
    total_specific_energy = load_data.calculate_total_specific_energy(stir_data) + stir_data['flash', 'gpot'].value
    enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
    pns_masscut_index = np.min(np.where(total_specific_energy >= 0))
    min_tracer_mass = enclosed_mass[pns_masscut_index]
    print("Min Tracer Mass:", min_tracer_mass)

    # Set the maximum tracer mass as the outer edge of the shock at the final time
    enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
    max_tracer_mass = enclosed_mass[np.argmin(np.abs(stir_data['gas', 'r'].value - shock_radius[-1]))]
    print("Max Tracer Mass:", max_tracer_mass)
    print(f"Total Tracer Mass: {max_tracer_mass - min_tracer_mass}")

    print("Loading the progenitor")
    progenitor = load_data.load_kepler_progenitor("sukhbold_2016", zams_mass)
    
    # TODO: Interpolate ye_weights to reduce it's array length to the specified number of tracers
    if num_tracers == None:
        ye_weights = 1 - (np.abs(stir_data["ye  "].value - 0.5) / 0.5)
        tracer_mass = (ye_weights / np.sum(ye_weights)) * (max_tracer_mass - min_tracer_mass)
        mass_grid = np.cumsum(tracer_mass) + min_tracer_mass
        print("Using variable tracer mass, total tracer count: ", len(mass_grid))
        
    else:
        mass_grid = np.linspace(min_tracer_mass, max_tracer_mass, num_tracers)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    color = 'tab:red'
    axs[0].set_xlabel('Zone')
    axs[0].set_ylabel('Tracer Mass (M_sun)', color=color)
    axs[0].plot(range(len(tracer_mass)), tracer_mass, color = color)
    axs[0].tick_params(axis = 'y', labelcolor = color)
    #axs[0].set_title("")

    color = 'tab:blue'
    ax2 = axs[0].twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel('y_e', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(tracer_mass)), stir_data["ye  "], color = color)
    ax2.axhline(0.5, color = color, linestyle = "--")
    ax2.tick_params(axis = 'y', labelcolor = color)
    
    axs[1].hist(tracer_mass, bins=20, color='tab:blue')
    axs[1].set_xlabel('Tracer Mass (M_sun)')
    axs[1].set_ylabel('Count')
    axs[1].semilogy()
    
    fig.suptitle(f"{np.median(tracer_mass):.2e} ± {np.std(tracer_mass):.2e} M_sun Resolution")
    fig.tight_layout()
    plt.savefig("mass_grid.png")
    
    tracer_string = f"n{num_tracers}" if num_tracers != None else "variable"
    if num_tracers == None: num_tracers = len(mass_grid)

    nucleo_output_path = f"{config.nucleo_results_directory}/{run_date}_m{zams_mass}_a{alpha}_{tracer_string}"
    stitched_output_path = f"{config.stitched_output_directory}/stitched_{model_name}_{tracer_string}"

    # If this model has already been run completely, skip this model
    if os.path.exists(stitched_output_path) and not(rerun_stitching):
        print(f"Complete data already exists. Skipping the {zams_mass} model with {tracer_string} tracers...")
        return

    start_time = time.time()

    print("Loading tracer data")
    tracers = fb.load_save.get_tracers( # TODO: Make it so flashbang can handle custom save locations for cached tracers
        run = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}", 
        model = f"run_{zams_mass}",
        model_set = f"run_{run_date}_a{alpha}",
        mass_grid = mass_grid,
        reload = rerun_tracers,
        config = "stir"
    )
    
    nuc_start = time.time()

    print("Beginning nucleosynthesis calculations")
    output = nuc.do_nucleosynthesis(
        model_path = base_path, 
        stir_model = model_name, 
        progenitor = progenitor["model"],
        domain_radius = 1e9,
        tracers = tracers,
        output_path = f"./skynet_output/{run_date}_a{alpha}_run_{zams_mass}_{tracer_string}",
        isotopes_file = f"{config.isotope_list_file}",
        verbose = 1
    )
    
    print(f"Nucleosynthesis took {time.time() - nuc_start:.2f} seconds to load")

    output.to_netcdf(nucleo_output_path)

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
        stir_profile_path = last_checkpoint,
        post_proc_nuc = nucleo_output_path,
        verbose=False
    )

    # Add the progenitor data outside the stir domain onto the stir output
    print("Stitching progenitor onto STIR profiles")
    stitched = stitching.combine_data(stir_data, progenitor, stir_portion=0.9)

    # Save the final stitched data into it's own file.
    print("Saving final stitched data")
    save_data.save_fixed_width(stitched["profiles"], stitched_output_path)

if __name__ == "__main__":
    
    # Parses command line arguments so that we can post-process multiple models in sequence if desired.
    # Note: To do this much faster when submitting jobs to a cluster, you'll want to run nucleosynthesis on each model separately
    #       You can do this by running (and possibly modifying) the bulk_nucleo.py script.
    parser = argparse.ArgumentParser(description='Run nucleosynthesis calculations')
    parser.add_argument('-t', '--tracer-count', type=int, help='Number of tracers, leave blank to use variable tracer mass')
    parser.add_argument('-m', '--mass', type=str, required=True, help='ZAMS mass')
    parser.add_argument('-a', '--alpha', type=float, default=1.25, help='Alpha values (default: 1.25)')
    parser.add_argument('-r', '--recreate-tracers', action='store_true', help='Forces all tracer data to be recreated even if there is existing usable data cached.')
    args = parser.parse_args()
    
    print(f"Running for mass: {args.mass}")
    print(f"Running for alpha: {args.alpha}")
    print(f"Running for num_tracers: {args.tracer_count}")

    # Post-process for each model requested, one by one
    run_model(args.mass, args.alpha, args.tracer_count, rerun_tracers = args.recreate_tracers)
    
    print(f"Completed mass {args.mass}, alpha {args.alpha}, tracers {args.tracer_count}")


