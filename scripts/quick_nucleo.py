from osnap import *
import os
import os.path
import flashbang as fb
import nucleosynth.nucleo as nuc
import numpy as np
import time
import argparse

# Hide the yt output
import yt
yt.set_log_level(50)

def run_model(zams_mass, alpha, num_tracers, rerun_tracers=False, rerun_nucleo=False, rerun_stitching=False):

    # Generate paths to the different data files
    run_date = "14may19"
    base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{alpha}/run_{zams_mass}"
    model_name = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}"
    nucleo_output_path = f"{config.nucleo_results_directory}/{run_date}_m{zams_mass}_a{alpha}_n{num_tracers}"
    stitched_output_path = f"{config.stitched_output_directory}/stitched_{model_name}_n{num_tracers}"

    # If this model has already been run completely, skip this model
    if os.path.exists(stitched_output_path) and not(rerun_stitching):
        print(f"Complete data already exists. Skipping the {zams_mass} model with {num_tracers} tracers...")
        return

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
    pns_masscut_index = np.min(np.where(total_specific_energy >= 0)) - 1
    min_tracer_mass = enclosed_mass[pns_masscut_index]
    print("Min Tracer Mass:", min_tracer_mass)

    # Set the maximum tracer mass as the outer edge of the shock at the final time
    enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
    max_tracer_mass = enclosed_mass[np.argmin(np.abs(stir_data['gas', 'r'].value - shock_radius[-1]))]
    print("Max Tracer Mass:", max_tracer_mass)

    print("Loading the progenitor")
    progenitor = load_data.load_kepler_progenitor("sukhbold_2016", zams_mass)

    if not(os.path.exists(nucleo_output_path)) or rerun_nucleo or rerun_stitching or rerun_tracers:

        start_time = time.time()

        print("Loading tracer data")
        tracers = fb.load_save.get_tracers( # TODO: Make it so flashbang can handle custom save locations
            run = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}", 
            model = f"run_{zams_mass}",
            model_set = f"run_{run_date}_a{alpha}",
            mass_grid = np.linspace(min_tracer_mass, max_tracer_mass, num_tracers), 
            reload = rerun_tracers,
            config = "stir"
        )

        print("Beginning nucleosynthesis calculations")
        output = nuc.do_nucleosynthesis(
            model_path = base_path, 
            stir_model = model_name, 
            progenitor = progenitor["model"],
            domain_radius = 1e9,
            tracers = tracers,
            output_path = f"./skynet_output/{run_date}_a{alpha}_run_{zams_mass}_n{num_tracers}",
            verbose = 1
        )

        save_data.save_fixed_width(output, nucleo_output_path)

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

    else:
        print("Nucleosynthesis data already exists. Skipping post_processing...")

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
    
    parser = argparse.ArgumentParser(description='Run nucleosynthesis calculations')
    parser.add_argument('-t', '--tracer-count', nargs='+', type=int, default=[100],
                        help='Number of tracers (can specify multiple values for multiple runs), default: 100')
    parser.add_argument('-m', '--masses', nargs='+', type=str, required=True,
                        help='ZAMS masses (can specify multiple values, or use "all" for all masses)')
    parser.add_argument('-a', '--alphas', nargs='*', type=float, default=[1.25],
                        help='Alpha values (can specify multiple values, default: 1.25)')
    parser.add_argument('-f', '--force-reload', action='store_true',
                        help='Forces all tracer data, nucleosynthesis, and stiching to be rerun even if there is existing usable data cached.')
    
    args = parser.parse_args()
    
    # Handle "all" for masses
    if args.masses[0] == "all":
        masses = np.array(["9.0", "9.25", "9.5", "9.75", "10.0", "10.25", "10.5", "10.75", "11.0", "11.25", "11.5", "11.75", 
                    "12.0", "12.25", "12.5", "12.75", "13.0", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", 
                    "13.8", "13.9", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.8", "14.9", 
                    "15.0", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8", "15.9", "16.0", "16.1", 
                    "16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "17.0", "17.1", "17.2", "17.3", 
                    "17.4", "17.5", "17.6", "17.7", "17.8", "17.9", "18.0", "18.1", "18.2", "18.3", "18.4", "18.5", 
                    "18.6", "18.7", "18.8", "18.9", "19.0", "19.1", "19.2", "19.3", "19.4", "19.5", "19.6", "19.7", 
                    "19.8", "19.9", "20.0", "20.1", "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "20.8", "20.9", 
                    "21.0", "21.1", "21.2", "21.3", "21.4", "21.5", "21.6", "21.7", "21.8", "21.9", "22.0", "22.1", 
                    "22.2", "22.3", "22.4", "22.5", "22.6", "22.7", "22.8", "22.9", "23.0", "23.1", "23.2", "23.3", 
                    "23.4", "23.5", "23.6", "23.7", "23.8", "23.9", "24.0", "24.1", "24.2", "24.3", "24.4", "24.5", 
                    "24.6", "24.7", "24.8", "24.9", "25.0", "25.1", "25.2", "25.3", "25.4", "25.5", "25.6", "25.7", 
                    "25.8", "25.9", "26.0", "26.1", "26.2", "26.3", "26.4", "26.5", "26.6", "26.7", "26.8", "26.9", 
                    "27.0", "27.1", "27.2", "27.3", "27.4", "27.5", "27.6", "27.7", "27.8", "27.9", "28.0", "28.1", 
                    "28.2", "28.3", "28.4", "28.5", "28.6", "28.7", "28.8", "28.9", "29.0", "29.1", "29.2", "29.3", 
                    "29.4", "29.5", "29.6", "29.7", "29.8", "29.9", "30.0", "31", "32", "33", "35", "40", "45", "50", 
                    "55", "60", "70", "80", "100", "120"])
    else:
        masses = np.array(args.masses, dtype=str)
    
    print(f"Running for masses: {masses}")
    print(f"Running for alphas: {args.alpha}")
    print(f"Running for num_tracers: {args.num_tracers}")

    # Run nested loops for all combinations
    for alpha in args.alpha:
        for num_tracers in args.num_tracers:
            for zams_mass in masses:
                run_model(zams_mass, alpha, num_tracers, 
                         rerun_stitching=args.force_reload, 
                         rerun_nucleo=args.force_reload, 
                         rerun_tracers=args.force_reload)
                print(f"Completed mass {zams_mass}, alpha {alpha}, tracers {num_tracers}")


