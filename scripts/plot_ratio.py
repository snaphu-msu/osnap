from osnap import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib.lines as mlines
import yt
yt.set_log_level(50)

# Plot the yields for a given isotope (or set of isotopes) vs the specified variable
if __name__ == "__main__":

    run_date = "14may19"

    # Allows the user to specify which models are plotted, as well as how it's displayed and saved
    # TODO: Implement using atomic number for x-axis.
    parser = argparse.ArgumentParser(description='Plot mass fractions for given isotopes, masses, and alpha values.')
    parser.add_argument('-n', '--name', nargs=1, type=str, required = True, help='Name of the saved plot file.')
    parser.add_argument('-i', '--isotopes', nargs=2, type=str, required = True, help='Isotopes to plot the ratio of. First is on the x-axis, second is on the y-axis.')
    parser.add_argument('-m', '--masses', nargs='*', type=str, default=["20.0"], 
                        help='ZAMS masses (can specify multiple values, or use "all" for all masses listed in the config file, default: "20.0")')
    parser.add_argument('-a', '--alpha', type=str, default="1.25", help='Alpha values (default: "1.25")')
    parser.add_argument('-t', '--tracer-count', type=str, default="1000", help='Number of tracers (default: "1000")')
    parser.add_argument('-e', '--external-yields', type=str, help='Name of an optional external yields file. Must be in main data directory as specified in the config file.')
    args = parser.parse_args()

    if args.masses[0] == "all":
        args.masses = config.all_masses

    yields = pd.DataFrame(columns=["set", "zams_mass", args.isotopes[0], args.isotopes[1], f"{args.isotopes[0]}_error", f"{args.isotopes[1]}_error"])
    for mass in args.masses:

        # Load in the stitched data for the given parameters
        path = f"{config.stitched_output_directory}/stitched_stir2_{run_date}_s{mass}_alpha{args.alpha}_n{args.tracer_count}"
        if not os.path.exists(path):
            print(f"stitched_stir2_{run_date}_s{mass}_alpha{args.alpha}_n{args.tracer_count} does not exist. Skipping...")
            continue
        data = pd.read_csv(path, sep='\s+')

        # TODO: Cleanup by ideally storing this as metadata in the stitched files, and then accesing that metadata here
        # Load in the STIR checkpoint data
        base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{args.alpha}/run_{mass}"
        model_name = f"stir2_{run_date}_s{mass}_alpha{args.alpha}"
        _, shock_radius = np.loadtxt(base_path + "/" + model_name + ".dat", unpack=True, usecols=(0, 11))
        last_checkpoint = base_path + "/output/" + sorted([f for f in os.listdir(base_path + "/output") if "chk" in f])[-1]
        stir_data = yt.load(last_checkpoint).all_data()

        # Calculate the PNS mass and exclude it from the ejecta mass calculations
        total_specific_energy = load_data.calculate_total_specific_energy(stir_data) + stir_data['flash', 'gpot'].value
        enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
        pns_masscut_index = np.min(np.where(total_specific_energy >= 0))
        pns_mass = enclosed_mass[pns_masscut_index]
        data = data[data["enclosed_mass"] > pns_mass]

        # Calculate the total eject mass for the given isotopes
        ejecta_mass_1 = np.sum(data[args.isotopes[0]] * data["density"] * data["cell_volume"]) / config.M_sun
        ejecta_mass_2 = np.sum(data[args.isotopes[1]] * data["density"] * data["cell_volume"]) / config.M_sun
        yields = pd.concat([yields, pd.DataFrame([["STIR", float(mass), ejecta_mass_1, ejecta_mass_2, 0, 0]], columns=yields.columns)], ignore_index=True)

    if args.external_yields:
        external_yields = pd.read_csv(
            f'{config.main_data_directory}/{args.external_yields}', sep='\s+', 
            usecols=["set", "zams_mass", args.isotopes[0], args.isotopes[1], f"{args.isotopes[0]}_error", f"{args.isotopes[1]}_error"])
        yields = pd.concat([yields, external_yields], ignore_index=True)

    plt.figure(figsize=(10, 5))
    sets = yields['set'].unique()
    markers = ['x', '^', 'v', "s", "o"]

    min_mass = np.min(yields['zams_mass'])
    max_mass = np.max(yields['zams_mass'])
    cmap = plt.get_cmap('Spectral')
    labels = []

    for set in sets:
        set_index = np.where(sets == set)[0][0]
        current_set = yields[yields['set'] == set]

        has_error_bars = np.max(current_set[f"{args.isotopes[0]}_error"]) > 0 or np.max(current_set[f"{args.isotopes[1]}_error"]) > 0

        # Add a lable for the plot legend
        labels.append(mlines.Line2D([], [], color='black', marker=markers[set_index],
                        markersize=5, label=f"{set}", 
                        linestyle = 'solid' if has_error_bars else 'none'))
        
        # Plot the error bars
        if has_error_bars:
            error_bar_colors = cmap((current_set['zams_mass'] - min_mass)  / (max_mass - min_mass))
            plt.errorbar(
                current_set[args.isotopes[0]], current_set[args.isotopes[1]],
                fmt = 'none', ecolor = error_bar_colors,
                xerr = current_set[f"{args.isotopes[0]}_error"],
                yerr = current_set[f"{args.isotopes[1]}_error"])

        plt.scatter(
            current_set[args.isotopes[0]], current_set[args.isotopes[1]],
            marker=markers[set_index], s = 26,
            vmin = min_mass, vmax = max_mass, c=current_set['zams_mass'], cmap = cmap)

    cbar = plt.colorbar()
    cbar.set_label('ZAMS Mass [Msun]')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend(handles=labels)
    plt.title("Isotope Ratios")
    plt.xlabel(f"{args.isotopes[0]} [Msun]")
    plt.ylabel(f"{args.isotopes[1]} [Msun]")
    plt.savefig(f"{config.plot_directory}/{args.name[0]}.png")