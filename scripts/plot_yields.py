from osnap import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Plot the yields for a given isotope (or set of isotopes) vs the specified variable
if __name__ == "__main__":

    run_date = "14may19"

    # Allows the user to specify which models are plotted, as well as how it's displayed and saved
    # TODO: Implement using atomic number for x-axis.
    parser = argparse.ArgumentParser(description='Plot mass fractions for given isotopes, masses, and alpha values.')
    parser.add_argument('-n', '--name', nargs=1, type=str, required = True, help='Name of the saved plot file.')
    parser.add_argument('-i', '--isotope', type=str, required = True, help='Isotope to plot')
    parser.add_argument('-v', '--variable', type=str, default="mass", choices=["mass", "atomic_number"], help="The x-axis variable. Options: 'mass', 'atomic_number'")
    parser.add_argument('-m', '--masses', nargs='*', type=str, default=["20.0"], 
                        help='ZAMS masses (can specify multiple values, or use "all" for all masses listed in the config file, default: "20.0")')
    parser.add_argument('-a', '--alpha', type=str, default="1.25", help='Alpha values (default: "1.25")')
    parser.add_argument('-t', '--tracer-count', type=str, default="1000", help='Number of tracers (default: "1000")')
    parser.add_argument('-e', '--external-yields', type=str, help='Name of an optional external yields file. Must be in main data directory as specified in the config file.')
    parser.add_argument('-l', '--log-scale', action='store_true', help='Plot on a log scale.')
    args = parser.parse_args()

    if args.masses[0] == "all":
        args.masses = config.all_masses

    yields = pd.DataFrame(columns=["set", "zams_mass", args.isotope])
    for mass in args.masses:
        path = f"{config.stitched_output_directory}/stitched_stir2_{run_date}_s{mass}_alpha{args.alpha}_n{args.tracer_count}"
        if not os.path.exists(path):
            print(f"stitched_stir2_{run_date}_s{mass}_alpha{args.alpha}_n{args.tracer_count} does not exist. Skipping...")
            continue
        data = pd.read_csv(path, sep='\s+')
        ejecta_mass = np.sum(data[args.isotope] * data["density"] * data["cell_volume"]) / config.M_sun
        yields = pd.concat([yields, pd.DataFrame([["STIR", float(mass), ejecta_mass]], columns=yields.columns)], ignore_index=True)

    if args.external_yields:
        external_yields = pd.read_csv(f'{config.main_data_directory}/{args.external_yields}', sep='\s+', usecols=["set", "zams_mass", args.isotope])
        yields = pd.concat([yields, external_yields], ignore_index=True)

    plt.figure(figsize=(10, 5))
    sets = yields['set'].unique()
    markers = ['*', 'x', '+', '.']
    colors = ['red', 'green', 'blue', 'black']

    # Iterate over all rows of each set
    for set in sets:
        set_index = np.where(sets == set)[0][0]
        current_set = yields[yields['set'] == set]
        zams_masses = current_set['zams_mass']
        set_yields = current_set[args.isotope]
        plt.scatter(zams_masses, set_yields, label=set, marker=markers[set_index], color=colors[set_index])

    if args.log_scale: plt.loglog()
    plt.legend()
    plt.title(f"{args.isotope} Yields")

    if args.variable == "mass":
        plt.xlabel("log(ZAMS Mass [Msun])" if args.log_scale else "ZAMS Mass [Msun]")
    elif args.variable == "atomic_number":
        plt.xlabel("log(Atomic Number)" if args.log_scale else "Atomic Number")

    plt.ylabel(f"log(Ejecta Mass [Msun])" if args.log_scale else "Ejecta Mass [Msun]")
    plt.savefig(f"{config.plot_directory}/{args.name[0]}.png")