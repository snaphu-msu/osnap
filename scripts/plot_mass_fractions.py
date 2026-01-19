from osnap import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Plot the mass fractions of the given isotopes vs enclosed mass for the requested models
if __name__ == "__main__":

    run_date = "14may19"

    # Allows the user to specify which models are plotted, as well as how it's displayed and saved
    parser = argparse.ArgumentParser(description='Plot mass fractions for given isotopes, masses, and alpha values.')
    parser.add_argument('-n', '--name', nargs=1, type=str, required = True, help='Name of the saved plot file.')
    parser.add_argument('-m', '--masses', nargs='*', type=str, default=["20.0"], 
                        help='ZAMS masses (can specify multiple values, or use "all" for all masses listed in the config file, default: "20.0")')
    parser.add_argument('-a', '--alphas', nargs='*', type=float, default=["1.25"], help='Alpha values (can specify multiple values, default: "1.25")')
    parser.add_argument('-i', '--isotopes', nargs='+', type=str, required = True, help='Isotopes to plot (can specify multiple isotopes)')
    parser.add_argument('-t', '--tracer-count', nargs='*', type=int, default=[1000], help='Number of tracers (can specify multiple values, default: 1000)')
    parser.add_argument('-x', '--x-range', nargs=2, type=float, default=[1.8, 3], help='Range of enclosed mass to plot (default: [1.8, 3])')
    parser.add_argument('-y', '--y-range', nargs=2, type=float, default=None, help='Range of log mass fractions to plot (default: None)')
    args = parser.parse_args()

    if args.masses[0] == "all":
        args.masses = config.all_masses
    
    # Loops over all supplied tracer counts, alpha values, and ZAMS masses to plot all requested models
    plt.figure(figsize=(10, 5))
    for tracers in args.tracer_count:
        for alpha in args.alphas:
            for mass in args.masses:

                # Loads the stitched data for the given parameters
                path = f"{config.stitched_output_directory}/stitched_stir2_{run_date}_s{mass}_alpha{alpha}_n{tracers}"
                if not os.path.exists(path):
                    print(f"stitched_stir2_{run_date}_s{mass}_alpha{alpha}_n{tracers} does not exist. Skipping...")
                    continue
                data = pd.read_csv(path, sep='\s+')

                # Plots each isotope's abundance and calculates their total mass
                for isotope in args.isotopes:
                    if isotope not in data.columns:
                        print(f"{isotope} is not a valid isotope.")
                    else:
                        total_mass = np.sum(data[isotope] * data["density"] * data["cell_volume"]) / config.M_sun

                        # Build the this isotope's label
                        label = ""
                        if len(args.isotopes) > 1 or (len(args.masses) == 1 and len(args.alphas) == 1 and len(args.tracer_count) == 1): label += isotope
                        if len(args.masses) > 1: 
                            if len(label) > 0: label += ", "
                            label += f"m={mass}"
                        if len(args.alphas) > 1: 
                            if len(label) > 0: label += ", "
                            label += f"a={alpha}"
                        if len(args.tracer_count) > 1: 
                            if len(label) > 0: label += ", "
                            label += f"n={tracers}"

                        plt.plot(data["enclosed_mass"], data[isotope], label = f"{label} (M_total={total_mass:.2e})")

    # Build the plot's title
    title = ""
    if len(args.isotopes) == 1: title += args.isotopes[0]
    if len(args.masses) == 1: 
        if len(title) > 0: title += ", "
        title += f"{args.masses[0]} Msun"
    if len(args.alphas) == 1: 
        if len(title) > 0: title += ", "
        title += f"{args.alphas[0]} alpha"
    if len(args.tracer_count) ==1: 
        if len(title) > 0: title += ", "
        title += f"{args.tracer_count[0]} tracers"

    plt.semilogy()
    plt.title(f"Final Mass Fractions ({title})")
    plt.xlabel("Enclosed Mass [Msun]")
    plt.ylabel(f"Log Mass Fraction")
    plt.xlim(args.x_range)
    if args.y_range is not None: 
        plt.ylim(args.y_range)
    plt.legend()
    plt.savefig(f"{config.plot_directory}/{args.name[0]}.png")