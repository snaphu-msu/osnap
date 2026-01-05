import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from osnap.config import config

# Plot the abundance of the given isotopes vs enclosed mass for a given stitched model
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No path or isotope were specified.")
    else:
        path = str(sys.argv[1])
        data = pd.read_csv(config.nucleo_results_directory + "/" + path, sep='\s+')

        if len(sys.argv) == 2:
            useful_columns = []
            print(f"No isotope was specified. The possible isotopes are: {data.columns[2:]}")
        else:
            plt.figure(figsize=(10, 5))
            isotopes = sys.argv[2:]
            isotope_string = ""
            for isotope in isotopes:
                if isotope not in data.columns:
                    print(f"{isotope} is not a valid isotope. The possible isotopes are: {data.columns[2:]}")
                else:
                    isotope_string += f"_{isotope}"
                    plt.plot(data["enclosed_mass"], data[isotope], label = isotope)
                    total_mass = np.sum(data[isotope] * data["density"] * data["cell_volume"]) / 1.98e33
                    print(f"Total mass of {isotope} is {total_mass} M_sun")
            plt.semilogy()
            plt.title(f"Final Mass Fraction Profiles")
            plt.xlabel("Mass [M_sun]")
            plt.ylabel(f"Log Mass Fraction")
            plt.ylim(1e-5, 1)
            plt.xlim(1.8, 3)
            #plt.axvline(3.03, ls = "--", color = "red")
            plt.legend()
            plt.savefig(f"{config.plot_directory}/{path}_{isotope_string}.png")