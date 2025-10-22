import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


### Stitch on progenitor, then plot total mass of each isotope vs atomic mass number
### Plot nickel-56 yields vs zams mass
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No path or isotope were specified.")
    else:
        path = str(sys.argv[1])
        data = pd.read_csv(path)
        if len(sys.argv) == 2:
            useful_columns = []
            print(f"No isotope was specified. The possible isotopes are: {data.columns[2:]}")
        else:
            isotopes = sys.argv[2:]
            isotope_string = ""
            for isotope in isotopes:
                if isotope not in data.columns:
                    print(f"{isotope} is not a valid isotope. The possible isotopes are: {data.columns[2:]}")
                else:
                    isotope_string += f"_{isotope}"
                    plt.plot(data["mass"], np.log10(data[isotope]), label = isotope)
            plt.title(f"Final Mass Fraction Profiles")
            plt.xlabel("Mass [M_sun]")
            plt.ylabel(f"Log Mass Fraction")
            plt.legend()
            plt.savefig(f"plots/{path[:-4]}{isotope_string}.png")