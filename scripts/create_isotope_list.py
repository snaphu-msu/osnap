import argparse
import pandas as pd
import osnap.config as config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create a list of isotopes for use in nucleosynthesis calculations.')
    parser.add_argument('-z', '--max-z', nargs=1, type=int, default=999, help='Maximum atomic mass number to include in the list.')
    parser.add_argument('-a', '--max-a', nargs=1, type=int, default=999, help='Maximum atomic number to include in the list.')
    parser.add_argument('-n', '--filename', nargs=1, type=str, default="isotopes", help='Name of the output file.')
    args = parser.parse_args()

    elements = pd.read_csv(f"{config.main_data_directory}/elements.csv", delimiter='\s+')
    isotopes = []
    for index, row in elements.iterrows():
        for z in range(row['Min_Z'], row['Max_Z'] + 1):
            if z <= args.max_z and row['A'] <= args.max_a:
                isotopes.append(f"{row['Symbol'].lower()}{z}")

    # Write the list of isotopes to a file with each isotope on its own line
    with open(f"{config.main_data_directory}/{args.filename}", 'w', encoding='utf-8') as f:
        f.write(f"n\n")
        for isotope in isotopes:
            if isotope == "h1":
                f.write(f"p\n")
            elif isotope == "h2":
                f.write(f"d\n")
            elif isotope == "h3":
                f.write(f"t\n")
            else:   
                f.write(f"{isotope}\n")
