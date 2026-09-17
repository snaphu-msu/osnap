from osnap import *
import os
import nucleosynth.nucleo as nuc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import yt
import argparse
yt.set_log_level(50)

def get_tracer_data():
    """Compiles the Hermansen tracer data into an xarray dataset."""

    # Gets all relevant tracer files from directory and sorts them by tracer number
    files = sorted(os.listdir("../tools/hermansen_code/inputs/Lagrangian_nu"), 
                key = lambda x: int(x.split("tracer")[1].split(".")[0]))

    # Goes through each tracer file to compile the data
    times, masses, radii, temps, densities, yes, enues, enuas, fnues, fnuas = [], [], [], [], [], [], [], [], [], []
    composition = {"r": []}
    for file in files:
        
        # Only look at DAT files
        if file.endswith(".inp"):
            continue
            
        # Reads the enclosed mass from the top of the file
        with open(f"../tools/hermansen_code/inputs/Lagrangian_nu/{file}", 'r') as f:
            masses.append(float(f.readline().split(" ")[3]))

        # Loads the time, temperature, density, radius, and electron fraction from the file
        time, temp, dens, r, ye, enue, enua, fnue, fnua = np.loadtxt(f"../tools/hermansen_code/inputs/Lagrangian_nu/{file}", 
                                                    usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8), 
                                                    skiprows = 2, 
                                                    unpack = True)
        
        radii.append(r)
        temps.append(temp)
        densities.append(dens)
        yes.append(ye)
        enues.append(enue)
        enuas.append(enua)
        fnues.append(fnue)
        fnuas.append(fnua)
        times = time
        
        composition["r"].append(r[0])
        isotopes = []
        with open(f"../tools/hermansen_code/inputs/Lagrangian_nu/{file}.inp", 'r') as f:
            line = f.readline()
            while len(line) > 0:
                split_line = line.split(" ")
                isotope = split_line[0]
                mass_fraction = float(split_line[1].strip())
                if isotope not in composition.keys():
                    composition[isotope] = []
                composition[isotope].append(mass_fraction)
                isotopes.append(isotope)
                line = f.readline()
            for isotope in composition.keys():
                if isotope not in isotopes and isotope != "r":
                    composition[isotope].append(0.0)
                    
    
    composition = pd.DataFrame(composition).rename(columns = {"h1": "p", "h2": "d", "h3": "t"})
        
    # Compiles the data into an xarray dataset and returns it
    data = xr.Dataset({
        "r": (("chk", "mass"), np.array(radii).T),
        "temp": (("chk", "mass"), np.array(temps).T),
        "dens": (("chk", "mass"), np.array(densities).T),
        "ye  ": (("chk", "mass"), np.array(yes).T),
        "enue": (("chk", "mass"), np.array(enues).T),
        "enua": (("chk", "mass"), np.array(enuas).T),
        "fnue": (("chk", "mass"), np.array(fnues).T),
        "fnua": (("chk", "mass"), np.array(fnuas).T) 
    }, coords = { "chk": np.arange(198), "mass": masses, "time": times })
    
    return data, composition

def get_mass_fractions(file_ending, mass_chains, time):
    
    
    # output = []
    # path = f"{config.stitched_output_directory}/stitched_stir2_14may19_s12.0_alpha1.25_{file_ending}"
    # data = pd.read_csv(path, sep='\s+')
    # data = data[data["enclosed_mass"] > pns_mass]
    # data = data[data["enclosed_mass"] <= max_mass]
    
    # #print(np.sum(data["density"] * data["cell_volume"]) / config.M_sun)

    # # Plots each isotope's abundance and calculates their total mass
    # for isotope in isotopes:
    #     isotope_mass = np.sum(data[isotope] * data["density"] * data["cell_volume"]) / config.M_sun
    #     if lumped_isotopes[isotopes.index(isotope)] is not None:
    #         isotope_mass += np.sum(data[lumped_isotopes[isotopes.index(isotope)]] * data["density"] * data["cell_volume"]) / config.M_sun
    #     output.append(isotope_mass / (max_mass - pns_mass))
        
    composition = xr.load_dataset(f"{config.nucleo_results_directory}/14may19_m12.0_a1.25_{file_ending}")

    sum_X = []
    isotope_masses = np.zeros(len(composition.coords["isotope"]))
    total_mass = 0
    for mass_index in range(len(composition.coords["mass"])):
        if mass_index > 99: continue
        dm = 0.089618957999258 if mass_index > 99 else 0.00453703704
        for isotope_index in range(len(composition.coords["isotope"])):
            isotope_masses[isotope_index] += composition["X"].values[mass_index, isotope_index, time] * dm
        total_mass += dm
    sum_X = isotope_masses / total_mass
    
    output = []
    for chain in mass_chains:
        mass_frac = 0
        for isotope in chain:
            isotope_index = np.argwhere(composition.coords["isotope"].values == isotope)[0][0]
            mass_frac += sum_X[isotope_index]
        output.append(mass_frac)
        
    return np.array(output)

def process_data(base_path, model_name, name):
    
    # Generate paths to the different data files
    last_checkpoint = base_path + "/output/" + sorted([f for f in os.listdir(base_path + "/output") if "chk" in f])[-1]

    print("Loading the progenitor")
    
    # TODO: Make this more generic. This is temporarily set up for only one specific mass model.
    prog = pd.read_csv(f"{config.progenitor_directory}/sukhbold_2016/s12.0_presn_full", skiprows=3, delimiter="\s+")
    prog = prog.rename(columns={"nt1": "n", "h1": "p", "h2": "d", "h3": "t", "luminosity": "L", "radius": "r", "velocity": "v", "temperature": "temp", "mass": "enclosed_mass"})
    prog["enclosed_mass"] = np.cumsum(prog["enclosed_mass"]) / config.M_sun
    prog_composition = prog.drop(columns = ["grid", "enclosed_mass", "v", "density", "temp", "pressure", "specific-entropy", "Abar", "Ye", "stability", "network"])
    
    # Grab the total specific energy and gravitational potential from the original progenitor data
    # TODO: Make this more general. Also, currently have to skip the last entry cause original progenitor has one more zone somehow.
    #       So we'll need to find some way to interpolate the progenitor data to match the STIR data, or vice versa, so that we can stitch them together properly.
    old_progenitor = load_data.load_kepler_progenitor("sukhbold_2016", 12.0)
    prog["ener"] = old_progenitor["profiles"]["ener"].values[:-1]
    prog["gpot"] = old_progenitor["profiles"]["gpot"].values[:-1]
    
    progenitor = {"profiles": prog }
    nucleo_output_path = f"{config.nucleo_results_directory}/14may19_m12.0_a1.25_hermansen{name}"
    stitched_output_path = f"{config.stitched_output_directory}/stitched_{model_name}_hermansen"

    print("Loading tracer data")
    tracers, prog_composition = get_tracer_data()

    print("Beginning nucleosynthesis calculations")
    output = nuc.do_nucleosynthesis(
        model_path = base_path, 
        stir_model = model_name, 
        progenitor = prog_composition,
        domain_radius = 1e9,
        tracers = tracers,
        output_path = f"{config.skynet_output_directory}/14may19_a1.25_run_12.0_hermansen",
        isotopes_file = f"{config.isotope_list_file}",
        log_level = 1
    )
    
    print("Saving nucleosynthesis output to netcdf")

    output.to_netcdf(nucleo_output_path)

def create_plots(name):
    
    # Specifies which mass chains will be plotted and what their values are from Hermansen and Maruf's works
    mass_chains = [["k43", "ar43"], ["ti44", "v44"], ["sc47", "ca47"], ["v48", "cr48"], ["v49", "cr49"], ["cr51", "mn51"], ["mn52", "fe52"], 
                ["mn53", "fe53"], ["fe55", "co55"], ["co56", "ni56"], ["co57", "ni57"], ["fe59", "mn59"], ["ni59", "cu59"]]
    chain_labels = ["$^{43}$K + $^{43}$Ar", "$^{44}$Ti + $^{44}$V", "$^{47}$Sc + $^{47}$Ca", "$^{48}$V + $^{48}$Cr", 
                    "$^{49}$V + $^{49}$Cr", "$^{51}$Cr + $^{51}$Mn", "$^{52}$Mn + $^{52}$Fe", "$^{53}$Mn + $^{53}$Fe", 
                    "$^{55}$Fe + $^{55}$Co", "$^{56}$Co + $^{56}$Ni", "$^{57}$Co + $^{57}$Ni", 
                    "$^{59}$Fe + $^{59}$Mn", "$^{59}$Ni + $^{59}$Cu"]
    hermansen = np.array([1.4e-8, 3.01e-5, 6.12e-8, 9.24e-5, 6.08e-6, 1.85e-5, 0.001, 0.000107, 0.000495, 0.0875, 0.00256, 3.92e-5, 0.000162])
    maruf = np.array([1.94e-8, 3.42e-5, 6.93e-8, 9.47e-5, 4.58e-6, 1.39e-5, 8.60e-4, 9.23e-5, 4.32e-4, 9.38e-2, 3.60e-3, 4.34e-5, 1.43e-4])

    # Plots the mass fractions of each isotope for each set of tracers, normalized to Hermansen's values
    plt.figure(figsize=(10, 5))
    plt.rcParams['axes.axisbelow'] = True
    plt.grid(which = 'both', axis = 'both', color = 'lightgray')
    plt.axhline(y=1, color = 'black', lw = 1)
    plt.scatter(chain_labels, maruf / hermansen, marker='o', label='Maruf')
    
    results = get_mass_fractions("hermansen", mass_chains, -1)
    plt.scatter(chain_labels, results / hermansen, marker='o', label=f'John')
        
    plt.xlabel('Mass Chain', fontweight='bold', fontsize=12)
    plt.ylabel('X Ratio to Hermansen', fontweight='bold', fontsize=12)
    plt.xticks(chain_labels, chain_labels, rotation=45)
    plt.ylim(0.1, 10)
    plt.semilogy()
    plt.legend()
    # BUG: For some reason the below two lines cause a floating point error, but only if run immediately after process_data()
    plt.tight_layout()
    plt.savefig(f"{config.plot_directory}/hermansen_isotope_comparison{name}.png", dpi=400)

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description='Run nucleosynthesis calculations')
    parser.add_argument('-p', '--plot', action='store_true', help='Create plots of the nucleosynthesis results')
    parser.add_argument('-n', '--name', nargs=1, help='What to end the file name with')
    args = parser.parse_args()
    
    base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_14may19_a1.25/run_12.0"
    model_name = f"stir2_14may19_s12.0_alpha1.25"
    name = "_" + args.name[0] if args.name is not None else ""

    if not(args.plot):
        process_data(base_path, model_name, name)
        
    create_plots(name)
        
    
