from osnap import *
import os.path
import flashbang as fb
import nucleosynth.nucleo as sky
import xarray as xr
import numpy as np
import progs.progs as progs
import matplotlib.pyplot as plt
import time

# Hide the yt output
import yt
yt.set_log_level(50)

force_reload_tracers = False
max_tracer_mass = 1.39 # TODO: Set this to the last mass element with a shock marker
num_tracers = 1000
zams_mass = 9.0
alpha = 1.25

run_date = "14may19"
base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{alpha}/run_{zams_mass}"
model_name = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}"

# Set the minimum tracer mass as the PNS mass, which is determined as the first unbound mass element
last_chk_file = base_path + "/output/" + sorted([f for f in os.listdir(base_path + "/output") if "chk" in f])[-1]
stir_data = yt.load(last_chk_file).all_data()
total_specific_energy = load_data.calculte_total_specific_energy(stir_data) + stir_data['flash', 'gpot'].value
enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / config.M_sun
pns_masscut_index = np.min(np.where(total_specific_energy >= 0)) - 1
min_tracer_mass = enclosed_mass[pns_masscut_index]

### 
# TODO:
# 1. Find which models explode.
# 2. Find ideal tracer count. Use high number as most accurate ground truth.
#    Run for varying num_tracers values, recording the abundances of each element for each value.
#    Find point where error (M_truth - M_test) / M_truth is below a threshold
#    Log-log plot of error vs tracer count
###
start_time = time.time()

progenitor = progs.ProgModel(zams_mass, "sukhbold_2016")

tracers = fb.load_save.get_tracers(
    run = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}", 
    model = f"run_{zams_mass}",
    model_set = f"run_{run_date}_a{alpha}",
    mass_grid = np.linspace(min_tracer_mass, max_tracer_mass, num_tracers), 
    reload = force_reload_tracers,
    config = "stir"
)

output = sky.do_nucleosynthesis(
    model_path = base_path, 
    stir_model = model_name, 
    progenitor = progenitor,
    domain_radius = 1e9,
    tracers = tracers,
    output_path = f"./skynet_output/{run_date}_a{alpha}_run_{zams_mass}",
    verbose = 1
)

# Print the time taken to run the nucleosynthesis
time_elapsed = time.time() - start_time
hours, rem = divmod(time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
hour_string = f"{hours:.0f}" if hours >= 10 else f"0{hours}"
min_string = f"{minutes:.0f}" if minutes >= 10 else f"0{minutes}"
sec_string = f"0{seconds:.2f}" if seconds >= 10 else f"0{seconds:.2f}"
time_elapsed_str = f"{hour_string}:{min_string}:{sec_string}"
print(f"Time Taken: {time_elapsed_str}", f"(per tracer: {(time_elapsed / num_tracers):.2f}s)")