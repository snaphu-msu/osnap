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
run_date = "14may19"
base_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}"
min_tracer_mass = 1.34
max_tracer_mass = 1.39
num_tracers = 1000
zams_mass = 9.0
alpha = 1.25

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
    model_path = f"/mnt/research/SNAPhU/STIR/run_sukhbold/run_{run_date}_a{alpha}/run_{zams_mass}", 
    stir_model = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}", 
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