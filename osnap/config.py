import yaml
from os import path, pardir

# Load configuration
root_path = path.join(path.dirname(__file__), pardir)
stream = open(path.join(root_path, 'config.yaml'), 'r')
configs = yaml.safe_load(stream)

# File paths
relative_paths = configs['relative_paths']
if relative_paths:
    progenitor_directory = path.join(root_path, configs['progenitor_directory'])
    stir_profiles_directory = path.join(root_path, configs['stir_profiles_directory'])
    stitched_output_directory = path.join(root_path, configs['stitched_output_directory'])
    nucleo_results_directory = path.join(root_path, configs['nucleo_results_directory'])
    mesa_export_directory = path.join(root_path, configs['mesa_export_directory'])
    plot_directory = path.join(root_path, configs['plot_directory'])
    eos_file_path = path.join(root_path, configs['eos_file_path'])
else:
    progenitor_directory = configs['progenitor_directory']
    stir_profiles_directory = configs['stir_profiles_directory']
    stitched_output_directory = configs['stitched_output_directory']
    nucleo_results_directory = configs['nucleo_results_directory']
    mesa_export_directory = configs['mesa_export_directory']
    plot_directory = configs['plot_directory']
    eos_file_path = configs['eos_file_path']

# Optional suffixes
progenitor_suffix = configs['progenitor_suffix']
stir_profiles_suffix = configs['stir_profiles_suffix']
output_suffix = configs['output_suffix']

# Velocity options
cell_edge_velocity = configs['cell_edge_velocity']
velocity_name = configs['velocity_name']

# Plotting options
default_plotted_profiles = configs['default_plotted_profiles']
log_plots = configs['log_plots']

# Default list of masses being used in nucleo calculations and plotting when the "all" option is used.
all_masses = configs['all_masses']

# Constants in CGS units
M_sun = 1.989E33 # Mass of the sun in grams
R_sun = 6.959E10 # Radius of the sun in centimeters
sigma_b = 5.669E-5 # Stefan-Boltzmann constant
G = 6.67430E-8 # Gravitational constant