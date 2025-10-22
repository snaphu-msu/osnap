import yaml
from os import path, pardir

# Load configuration
stream = open(path.join(path.dirname(__file__), pardir, 'config.yaml'), 'r')
configs = yaml.safe_load(stream)
progenitor_directory = configs['progenitor_directory']
progenitor_suffix = configs['progenitor_suffix']
stir_profiles_directory = configs['stir_profiles_directory']
stir_profiles_suffix = configs['stir_profiles_suffix']
output_directory = configs['output_directory']
output_suffix = configs['output_suffix']
cell_edge_velocity = configs['cell_edge_velocity']
eos_file_path = configs['eos_file_path']
velocity_name = configs['velocity_name']
default_plotted_profiles = configs['default_plotted_profiles']
log_plots = configs['log_plots']

# Constants in CGS units
M_sun = 1.989E33 # Mass of the sun in grams
R_sun = 6.959E10 # Radius of the sun in centimeters
sigma_b = 5.669E-5 # Stefan-Boltzmann constant
G = 6.67430E-8 # Gravitational constant