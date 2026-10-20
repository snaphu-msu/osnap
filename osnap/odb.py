########################################################################################
### odb.py
###
### For the abstract data layer of OSNAP, which allows loading many
### kinds of supernova simulation data, creating a consistent format
### and saving that data to an HDF5 file.
########################################################################################

import h5py
import numpy as np
from osnap import units, info, config
from os import path, listdir
import pandas as pd
import yaml
from datetime import datetime, timezone
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import astropy.units as u

#########################################################################################
### Notes about what fields the ODB should contain:
### attributes: 
###     osnap_info (version number of OSNAP last used to write to the ODB)
###     progenitor_info (software and version used to create the progenitor)
###     collapse_info (software and version used to explode the star)
###     pns_mass, pns_radius, explosion_energy
### profiles: (dataset containing the current final profiles for the star)
###     density, temperature, ye, velocity, entropy, pressure, specific_energy
### progenitor: (dataset containing the initial progenitor profiles for the star)
### composition: (dataset containing the current final composition for the star)
### trajectories: (dataset containing any generated trajectories from the star)
### light_curves: (dataset containing any generated light curves from the star)
### spectra: (dataset containing any generated spectra from the star)
#########################################################################################


CONFIG = config.load()

class ODB():
    """
    Abstract Data Layer for OSNAP.

    This class provides a consistent interface for loading and saving supernova simulation 
    data from various sources. It allows for the creation of a unified data format that can 
    be easily accessed and manipulated.
    """
    
    def __init__(self, file_path):
        """
        Creates an access point for the HDF5 file where data is stored.

        Args:
            file_path (str): Path to the HDF5 file where data should be stored.
        """
        
        # Sets the path to the HDF5 file where data is stored
        self.file_path = file_path + ".odb"
    
    
    def read(self, group, dataset):
        """
        Reads data from a field in the HDF5 file.
        
        Args:
            group (str): The group in the HDF5 file where the dataset is located.
            dataset (str): The name of the dataset to read.
        """
        
        with h5py.File(self.file_path, 'r') as f:
            
            key = f"{group}/{dataset}"
            if key in f:
                data = units.Variable(key, 
                                      f[key][:], 
                                      f[key].attrs.get("units", ""), 
                                      f[key].attrs.get("location", "average"))
                return data
            else:
                raise KeyError(f"Key '{key}' not found in the ODB file at '{self.file_path}'.")
    
    
    def write(self, group, variable): 
        """
        Writes data to a field in the HDF5 file.

        Args:
            key (str): The key for the data field you want to write to.
            variable (units.Variable): The data to be written to the specified field.
        """
        
        with h5py.File(self.file_path, 'a') as f:
            
            key = f"{group}/{variable.name}"
            
            # If the key already exists, get rid of the old data
            if key in f:
                del f[key]
            
            # Write the new data and update the OSNAP version
            f.create_dataset(key, data = variable.value)
            f[key].attrs["units"] = variable.unit
            f[key].attrs["location"] = variable.loc
            self.update_info(f)
            
            
    def write_group(self, group_path, variables, attributes = {}):
        """
        Writes multiple variables to a group in the HDF5 file.

        Args:
            group_path (str): The group in the HDF5 file where the variables should be written.
            variables (list of units.Variable): A list of Variable instances to be written.
        """
        
        with h5py.File(self.file_path, 'a') as f:
            
            # If group already exists, get rid of the old data
            if group_path in f:
                del f[group_path]
                
            group = f.create_group(group_path)
            
            # Write the new data
            for variable in variables:
                key = f"{group_path}/{variable.name}"
                f.create_dataset(key, data = variable.value)
                f[key].attrs["units"] = f"{variable.unit}"
                f[key].attrs["location"] = variable.loc
            
            # Writes any specified attributes to the group
            for attr_name, attr_value in attributes.items():
                group.attrs[attr_name] = attr_value
            
            self.update_info(f, group)
            
    
    def write_metadata(self, path = "", name = "", value = 0): 
        """
        Adds or updates an attribute to the ODB file.

        Args:
            path (str): The path in the HDF5 file where the attribute should be added.
            name (str): The name of the attribute to add or update.
            value (object): The value of the attribute to add or update.
        """
        
        with h5py.File(self.file_path, 'a') as f:
            
            # If path is not empty grab the group/dataset at that path, otherwise use the root group
            location = f[path] if path else f
            
            # Write the attribute and update the OSNAP version
            location.attrs[name] = value
            self.update_info(f)
       
            
    def update_info(self, hdf5_file, group = None):
        """
        Updates the ODB metadata with the current OSNAP version and a timestamp of last update.
        
        Args:
            hdf5_file (h5py.File): The HDF5 file object to update.
        """
        
        hdf5_file.attrs["osnap_version"] = info.osnap_version
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        hdf5_file.attrs["last_updated"] = now
        if group is not None:
            group.attrs["last_updated"] = now
        
    
    def load_progenitor(self, progenitor_type, progenitor_path):
        """
        Loads the progenitor data into this ODB instance.

        Args:
            progenitor_type (str): Type of the progenitor data (e.g., 'kepler', 'mesa').
            progenitor_path (str): Path to the progenitor data.
        """
        
        self.load_data("progenitor", progenitor_path, progenitor_type)
    
    
    def load_collapse(self, collapse_type, collapse_path):
        """
        Loads the collapse data into this ODB instance.

        Args:
            collapse_type (str): Type of the collapse data (e.g., 'stir', 'mesa').
            collapse_path (str): Path to the collapse data.
        """
        
        self.load_data("profiles", collapse_path, collapse_type)
        self.constrain_PNS()
    
    
    def load_data(self, odb_path, data_type, file_path):
        """
        Load data from a file into this ODB instance.
        
        Args:
            data_type (str): The type of data to load (e.g., 'kepler', 'mesa', etc).
            file_path (str): The path to the file containing the data.
        """
        
        print(f"Loading '{data_type}' data from '{file_path}'")
        
        # Get the definition settings for the specified data type
        definition = get_definition(data_type)
        
        # Load the data using the method given in the definition file
        if definition['file_type'] == 'csv':
            loaded_data = load_csv(file_path, definition)
        elif definition['file_type'] == 'hdf5':
            loaded_data = load_hdf5(file_path, definition)
        else:
            raise ValueError(f"Unsupported file type '{definition['file_type']}' for data type '{data_type}'")
            
        # If the zone order is descending, reverse it to be ascending order
        if definition['zone_order'] == 'descending':
            loaded_data = loaded_data.iloc[::-1].reset_index(drop = True)
    
        # Make all column names lowercase and removes bad characters (like single quotes)
        loaded_data.columns = [col.strip("'").lower() for col in loaded_data.columns]
        
        # Create a dictionary to store the loaded data
        data = {}
        for col in definition['column_info']:
            
            # Try to get settings for each column, skipping ill-defined or unnamed columns
            info = definition['column_info'][col]
            if info == [] or info[0] == '': continue
            
            # Ensure the given column name is present in the loaded data
            name = info[0].lower()
            if name not in loaded_data.columns:
                raise ValueError(f"Column '{name}' not found in loaded data of type '{data_type}'")
            
            data[col] = units.Variable(col, loaded_data[name].to_numpy(), info[1], info[2])
        
        # Adds values for neutrons, protons, deuterium, and tritium
        comp_type = definition['composition_type']
        for col in ['neutrons', 'protons', 'deuterium', 'tritium']:
            if col in definition and definition[col] != '':
                values = loaded_data[definition[col].lower()].to_numpy()
                data[col] = units.Variable(col, values, comp_type, 'average')
        
        # Loads in values for all isotopes listed in the definition file
        for iso in definition['isotopes']:
            element = ''.join([c for c in iso if not c.isdigit()]).lower()
            mass_number = ''.join([c for c in iso if c.isdigit()])
            iso_name = f"{element}{mass_number}"
            if iso_name not in data:
                values = loaded_data[iso.lower()].to_numpy()
                data[iso_name] = units.Variable(iso_name, values, comp_type, 'average')
            
        # Loads in values for all elements with no mass number
        # TODO: May handle these differently later
        for element in definition['element_groups']:
            element = element.lower()
            if element not in data:
                values = loaded_data[element].to_numpy()
                data[element] = units.Variable(element, values, comp_type, 'average')
                
        # Clean the data by removing any columns with all NaN values
        for col in list(data):
            if np.all(np.isnan(data[col].value)):
                print(f"Warning: Column '{col}' contains all NaN values and will be ignored.")
                del data[col]
            
        # Ensure all necessary data is present and in the correct format for ODB
        data = units.correct_units(data)
        data = calculate_missing_quantities(data)
        data = units.correct_units(data) # DEBUG: This is just to inform us if any units are changed post-derivation
        
        # Writes the loaded data to the ODB filee
        print(f"Saving loaded data to ODB in group '{odb_path}'")
        self.write_group(odb_path, list(data.values()), 
                         attributes = {"type": data_type, "file": file_path})
    
    
    def constrain_PNS(self):
        """
        Determines the proto-neutron star's mass and radius based on current data.
        """
        
        # Reads relevant values from the ODB file
        radius = self.read("profiles", "radius")
        total_specific_energy = self.read("profiles", "total_specific_energy")
        enclosed_mass = self.read("profiles", "enclosed_mass")

        # Determines the PNS to end where matter becomes unbound (total specific energy >= 0)
        pns_masscut_index = np.min(np.where(total_specific_energy.value >= 0))
        pns_mass = enclosed_mass.value[pns_masscut_index]
        pns_radius = radius.value[pns_masscut_index]
        
        # Writes the calculated PNS mass and radius to the ODB file as attributes
        self.write_metadata("profiles", "pns_mass", pns_mass)
        self.write_metadata("profiles", "pns_mass_units", enclosed_mass.unitstr)
        self.write_metadata("profiles", "pns_radius", pns_radius)
        self.write_metadata("profiles", "pns_radius_units", radius.unitstr)
    

def load_csv(file_path, definition):
    """
    Load data from a CSV file and return it as a DataFrame.
    
    Args:
        file_path (str): The path to the CSV file containing the data.
        
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data from the CSV file.
    """
    
    # If using whitespace delimiter, set the separator as whitespace of 2 or more characters
    # This prevents issues when column names include spaces
    delimiter = definition['delimiter']
    if delimiter == "whitespace":
        delimiter = r'\s{2,}'
    
    # Load the CSV file into a pandas DataFrame
    csv_data = pd.read_csv(file_path, 
                           skiprows = definition['skiprows'], 
                           sep = delimiter,
                           na_values = definition['missing_data'],
                           engine = "python",
                           usecols = definition['all_columns'])
    
    return csv_data


def load_hdf5(file_path, definition):
    """
    Load data from an HDF5 file and return it as a DataFrame.
    
    Args:
        file_path (str): The path to the HDF5 file (or folder of files) containing the data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data from the HDF5 file.
    """
    
    data_dict = {}
    
    # Load the HDF5 file into a pandas DataFrame
    # TODO: Finish setting this up to interpret AMR blocks
    with h5py.File(file_path, 'r') as f:
        
        if definition['uses_amr']:
            
            # If no node type column was specified or found, throw an error
            if definition['node_type'] == '' or definition['node_type'] not in f:
                raise ValueError(f"No dataset named '{definition['node_type']}' could be found in the HDF5 file at '{file_path}'")
            
            # If no bounding box column was specified or found, throw an error
            if definition['bounding_box'] == '' or definition['bounding_box'] not in f:
                raise ValueError(f"No dataset named '{definition['bounding_box']}' could be found in the HDF5 file at '{file_path}'")
            
            # Create a list of the indices for blocks that are leaf nodes
            leaves = np.where(f[definition['node_type']][:] == 1)[0]
            if len(leaves) == 0:
                raise ValueError(f"No leaf nodes were found in the HDF5 file at '{file_path}'")
                        
            # Determine the inner and outer radius of each leaf node using the bounding box
            bounding_boxes = f[definition['bounding_box']][leaves]
            inner_radii = bounding_boxes[:, definition['bounding_box_radial_index'], 0]
            outer_radii = bounding_boxes[:, definition['bounding_box_radial_index'], 1]
            
            # Sort the leaves by their radii in ascending order
            sort_order = np.argsort(inner_radii)
            inner_radii = inner_radii[sort_order]
            outer_radii = outer_radii[sort_order]
            leaves = leaves[sort_order]

            # Compile data from all specified columns in the definition file into a dictionary
            # Skips the bounding box and node type columns
            for key in f.keys():
                if key in definition['all_columns']:
                    if key != definition['bounding_box'] and key != definition['node_type']:
                        data_dict[key] = f[key][leaves].flatten()
            
            # TODO: This gets the number of zones in a block by looking at temperature. May need to change this to be more general in the future.
            #       I don't want to make zones per block part of the definition file, since it may change between models, I assume.
            zones_per_block = f[definition['column_info']['temperature'][0]].shape[definition['radial_axis']]
            
            # Determine the radius of each zone
            zone_width = (outer_radii - inner_radii) / zones_per_block
            zone_radii = []
            for i in range(len(inner_radii)):
                for j in range(zones_per_block):
                    zone_radii.append(inner_radii[i] + zone_width[i] * (j + 1))
                
            data_dict['radius'] = np.array(zone_radii)
            definition['column_info']['radius'] = ['radius', definition['bounding_box_units'], 'outer']
            
        # TODO: Add support for non-AMR HDF5 data
        
    return pd.DataFrame(data_dict)


def get_definition(data_type):
    """
    Get settings from the definition file for a data type.
    
    Args:
        data_type (str): The type of data being loaded (e.g., 'kepler', 'mesa', etc).
        
    Returns:
        dict: A dictionary containing the settings from the definition file, or None 
            if the file does not exist.
    """
    
    # Load the definition file for the specified data type
    def_path = path.join(path.dirname(__file__), f'../definitions/{data_type}.yaml')
    if path.exists(def_path):
        with open(def_path, 'r') as f:
            loaded_def = yaml.safe_load(f)
            
            # From the loaded definition, create a list of all column names that will be used in the ODB
            loaded_def['all_columns'] = []
            for col in loaded_def['column_info']:
                if len(loaded_def['column_info'][col]) > 0:
                    loaded_def['all_columns'].append(loaded_def['column_info'][col][0])
                    
            # Including lists that are not part of the column_info variable
            loaded_def['all_columns'] += list(loaded_def['isotopes']) + list(loaded_def['element_groups'])
            
            # As well as single value column names
            for col in ['node_type', 'bounding_box', 'neutrons', 'protons', 'deuterium', 'tritium']:
                loaded_def['all_columns'].append(loaded_def[col])
                
            # But remove any empty strings and duplicates
            loaded_def['all_columns'] = list(set([col for col in loaded_def['all_columns'] if col != '']))
            
            return loaded_def 
        
    # If the requested definition file does not exist, 
    # raise an error and list all available data types
    else:
        def_dir = path.join(path.dirname(__file__), '../definitions')
        all_defs = [f.split('.')[0] for f in listdir(def_dir) if f.endswith('.yaml')]
        raise ValueError(f"Definition file for data type '{data_type}' not found. "
                         + f"Please use one of the following types: {all_defs}")

    
def calculate_missing_quantities(data):
    """
    Calculate any missing quantities in the data.
    
    Args:
        data (dict): A dictionary containing the existing data.
        
    Returns:
        dict: The updated data dictionary with missing quantities calculated and added.
    """

    # Tries to calculate the cell volume if it's missing
    if "cell_volume" not in data:
        if "radius" in data:
            # TODO: Temporary solution. Need to properly calculate volume of the first and last zones
            #       Will also need to account for radius location (centered vs outer) when calculating volume
            volume = np.concat(([0], (4/3) * np.pi * (data["radius"].value[1:] ** 3 - data["radius"].value[:-1] ** 3)))
            data["cell_volume"] = units.Variable("cell_volume", volume, "cm^3", data["radius"].loc)
        else:
            raise ValueError("Cannot calculate cell_volume: 'radius' is missing.")
    
    # Tries to calcualte the cell mass using either the density and volume, or the enclosed mass
    if "cell_mass" not in data:
        if "density" in data:
            cell_mass = data["density"].value * data["cell_volume"].value
        elif "enclosed_mass" in data:
            cell_mass = np.diff(data["enclosed_mass"].value, prepend = data["enclosed_mass"].value[0])
        else:
            raise ValueError("Cannot calculate cell_mass: data must include either 'density' or 'enclosed_mass'")
        data["cell_mass"] = units.Variable("cell_mass", cell_mass, "g", data["cell_volume"].loc)
        
    # If enclosed mass is missing, calculate it using the cell masses
    if "enclosed_mass" not in data:
        enclosed_mass = np.cumsum(data["cell_mass"])
        data["enclosed_mass"] = units.Variable("enclosed_mass", enclosed_mass, "g", data["cell_mass"].loc)
        
    # If gravitational potential is missing, calculate it using the density and radius
    # Uses the equation for the newtonian gravitational potential of a spherically symmetric mass distribution
    # TODO: Possibly need to account for the location of the radius (centered vs outer)
    if "grav_potential" not in data:
        inner_term = np.concatenate(([0.0], np.cumsum(data["density"].value * data["radius"].value ** 3)[:-1]))
        inner_term /= 3 * data["radius"].value
        outer_term = np.cumsum((data["density"].value * data["radius"].value ** 2)[::-1])[::-1] / 2
        grav_potential = -4 * np.pi * u.G * (inner_term + outer_term)
        data["grav_potential"] = units.Variable("grav_potential", grav_potential, "erg/g", data["radius"].loc)
    
    # If total specific energy is missing, calculate it using and EOS and existing data
    if "total_specific_energy" not in data:
        data["total_specific_energy"] = calculate_total_specific_energy(data)

    return data


# TODO: Fix. Currently results in all NaN values for some reason.
# TODO: Units may be funky here
def calculate_total_specific_energy(data):
    """
    Calculates the total specific energy for each zone in the star.
    
    Args:
        data (dict): A dictionary that includes the electron fraction, temperature, density, \
            radial velocity, and gravitational potential for each zone.
            
    Returns:
        units.Variable: A Variable instance containing the total specific energy for each zone.
    """

    # Load the equation of state as an interpolator
    with h5py.File(CONFIG["EOS_PATH"], 'r') as EOS:
        logenergy = RegularGridInterpolator((EOS['ye'], EOS['logtemp'], EOS['logrho']), 
                                                EOS['logenergy'][:,:,:], bounds_error = False)
        
        # The EOS offsets its energy values by a constant energy_shift to ensure no values are negative.
        # So, we need to offset it back down by the same amount to get the correct total specific energy.
        energy_shift = EOS['energy_shift'][0] * u.erg / u.g

    # Use the EOS to calculate the total specific energy
    llogtemp = np.log10((data["temperature"] * const.k_B.to(u.MeV / u.K)).value)
    llogrho = np.log10(data["density"].value)
    energy = 10.0 ** logenergy(np.array([data["Y_e"].value, llogtemp, llogrho]).T) * u.erg / u.g
    energy0 = 10.0 ** logenergy(np.array([data["Y_e"].value, llogtemp * 0.0 - 2.0, llogrho]).T) * u.erg / u.g

    # Calculate the total specific energy and return it as a Variable
    ener = 0.5 * data["radial_velocity"] ** 2 + (energy - energy0) + data["grav_potential"] - energy_shift
    return units.Variable("total_specific_energy", ener, "erg/g", data["radial_velocity"].loc)