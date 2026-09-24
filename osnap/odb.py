########################################################################################
### odb.py
###
### For the abstract data layer of OSNAP, which allows loading many
### kinds of supernova simulation data, creating a consistent format
### and saving that data to an HDF5 file.
########################################################################################

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from config import CONFIG
import units
import os
import info

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


def ODB():
    """
    Abstract Data Layer for OSNAP.

    This class provides a consistent interface for loading and saving supernova simulation 
    data from various sources. It allows for the creation of a unified data format that can 
    be easily accessed and manipulated.
    """
    
    def __init__(self, file_path):
        """
        Creates an access point for the HDF5 file where data is stored. If the file does not 
        exist, it will be created.

        Args:
            file_path (str): Path to the HDF5 file where data should be stored.
        """
        
        # Sets the path to the HDF5 file where data is stored
        self.file_path = file_path
        
        # If the file doesn't already exist, create an empty one
        if not os.path.exists(file_path):
            with h5py.File(file_path, 'w') as f:
                pass
        
    def read(self, keys):
        """
        Grabs data from one or more fields in the HDF5 file.

        Args:
            keys (list[str]): The keys for the data fields you want to retrieve.

        Returns:
            dict: A dictionary containing the requested rows of data.
        """
        
        with h5py.File(self.file_path, 'r') as f:
            data = {}
            for key in keys:
                if key in f:
                    data[key] = f[key][:]
                else:
                    raise KeyError(f"Key '{key}' not found in the HDF5 file.")
            return data
        
    def write(self, key, data): 
        """
        Writes data to a field in the HDF5 file.

        Args:
            key (str): The key for the data field you want to write to.
            data (array-like): The data to be written to the specified field.
        """
        
        with h5py.File(self.file_path, 'a') as f:
            if key not in f:
                f.create_dataset(key, data=data[key])
            else:
                f[key][:] = data[key]
            f.attr["osnap_info"] = info.osnap_version
        
    def constrain_PNS(self):
        """
        Determines the proto-neutron star's mass and radius based on current data.
        """
        last_checkpoint = base_path + "/output/" + sorted([f for f in os.listdir(base_path + "/output") if "chk" in f])[-1]
        stir_data = yt.load(last_checkpoint).all_data()
        total_specific_energy = calculate_total_specific_energy(stir_data) + stir_data['flash', 'gpot'].value
        enclosed_mass = np.cumsum(stir_data['flash', 'cell_volume'].value * stir_data['gas', 'density'].value) / units.M_SUN
        pns_masscut_index = np.min(np.where(total_specific_energy >= 0))
        pns_mass = enclosed_mass[pns_masscut_index]
        data = data[data["enclosed_mass"] > pns_mass]
    
        
    def calculate_total_specific_energy(ye, temp, dens, vel):

        # Load the equation of state used by STIR
        with h5py.File(CONFIG["EOS_PATH"], 'r') as EOS:
            mif_logenergy = RegularGridInterpolator((EOS['ye'], EOS['logtemp'], EOS['logrho']), 
                                                    EOS['logenergy'][:,:,:], bounds_error=False)
            energy_shift = EOS['energy_shift'][0]

        # Use the EOS to calculate the total specific energy
        llogtemp = np.log10(temp * 8.61733326e-11)
        llogrho = np.log10(dens)
        energy = 10.0 ** mif_logenergy(np.array([ye, llogtemp, llogrho]).T)
        llogtemp = llogtemp * 0.0 - 2.050
        energy0 = 10.0 ** mif_logenergy(np.array([ye, llogtemp, llogrho]).T)

        ener = (0.5 * vel ** 2 + (energy - energy0) * yt.units.erg / yt.units.g).v
        
        # The EOS offsets its energy values by a constant energy_shift to ensure no values are negative.
        # So, we need to offset it back down by the same amount to get the correct total specific energy.
        ener -= energy_shift
        
        return ener

    def calculate_volumes(radii):
        """
        Calculate the volumes of spherical shells given their radii.

        Args:
            radii (array-like):  An array of radii defining the boundaries of the shells.

        Returns:
            array: An array of volumes corresponding to each shell.
        """
        
        radii = np.asarray(radii)
        return (4/3) * np.pi * (radii[1:] ** 3 - radii[:-1] ** 3)
