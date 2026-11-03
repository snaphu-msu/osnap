###########################################################################
### viz.py
###
### Methods for quickly creating visualizations.
###########################################################################

import matplotlib.pyplot as plt
from osnap import odb
import astropy.units as u

def plot_profile(data, group_path, profile, x_axis = "enclosed_mass", save_path = None, log = False, x_units = None, y_units = None):
    """
    Plots a given profile from an ODB object or file

    Args:
        data (odb.ODB or str): The odb object containing the profile data, or the path to an ODB file.
        group_path (str): Path to the group that contains the relevant datasets (ex. "progenitor" or "profiles").
        profile (str): Name of the dataset to plot on the y-axis (ex. "density" or "temperature").
        x_axis (str, optional): Name of the dataset to use on the x-axis. Generally "enclosed_mass", "radius", or "time". \
            Defaults to "enclosed_mass".
        save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
        log (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to False.
    """
    
    # If the data provided is a path, load the ODB at that path
    if type(data) == str:
        data = odb.ODB(data)
    
    # If the group doesn't exist, raise an error
    groups = data.list_groups()
    if group_path not in groups:
        raise ValueError(f"Group '{group_path}' not found in ODB. Available groups: {groups}")
    
    # If either dataset doesn't exist, raise an error
    datasets = data.list_datasets(group_path)
    if x_axis not in datasets:
        raise ValueError(f"X-axis dataset '{x_axis}' not found in group '{group_path}'. Available datasets: {datasets}")
    if profile not in datasets:
        raise ValueError(f"Y-axis dataset '{profile}' not found in group '{group_path}'. Available datasets: {datasets}")
    
    plt.figure(figsize=(10, 6))
    
    x_data = data.read(group_path, x_axis)
    y_data = data.read(group_path, profile)
    
    # Conver the units if they are specified
    if x_units:
        x_data = x_data.to(u.Unit(x_units))
    if y_units:
        y_data = y_data.to(u.Unit(y_units))
    
    plt.plot(x_data.value, y_data.value)
    
    if log:
        plt.yscale('log')
    
    plt.title(f"{group_path}: {profile} vs {x_axis}")
    plt.xlabel(x_axis + f" ({x_data.unit.to_string()})")
    plt.ylabel(profile + f" ({y_data.unit.to_string()})")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()