###########################################################################
### viz.py
###
### Methods for quickly creating visualizations.
###########################################################################

import matplotlib.pyplot as plt
from osnap import odb, units
import astropy.units as u
import numpy as np

def plot_profile(data, group_path, profile, x_axis = "enclosed_mass", save_path = None, log_x = False, log_y = False, x_units = None, y_units = None):
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
    if profile == x_axis:
        y_data = units.Variable(y_data.name, np.diff(x_data.value, prepend=x_data.value[0]), x_data.unit, x_data.loc)
    
    # Conver the units if they are specified
    if x_units:
        x_data = x_data.to(u.Unit(x_units))
    if y_units:
        y_data = y_data.to(u.Unit(y_units))
    
    plt.plot(x_data.value, y_data.value)
    
    if log_x:
        plt.xscale('log')
    
    if log_y:
        plt.yscale('log')
    
    plt.title(f"{group_path}: {profile} vs {x_axis}")
    plt.xlabel(x_axis + f" ({x_data.unit.to_string()})")
    plt.ylabel(profile + f" ({y_data.unit.to_string()})")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
def plot_tracers(data, profile, zone = 0, save_path = None, log_x = False, log_y = False, y_units = None):
    """
    Plots a tracer over time for a given profile.

    Args:
        data (odb.ODB or str): The odb object containing the profile data, or the path to an ODB file.
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
    if "tracers" not in groups:
        raise ValueError(f"Group 'tracers' not found in ODB. Available groups: {groups}")
    
    # If either dataset doesn't exist, raise an error
    datasets = data.list_datasets('tracers')
    if "time" not in datasets:
        raise ValueError(f"X-axis dataset 'time' not found in group 'tracers'. Available datasets: {datasets}")
    if profile not in datasets:
        raise ValueError(f"Y-axis dataset '{profile}' not found in group 'tracers'. Available datasets: {datasets}")
    
    plt.figure(figsize=(10, 6))
    
    x_data = data.read("tracers", "time")
    y_data = data.read("tracers", profile)
    enclosed_mass = data.read("tracers", "enclosed_mass")
    
    # Conver the units if they are specified
    if y_units:
        y_data = y_data.to(u.Unit(y_units))
    
    plt.plot(x_data.value, y_data.value[:, zone])
    
    if log_x:
        plt.xscale('log')
    
    if log_y:
        plt.yscale('log')
    
    plt.title(f"tracers: {profile} vs time ({enclosed_mass.value[zone]:.4f} {enclosed_mass.unit.to_string()})")
    plt.xlabel("time" + f" ({x_data.unit.to_string()})")
    if y_data.unit.to_string() == "":
        plt.ylabel(profile)
    else:
        plt.ylabel(profile + f" ({y_data.unit.to_string()})")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()