###########################################################################
### units.py
###
### Holds conversion factors and constants for use throughout the code.
###########################################################################

import astropy.units as u

# Constants
M_SUN = 1.989E33        # Mass of the sun in grams
R_SUN = 6.959E10        # Radius of the sun in centimeters
SIGMA_B = 5.669E-5      # Stefan-Boltzmann constant
G = 6.67430E-8          # Gravitational constant
K_B = 8.61733326e-11    # Boltzmann constant in MeV/K

desired_units = {
    "cell_mass": "g",
    "enclosed_mass": "g",
    "radius": "cm",
    "radial_velocity": "cm/s",
    "cell_volume": "cm^3",
    "grav_potential": "erg/g",
    "angular_velocity": "rad/s",
    "density": "g/cm^3",
    "temperature": "K",
    "pressure": "dyne/cm^2",
    "specific_energy": "erg/g",
    "specific_entropy": "erg/g/K",
    "total_specific_energy": "erg/g",
    "A_bar": "",
    "Y_e": ""
}

special_units = ["mass fraction", "number fraction"]


def correct_units(data):
    """
    Correct the units of the data to match the OSNAP data specification.
    
    Args:
        data (dict): A dictionary containing instances of Variable.
    """
    
    for column in data:
        
        # If a desired unit is given, ensure the data is in that unit.
        if column in desired_units:
            current_units = data[column].units
            target_units = desired_units[column]
            if current_units != target_units:
                print(f"Converting {column} from {current_units} to {target_units}.")
                data[column].values = convert(data[column].values, current_units, target_units)
                data[column].units = target_units
                
        # If not, and its not a special kind of unit, print a warning
        elif data[column].units not in special_units:
            print(f"Warning: No units specified by OSNAP for {column}. Skipping unit correction.")
    
    return data


def convert(values, from_units, to_units):
    """
    Convert data from one set of units to another.

    Args:
        values (array-like): The values of data to be converted.
        from_units (str): The current units of the data.
        to_units (str): The desired units for the data.
        
    Returns:
        array-like: The data converted to the desired units.
    """

    current_quantity = values * u.Unit(from_units)
    target_quantity = current_quantity.to(u.Unit(to_units))
    return target_quantity.value