###########################################################################
### units.py
###
### Holds conversion factors and constants for use throughout the code.
###########################################################################

import astropy.units as u
import astropy.constants as const

# Constants
#M_SUN = 1.989E33        # Mass of the sun in grams
#R_SUN = 6.959E10        # Radius of the sun in centimeters
#SIGMA_B = 5.669E-5      # Stefan-Boltzmann constant
#G = 6.67430E-8          # Gravitational constant
#K_B = 8.61733326e-11    # Boltzmann constant in MeV/K

# Custom units
u.add_enabled_units([
    u.def_unit('k_B', 1.0 * const.k_B, format={'latex': r'k_B', 'unicode': 'k_B'}),
    u.def_unit('baryon'),
    u.def_unit('mass_fraction'),
    u.def_unit('number_fraction')
])

desired_units = {
    "cell_mass": "g",
    "enclosed_mass": "g",
    "radius": "cm",
    "radial_velocity": "cm/s",
    "cell_volume": "cm^3",
    "grav_potential": "erg/g",
    "angular_velocity": "rad/s",
    "density": "M_sun/cm^3",
    "temperature": "K",
    "pressure": "dyne/cm^2",
    "specific_energy": "erg/g",
    "specific_entropy": "k_B/baryon",
    "total_specific_energy": "erg/g",
    "A_bar": "",
    "Y_e": ""
}


class Variable(u.Quantity):
    
    def __new__(cls, name, values, units, location="average", **kwargs):
        """
        Initializes a Variable instance.

        Args:
            name (str): The name of the variable.
            values (array-like): The values associated with the variable.
            units (str): The units of the variable.
            location (str, optional): The location of the variable ('outer', 'center', or 'average')
        """
        
        unit = u.Unit(units) if type(units) is str else units
        obj = super().__new__(cls, values, unit=unit, **kwargs)
        obj.name = name
        obj.location = location
        return obj
    
    @property
    def v(self): return self.value

    @property
    def u(self): return self.unit

    @property
    def l(self): return self.location

    @l.setter
    def l(self, location): self.shift_location(self, location)

    @property
    def loc(self): return self.location

    @l.setter
    def loc(self, location): self.shift_location(self, location)
    
    def shift_location(self, new_location):
        """
        Shift the location of the variable.

        Args:
            new_location (str): The new location ('outer', 'center', or 'average').
        """
        self.location = new_location


def correct_units(
    data):
    
    """
    Correct the units of the data to match the OSNAP data specification.
    
    Args:
        data (dict): A dictionary containing instances of Variable.
    """
    
    for column in data:
        
        # If a desired unit is given, ensure the data is in that unit.
        if column in desired_units:
            current_units = data[column].u
            target_units = u.Unit(desired_units[column])
            if current_units != target_units:
                print(f"Converting {column} from {current_units} to {target_units}.")
                data[column] = data[column].to(target_units)
                print(type(data[column]))
                # data[column].values = convert(data[column].values, current_units, target_units)
                # data[column].units = target_units
                
        # If not, print a warning
        else:
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