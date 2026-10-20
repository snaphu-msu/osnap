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
    "density": "g/cm^3",
    "temperature": "K",
    "pressure": "dyne/cm^2",
    "specific_energy": "erg/g",
    "specific_entropy": "k_B/baryon",
    "total_specific_energy": "erg/g",
    "A_bar": "",
    "Y_e": ""
}


class Variable(u.Quantity):
    """
    An object which stores the name, location, values and units of an ODB variable. 
    Overrides the astropy.units.Quantity class to include name and location attributes.
    
    Attributes:
        name (str): The name of the variable.
        value (array-like): The values associated with the variable.
        unit (astropy.units.Unit): The units of the variable.
        location/loc (str): The location of the variable ('outer', 'center', or 'average'). Also accessible by 'loc'.
    """
    
    def __new__(cls, name, values, units, location, **kwargs):
        """
        Initializes a Variable instance.

        Args:
            name (str): The name of the variable.
            values (array-like): The values associated with the variable.
            units (str or astropy.units.Unit): The units of the variable.
            location (str): The location of the variable ('outer', 'center', or 'average')
        """
        
        unit = u.Unit(units) if type(units) is str else units
        obj = super().__new__(cls, values, unit=unit, **kwargs)
        obj.name = name
        obj.location = location
        return obj
    
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if hasattr(self, "name"):
            result.name = self.name
        if hasattr(self, "location"):
            result.location = self.location
        return result
    
    to.__doc__ = ("Overrides the astropy.units.Quantity.to() method to ensure the location and \
        name are kept during conversions.\n\n" + u.Quantity.to.__doc__)

    @property
    def loc(self): return self.location

    @loc.setter
    def loc(self, location): self.shift_location(self, location)

    @property
    def unitstr(self): return f"{self.unit}"
    
    # TODO: Implement shifting locations. This currently only changes the location label without changing any values.
    def shift_location(self, new_location):
        """
        Shift the location of the variable.

        Args:
            new_location (str): The new location ('outer', 'center', or 'average').
        """
        self.location = new_location


def correct_units(data):
    
    """
    Correct the units of the data to match the OSNAP data specification.
    
    Args:
        data (dict): A dictionary containing instances of Variable.
    """
    
    for column in data:
        
        # If a desired unit is given, ensure the data is in that unit.
        if column in desired_units:
            current_units = data[column].unit
            target_units = u.Unit(desired_units[column])
            if current_units != target_units:
                print(f"Converting {column} from {current_units} to {target_units}.")
                data[column] = data[column].to(target_units)
                
        # If not, print a warning
        else:
            print(f"Warning: No units specified by OSNAP for {column}. Skipping unit correction.")
    
    return data