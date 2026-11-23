###########################################################################
### units.py
###
### Holds conversion factors and constants for use throughout the code.
###########################################################################

import astropy.units as u
import astropy.constants as const
import numpy as np
from scipy.interpolate import interp1d

# Custom units
u.add_enabled_units([
    u.def_unit('k_B', 1.0 * const.k_B, format={'latex': r'k_B', 'unicode': 'k_B'}),
    u.def_unit('baryon'),
    u.def_unit('mass fraction'),
    u.def_unit('number fraction')
])

desired_units = {
    "cell_mass": "g",
    "enclosed_mass": "M_sun",
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
    "Y_e": "",
    "neutrons": "mass fraction",
    "protons": "mass fraction"
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

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is not None:
            for attribute in ("name", "location"):
                if hasattr(obj, attribute):
                    setattr(self, attribute, getattr(obj, attribute))
    
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
    def loc(self, new_location): self.location = new_location

    @property
    def unitstr(self): return f"{self.unit}"
    
    # TODO: Finish implementing this
    def shifted(self, to, coords = None):
        
        # If already correct or not applicable, return the values as is
        if self.location == "N/A" or self.location == to:
            return self
        
        # Ensure the target location is valid
        if to not in ["inner", "outer", "center", "average"]:
            raise ValueError(f"Invalid target location '{to}'. Must be 'inner', 'outer', 'center', or 'average'.")
        
        # If no coords supplied, use the zone indices as the coords
        zones = np.arange(len(self.value))
        if coords is None:
            coords = zones
            
        # Create a finer grid of coordinates and interpolate the values to this finer grid
        coords_interp_func = interp1d(zones, coords, fill_value = "extrapolate")
        finer_zones = np.arange(0, len(self.value) + 1, 0.5)
        finer_coords = coords_interp_func(finer_zones)
        interp_func = interp1d(coords, self.value, fill_value = "extrapolate")
        # with np.printoptions(precision = 1, suppress = True):
        #     print(f"{finer_coords[(2 * (zones + 0.5)).astype(int)]}")
            
        # TODO: Unverified
        if to == "inner":
            if self.location == "center" or self.location == "average":
                values = np.concatenate(([0.0], interp_func(finer_coords[(2 * (zones[1:] - 0.5)).astype(int)])))
            elif self.location == "outer":
                values = np.concatenate(([0.0], interp_func(finer_coords[(2 * (zones[1:] - 1.0)).astype(int)])))
                
        # Verified to be working
        elif to == "outer":
            if self.location == "center" or self.location == "average":
                values = interp_func(finer_coords[(2 * (zones + 0.5)).astype(int)])
            elif self.location == "inner":
                values = interp_func(finer_coords[(2 * (zones + 1.0)).astype(int)])
                
        # TODO: Unverified
        elif to == "center" or self.location == "average":
            if self.location == "inner":
                values = interp_func(finer_coords[(2 * (zones + 0.5)).astype(int)])
            elif self.location == "outer":
                values = interp_func(finer_coords[(2 * (zones - 0.5)).astype(int)])
        
        return Variable(self.name, values, self.unit, to)


def correct_units(data, log = True):
    
    """
    Correct the units of the data to match the OSNAP data specification.
    
    Args:
        data (dict): A dictionary containing instances of Variable.
        log (bool): If True, logs warnings and information about unit conversions. Default is True.
    """
    
    unknown_units = []
    for column in data:
        
        # If a desired unit is given, ensure the data is in that unit.
        if column in desired_units:
            current_units = data[column].unit
            target_units = u.Unit(desired_units[column])
            if current_units != target_units:
                if log:
                    print(f"Converting {column} from {current_units} to {target_units}.")
                try:
                    data[column] = data[column].to(target_units)
                except:
                    print(f"Error: Could not convert {column} from {current_units} to {target_units}.")
                
        # If no desired units, ignore this column
        else:
            unknown_units.append(column)
            
    # Let user know if there are any columns whose units are not handled by OSNAP
    if log and unknown_units:
        print(f"Warning: No preferred units specified by OSNAP for the following columns - {unknown_units}")
    
    return data
