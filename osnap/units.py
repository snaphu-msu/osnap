###########################################################################
### units.py
###
### Holds conversion factors and constants for use throughout the code.
###########################################################################

import astropy.units as u
import astropy.constants as const
import numpy as np

# Custom units
u.add_enabled_units([
    u.def_unit('k_B', 1.0 * const.k_B, format={'latex': r'k_B', 'unicode': 'k_B'}),
    u.def_unit('baryon'),
    u.def_unit('mass fraction'),
    u.def_unit('number fraction')
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
    
#   def shift(self, new_location, coords, *, boundary="constant"):
#         """
#         Relocate a one-dimensional profile in place and return this Variable.

#         Values must be a nonempty, one-dimensional floating-point array in
#         increasing zone order. The coordinate is radius or enclosed mass.
#         All three locations retain one value per cell: ``outer`` samples its
#         outer face and ``center`` samples its midpoint in the chosen coordinate.
#         By default the averaging measure follows the coordinate:

#         * radius: volume average, integral(q(r) r**2 dr) / integral(r**2 dr)
#         * mass: mass average, integral(q(m) dm) / (m_outer - m_inner)

#         Point samples use piecewise linear interpolation in the chosen
#         coordinate. Conversion to averages integrates that interpolant over
#         each cell. Averages use a cellwise linear reconstruction about the
#         corresponding centroid with minmod-limited slopes, preserving each
#         supplied cell average. For mass averages in mass coordinates the
#         centroid is the midpoint, so average-to-center retains the values
#         under this linear reconstruction. When coordinate and measure differ,
#         supply both radius and mass boundaries. Their relation is approximated
#         as linear within each cell. Selecting a coordinate must match the
#         stored sampling; it does not convert between the two grids.
#         For average-to-outer conversion, the result is the reconstructed value
#         approached from inside that cell (the inner-side trace at its outer
#         face); the two sides of a discontinuity need not agree.

#         These are approximations: location changes generally do not round-trip
#         exactly, and the average of a nonlinear expression is not generally the
#         expression evaluated with averaged inputs. Cell integrals such as mass
#         or volume should not be relocated as though they were point samples.

#         Args:
#             new_location (str): The new location ('outer', 'center', or 'average').
#             coordinates (array-like or astropy.units.Quantity): N+1 face
#                 coordinates, including the innermost boundary, for N
#                 values. Must be finite, nonnegative and strictly increasing.
#                 Quantities must have length units for radius or mass units for
#                 enclosed mass. Bare arrays may use any consistent coordinate
#                 unit. A coordinate Variable must have location 'outer'.
#                 Use enclosed masses at faces, rather than individual cell
#                 masses. Centered coordinates alone do not specify boundaries.
#                 Required unless the location is unchanged. For an N-element
#                 outer-face profile, explicitly prepend the inner coordinate.
#             boundary (str): 'constant' (default) holds point samples constant
#                 outside their sampled interval and sets the first/last average
#                 reconstruction slopes to zero. 'linear' uses the nearest two
#                 samples for endpoint slopes/extrapolation, which can overshoot.
#                 A single cell always uses a constant reconstruction.

#         Returns:
#             Variable: A new object with the same name and units, but with a new location and values \
#                 shifted to that location.
#         """
        
#         shifted = self.copy()
#         if new_location == shifted.location:
#             return shifted
        
#         average = "volume" if coords.name == "radius" else "enclosed_mass"
        
#         edges = coords.values / coords.values[-1]
#         centers = edges[:-1] + np.diff(edges) / 2
#         measure_edges = None
#         if "average" in (shifted.location, new_location):
#             measure_coordinate = "radius" if average == "volume" else "enclosed_mass"
#             if coordinate == measure_coordinate:
#                 measure_edges = edges
#             else:
#                 companion_edges = radius_edges if average == "volume" else mass_edges
#                 measure_edges = _coordinate_edges(companion_edges, shifted.size, measure_coordinate)
#         centers = edges[:-1] + np.diff(edges) / 2
#         values = np.asarray(shifted.value, dtype=float)
#         target = edges[1:] if new_location == "outer" else centers
#         if shifted.location == "average":
#             centroids = _cell_centroid(edges[:-1], edges[1:],
#                                        measure_edges[:-1], measure_edges[1:], average)
#             slopes = np.zeros_like(values)
#             if shifted.size > 1:
#                 secants = np.diff(values) / np.diff(centroids)
#                 left, right = secants[:-1], secants[1:]
#                 # Minmod suppresses spurious extrema near discontinuities.
#                 slopes[1:-1] = np.where(
#                     np.sign(left) == np.sign(right),
#                     np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
#                 if boundary == "linear":
#                     slopes[0], slopes[-1] = secants[0], secants[-1]
#             converted = values + slopes * (target - centroids)
#         else:
#             samples = edges[1:] if shifted.location == "outer" else centers
#             if new_location == "average":
#                 converted = _cell_average(samples, values, edges, measure_edges, boundary, average)
#             else:
#                 converted = _interpolate_points(samples, values, target, boundary)
                
                
#         shifted.value[...] = converted
#         shifted.location = new_location
#         return shifted


# def _cell_centroid(inner, outer, measure_inner, measure_outer, average):
#     """Mean coordinate under the measure, using a cellwise linear mapping."""
#     fraction = 0.5
#     if average == "volume":
#         # The spherical radial centroid is offset from the radial midpoint.
#         # This ratio form avoids subtracting powers or squaring physical radii.
#         midpoint = measure_inner + (measure_outer - measure_inner) / 2
#         ratio = (measure_outer - measure_inner) / (2 * midpoint)
#         fraction += ratio / (3 + ratio**2)
#     return inner + (outer - inner) * fraction


# def _interpolate_points(samples, values, target, boundary):
#     result = np.interp(target, samples, values)
#     if boundary == "linear" and len(samples) > 1:
#         for first, second, mask in (
#                 (0, 1, target < samples[0]),
#                 (-1, -2, target > samples[-1])):
#             result[mask] = values[first] + (target[mask] - samples[first]) * (
#                 (values[second] - values[first]) / (samples[second] - samples[first]))
#     return result


# def _cell_average(samples, values, edges, measure_edges, boundary, average):
#     """Integrate a piecewise linear interpolant with the selected measure."""
#     # Split at every cell face and sample, so each segment is affine in coordinate.
#     knots = np.unique(np.concatenate((edges, samples)))
#     inner, outer = knots[:-1], knots[1:]
#     measure_knots = np.interp(knots, edges, measure_edges)
#     measure_inner, measure_outer = measure_knots[:-1], measure_knots[1:]
#     centroids = _cell_centroid(inner, outer, measure_inner, measure_outer, average)
#     segment_average = _interpolate_points(samples, values, centroids, boundary)
#     weight = measure_outer - measure_inner
#     if average == "volume":
#         # 4*pi/3 cancels; factoring avoids subtracting nearly equal cubes.
#         weight = weight * (measure_inner**2 + measure_inner * measure_outer + measure_outer**2)
#     cells = np.searchsorted(edges, inner, side="right") - 1
#     integral = np.bincount(cells, weights=segment_average * weight, minlength=len(edges) - 1)
#     total_weight = np.bincount(cells, weights=weight, minlength=len(edges) - 1)
#     return integral / total_weight


def correct_units(data, log_warnings = True):
    
    """
    Correct the units of the data to match the OSNAP data specification.
    
    Args:
        data (dict): A dictionary containing instances of Variable.
        log_warnings (bool): If True, log warnings for any columns whose units are not handled by OSNAP.
    """
    
    unknown_units = []
    for column in data:
        
        # If a desired unit is given, ensure the data is in that unit.
        if column in desired_units:
            current_units = data[column].unit
            target_units = u.Unit(desired_units[column])
            if current_units != target_units:
                print(f"Converting {column} from {current_units} to {target_units}.")
                try:
                    data[column] = data[column].to(target_units)
                except:
                    print(f"Error: Could not convert {column} from {current_units} to {target_units}.")
                
        # If no desired units, ignore this column
        else:
            unknown_units.append(column)
            
    # Let user know if there are any columns whose units are not handled by OSNAP
    if log_warnings and unknown_units:
        print(f"Warning: No preferred units specified by OSNAP for the following columns - {unknown_units}")
    
    return data
