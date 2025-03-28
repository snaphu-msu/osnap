import numpy as np
from astropy import units
from astropy import constants as const

cm_to_1k_km = units.cm.to(1e3 * units.km)
g_to_msun = units.g.to(units.M_sun)
sb = const.sigma_sb.cgs.value


def get_enclosed_mass(zone_mass):
    """Calculate enclosed mass by integrating zone mass

    Returns: np.array

    parameters
    ----------
    zone_mass : []
    """
    zone_mass = np.array(zone_mass)
    enc_mass = [zone_mass[0]]

    for zm in zone_mass[1:]:
        enc_mass += [enc_mass[-1] + zm]

    enc_mass = np.array(enc_mass)

    return enc_mass


def get_centered_mass(mass_edge, radius_edge, radius_center, density):
    """Calculate cell-centered enclosed mass

    Returns: np.array

    parameters
    ----------
    mass_edge : []
        cell-outer enclosed mass (Msun)
    radius_edge : []
        cell-outer radius (cm)
    radius_center : []
        cell-center radius (cm)
    density : []
        cell-average density (g/cm^3)
    """
    mass_edge = np.array(mass_edge)
    radius = np.array(radius_edge)

    # cell-inner radius
    radius_inner = np.zeros(len(radius))
    radius_inner[1:] = radius[:-1]

    # cell-inner enclosed mass
    mass_inner = np.zeros(len(mass_edge))
    mass_inner[1:] = mass_edge[:-1]

    # left-half mass of cell
    vol_lhalf = 4/3 * np.pi * (radius_center**3 - radius_inner**3)
    mass_lhalf = vol_lhalf * density * g_to_msun

    mass_center = mass_inner + mass_lhalf

    return mass_center


def get_centered_radius(radius_edge):
    """Calculate cell-centered radius from cell-outer radius

    Returns: np.array

    parameters
    ----------
    radius_edge : []
        cell-outer radius
    """
    radius_edge = np.array(radius_edge)

    dr = np.array(radius_edge)
    dr[1:] = np.diff(dr)  # cell width

    r_center = radius_edge - (0.5 * dr)

    return r_center


def get_interp_center(var_outer, radius_outer):
    """Interpolate cell-outer quantity to cell-center

    Returns: np.array

    parameters
    ----------
    var_outer : []
        cell-outer variable to interpolate
    radius_outer : []
        cell-outer radius
    """
    var_outer = np.array(var_outer)
    radius_outer = np.array(radius_outer)
    radius_center = get_centered_radius(radius_outer)

    var_center = np.interp(x=radius_center,
                           xp=radius_outer,
                           fp=var_outer,
                           left=0)

    # linearly-extrapolate innermost cell-center
    dv_dr = np.diff(var_outer[:2]) / np.diff(radius_outer[:2])
    var_center[0] = var_outer[0] + dv_dr * (radius_center[0] - radius_outer[0])

    return var_center


def get_xi(mass, radius):
    """Calculate compactness parameter

    Returns: np.array

    parameters
    ----------
    mass : []
        Enclosed mass coordinate (Msun)
    radius : []
        radius coordinate (cm)
    """
    mass = np.array(mass)
    radius = np.array(radius)

    xi = mass / (radius * cm_to_1k_km)

    return xi


def get_luminosity(radius, temperature):
    """Calculate blackbody luminosity

    Returns: np.array

    parameters
    ----------
    radius : []
        radius coordinate (cm)
    temperature : []
        Temperature coordinate (K)
    """
    radius = np.array(radius)
    temp = np.array(temperature)

    lum = 4 * np.pi * sb * radius**2 * temp**4

    return lum


def get_velz(radius, ang_vel):
    """Calculate tangential velocity (velz) from angular velocity

    Returns: np.array
        tangential velocity [cm/s]

    parameters
    ----------
    radius : []
        radius coordinate (cm)
    ang_vel : []
        angular velocity [rad/s]
    """
    radius = np.array(radius)
    ang_vel = np.array(ang_vel)

    velz = radius * ang_vel

    return velz


def get_vkep(radius, mass):
    """Calculate keplerian velocity

    Returns: np.array
        tangential velocity [cm/s]

    parameters
    ----------
    radius : []
        radius coordinate [cm]
    mass : []
        enclosed mass coordinate [Msun]
    """
    radius = np.array(radius)
    mass = np.array(mass)
    G = const.G.to(units.cm**3 / (units.M_sun * units.s**2))

    vkep = np.sqrt(G * mass / radius)

    return vkep.value
