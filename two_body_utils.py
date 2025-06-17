"""
Constants, Variables, and Functions used in two_body.ipynb 
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyasl
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from datetime import datetime


"""
Constants and Variables
"""
# Number of time intervals in array
n_int = 10000

# Solar Mass
M_SUN = const.M_sun.to(u.kg)

# Masses of orbiting bodies in solar masses
m_primary = 50
m_secondary = 1

# Distance to Omega Centauri center (km)
distance_kpc = 5.43 * u.kpc
distance_km= distance_kpc.to(u.km)

# M is the mean anomaly
M = 0.75

# m2m1 is the ratio of secondary to primary masses
m2m1 = m_secondary/m_primary

# tau is time of periapsis (days)
tau = 0.0

# mtot is the total mass of the system in solar masses
mtot = m_primary 

# semi_major_sample (In km!!!!) is a sample semi-major axis relative to COM in kilometers
# sample uses the two highest velocity omega centauri stars
# using average angular distance from central mass between the two stars
# apporximately 0.5" at a cluster distance 5.43kpc
angular_distance_rad = (0.5*u.arcsec).to(u.rad).value
semi_major_sample = angular_distance_rad*distance_km

# Other semi-major axis speculations:
semi_major_sample = (200*u.au).to(u.km)


# semi major axis used in IMBH_av_plots in kilometers
a_IMBH_av_plots = (0.048*u.pc).to(u.km)
# semi_major_primary is the semi major axis of M_secondary's orbit in kilometers
semi_major_primary = 0*u.km
# a_secondary is the semi-major axis of M_secondary's orbit in kilometers
semi_major_secondary = semi_major_sample

# v_xyz_sample is a sample 3D velocity relative to the COM in meters per second
# sample uses the two highest velocity omega centauri stars
# using average proper motion from central mass between the two stars
# apporximately 100 km/s at a cluster distance 5.43kpc
v_xyz_sample = 100*u.km/u.s

# Omega is the longitude of the ascending node in degrees
# Describes the angle between the reference direction (usually the vernal equinox) and the ascending node of the orbit
# plane of sky components are best approximated using magnitude*cos(i) when Omega = 0
Omega = 0

# w is the argument of periapsis in degrees
# (between where the orbiting body crosses the reference plane going north, and the periapsis)
# plane of sky components are best approximated using magnitude*cos(i) when w = 0
w = 0

# i is the orbital inclination in degrees
i = 0

# nu is the true annomaly
nu = None


"""
Variables better  defined in two_body.ipynb
"""

# e is eccentricity
# 100 random eccentricities from the thermal distribution 
# The CDF (F(e)) gives probability
# The inverse CDF turns uniform random samples into samples that match probability density
# f(e) = 2e, so more eccentric orbits are more likely than circular
# u=F(e)=e, each proability value is equally likely and maps to some value of e 
p = np.random.uniform(0, 1, n_int)
e = np.sqrt(p)





"""
Collection of Utilities
"""

def eccentric_annomaly(M = M, e=e):
    """
    Uses solver for Kepler's Equation for a set
    of mean anomaly and eccentricity.
    """
    ks = pyasl.MarkleyKESolver()
    E = ks.getE(M,e)
    return E


def xy_orbital_acceleration_secondary(m_primary = m_primary, rd = None, i = i):
    """
    Compute the component of orbital acceleration in the plane of the sky (xy)
    using Newton's law of gravitation 

    Parameters:
        m_primary: mass of primary body (kg)
        rd: relative distance between the two bodies (m)
        i: orbital inclination of the binary relative to xy (rad)

    Returns:
        The magnitudes (km/s²) of the XY acceleration as an array
    """
    
    # xyz acceleration vector
    a_xyz_secondary = const.G * m_primary*M_SUN / rd.to(u.m)**2

    # x,y components
    a_xy_secondary = a_xyz_secondary*np.cos(i)

    # in km/s²
    a_xy_secondary /= 1000
    
    return a_xy_secondary


def true_anomaly(E, e):
    """
    Calculate the true anomaly (nu) from eccentric anomaly E and eccentricity e.
    """
    sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
    nu = np.arctan2(sin_nu, cos_nu)
    return np.mod(nu, 2*np.pi)  # convert to [0, 2pi)


def com_radius(a=semi_major_sample, e=e, nu=nu):
    """
    Compute the relative distance of a body from the center of mass in a Keplerian orbit.

    Parameters
    ----------
    a : Semi-major axis of the orbit [same units as output = m].
    e : Orbital eccentricity (0 <= e < 1).
    nu : True anomaly in radians.

    Returns
    -------
    Distance from the center of mass to the orbiting body at true anomaly in same units as a
    """
    return a * (1 - e**2) / (1 + e * np.cos(nu))


def relative_distance(a_primary, a_secondary, e, nu):
    """
    Compute the distance between two orbiting bodies at a given true anomaly,
    using the com_radius function to get their individual distances from the center of mass. 
    Inputs must be meters!!!

    Parameters
    ----------
    a_primary : Semi-major axis of the primary body.
    a_secondary : Semi-major axis of the secondary body.
    e : Orbital eccentricity.
    nu : True anomaly in radians.

    Returns
    -------
    Distance between the two bodies at true anomaly `nu` in meters.
    """
    r1 = com_radius(a_primary, e, nu)
    r2 = com_radius(a_secondary, e, nu)
    
    return np.abs(r1 - r2)


def masyr2_to_kms2(a_masyr2=None, distance_km=distance_km):
    """
    Convert angular acceleration from milliarcseconds per year squared (mas/yr²)
    to linear acceleration in kilometers per second squared (km/s²).

    Parameters
    ----------
    a_masyr2 : Angular acceleration 
    distance_km : Distance to the object in kilometers 
    Returns
    -------
    Quantity
        Linear acceleration in km/s².
    """
    # Convert mas/yr² to rad/yr²
    a_radyr2 = a_masyr2.to(u.rad / u.yr**2, equivalencies=u.dimensionless_angles())

    # a (rad/yr²) × distance (km) = km/yr²
    # a_kmyr2 = a_radyr2 * distance_km

    # a (rad/yr²) × distance (km) = km/yr²
    a_kmyr2 = a_radyr2.value * distance_km.value * u.km / u.yr**2

    # Convert km/yr² to km/s²
    a_kms2 = a_kmyr2.to(u.km / u.s**2)

    return a_kms2

def circular_period(semi_major, speed):
    """
    Calculate the orbital period in seconds.
    
    Parameters:
        semi_major: semi major axis (kilometers)
        speed: Orbital speed (km/s)

    Returns:
         Orbital period (seconds)
    """
    circ = 2 * np.pi * semi_major
    return circ/speed


def orbital_speed(a, e, nu, m_primary=m_primary*M_SUN):
    """
    Calculate orbital velocity magnitude at a given true anomaly using the vis-viva equation.

    Parameters
    ----------
    a : Semi-major axis of orbiting body[m]
    e : Orbital eccentricity
    nu : True anomaly [radians]
    m_primary : Mass of the central body [kg]

    Returns
    -------
    Orbital speed [m/s]
    """
    # Gravitational parameter μ = G * M
    mu = const.G * m_primary

    # Radial distance at true anomaly nu
    r = com_radius(a, e, nu)

    # Vis-viva equation
    v = np.sqrt(mu * (2 / r - 1 / a))

    return v


def true_anomaly_to_time(nu, e, a, m_primary=m_primary*M_SUN):
    """
    Convert true anomaly to time since periastron/periapsis (tau) for a Keplerian orbit.
    Assume tau is zero.

    Parameters
    ----------
    nu : True anomaly [radians]
    e : Orbital eccentricity (0 <= e < 1)
    a : Semi-major axis [meters]
    m_primary : Mass of the central body [kg], default is solar mass

    Returns
    -------
    t : Time since periastron passage [seconds]
    """

    # Compute eccentric anomaly E from true anomaly ν
    E = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(nu / 2))

    # Ensure E is between 0 and 2π
    E = np.mod(E, 2 * np.pi)

    # Compute mean anomaly M from eccentric anomaly
    M = E - e * np.sin(E)

    # Compute mean motion n (rad/s)
    mu = const.G * m_primary  # Gravitational parameter [m³/s²]
    n = np.sqrt(mu / a**3)    # Mean motion

    # Step 4: Compute time since periastron
    t = M / n  # Time since tau [seconds]

    return t


def orbital_period_days(semi_major, speed):
    """
    Compute the orbital period in days given a semi-major axis and orbital speed.

    Parameters
    ----------
    semi_major : Semi-major axis of the orbit (astropy Quantity with units of length, e.g., km).
    speed : Orbital speed (astropy Quantity with units of velocity, e.g., km/s).

    Returns
    -------
    period_days : Orbital period in days (astropy Quantity).
    """
    # Ensure input units are correct
    semi_major = semi_major.to(u.km)
    speed = speed.to(u.km / u.s)
    
    # Period T = 2πa / v for approximate circular orbit
    period_sec = (2 * np.pi * semi_major / speed).to(u.s)
    
    # Convert to days
    return period_sec.to(u.day)

