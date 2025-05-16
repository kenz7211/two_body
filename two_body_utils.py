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
n_int = 10

# Gravitational Constant
G = 6.67430e-11

# Solar Mass
M_SUN = 1.989e30 

# Masses of orbiting bodies
m_primary = 8200*M_SUN
m_secondary = M_SUN

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
mtot = 8200*M_SUN

# semi_major_sample is a sample semi-major axis relative to COM in kilometers
# sample uses the two highest velocity omega centauri stars
# using average angular distance from central mass between the two stars
# apporximately 0.5" at a cluster distance 5.43kpc
angular_distance_rad = (0.5*u.mas).to(u.rad).value
semi_major_sample = angular_distance_rad*distance_km

# semi_major_primary is the semi major axis of M_secondary's orbit in kilometers
semi_major_primary = 0*u.km
# a_secondary is the semi-major axis of M_secondary's orbit in kilometers
semi_major_secondary = 1*u.km

# v_xyz_sample is a sample 3D velocity relative to the COM in meters per second
# sample uses the two highest velocity omega centauri stars
# using average proper motion from central mass between the two stars
# apporximately 100 km/s at a cluster distance 5.43kpc
v_xyz_sample = 100*u.km/u.s

# Omega is the longitude of the ascending node in degrees
Omega = 70

# w is the argument of periapsis in degrees
# (where the orbiting body crosses the reference plane going north, and the periapsis)
w = 30

# i is the orbital inclination in degrees
i = 50

#nu is the true annomaly
nu = None


"""
Variables better  defined in two_body.ipynb
"""

# per_sample is a sample period in s
# per_sample = circular_period(semi_major = semi_major_sample, speed = v_xyz_sample)

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

# E is the eccentric annomaly of the orbit
def eccentric_annomaly(M = M, e=e):
    """
    Uses solver for Kepler's Equation for a set
    of mean anomaly and eccentricity.
    """
    ks = pyasl.MarkleyKESolver()
    E = ks.getE(M,e)
    return E


# XY acceleration of m_secondary in km/s^2
def xy_orbital_acceleration_secondary(m_primary = m_primary, rd = None, i = i):
    """
    Compute the component of orbital acceleration in the plane of the sky (xy)
    using Newton's law of gravitation 

    Parameters:
        m_primary = mass of primary body (kg)
        rd = relative distance between the two bodies (m)
        i = orbital inclination of the binary relative to xy (rad)

    Returns:
        a_xy_secondary = The magnitudes (km/s²) of the XY acceleration as an array
    """
    
    # xyz acceleration vector
    a_xyz_secondary = G * m_primary / rd**2

    # x,y components
    a_xy_secondary = a_xyz_secondary*np.cos(i)

    # in km/s²
    a_xy_secondary /= 1000
    
    return a_xy_secondary

# True anomaly
def true_anomaly(E, e):
    """
    Calculate the true anomaly (nu) from eccentric anomaly and eccentricity
    using the true anomaly formula.

    Parameters:
    E = Eccentric anomaly in radians
    e = Eccentricity (0 <= e < 1)

    Returns:
        nu = true anomaly in radians
    """
    cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
    nu = np.arccos(cos_nu)

    return nu

# Distance from center of mass
def com_radius(a=semi_major_secondary, e=e, nu=nu):
    """
    Compute the relative distance of a body from the center of mass in a Keplerian orbit.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit [same units as output = m].
    e : float
        Orbital eccentricity (0 <= e < 1).
    nu : float
        True anomaly in radians.

    Returns
    -------
    float
        Distance from the center of mass to the orbiting body at true anomaly
    """
    return a * (1 - e**2) / (1 + e * np.cos(nu))

# Distance between orbiting bodies
def relative_distance(a_primary=semi_major_primary, a_secondary=semi_major_secondary, nu=nu, e=e):
    """
    Compute the relative distance between two orbiting bodies at a given true anomaly.

    Assumes both bodies share the same eccentricity and true anomaly, but have different semi-major axes.

    Parameters
    ----------
    a_primary : float
        Semi-major axis of the primary body.
    a_secondary : float
        Semi-major axis of the secondary body.
    e : float
        Orbital eccentricity.
    nu : float
        True anomaly in radians.

    Returns
    -------
    float
        Distance between the two bodies at true anomaly `nu`.
    """
    r_primary = com_radius(a_primary, e, nu)
    r_secondary = com_radius(a_secondary, e, nu)

    x_primary = r_primary * np.cos(nu)
    y_primary = r_primary * np.sin(nu)
    x_secondary = r_secondary * np.cos(nu)
    y_secondary = r_secondary * np.sin(nu)

    return np.sqrt((x_primary - x_secondary)**2 + (y_primary - y_secondary)**2)

# Convert angular accelerration to linear acceleration
def masyr2_to_kms2(a_masyr2=None, distance_km=distance_km):
    """
    Convert angular acceleration from milliarcseconds per year squared (mas/yr²)
    to linear acceleration in kilometers per second squared (km/s²).

    Parameters
    ----------
    a_masyr2 : 
        Angular acceleration 
    distance_km : 
        Distance to the object in kilometers 
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

# Calculate the period of a circular  orbit in seconds
def circular_period(semi_major, speed):
    """
    Calculate the orbital period in seconds.
    
    Parameters:
        semi_major (float): semi major axis (kilometers)
        speed (float): Orbital speed (km/s)

    Returns:
         Orbital period (seconds)
    """
    circ = 2 * np.pi * semi_major
    return circ/speed


