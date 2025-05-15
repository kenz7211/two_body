"""
Constants, Parameters, and Functions used in two_body.ipynb 
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyasl
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from datetime import datetime

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
m2m1 = 1/8200

# tau is time of periapsis (days)
tau = 0.0

# mtot is the total mass of the system in solar masses
mtot = 8200*M_SUN

# per is period in s
per = 1

# Omega is the longitude of the ascending node in degrees
Omega = 0

# w is the argument of periapsis in degrees
# (where the orbiting body crosses the reference plane going north, and the periapsis)
w = 0

# i is the orbital inclination
i = 0

#nu is the true annomaly
nu = None

# a is the semi-major axis relative to COM in km
a = 1
# a_primary is the semi-major axis of m_primary's orbit
a_primary = 0
# a_secondary is the semi-major axis of M_secondary's orbit
a_secondary = 1

# e is eccentricity
# 100 random eccentricities from the thermal distribution 
# The CDF (F(e)) gives probability
# The inverse CDF turns uniform random samples into samples that match probability density
# f(e) = 2e, so more eccentric orbits are more likely than circular
# u=F(e)=e, each proability value is equally likely and maps to some value of e 
p = np.random.uniform(0, 1, n_int)
e = np.sqrt(p)


# E is the eccentric annomaly of the orbit
def eccentric_annomaly(M = M, e = e):
    """
    Uses solver for Kepler's Equation for a set
    of mean anomaly and eccentricity.
    """
    ks = pyasl.MarkleyKESolver()
    E = ks.getE(M,e)
    return E


# XY acceleration of m_secondary
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
def com_radius(a=a, e=e, nu=nu):
    """
    Compute the relative distance of a body from the center of mass in a Keplerian orbit.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit [same units as output].
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
def relative_distance(a_primary=a_primary, a_secondary=a_secondary, e=e, nu=nu):
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




