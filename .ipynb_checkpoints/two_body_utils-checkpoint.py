"""
Constants, Parameters, and Functions used in two_body.ipynb 
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyasl

# Number of time intervals in array
n_int = 200

# Gravitational Constant
G = 6.67430e-11

# Solar Mass
M_SUN = 1.989e30 

# Masses of orbiting bodies
m_primary = 9000*M_SUN
m_secondary = M_SUN


# M is the mean anomaly
M = 0.75
# m2m1 is the ratio of secondary to primary masses
m2m1 = 1e-9
# tau is time of periapsis (days)
tau = 0.0
# mtot is the total mass of the system in solar masses
mtot = 2.3
# per is period
per = 2
# e is eccentricity
e = 0.5
# Omega is the longitude of the ascending node in degrees
Omega = 180
# w is the argument of periapsis in degrees
w = 0
# i is the orbital inclination
i = 0
# rd is the relative distance between m_primary and m_secondary
 
#Solver for Kepler's equation
# Solves Kepler's Equation for a set
# of mean anomaly and eccentricity.
ks = pyasl.MarkleyKESolver()
print("Eccentric anomaly: ", ks.getE(M,e))

# XY acceleration of m_secondary
def xy_orbital_acceleration_secondary(m_primary = m_primary, rd = 1, i = i):
    """
    Compute the component of orbital acceleration in the plane of the sky (xy)
    using Newton's law of gravitation 

    Args:
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







