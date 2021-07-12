#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes an isothermal Parker (planetary) wind model.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy.optimize as so
from p_winds import tools


__all__ = ["sound_speed", "radius_sonic_point", "density_sonic_point",
           "structure"]


# Speed of sound
def sound_speed(temperature, h_he_fraction, ion_fraction=0.0):
    """
    Speed of sound in an isothermal ideal gas.

    Parameters
    ----------
    temperature (``float``):
        Constant temperature of the gas in Kelvin. Assumed to be close to the
        maximum thermospheric temperature (see Oklopčić & Hirata 2018 and
        Lampón et al. 2020 for more details).

    h_he_fraction (``float``):
        Average H/He fraction of the upper atmosphere.

    ion_fraction (``float``):
        Average ionization fraction of the upper atmosphere.

    Returns
    -------
    sound_speed (``float``):
        Sound speed in the gas in unit of km / s.
    """
    # H has one proton and one electron
    # He has 2 protons and 2 neutrons, and 2 electrons
    he_h_fraction = 1 - h_he_fraction
    mu = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + ion_fraction)
    m_h = 1.67262192369e-27  # Hydrogen mass in kg
    mean_molecular_weight = m_h * mu
    k_b = 1.380649e-29  # Boltzmann constant in km ** 2 / s ** 2 * kg / K
    return (k_b * temperature / mean_molecular_weight) ** 0.5


# Radius of sonic point
def radius_sonic_point(planet_mass, sound_speed_0):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound.

    Parameters
    ----------
    planet_mass (``float``):
        Planetary mass in unit of Jupiter mass.

    sound_speed (``float``):
        Constant speed of sound in unit of km / s.

    Returns
    -------
    radius_sonic_point (``float``):
        Radius of the sonic point in unit of Jupiter radius.
    """
    grav = 1772.0378503888546  # Gravitational constant in unit of
    # jupiterRad * km ** 2 / s ** 2 / jupiterMass
    return grav * planet_mass / 2 / sound_speed_0 ** 2


# Density at the sonic point
def density_sonic_point(mass_loss_rate, radius_sp, sound_speed_0):
    """
    Density at the sonic point, where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    mass_loss_rate (``float``):
        Total mass loss rate of the planet in units of g / s.

    radius_sp (``float``):
        Radius at the sonic point in unit of Jupiter radius.

    sound_speed_0 (``float``):
        Speed of sound, assumed to be constant, in units of km / s.

    Returns
    -------
    rho_sp (``float``):
        Density at the sonic point in units of g / cm ** 3.
    """
    vs = sound_speed_0 * 1E5  # Convert sound speed to cm / s
    rs = radius_sp * 7.1492E+09  # Convert radius to cm
    rho_sp = mass_loss_rate / 4 / np.pi / rs ** 2 / vs
    return rho_sp


# Velocity and density in function of radius at the sonic point
def structure(r):
    """
    Calculate the velocity and density of the atmosphere in function of radius
    at the sonic point (r_s), and in units of sound speed (v_s) and density at
    the sonic point (rho_s), respectively.

    Parameters
    ----------
    r (``numpy.ndarray`` or ``float``):
        Radius at which to sample the velocity in unit of radius at the sonic
        point.

    Returns
    -------
    velocity_r (``numpy.ndarray`` or ``float``):
        `numpy` array or a single value of velocity at the given radius or radii
        in unit of sound speed.

    density_r (``numpy.ndarray`` or ``float``):
        Density sampled at the radius or radii `r` and in unit of density at the
        sonic point.

    """
    def _eq_to_solve(v, r_k):
        eq = v * np.exp(-0.5 * v ** 2) - (1 / r_k) ** 2 * np.exp(
            -2 * 1 / r_k + 3 / 2)
        return eq

    # The transcendental equation above has many solutions. In order to converge
    # to the physical solution for a planetary wind, we need to provide
    # different first guesses depending if the radius at which we are evaluating
    # the velocity is either above or below the sonic radius. If the radius is
    # below, the first guess is 0.1. If the radius is above, the first guess
    # is 2.0. This is a hacky solution, but it seems to work well.

    if isinstance(r, np.ndarray):
        # If r is a ``numpy.ndarray``, we do a dirty little hack to setup an
        # array of first guesses `x0` whose values are 0.1 for r <= 1, and 2 if
        # r > 1.
        ind = tools.nearest_index(r, 1.0)  # Find the index where r == 1.0
        x0_array = np.ones_like(r) * 0.1   # Setup the array of first guesses
        x0_array[ind:] *= 20.0
        velocity_r = np.array([so.newton(_eq_to_solve, x0=x0_array[k],
                                         args=(r[k],)) for k in range(len(r))])
    # If r is float, just do a simple if statement
    elif r <= 1.0:
        velocity_r = so.newton(_eq_to_solve, x0=1E-1, args=(r,))
    else:
        velocity_r = so.newton(_eq_to_solve, x0=2.0, args=(r,))

    density_r = np.exp(2 * 1 / r - 3 / 2 - 0.5 * velocity_r ** 2)

    return velocity_r, density_r
