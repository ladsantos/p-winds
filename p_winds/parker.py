#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes an isothermal Parker (planetary) wind model.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
import scipy.optimize as so


__all__ = ["sound_speed", "radius_sonic_point", "density_sonic_point",
           "structure"]


# Speed of sound
def sound_speed(temperature, h_he_fraction):
    """
    Speed of sound in an isothermal ideal gas. The input values must be
    `astropy.Quantity`.

    Parameters
    ----------
    temperature (``float``): Constant temperature of the gas. Assumed to be
        close to the maximum thermospheric temperature (see Oklopčić & Hirata
        2018 and Lampón et al. 2020 for more details).

    h_he_fraction (``float``): Average H/He fraction of the upper atmosphere.

    Returns
    -------
    sound_speed (``float``): Sound speed in the gas

    """
    # H has one proton, He has 2 protons and 2 neutrons
    mean_molecular_weight = c.m_p * (h_he_fraction + (1 - h_he_fraction) * 4)
    return (c.k_B * temperature / mean_molecular_weight) ** 0.5


# Radius of sonic point
def radius_sonic_point(planet_mass, sound_speed_0):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    planet_mass (``float``): Planetary mass.

    sound_speed (``float``): Constant speed of sound.

    Returns
    -------
    radius_sonic_point (``float``): Radius of the sonic point

    """
    return c.G * planet_mass / 2 / sound_speed_0 ** 2


# Density at the sonic point
def density_sonic_point(mass_loss_rate, radius_sp, sound_speed_0):
    """
    Density at the sonic point, where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    mass_loss_rate (``float``): Total mass loss rate of the planet

    radius_sp (``float``):

    sound_speed_0 (``float``):

    Returns
    -------

    """
    rho_sp = mass_loss_rate / 4 / np.pi / radius_sp ** 2 / sound_speed_0
    return rho_sp


# Velocity and density in function of radius at the sonic point
def structure(r):
    """
    Calculate the velocity and density of the atmosphere in function of radius
    at the sonic point (r_s), and in units of sound speed (v_s) and density at
    the sonic point (rho_s), respectively.

    Parameters
    ----------
    r (``np.ndarray`` or ``float``): Radius at which to sample the velocity in
        unit of radius at the sonic point.

    Returns
    -------
    velocity_r (``np.ndarray`` or ``float``): `numpy` array or a single value
        of velocity at the given radius or radii in unit of sound speed.

    density_r (``numpy.ndarray`` or ``float``): Density sampled at the radius
        or radii `r` and in unit of density at the sonic point.

    """
    def _eq_to_solve(v, r_k):
        eq = v * np.exp(-0.5 * v ** 2) - (1 / r_k) ** 2 * np.exp(
            -2 * 1 / r_k + 3 / 2)
        return eq

    try:
        velocity_r = np.array([so.newton(_eq_to_solve, x0=1E-1, args=(rk,))
                               for rk in r])
    except TypeError:
        velocity_r = so.newton(_eq_to_solve, x0=1E-1, args=(r,))

    density_r = np.exp(2 * 1 / r - 3 / 2 - 0.5 * velocity_r ** 2)

    return velocity_r, density_r
