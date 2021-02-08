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


__all__ = ["sound_speed", "radius_sonic_point", "velocity", "density"]


# Speed of sound
def sound_speed(temperature, mean_molecular_weight):
    """
    Speed of sound in an isothermal ideal gas. The input values must be
    `astropy.Quantity`.

    Parameters
    ----------
    temperature (``float``): Constant temperature of the gas. Assumed to be
        close to the maximum thermospheric temperature (see Oklopčić & Hirata
        2018 and Lampón et al. 2020 for more details).

    mean_molecular_weight (``float``): Mean molecular weight of the gas.

    Returns
    -------
    sound_speed (``float``): Sound speed in the gas

    """
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
    Density at the sonic point, where the wind speed matches the speed of sound.
    The input values must be `astropy.Quantity`.

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


# Velocity in function of radius
def velocity(r):
    """
    Calculate the velocity of the atmosphere in function of radius in unit of
    sound speed.

    Parameters
    ----------
    r (``np.ndarray`` or ``float``): Radius at which to sample the velocity in
        unit of radius at the sonic point.

    Returns
    -------
    velocity_r (``np.ndarray`` or ``float``): `numpy` array or a single value
        of velocity at the given radius or radii in unit of sound speed.

    """
    def _eq_to_solve(v, r):
        eq = v * np.exp(-0.5 * v ** 2) - (1 / r) ** 2 * np.exp(
            -2 * 1 / r + 3 / 2)
        return eq

    try:
        velocity_r = np.array([so.newton(_eq_to_solve, x0=1E-1, args=(rk,))
                        for rk in r])
    except TypeError:
        velocity_r = so.newton(_eq_to_solve, x0=1E-1, args=(r,))
    return velocity_r


# Density in function of radius
def density(r):
    """
    Calculate the density profile of the atmosphere in function of radius and in
    unit of density at the sonic point.

    Parameters
    ----------
    r (``numpy.ndarray`` or ``float``): Radius in unit of radius of the sonic
        point.

    Returns
    -------
    density_r (``numpy.ndarray`` or ``float``): Density sampled at the radius
        or radii `r` and in unit of density at the sonic point.

    """
    v_r = velocity(r)
    density_r = np.exp(2 * 1 / r - 3 / 2 - 0.5 * v_r ** 2)
    return density_r
