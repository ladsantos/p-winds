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


__all__ = ["sound_speed", "radius_sonic_point", "velocity_profile",
           "density_profile"]


# Speed of sound
def sound_speed(T_0, mean_molecular_weight):
    """
    Speed of sound in an isothermal ideal gas.

    Parameters
    ----------
    T_0 (``float``): Constant temperature of the gas.

    mean_molecular_weight (``float``): Mean molecular weight of the gas.

    Returns
    -------
    sound_speed (``float``): Sound speed in the gas

    """
    return (c.k_B * T_0 / mean_molecular_weight) ** 0.5


# Radius of sonic point
def radius_sonic_point(planet_mass, sound_speed_0):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound.

    Parameters
    ----------
    planet_mass (``float``): Planetary mass.

    sound_speed (``float``): Constant speed of sound.

    Returns
    -------
    radius_sonic_point (``float``): Radius of the sonic point

    """
    return c.G * planet_mass / 2 / sound_speed_0 ** 2


# Velocity profile
def velocity_profile(r_array, sound_speed_0, radius_sp):
    """
    Calculate the velocity profile of the atmosphere in function of radius.

    Parameters
    ----------
    r_array (``np.ndarray`` or ``float``): `numpy` array or a single value of
        radius at which to evaluate the velocity.

    sound_speed_0 (``float``): Constant speed of sound.

    radius_sp (``float``): Radius of the sonic point.

    Returns
    -------
    velocity (``np.ndarray`` or ``float``): `numpy` array or a single value of
        velocity at the given radius or radii.

    """
    # This function needs to do an optimization, which is not possible with
    # arrays containing astropy units. So we need to remove the units first in
    # case the input arrays contain them.
    try:
        vs = sound_speed_0.to(u.km / u.s).value
        rs = radius_sp.to(u.jupiterRad).value
        r = r_array.value
        v_unit = u.km / u.s
    except AttributeError:
        vs = sound_speed_0
        rs = radius_sp
        r = r_array
        v_unit = 1.0

    def _eq_to_solve(v, r):
        eq = v / vs * np.exp(-0.5 * (v / vs) ** 2) - (rs / r) ** 2 * np.exp(
            -2 * rs / r + 3 / 2)
        return eq

    # If the radius is a `float`
    if isinstance(r, float):
        velocity = so.newton(_eq_to_solve, x0=1E-1, args=(r,))
    # If the radius is a `numpy` array
    elif isinstance(r, np.ndarray):
        velocity = np.array([so.newton(_eq_to_solve, x0=1E-1,
                                       args=(rk,)) for rk in r])
    else:
        raise TypeError('The radius has to be `float` or `numpy.ndarray`.')
    return velocity * v_unit


# Density profile
def density_profile(r, v, radius_sp, sound_speed_0, density_sp):
    """
    Calculate the density profile of the atmosphere in function of radius.

    Parameters
    ----------
    r (``numpy.ndarray`` or ``float``): Radius.

    v (``numpy.ndarray`` or ``float``): Velocity sampled at the radius or radii
        `r`.

    radius_sp (``float``): Radius of the sonic point.

    sound_speed_0 (``float``): Constant sound speed.

    density_sp (``float``): Density at the sonic point.

    Returns
    -------
    density (``numpy.ndarray`` or ``float``): Density sampled at the radius or
        radii `r`.

    """
    vs = sound_speed_0
    rs = radius_sp
    rhos = density_sp
    density = rhos * np.exp(2 * rs / r - 3 / 2 - 0.5 * (v / vs) ** 2)
    return density
