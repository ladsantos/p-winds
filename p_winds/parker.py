#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes an isothermal Parker (planetary) wind model.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.constants as c
import scipy.optimize as so
from p_winds import tools


__all__ = ["sound_speed", "radius_sonic_point", "density_sonic_point",
           "structure"]


# Speed of sound
def sound_speed(temperature, h_he_fraction, ion_fraction=0.0):
    """
    Speed of sound in an isothermal ideal gas. The input values must be
    ``astropy.Quantity``.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Constant temperature of the gas. Assumed to be close to the maximum
        thermospheric temperature (see Oklopčić & Hirata 2018 and Lampón et al.
        2020 for more details).

    h_he_fraction (``float``):
        Average H/He fraction of the upper atmosphere.

    ion_fraction (``float``):
        Average ionization fraction of the upper atmosphere.

    Returns
    -------
    sound_speed (``astropy.Quantity``):
        Sound speed in the gas.
    """
    # H has one proton and one electron
    # He has 2 protons and 2 neutrons, and 2 electrons
    he_h_fraction = 1 - h_he_fraction
    mu = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + ion_fraction)
    mean_molecular_weight = c.m_p * mu
    return (c.k_B * temperature / mean_molecular_weight) ** 0.5


# Radius of sonic point
def radius_sonic_point(planet_mass, sound_speed_0):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    planet_mass (``astropy.Quantity``):
        Planetary mass.

    sound_speed (``astropy.Quantity``):
        Constant speed of sound.

    Returns
    -------
    radius_sonic_point (``astropy.Quantity``):
        Radius of the sonic point.
    """
    return c.G * planet_mass / 2 / sound_speed_0 ** 2


# Density at the sonic point
def density_sonic_point(mass_loss_rate, radius_sp, sound_speed_0):
    """
    Density at the sonic point, where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    mass_loss_rate (``astropy.Quantity``):
        Total mass loss rate of the planet.

    radius_sp (``astropy.Quantity``):
        Radius at the sonic point.

    sound_speed_0 (``astropy.Quantity``):
        Speed of sound, assumed to be constant.

    Returns
    -------
    rho_sp (``astropy.Quantity``):
        Density at the sonic point.

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
