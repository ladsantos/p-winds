#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes an isothermal Parker (planetary) wind model.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy.optimize as so
from scipy.integrate import simps, trapz
from astropy import units as u, constants as c

__all__ = ["average_molecular_weight", "sound_speed", "radius_sonic_point",
           "density_sonic_point", "structure", "radius_sonic_point_tidal",
           "structure_tidal"]


# Average mean-molecular weight following the formulation of Lampón et al. 2020
def average_molecular_weight(ion_fraction_profile, r_profile, v_profile,
                             planet_mass, temperature, he_h_fraction=0.1 / 0.9):
    """
    Calculates the "average" mean molecular weight of the upper atmosphere
    following Eq. A.3 in Lampón et al. 2020, in unit of proton mass.

    Parameters
    ----------
    ion_fraction_profile : ``numpy.ndarray``
        Hydrogen ion fraction in function of radial distance.

    r_profile : ``numpy.ndarray``
        Radial distance profile in unit of Jupiter radii. This is the
        independent variable over which the profiles are described.

    v_profile : ``numpy.ndarray``
        Velocity profile in units of km / s in function of radial distance.

    planet_mass : ``float``
        Planetary mass in unit of Jupiter mass.

    temperature : ``float``
        Isothermal temperature of the outflow in unit of K.

    he_h_fraction : ``float``, optional
        Number fraction of He particles in relation to H particles. Default is
        0.1 / 0.9.

    Returns
    -------
    mu_bar : ``float``
        "Average" mean molecular weight as defined by Eq. A.3 of Lampón et al.
        2020, in unit of proton mass.
    """
    # Converting units
    m_planet = planet_mass * 1.8981246e+27  # Planet mass in kg
    r = r_profile * 71492000.0  # Radius profile in m
    v_r = v_profile * 1000  # Velocity profile in unit of m / s

    # Physical constants
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    grav = 6.6743e-11  # Gravitational constant in m ** 3 / kg / s ** 2

    # Mean molecular weight in function of radial distance r
    mu_r = (1 + 4 * he_h_fraction) / \
        (1 + he_h_fraction + ion_fraction_profile)

    # Eq. A.3 of Lampón et al. 2020 is a combination of several integrals, which
    # we calculate here
    int_1 = simps(mu_r / r ** 2, r)
    int_2 = simps(mu_r * v_r, v_r)
    int_3 = trapz(mu_r, 1 / mu_r)
    int_4 = simps(1 / r ** 2, r)
    int_5 = simps(v_r, v_r)
    int_6 = 1 / mu_r[-1] - 1 / mu_r[0]
    term_1 = grav * m_planet * int_1 + int_2 + k_b * temperature * int_3
    term_2 = grav * m_planet * int_4 + int_5 + k_b * temperature * int_6
    mu_bar = term_1 / term_2

    return mu_bar


# Speed of sound
def sound_speed(temperature, mean_molecular_weight=1.0):
    """
    Speed of sound in an isothermal ideal gas.

    Parameters
    ----------
    temperature : ``float``
        Constant temperature of the gas in Kelvin. Assumed to be close to the
        maximum thermospheric temperature (see Oklopčić & Hirata 2018 and
        Lampón et al. 2020 for more details).

    mean_molecular_weight : ``float``
        Mean molecular weight of the atmosphere in unit of proton mass. Default
        value is 1.0 (100% neutral H).

    Returns
    -------
    cs : ``float``
        Sound speed in the gas in unit of km / s.
    """
    m_h = 1.67262192369e-27  # Hydrogen mass in kg
    k_b = 1.380649e-29  # Boltzmann constant in km ** 2 / s ** 2 * kg / K
    cs = (k_b * temperature / mean_molecular_weight / m_h) ** 0.5
    return cs


# Radius of sonic point
def radius_sonic_point(planet_mass, sound_speed_0):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound.

    Parameters
    ----------
    planet_mass : ``float``
        Planetary mass in unit of Jupiter mass.

    sound_speed_0 : ``float``
        Constant speed of sound in unit of km / s.

    Returns
    -------
    radius_sonic_point : ``float``
        Radius of the sonic point in unit of Jupiter radius.
    """
    grav = 1772.0378503888546  # Gravitational constant in unit of
    # jupiterRad * km ** 2 / s ** 2 / jupiterMass
    radius_sonic_point = grav * planet_mass / 2 / sound_speed_0 ** 2
    return radius_sonic_point


# Density at the sonic point
def density_sonic_point(mass_loss_rate, radius_sp, sound_speed_0):
    """
    Density at the sonic point, where the wind speed matches the speed of
    sound. The input values must be `astropy.Quantity`.

    Parameters
    ----------
    mass_loss_rate : ``float``
        Total mass loss rate of the planet in units of g / s.

    radius_sp : ``float``
        Radius at the sonic point in unit of Jupiter radius.

    sound_speed_0 : ``float``
        Speed of sound, assumed to be constant, in units of km / s.

    Returns
    -------
    rho_sp : ``float``
        Density at the sonic point in units of g / cm ** 3.
    """
    vs = sound_speed_0 * 1E5  # Convert sound speed to cm / s
    rs = radius_sp * 7.1492E+09  # Convert radius to cm
    rho_sp = mass_loss_rate / 4 / np.pi / rs ** 2 / vs
    return rho_sp


# Velocity and density in function of radius at the sonic point
def structure(r, v_guess=None):
    """
    Calculate the velocity and density of the atmosphere in function of radius
    at the sonic point, and in units of sound speed and density at the sonic
    point, respectively.

    Parameters
    ----------
    r : ``numpy.ndarray`` or ``float``
        Radius at which to sample the velocity in unit of radius at the sonic
        point.
        
    v_guess : ``numpy.ndarray`` or ``float``, optional
        Guessed value(s) of velocity, in unit of sound speed, corresponding to 
        the radius(ii) ``r``. If ``None``, then the code assumes a standard 
        guess for the velocity. If not ``None``, ``v_guess`` must have the same
        shape as ``r``. Default is ``None``.

    Returns
    -------
    velocity_r : ``numpy.ndarray`` or ``float``
        `numpy` array or a single value of velocity at the given radius or radii
        in unit of sound speed.

    density_r : ``numpy.ndarray`` or ``float``
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
    # the velocity is either above or below the sonic radius. The user has the
    # option of passing a `v_guess`.

    if v_guess is not None:
        velocity_r = so.newton(_eq_to_solve, x0=v_guess, args=(r,))

    # Otherwise, we set it automatically: If the radius is below the sonic
    # radius, the first guess is 0.1. If the radius is above, the first
    # guess is 2.0. This is a hacky solution, but it seems to work well.
    elif isinstance(r, np.ndarray):
        # If r is a ``numpy.ndarray``, we do a dirty little hack to setup an
        # array of first guesses `x0` whose values are 0.1 for r <= 1, and 2.1
        # if r > 1.
        v_init = np.array(r > 1, dtype=int) * 2 + 0.1
        velocity_r = so.newton(_eq_to_solve, x0=v_init, args=(r,))
    # If r is float, just do a simple if statement
    else:
        if r <= 1.0:
            velocity_r = so.newton(_eq_to_solve, x0=1E-1, args=(r,))
        else:
            velocity_r = so.newton(_eq_to_solve, x0=2.0, args=(r,))

    density_r = np.exp(2 * 1 / r - 3 / 2 - 0.5 * velocity_r ** 2)

    return velocity_r, density_r

def radius_sonic_point_tidal(planet_mass, sound_speed_0, star_mass,
                             semi_major_axis):
    """
    Radius of the sonic point, i.e., where the wind speed matches the speed of
    sound, accounting for the tidal gravity of the host star.

    Parameters
    ----------
    planet_mass : ``float``
        Planetary mass in unit of Jupiter mass.

    sound_speed_0 : ``float``
        Constant speed of sound in unit of km / s.

    star_mass : ``float``
        Stellar mass in unit of solar mass.

    semi_major_axis : ``float``
        Planetary semimajor axis in unit of AU.

    Returns
    -------
    radius_sonic_point : ``float``
        Radius of the sonic point in units of Jupiter radius.
    """
    grav = 1772.0378503888546  # Gravitational constant in unit of
    # jupiterRad * km ** 2 / s ** 2 / jupiterMass

    # Convert stellar mass unit to Jupiter mass and semi_major_axis unit to
    # Jupiter radius
    star_mass = 1047.56551466 * star_mass
    semi_major_axis = 2092.51203911 * semi_major_axis

    # Calculate the radius at the sonic point
    v_K = np.sqrt(grav * star_mass / semi_major_axis)
    m1 = np.sqrt(planet_mass ** 2 / 4 + 8 * star_mass ** 2 / 81 *
                 (sound_speed_0 / v_K) ** 6)
    radius_sonic_point = \
        semi_major_axis * (((m1 + planet_mass / 2)/(3 * star_mass)) ** (1 / 3) \
        - ((m1 - planet_mass / 2) / (3 * star_mass)) ** (1 / 3))
    return radius_sonic_point

# Velocity and density in function of radius at the sonic point
def structure_tidal(r, sound_speed_0, radius_sonic_point, planet_mass,
                    star_mass, semi_major_axis):
    """
    Calculate the velocity and density of the atmosphere in function of radius
    at the sonic point, and in units of sound speed and density at the sonic
    point, respectively. This version accounts for the tidal gravity of the host
    star.

    Parameters
    ----------
    r : ``numpy.ndarray`` or ``float``
        Radius at which to sample the velocity in unit of radius at the sonic
        point.

    sound_speed_0 : ``float``
        Constant speed of sound in unit of km / s.

    radius_sonic_point : ``float``
        Sonic radius in unit of Jupiter radius. Note: ensure that this is
        computed with ``radius_sonic_point_tidal``.

    planet_mass : ``float``
        Planetary mass in unit of Jupiter mass.

    star_mass : ``float``
        Stellar mass in unit of solar mass.

    semi_major_axis : ``float``
        Planetary semimajor axis in unit of AU.

    Returns
    -------
    velocity_r : ``numpy.ndarray`` or ``float``
        `numpy` array or a single value of velocity at the given radius or radii
        in unit of sound speed.

    density_r : ``numpy.ndarray`` or ``float``
        Density sampled at the radius or radii `r` and in unit of density at the
        sonic point.
    """

    # First make all quantities cgs, then work with values only
    sound_speed_0 = 100000. * sound_speed_0
    radius_sonic_point = 7.1492e+09 * radius_sonic_point
    planet_mass = 1.8981246e+30 * planet_mass
    star_mass = 1.98840987e+33 * star_mass
    semi_major_axis = 1.49597871e+13 * semi_major_axis
    grav = c.G.to(u.cm ** 3 / u.g / u.s ** 2).value

    # Equation for the velocity profile
    def _eq_to_solve(v, r, c_s, r_s, M_p, M_star, a):
        eq = v * np.exp(-v ** 2 / 2) - (1 / r) ** 2 * np.exp(
            - grav * M_p / (c_s ** 2 * (r * r_s)) + grav * M_p / (c_s ** 2 * r_s) \
            - 3 * grav * M_star * (r * r_s) ** 2 / (2 * a ** 3 * c_s ** 2) \
            + 3 * grav * M_star * r_s ** 2 / (2 * a ** 3 * c_s ** 2) - 0.5
        )
        return eq

    if isinstance(r, np.ndarray):
        # One line version of Leo's initial value hack gives 0.1 below sonic
        # point and 2 above
        v_init = (np.array(r > 1, dtype=int) * 2 + 0.1)

        # Compute velocity profile
        velocity_r = so.newton(_eq_to_solve, x0=v_init,
            args=(r, sound_speed_0, radius_sonic_point, planet_mass,
                  star_mass, semi_major_axis), maxiter=1000)
    elif r <= 1.0:
        velocity_r = so.newton(_eq_to_solve, x0=1E-1,
                               args=(r, sound_speed_0, radius_sonic_point,
                                     planet_mass, star_mass, semi_major_axis))
    else:
        velocity_r = so.newton(_eq_to_solve, x0=2.0,
                               args=(r, sound_speed_0, radius_sonic_point,
                                     planet_mass, star_mass, semi_major_axis))

    # Some useful definitions to make the code cleaner
    k1 = sound_speed_0 ** 2 * (r * radius_sonic_point)
    k2 = sound_speed_0 ** 2 * radius_sonic_point
    k3 = 2 * semi_major_axis ** 3 * sound_speed_0 ** 2
    density_r = np.exp(grav * planet_mass / k1 - grav * planet_mass / k2 \
        + 3 * grav * star_mass * (r * radius_sonic_point) ** 2 / k3 \
        - 3 * grav * star_mass * radius_sonic_point ** 2 / k3 + 0.5 - \
        np.array(velocity_r) ** 2 / 2)

    return velocity_r, density_r


