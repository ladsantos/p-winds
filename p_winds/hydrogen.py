#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of H in the
upper atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp
from p_winds import parker, tools


__all__ = ["photoionization", "recombination", "ion_fraction"]


# Hydrogen photoionization
def photoionization(spectrum_at_planet):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom).

    Returns
    -------
    phi (``astropy.Quantity``):
        Ionization rate of hydrogen at null optical depth.

    a_0 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value

    # Wavelength corresponding to the energy to ionize H
    wl_break = (c.h * c.c / (13.6 * u.eV)).to(u.angstrom).value

    # Index of the lambda_0 in the wavelength array
    i_break = tools.nearest_index(wavelength, wl_break)

    # Auxiliary definitions
    wavelength_cut = wavelength[:i_break + 1]
    flux_lambda_cut = flux_lambda[:i_break + 1]
    epsilon = (wl_break / wavelength_cut - 1) ** 0.5

    # Photoionization cross-section in function of frequency
    a_lambda = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
        (1 - np.exp(-2 * np.pi / epsilon)) *
        (wavelength_cut / wl_break) ** 4) * u.cm ** 2

    # Flux-averaged photoionization cross-section
    a_0 = simps(flux_lambda_cut * a_lambda, wavelength_cut) / \
        simps(flux_lambda_cut, wavelength_cut) * u.cm ** 2

    # Finally calculate the photoionization rate
    phi = simps(flux_lambda_cut * a_lambda, wavelength_cut) / u.s
    return phi, a_0


# Case-B hydrogen recombination
def recombination(temperature):
    """
    Calculates the case-B hydrogen recombination rate for a gas at a certain
    temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    alpha_rec (``astropy.Quantity``):
        Recombination rate of hydrogen.
    """
    alpha_rec = 2.59E-13 * (temperature.to(u.K).value / 1E4) ** (-0.7) * \
        u.cm ** 3 / u.s
    return alpha_rec


# Fraction of ionized hydrogen vs. radius profile
def ion_fraction(radius_profile, planet_radius, temperature, h_he_fraction,
                 mass_loss_rate, planet_mass, spectrum_at_planet,
                 average_ion_fraction=0.0, initial_state=np.array([1.0, 0.0])):
    """
    Calculate the fraction of ionized hydrogen in the upper atmosphere in
    function of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        Radius in unit of planetary radii. Not a ``astropy.Quantity``.

    planet_radius (``astropy.Quantity``):
        Planetary radius.

    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    h_he_fraction (``float``):
        H/He fraction of the atmosphere.

    mass_loss_rate (``astropy.Quantity``):
        Mass loss rate of the planet.

    planet_mass (``astropy.Quantity``):
        Planetary mass.

    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``.

    average_ion_fraction (``float``):
        Average ion fraction in the upper atmosphere.

    initial_state (``numpy.ndarray``, optional):
        The initial state is the `y0` of the differential equation to be solved.
        This array has two items: the initial value of `f_ion` (ionization
        fraction) and `tau` (optical depth) at the outer layer of the
        atmosphere. The standard value for this parameter is
        ``numpy.array([1.0, 0.0])``, i.e., completely ionized at the outer layer
        and with null optical depth.

    Returns
    -------
    f_r (``numpy.ndarray``):
        Values of the fraction of ionized hydrogen in function of the radius.

    tau_r (``numpy.ndarray``):
        Values of the optical depth in function of the radius.
    """
    # First calculate the sound speed, radius at the sonic point and the
    # density at the sonic point. They will be useful to change the units of
    # the calculation aiming to avoid numerical overflows
    vs = parker.sound_speed(temperature, h_he_fraction, average_ion_fraction
                            ).to(u.km / u.s)
    rs = parker.radius_sonic_point(planet_mass, vs).to(u.jupiterRad)
    rhos = parker.density_sonic_point(mass_loss_rate, rs, vs).to(
        u.g / u.cm ** 3)

    # Hydrogen recombination rate in unit of rs ** 2 * vs
    alpha_rec_unit = (rs ** 2 * vs).to(u.cm ** 3 / u.s)
    alpha_rec = (recombination(temperature) / alpha_rec_unit).decompose().value

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * rs ** 3).to(u.g)
    m_h = (c.m_p / m_h_unit).decompose().value

    # Photoionization rate at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs
    phi_unit = (vs / rs).to(1 / u.s)
    phi, a_0 = photoionization(spectrum_at_planet)
    phi = (phi / phi_unit).decompose().value
    a_0 = (a_0 / rs ** 2).decompose().value

    # Multiplicative factor of Eq. 11 of Oklopcic & Hirata 2018
    k1 = h_he_fraction * a_0 / (1 + (1 - h_he_fraction) * 4) / m_h

    # Multiplicative factor of the second term in the right-hand side of Eq.
    # 13 of Oklopcic & Hirata 2018
    k2 = h_he_fraction / (1 + (1 - h_he_fraction) * 4) * alpha_rec / m_h

    # Now let's solve the differential eq. 13 of Oklopcic & Hirata 2018

    # The radius in unit of radius at the sonic point
    r = (radius_profile * planet_radius / rs).decompose().value
    # We are going to integrate from outside inwards, so it is useful to define
    # a new variable called theta, which is simply 1 / r
    _theta = np.flip(1 / r)

    # The differential equation
    def _fun(theta, y):
        f = y[0]  # Fraction of ionized gas
        t = y[1]  # Optical depth
        velocity, rho = parker.structure(1 / theta)
        # In terms 1 and 2 we use the values of k2 and phi from above
        term1 = (1. - f) / velocity * phi * np.exp(-t)
        term2 = k2 * rho * f ** 2 / velocity
        # The DE system is the following
        df_dtheta = term1 - term2
        dt_dtheta = k1 * (1. - f) * rho
        return np.array([df_dtheta, dt_dtheta])

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (_theta[0], _theta[-1],), initial_state,
                    t_eval=_theta)

    # Finally retrieve the ion fraction and optical depth arrays. Since we
    # integrated f and tau from the outside, we have to flip them back to the
    # same order as the radius variable
    f_r = np.flip(sol['y'][0])
    tau_r = np.flip(sol['y'][1])

    return f_r, tau_r
