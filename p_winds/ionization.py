#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of H and He in the
upper atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp
from . import parker, tools


def hydrogen_photoionization_rate(spectrum_at_planet):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``): Spectrum of the host star arriving at the
        planet covering fluxes at least up to the wavelength corresponding to
        the energy to ionize hydrogen (13.6 eV, or 911.65 Angstrom).

    Returns
    -------
    ionization_rate (``float``): Ionization rate of hydrogen at null optical
        depth.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value

    # Wavelength corresponding to the energy to ionize H
    wl_break = (c.h * c.c / (13.6 * u.eV)).to(u.angstrom).value

    # Index of the nu_0 in the frequency array
    i_break = tools.nearest_index(wavelength, wl_break)

    # Auxiliary definitions
    wavelength_cut = wavelength[:i_break + 1]
    flux_lambda_cut = flux_lambda[:i_break + 1]
    epsilon = (wl_break / wavelength_cut - 1) ** 0.5

    # Photoionization cross-section in function of frequency
    a_lambda = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
        (1 - np.exp(-2 * np.pi / epsilon)) *
        (wavelength_cut / wl_break) ** 4) * u.cm ** 2

    # Finally calculate the photoionization rate
    phi = simps(flux_lambda_cut * a_lambda , wavelength_cut) / u.s
    return phi


# Case-B hydrogen recombination rate
def hydrogen_recombination_rate(temperature):
    """
    Calculates the case-B hydrogen recombination rate for a gas at a certain
    temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``): Isothermal temperature of the upper
        atmosphere.

    Returns
    -------
    alpha_rec (``astropy.Quantity``): Recombination rate of hydrogen.

    """
    alpha_rec = 2.59E-13 * (temperature.to(u.K).value / 1E4) ** (-0.7) * \
        u.cm ** 3 / u.s
    return alpha_rec


# Hydrogen neutral-fraction profile
def neutral_fraction(radius_profile, planet_radius, temperature, h_he_fraction,
                     mass_loss_rate, planet_mass, spectrum_at_planet):
    """

    Parameters
    ----------
    radius_profile (``numpy.ndarray``): Radius in unit of planetary radii. Not
        a `astropy.Quantity`.

    planet_radius (``astropy.Quantity``): Planetary radius.

    temperature (``astropy.Quantity``): Isothermal temperature of the upper
        atmosphere.

    h_he_fraction (``float``): H/He fraction of the atmosphere.

    mass_loss_rate (``astropy.Quantity``): Mass loss rate of the planet.

    planet_mass (``astropy.Quantity``): Planetary mass.

    spectrum_at_planet (``dict``): Spectrum of the host star arriving at the
        planet covering fluxes at least up to the wavelength corresponding to
        the energy to ionize hydrogen (13.6 eV, or 911.65 Angstrom). Can be
        generated using `tools.make_spectrum_dict`.

    Returns
    -------
    f_r (``numpy.ndarray``): Values of the fraction of ionized hydrogen in
        function of the radius.

    tau_r (``numpy.ndarray``): Values of the optical depth in function of the
        radius.

    """
    # First calculate the sound speed, radius at the sonic point and the
    # density at the sonic point. They will be useful to change the units of
    # the calculation aiming to avoid numerical overflows
    vs = parker.sound_speed(temperature, h_he_fraction).to(u.km / u.s)
    rs = parker.radius_sonic_point(planet_mass, vs).to(u.jupiterRad)
    rhos = parker.density_sonic_point(mass_loss_rate, rs, vs).to(
        u.g / u.cm ** 3)

    # Hydrogen recombination rate in unit of rs ** 2 * vs
    alpha_rec_unit = (rs ** 2 * vs).to(u.cm ** 3 / u.s)
    alpha_rec = (hydrogen_recombination_rate(temperature) /
                 alpha_rec_unit).decompose().value

    # Hydrogen mass in unit of rhos * rs ** 3
    m_H_unit = (rhos * rs ** 3).to(u.g)
    m_H = (c.m_p / m_H_unit).decompose().value

    # Multiplicative factor of the second term in the right-hand side of Eq.
    # 13 of Oklopcic & Hirata 2018
    k2 = h_he_fraction / (1 + (1 - h_he_fraction) * 4) * alpha_rec / m_H

    # Photoionization rate at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs
    phi_unit = (vs / rs).to(1 / u.s)
    phi = (hydrogen_photoionization_rate(spectrum_at_planet) /
           phi_unit).decompose().value

    # Now let's solve the differential eq. 13 of Oklopcic & Hirata 2018

    # The radius in unit of radius at the sonic point
    r = (radius_profile * planet_radius / rs).decompose().value
    # We are going to integrate from outside inwards, so it is useful to define
    # a new variable called theta, which is simply 1 / r
    theta = np.flip(1 / r)

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
        dt_dtheta = (1. - f) * rho
        return np.array([df_dtheta, dt_dtheta])

    # At the outer border (theta[0]), assume the gas is completely ionized
    # (f = 1.0) and the optical depth is null (t = 0.0)
    y0 = np.array([1.0, 0.0])

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (theta[0], theta[-1],), y0, t_eval=theta)

    # Finally retrieve the neutral fraction and optical depth arrays. Since we
    # integrated f and tau from the outside, we have to flip them back to the
    # same order as the radius variable
    f_r = np.flip(sol['y'][0])
    tau_r = np.flip(sol['y'][1])

    return f_r, tau_r
