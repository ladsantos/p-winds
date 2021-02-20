#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of He in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp
from p_winds import parker, tools, hydrogen


__all__ = ["photoionization", "recombination", "collision", "ion_fraction"]


# Helium photoionization
def photoionization(spectrum_at_planet):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        helium (4.8 eV, or 911.65 Angstrom).

    Returns
    -------
    phi_1 (``astropy.Quantity``):
        Ionization rate of helium singlet at null optical depth.

    phi_3 (``astropy.Quantity``):
        Ionization rate of helium triplet at null optical depth.

    a_1 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium singlet.

    a_3 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium triplet.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value

    # Wavelength corresponding to the energy to ionize He in singlet and triplet
    wl_break_1 = (c.h * c.c / (24.6 * u.eV)).to(u.angstrom).value

    # Index of the lambda_1 and lambda_3 in the wavelength array
    i1 = tools.nearest_index(wavelength, wl_break_1)

    # Auxiliary definitions
    wavelength_cut_1 = wavelength[:i1 + 1]
    flux_lambda_cut_1 = flux_lambda[:i1 + 1]
    epsilon_1 = (wl_break_1 / wavelength_cut_1 - 1) ** 0.5
    i2 = tools.nearest_index(energy, 65.4)  # Threshold to excite He+ to n = 2

    # Photoionization cross-section of H in function of frequency (this is
    # important because the cross-section for He singlet can be derived from
    # that of H
    a_lambda_h = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon_1)) / epsilon_1) /
        (1 - np.exp(-2 * np.pi / epsilon_1)) *
        (wavelength_cut_1 / wl_break_1) ** 4) * u.cm ** 2

    # Photoionization cross-section of He singlet (some hard-coding here; the
    # numbers originate from the paper Brown 1971, ADS 1971ApJ...164..387B)
    scale = np.ones_like(wavelength_cut_1)
    scale[:i2] *= 37.0 - 19.1 * (energy[:i2] / 65.4) ** (-0.76)
    scale[i2:] *= 6.53 * (energy[i2:] / 24.6) - 0.22
    a_lambda_1 = a_lambda_h * scale

    # The photoionization cross-section of He triplet is hard-coded with the
    # values calculated by Norcross 1971 (ADS 1971JPhB....4..652N). The
    # differential oscillator strength is calculated for bins of wavelength that
    # are not necessarily the same as the stellar spectrum wavelength bins.
    data_path = '../data/He_2_3_S_diff_osc_strength.dat'
    wavelength_cut_3 = np.loadtxt(data_path, skiprows=1, usecols=(0,))
    differential_oscillator_strength = np.loadtxt(data_path, skiprows=1,
                                                  usecols=(1,))
    a_lambda_3 = 8.0670E-18 * differential_oscillator_strength
    # Let's interpolate the stellar spectrum to the bins of the cross-section
    # from Norcross 1971
    flux_lambda_cut_3 = np.interp(wavelength_cut_3, wavelength, flux_lambda)

    # Flux-averaged photoionization cross-sections
    a_1 = simps(flux_lambda_cut_1 * a_lambda_1, wavelength_cut_1) / \
        simps(flux_lambda_cut_1, wavelength_cut_1) * u.cm ** 2
    a_3 = simps(flux_lambda_cut_3 * a_lambda_3, wavelength_cut_3) / \
        simps(flux_lambda_cut_3, wavelength_cut_3) * u.cm ** 2

    # Calculate the photoionization rates
    phi_1 = simps(flux_lambda_cut_1 * a_lambda_1, wavelength_cut_1) / u.s
    phi_3 = simps(flux_lambda_cut_3 * a_lambda_3, wavelength_cut_3) / u.s
    return phi_1, phi_3, a_1, a_3


# Helium recombination
def recombination(temperature):
    """
    Calculates the helium singlet and triplet recombination rates for a gas at
    a certain temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    alpha_rec_1 (``astropy.Quantity``):
        Recombination rate of helium singlet.

    alpha_rec_3 (``astropy.Quantity``):
        Recombination rate of helium triplet.
    """
    # The recombination rates come from Benjamin et al. (1999,
    # ADS:1999ApJ...514..307B)
    alpha_rec_1 = 1.54E-13 * (temperature.to(u.K).value / 1E4) ** (-0.486) * \
        u.cm ** 3 / u.s
    alpha_rec_3 = 2.10E-13 * (temperature.to(u.K).value / 1E4) ** (-0.778) * \
        u.cm ** 3 / u.s
    return alpha_rec_1, alpha_rec_3


# Population of helium singlet and triplet through collisions
def collision(temperature):
    """
    Calculates the helium singlet and triplet collisional population rates for
    a gas at a certain temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    q_13 (``astropy.Quantity``):
        Rate of helium transition from singlet (1^1S) to triplet (2^3S) due to
        collisions with free electrons.

    q_31a (``astropy.Quantity``):
        Rate of helium transition from triplet (2^3S) to 2^1S due to collisions
        with free electrons.

    q_31b (``astropy.Quantity``):
        Rate of helium transition from triplet (2^3S) to 2^1P due to collisions
        with free electrons.

    big_q_he (``astropy.Quantity``):
        Rate of charge exchange between ionized helium and atomic hydrogen.

    big_q_he_plus (``astropy.Quantity``):
        Rate of charge exchange between helium singlet and ionized hydrogen.
    """
    # The effective collision strengths are hard-coded from the values provided
    # by Bray et al. (2000, ADS:2000A&AS..146..481B), which are binned to
    # specific temperatures. Thus, we need to interpolate to the specific
    # temperature of our gas.
    temperature = temperature.to(u.K).value
    # Parse the tabulated data
    data_path = '../data/He_collisional_strengths.dat'
    tabulated_temp = 10 ** np.loadtxt(data_path, skiprows=1, usecols=(0,))
    tabulated_gamma_13 = np.loadtxt(data_path, skiprows=1, usecols=(1,))
    tabulated_gamma_31a = np.loadtxt(data_path, skiprows=1, usecols=(2,))
    tabulated_gamma_31b = np.loadtxt(data_path, skiprows=1, usecols=(3,))
    # And interpolate to our desired temperature
    gamma_13 = np.interp(temperature, tabulated_temp, tabulated_gamma_13,
                         left=tabulated_gamma_13[0],
                         right=tabulated_gamma_13[-1])
    gamma_31a = np.interp(temperature, tabulated_temp, tabulated_gamma_31a,
                          left=tabulated_gamma_31a[0],
                          right=tabulated_gamma_31a[-1])
    gamma_31b = np.interp(temperature, tabulated_temp, tabulated_gamma_31b,
                          left=tabulated_gamma_31b[0],
                          right=tabulated_gamma_31b[-1])

    # Finally calculate the rates using the equations from Table 2 of Lampón et
    # al. (2020).
    kt = c.k_B.to(u.eV / u.K).value * temperature
    k1 = 2.10E-8 * (13.6 / kt) ** 0.5
    q_13 = k1 * gamma_13 * np.exp(-19.81 / kt) / u.s
    q_31a = k1 * gamma_31a / 3 * np.exp(-0.80 / kt) / u.s
    q_31b = k1 * gamma_31b / 3 * np.exp(-1.40 / kt) / u.s
    big_q_he = 1.75E-11 * (300 / temperature) ** 0.75 * \
        np.exp(-128E3 / temperature) / u.s
    big_q_he_plus = 1.25E-15 * (300 / temperature) ** (-0.25)

    return q_13, q_31a, q_31b, big_q_he, big_q_he_plus


# Fraction of ionized helium singlet and triplet vs. radius profile
def ion_fraction():
    # Some hard-coding here. The numbers come from Oklopcic & Hirata (2018) and
    # Lampón et al. (2020).
    big_q_31 = 5E-10 / u.s
    big_a_31 = 1.272E-4 / u.s
    pass
