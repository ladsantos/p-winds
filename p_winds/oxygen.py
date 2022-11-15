#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of O in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.special import exp1
from p_winds import tools, microphysics
import warnings


__all__ = ["radiative_processes", "electron_impact_ionization", "recombination"]


# Some hard coding based on the astrophysical literature
_SOLAR_OXYGEN_ABUNDANCE_ = 8.69  # Asplund et al. 2009
_SOLAR_OXYGEN_FRACTION_ = 10 ** (_SOLAR_OXYGEN_ABUNDANCE_ - 12.00)


# Photoionization of O I (neutral)
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of oxygen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        oxygen (13.62 eV, or 910 Angstrom).

    Returns
    -------
    phi_oi (``float``):
        Ionization rate of O I at null optical depth in unit of 1 / s.

    a_oi (``float``):
        Flux-averaged photoionization cross-section of O I in unit of cm ** 2.

    a_h_oi (``float``):
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by O I in unit of cm ** 2.

    a_he (``float``):
        Flux-averaged photoionization cross-section of He I in unit of cm ** 2.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value
    energy_erg = (energy * u.eV).to(u.erg).value

    # Auxiliary definitions
    parameters_dict = microphysics.sigma_properties_v1996()

    energy_threshold_oi = parameters_dict['O I'][0]  # Ionization threshold in
    # eV
    wl_break_oi = 12398.42 / energy_threshold_oi  # O I ionization threshold in
    # angstrom
    wl_break_he = 504  # He ionization threshold in angstrom
    i0 = tools.nearest_index(wavelength, wl_break_he)
    i1 = tools.nearest_index(wavelength, wl_break_oi)
    wavelength_cut_0 = wavelength[:i0]
    flux_lambda_cut_0 = flux_lambda[:i0]
    wavelength_cut_1 = wavelength[:i1]
    flux_lambda_cut_1 = flux_lambda[:i1]
    energy_cut_1 = energy_erg[:i1]

    # Calculate the photoionization cross-section
    a_lambda_oi = microphysics.general_cross_section(wavelength_cut_1,
                                                     species='O I')

    # The flux-averaged photoionization cross-section of O I
    a_oi = abs(simps(flux_lambda_cut_1 * a_lambda_oi, wavelength_cut_1) /
               simps(flux_lambda_cut_1, wavelength_cut_1))

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that O I see.
    a_lambda_h_oi = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_1)
    a_h_oi = abs(simps(flux_lambda_cut_1 * a_lambda_h_oi, wavelength_cut_1) /
                 simps(flux_lambda_cut_1, wavelength_cut_1))

    # Same for the He atoms, but only up to the He ionization threshold
    a_lambda_he = microphysics.helium_total_cross_section(wavelength_cut_0)
    a_he = abs(simps(flux_lambda_cut_0 * a_lambda_he, wavelength_cut_0) /
               simps(flux_lambda_cut_0, wavelength_cut_0))

    # Calculate the photoionization rates
    phi_oi = abs(simps(flux_lambda_cut_1 * a_lambda_oi / energy_cut_1,
                       wavelength_cut_1))

    return phi_oi, a_oi, a_h_oi, a_he


# Ionization rate of O by electron impact
def electron_impact_ionization(electron_temperature):
    """
    Calculates the electron impact ionization rate that consumes neutral O and
    produces singly-ionized O. Based on the formula of Voronov 1997
    (https://ui.adsabs.harvard.edu/abs/1997ADNDT..65....1V/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    ionization_rate_oi (``float``):
        Ionization rate of neutral O into singly-ionized O in unit of
        cm ** 3 / s.
    """
    boltzmann_constant = 8.617333262145179e-05  # eV / K
    electron_energy = boltzmann_constant * electron_temperature
    energy_ratio_oi = 11.3 / electron_energy
    ionization_rate_oi = 3.59E-8 * (0.073 + energy_ratio_oi) ** (-1) * \
        energy_ratio_oi ** 0.34 * np.exp(-energy_ratio_oi)
    return ionization_rate_oi


# Recombination of singly-ionized O into neutral O
def recombination(electron_temperature):
    """
    Calculates the rate of recombination of singly-ionized O with an electron to
    produce a neutral O atom. Based on the formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    alpha_rec_oi  (``float``):
        Recombination rate of O II into O I in units of cm ** 3 / s.
    """
    alpha_rec_oi = 3.25E-12 * (300 / electron_temperature) ** 0.66
    return alpha_rec_oi


# Charge transfer between O and H
def charge_transfer(temperature):
    """
    Calculates the charge exchange rates of O with H nuclei. Based on the
    formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract).

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_oi_hp (``float``):
        Charge transfer rate between neutral O and H+ in units of cm ** 3 / s.

    ct_rate_oii_h (``float``):
        Charge transfer rate between O+ and neutral H in units of cm ** 3 / s.
    """
    # Recombination of O II into O I
    ct_rate_oii_h = 5.66E-10 * (300 / temperature) ** (-0.36) * \
        np.exp(8.6 / temperature)

    # Ionization of O I into O II
    ct_rate_oi_hp = 7.31E-10 * (300 / temperature) ** (-0.23) * \
        np.exp(-226 / temperature)

    return ct_rate_oi_hp, ct_rate_oii_h


# Calculation the number fractions of O I and O II
def ion_fraction():
    pass
