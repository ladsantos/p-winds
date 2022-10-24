#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of C in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp, odeint, cumtrapz
from scipy.interpolate import interp1d
from scipy.special import exp1
from p_winds import tools, microphysics
import warnings


__all__ = []


_SOLAR_CARBON_ABUNDANCE_ = 8.43  # Asplund et al. 2009
_SOLAR_CARBON_FRACTION_ = 10 ** (_SOLAR_CARBON_ABUNDANCE_ - 12.00)


# Photoionization of C I (neutral) into C II (singly-ionized)
def radiative_processes_ci(spectrum_at_planet):
    """

    Parameters
    ----------
    spectrum_at_planet

    Returns
    -------

    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value
    energy_erg = (energy * u.eV).to(u.erg).value

    # Auxiliary definitions
    parameters_dict = microphysics.sigma_properties_v1996()
    energy_threshold = parameters_dict['C I'][0]  # Ionization threshold in eV
    wl_break = 12398.42 / energy_threshold  # C ionization threshold in angstrom
    wl_break_he = 504  # He ionization threshold in angstrom
    i0 = tools.nearest_index(wavelength, wl_break_he)
    i1 = tools.nearest_index(wavelength, wl_break)
    wavelength_cut_0 = wavelength[:i0]
    flux_lambda_cut_0 = flux_lambda[:i0]
    energy_cut_0 = energy_erg[:i0]
    wavelength_cut_1 = wavelength[:i1]
    flux_lambda_cut_1 = flux_lambda[:i1]
    energy_cut_1 = energy_erg[:i1]

    # Calculate the photoionization cross-section
    a_lambda = microphysics.general_cross_section(wavelength_cut_1,
                                                  species='C I')

    # The flux-averaged photoionization cross-section of C I
    a_ci = abs(simps(flux_lambda_cut_1 * a_lambda, wavelength_cut_1) /
               simps(flux_lambda_cut_1, wavelength_cut_1))

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the C atoms see.
    a_lambda_h = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_1)
    a_h = abs(simps(flux_lambda_cut_1 * a_lambda_h, wavelength_cut_1) /
              simps(flux_lambda_cut_1, wavelength_cut_1))

    # Same for the He atoms, but only up to the He ionization threshold
    a_lambda_he = microphysics.helium_total_cross_section(wavelength_cut_0)
    a_he = abs(simps(flux_lambda_cut_0 * a_lambda_he, wavelength_cut_0) /
               simps(flux_lambda_cut_0, wavelength_cut_0))

    # Calculate the photoionization rates
    phi_ci = abs(simps(flux_lambda_cut_1 * a_lambda / energy_cut_1,
                 wavelength_cut_1))
    phi_h = abs(simps(flux_lambda_cut_1 * a_lambda_h / energy_cut_1,
                wavelength_cut_1))
    phi_he = abs(simps(flux_lambda_cut_0 * a_lambda_he / energy_cut_0,
                 wavelength_cut_0))

    return phi_ci, phi_h, phi_he, a_ci, a_h, a_he


# Ionization rate of neutral C by electron impact
def electron_impact_ionization(electron_temperature):
    """
    Calculates the electron impact ionization rate that consumes neutral C and
    produces singly-ionized C. Based on the formula of Voronov 1997
    (https://ui.adsabs.harvard.edu/abs/1997ADNDT..65....1V/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    ionization_rate (``float``):
        Ionization rate of neutral C into singly-ionized C in unit of
        cm ** 3 / s.
    """
    boltzmann_constant = 8.617333262145179e-05  # eV / K
    electron_energy = boltzmann_constant * electron_temperature
    energy_ratio = 11.3 / electron_energy
    ionization_rate = 6.85E-8 * (0.193 + energy_ratio) ** (-1) * \
        energy_ratio ** 0.25 * np.exp(-energy_ratio)
    return ionization_rate


# Recombination of singly-ionized C into neutral C
def recombination(electron_temperature):
    """
    Calculates the rate of recombination of singly-ionized C with an electron to
    produce a neutral C atom. Based on the formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    alpha_rec (``float``):
        Recombination rate of C in units of cm ** 3 / s.
    """
    alpha_rec = 4.67E-12 * (300 / electron_temperature) ** 0.60
    return alpha_rec


# Charge transfer between a singly-ionized C and neutral H
def charge_transfer(temperature):
    """
    Calculates the rate of conversion of ionized C into neutral C by charge
    exchange with H, He and Si nuclei. Based on the formulation of Stancil et
    al. 1998 (https://ui.adsabs.harvard.edu/abs/1998ApJ...502.1006S/abstract),
    Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract) and
    Glover & Jappsen 2007
    (https://ui.adsabs.harvard.edu/abs/2007ApJ...666....1G/abstract).

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_h (``float``):
        Charge transfer rate between neutral C and H in units of cm ** 3 / s.

    ct_rate_hp (``float``):
        Charge transfer rate between ionized C and H in units of cm ** 3 / s.

    ct_rate_he (``float``):
        Charge transfer rate between neutral C and He in units of cm ** 3 / s.

    ct_rate_si (``float``):
        Charge transfer rate between ionized C and Si in units of cm ** 3 / s.
    """
    ct_rate_h = 1.31E-15 * (300 / temperature) ** (-0.213)
    ct_rate_hp = 6.30E-17 * (300 / temperature) ** (-1.96) * \
        np.exp(-1.7E5 / temperature)
    ct_rate_he = 2.5E-15 * (300 / temperature) ** (-1.597)
    ct_rate_si = 2.1E-9

    return ct_rate_h, ct_rate_hp, ct_rate_he, ct_rate_si


# Excitation of C atoms and ions by electron impact using the formulation of
# Suno & Kato 2006
def electron_impact_excitation(electron_temperature, excitation_energy,
                               statistical_weight, coefficients,
                               forbidden_transition=False):
    """
    Calculate th C ion excitation rates by electron impact following the
    Type 1 formulation of Suno & Kato 2006
    (https://ui.adsabs.harvard.edu/abs/2006ADNDT..92..407S/abstract).

    Parameters
    ----------
    electron_temperature
    excitation_energy
    statistical_weight
    coefficients
    forbidden_transition

    Returns
    -------

    """
    # Some auxiliary definitions
    ka, kb, kc, kd, ke = coefficients
    if forbidden_transition is True:
        ke = 0
    else:
        pass
    electron_energy = 1.380649E-16 * electron_temperature  # erg
    electron_energy_ev = 8.6173333E-5 * electron_temperature  # eV
    y = excitation_energy / electron_energy

    # We use the Type 1 excitation formula (Eq. 10 in Suno & Kato 2006)
    term1 = (ka / y + kc) + kd / 2 * (1 - y)
    term2 = np.exp(y) * exp1(y) * (kb - kc * y + kd / 2 * y ** 2 + ke / y)
    gamma = y * (term1 + term2)
    excitation_rate = 8.010E-8 * np.exp(-y) * gamma / statistical_weight / \
        electron_energy_ev ** 0.5  # cm ** 3 / s

    return excitation_rate


def singly_ion_fraction():
    """

    Returns
    -------

    """
    pass
