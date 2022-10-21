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
from p_winds import tools, microphysics
import warnings


__all__ = []


_SOLAR_CARBON_ABUNDANCE_ = 8.43  # Asplund et al. 2009
_SOLAR_CARBON_FRACTION_ = 10 ** (_SOLAR_CARBON_ABUNDANCE_ - 12.00)


# Photoionization of neutral C into singly-ionized C
def radiative_processes_cii(spectrum_at_planet, r_grid, density, f_r,
                            h_fraction, c_fraction=_SOLAR_CARBON_FRACTION_):
    """

    Parameters
    ----------
    spectrum_at_planet
    r_grid
    density
    f_r
    h_fraction
    c_fraction

    Returns
    -------

    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = (c.h * c.c).to(u.erg * u.angstrom).value / wavelength

    # Wavelength corresponding to the energy to ionize H
    wl_break = 911.65  # angstrom

    # Index of the lambda_0 in the wavelength array
    i_break = tools.nearest_index(wavelength, wl_break)

    # Auxiliary definitions
    wavelength_cut = wavelength[:i_break + 1]
    flux_lambda_cut = flux_lambda[:i_break + 1]
    energy_cut = energy[:i_break + 1]

    # 2d grid of radius and wavelength
    xx, yy = np.meshgrid(wavelength_cut, r_grid)
    # Photoionization cross-section in function of wavelength
    a_lambda = microphysics.hydrogen_cross_section(wavelength=xx)

    # Optical depth to hydrogen photoionization
    m_h = 1.67262192E-24  # Proton mass in unit of kg
    r_grid_temp = r_grid[::-1]
    # We assume that the atmosphere is made of only H + He and that other
    # species are trace elements
    he_fraction = 1 - h_fraction
    f_he_to_h = he_fraction / h_fraction
    mu = (1 + 4 * f_he_to_h) / (1 + f_r + f_he_to_h)

    n_tot = density / mu / m_h
    n_htot = 1 / (1 + f_r + f_he_to_h) * n_tot
    n_h = n_htot * (1 - f_r)
    n_hetot = n_htot * f_he_to_h
    n_he = n_hetot * (1 - f_r)
    n_ctot = n_htot * c_fraction

    n_h_temp = n_h[::-1]
    column_h = cumtrapz(n_h_temp, r_grid_temp, initial=0)
    column_density_h = -column_h[::-1]
    tau_rnu = column_density_h[:, None] * a_lambda

    # Optical depth to helium photoionization
    n_he_temp = n_he[::-1]
    column_he = cumtrapz(n_he_temp, r_grid_temp, initial=0)
    column_density_he = -column_he[::-1]
    a_lambda_he = microphysics.helium_total_cross_section(wavelength=xx)
    tau_rnu += column_density_he[:, None] * a_lambda_he

    # XXX Still working on this function XXX
    pass


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


def single_ion_fraction():
    """

    Returns
    -------

    """
    pass
