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
from scipy.integrate import simps, solve_ivp, odeint
from scipy.interpolate import interp1d
from p_winds import tools, microphysics
import warnings


__all__ = []


_SOLAR_CARBON_ABUNDANCE_ = 8.43  # Asplund et al. 2009
_SOLAR_CARBON_FRACTION_ = 10 ** (_SOLAR_CARBON_ABUNDANCE_ - 12.00)


def radiative_processes_exact(spectrum_at_planet, r_grid, density, f_r,
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
    pass


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


def c_ion_reactions():
    """

    Returns
    -------

    """
    pass


def c_neutral_reactions():
    """

    Returns
    -------

    """
    pass


