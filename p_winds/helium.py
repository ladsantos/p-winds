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


__all__ = ["photoionization", ]


# Hydrogen photoionization rate
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
    ionization_rate (``float``):
        Ionization rate of hydrogen at null optical depth.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value

    # Wavelength corresponding to the energy to ionize He in singlet and triplet
    wl_break_1 = (c.h * c.c / (24.6 * u.eV)).to(u.angstrom).value
    wl_break_3 = (c.h * c.c / (4.8 * u.eV)).to(u.angstrom).value

    # Index of the lambda_1 and lambda_3 in the wavelength array
    i1 = tools.nearest_index(wavelength, wl_break_1)
    i3 = tools.nearest_index(wavelength, wl_break_3)

    # Auxiliary definitions
    wavelength_cut_1 = wavelength[:i1 + 1]
    flux_lambda_cut_1 = flux_lambda[:i1 + 1]
    wavelength_cut_3 = wavelength[:i3 + 1]
    flux_lambda_cut_3 = flux_lambda[:i3 + 1]
    epsilon_1 = (wl_break_1 / wavelength_cut_1 - 1) ** 0.5
    i2 = tools.nearest_index(energy, 65.4)  # Threshold to excite He+ to n = 2

    # Photoionization cross-section of H in function of frequency (this is
    # important because the cross-section for He singlet can be derived from
    # that of H
    a_lambda_H = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon_1)) / epsilon_1) /
        (1 - np.exp(-2 * np.pi / epsilon_1)) *
        (wavelength_cut_1 / wl_break_1) ** 4) * u.cm ** 2

    # Photoionization cross-section of He singlet (lots of hard-coding here;
    # the numbers originate from the paper Brown 1971, ADS 1971ApJ...164..387B)
    scale = np.ones_like(wavelength_cut_1)
    scale[:i2] *= 37.0 - 19.1 * (energy[:i2] / 65.4) ** (-0.76)
    scale[i2:] *= 6.53 * (energy[i2:] / 24.6) - 0.22
    a_lambda_He_1 = a_lambda_H * scale
