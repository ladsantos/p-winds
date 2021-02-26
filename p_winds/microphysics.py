#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains some useful hard-coded data and equations for calculations
in the other modules.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c


__all__ = ["hydrogen_cross_section", "helium_singlet_cross_section",
           "helium_triplet_cross_section"]


# Photoionization cross-section of hydrogen
def hydrogen_cross_section(wavelength=None, energy=None):
    """
    Compute the photoionization cross-section of hydrogen in function of
    wavelength or energy.

    Parameters
    ----------
    wavelength (``float`` or ``numpy.ndarray``, optional):
        Wavelength in unit of angstrom. Default is ``None``. If ``None``,
        ``energy`` cannot be ``None``.

    energy (``float`` or ``numpy.ndarray``, optional):
        Energy in unit of electron-volt. Default is ``None``. If ``None``,
        ``wavelength`` cannot be ``None``.

    Returns
    -------
    a_lambda (``float`` or ``numpy.ndarray``):
        Cross-section in function of wavelength and in unit of cm ** 2. Only
        returned if wavelength was input.

    a_nu (``float`` or ``numpy.ndarray``):
        Cross-section in function of energy and in unit of cm ** 2. Only
        returned if energy was input.
    """
    if wavelength is not None:
        wl_break = (c.h * c.c / (13.6 * u.eV)).to(u.angstrom).value
        epsilon = (wl_break / wavelength - 1) ** 0.5

        # Photoionization cross-section in function of wavelength
        a_lambda = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
            (1 - np.exp(-2 * np.pi / epsilon)) * (wavelength / wl_break) ** 4)
        return a_lambda

    elif energy is not None:
        epsilon = (energy / 13.6 - 1) ** 0.5
        # Photoionization cross-section in function of energy
        a_nu = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
                (1 - np.exp(-2 * np.pi / epsilon)) * (13.6 / energy) ** 4)
        return a_nu

    else:
        raise ValueError('Either the wavelength or energy has to be provided.')


# Photoionization cross-section of helium singlet
def helium_singlet_cross_section(wavelength):
    """
    Compute the photoionization cross-section of helium singlet in function of
    wavelength.

    Parameters
    ----------
    wavelength (``float`` or ``numpy.ndarray``):
        Wavelength in unit of angstrom.

    Returns
    -------
    a_lambda_1 (``float`` or ``numpy.ndarray``):
        Cross-section in function of wavelength and in unit of cm ** 2.
    """
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value

    # The cross-sections for helium are partially hard-coded and based on those
    # of hydrogen
    a_lambda_h = hydrogen_cross_section(wavelength=wavelength)

    # For the singlet, we simply scale the cross-sections of H following the
    # results from Brown (1971; ADS:1971ApJ...164..387B)
    scale = np.ones_like(wavelength)
    scale *= 37.0 - 19.1 * (energy / 65.4) ** (-0.76)
    # Clip negative values of scale
    scale[scale < 0] = 0.0
    a_lambda_1 = a_lambda_h * scale

    return a_lambda_1


# Photoionization cross-section of helium triplet
def helium_triplet_cross_section():
    """
    Compute the photoionization cross-section of helium triplet in function of
    wavelength.

    Returns
    -------
    wavelength (``numpy.ndarray``):
        Wavelength in which the cross-section was sampled in Norcross (1971).

    a_lambda_3 (``numpy.ndarray``):
        Cross-section in function of wavelength and in unit of cm ** 2.
    """
    # The photoionization cross-section of He triplet is hard-coded with the
    # values calculated by Norcross 1971 (ADS:1971JPhB....4..652N). The
    # differential oscillator strength is calculated for bins of wavelength that
    # are not necessarily the same as the stellar spectrum wavelength bins.

    # Helium 2^3S differential oscillator strength
    data_array = np.array([[2593.01,     0.605],
                           [2528.27,     0.589],
                           [2275.74,     0.537],
                           [2023.15,     0.501],
                           [1655.63,     0.435],
                           [1214.41,     0.247],
                           [958.87,     0.1572],
                           [792.18,     0.1138],
                           [674.86,     0.0780],
                           [587.81,     0.0620],
                           [520.65,     0.0557],
                           [467.27,     0.0461],
                           [423.81,     0.0358],
                           [387.75,     0.0310],
                           [357.34,     0.0325],
                           [331.36,     0.0520],
                           [271.94,     0.343],
                           [271.21,     0.338],
                           [256.70,     0.274],
                           [243.01,     0.231],
                           [230.71,     0.200],
                           [219.59,     0.1750],
                           [209.49,     0.1537]
                           ])
    wavelength = np.flip(data_array[:, 0])
    differential_oscillator_strength = np.flip(data_array[:, 1])
    a_lambda_3 = 8.0670E-18 * differential_oscillator_strength
    return wavelength, a_lambda_3


# Collisional strengths for He
def he_collisional_strength():
    # log(T) [K]    gamma_13    gamma_31a   gamma_31b
    array = np.array([
        [3.75,            6.198E-2,    2.389,       7.965E-1],
        [4.00,            6.458E-2,    2.456,       9.579E-1],
        [4.25,            6.387E-2,    2.275,       1.042],
        [4.50,            6.157E-2,    1.916,       1.015],
        [4.75,            5.832E-2,    1.496,       8.950E-1],
        [5.00,            5.320E-2,    1.111,       7.265E-1],
        [5.25,            4.787E-2,    8.003E-1,    5.516E-1],
        [5.50,            4.018E-2,    5.660E-1,    3.948E-1],
        [5.75,            3.167E-2,    3.944E-1,    2.677E-1],
    ])
    return array
