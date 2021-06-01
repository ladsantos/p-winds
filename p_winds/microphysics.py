#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains some useful hard-coded data and equations for calculations
in the other modules.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np

from warnings import warn


__all__ = ["hydrogen_cross_section", "helium_singlet_cross_section",
           "helium_triplet_cross_section", "he_3_properties"]


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
        epsilon = (911.65 / wavelength - 1) ** 0.5
        # Photoionization cross-section in function of wavelength
        a_lambda = \
            (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
             (1 - np.exp(-2 * np.pi / epsilon)) * (wavelength / 911.65)
             ** 4)
        if isinstance(a_lambda, np.ndarray):
            a_lambda[np.isnan(a_lambda)] = 0.0  # Remove NaNs if necessary
        return a_lambda

    elif energy is not None:
        epsilon = (energy / 13.6 - 1) ** 0.5
        # Photoionization cross-section in function of energy
        a_nu = (6.3E-18 * np.exp(4 - (4 * np.arctan(epsilon)) / epsilon) /
                (1 - np.exp(-2 * np.pi / epsilon)) * (13.6 / energy) ** 4)
        if isinstance(a_nu, np.ndarray):
            a_nu[np.isnan(a_nu)] = 0.0  # Remove NaNs if necessary
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
    energy = 12398.41984332 / wavelength  # Energy in unit of eV

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
    """
    Returns a hard-coded array containing the helium collisional strengths in
    function of temperature.

    Returns
    -------
    array (``numpy.ndarray``):
        Collisional strengths array. Columns: 0 = temperature, 1 = gamma_13,
        2 = gamma_31a, 3 = gamma_31b.
    """
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


# Line properties of the 1.083 microns He triplet taken from the NIST database
# https://www.nist.gov/pml/atomic-spectra-database
def he_3_properties():
    """
    Returns the central wavelengths in air, oscillator strengths and the
    Einstein coefficient of the helium triplet in 1.083 microns. The values
    were taken from the NIST database:
    https://www.nist.gov/pml/atomic-spectra-database

    Returns
    -------
    lambda_0 (``float``):
        Central wavelength in air of line 0 in unit of m.

    lambda_1 (``float``):
        Central wavelength in air of line 1 in unit of m.

    lambda_2 (``float``):
        Central wavelength in air of line 2 in unit of m.

    f_0 (``float``):
        Oscillator strength of line 0 (unitless).

    f_1 (``float``):
        Oscillator strength of line 1 (unitless).

    f_2 (``float``):
        Oscillator strength of line 2 (unitless).

    a_ij (``float``):
        Einstein coefficient of the whole triplet in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_0 = 1.082909114 * 1E-6
    lambda_1 = 1.083025010 * 1E-6
    lambda_2 = 1.083033977 * 1E-6
    # Oscillator strengths
    f_0 = 5.9902e-02
    f_1 = 1.7974e-01
    f_2 = 2.9958e-01
    # Einstein coefficient in units of s ** (-1)
    a_ij = 1.0216e+07

    return lambda_0, lambda_1, lambda_2, f_0, f_1, f_2, a_ij


# Parametrized photoionization cross-section of atoms and ions from Verner+1996
def general_cross_section(energy, species):
    """
    Calculates the photoionization cross-section of an atomic or ion species
    using the parametrization of Verner et al. (1996)
    [https://ui.adsabs.harvard.edu/abs/1996ApJ...465..487V/abstract].

    Parameters
    ----------
    energy (``float`` or ``numpy.ndarray``)
        Photon energy in eV.

    species (``str``):
        String containing the species for which you request the cross-section in
        the format ``'X N'`` where ``X`` is the element and ``N`` is the
        ionization number. Example: ``'C II'``.

    Returns
    -------

    """
    parameters_dict = sigma_properties_v1996()
    energy_threshold, energy_max, energy_0, sigma_0, y_a, p, y_w, y_0, y_1 = \
        parameters_dict[species]

    # Check if the requested energy is within the energy range reported by
    # Verner et al. (1996)
    if isinstance(energy, float):
        if energy < energy_threshold or energy > energy_max:
            warn('The requested energy is outside the range of validity: '
                 '[{:.1f}, {:.1f}] eV'.format(energy_threshold, energy_max))
        else:
            pass
    elif isinstance(energy, np.ndarray):
        if energy[0] < energy_threshold or energy[-1] > energy_max:
            warn('The requested energy is outside the range of validity: '
                 '[{:.1f}, {:.1f}] eV'.format(energy_threshold, energy_max))
        else:
            pass
    else:
        raise ValueError('`energy` must be either `float` or `numpy.ndarray`.')

    mb = 1E-18  # cm ** (-2)
    x = energy / energy_0 - y_0
    y = (x ** 2 + y_1 ** 2) ** 0.5

    term1 = (x - 1) ** 2 + y_w ** 2
    term2 = y ** (0.5 * p - 5.5)
    term3 = (1 + (y / y_a) ** 0.5) ** (-p)
    function_y = term1 * term2 * term3

    cross_section = sigma_0 * function_y * mb  # cm ** (-2)

    return cross_section


def sigma_properties_v1996():
    """
    Function that hard-codes the cross-section parameters from Verner et al.
    (1996). We include only the species important for exoplanetary atmospheres.

    Returns
    -------
    parameters_dict (``dict``):
        Dictionary containing the parameters.
    """
    parameters_dict = {
        #         E_thresold, E_max, energy_0, sigma_0, y_a, p, y_w, y_0, y_1
        'C I':    [1.126E1, 2.910E2, 2.144E0, 5.027E2, 6.126E1, 5.101E0, 9.157E-2, 1.133E0, 1.607E0],
        'C II':   [2.438E1, 3.076E2, 4.058E-1, 8.709E0, 1.261E2, 8.578E0, 2.093E0, 4.929E1, 3.234E0],
        'C III':  [4.789E1, 3.289E2, 4.614E0, 1.539E4, 1.737E0, 1.593E1, 5.922E0, 4.378E-3, 2.528E-2],
        'N I':    [1.453E1, 4.048E2, 4.034E0, 8.235E2, 8.033E1, 3.928E0, 9.097E-2, 8.598E-1, 2.325E0],
        'O I':    [1.362E1, 5.380E2, 1.240E0, 1.745E3, 3.784E0, 1.764E1, 7.589E-2, 8.698E0, 1.271E-1],
        'O II':   [3.512E1, 5.581E2, 1.386E0, 5.967E1, 3.175E1, 8.943E0, 1.934E-2, 2.131E1, 1.503E-2],
        'O III':  [5.494E1, 5.840E2, 1.723E-1, 6.753E2, 3.852E2, 6.822E0, 1.191E-1, 3.839E-3, 4.569E-1],
        'Mg I':   [7.646E0, 5.490E1, 1.197E1, 1.372E8, 2.228E-1, 1.574E1, 2.805E-1, 0, 0],
        'Mg II':  [1.504E1, 6.569E1, 8.139E0, 3.278E0, 4.341E7, 3.610E0, 0, 0, 0],
        'Si I':   [8.152E0, 1.060E2, 2.317E1, 2.506E1, 2.057E1, 3.546E0, 2.837E-1, 1.672E-5, 4.207E-1],
        'Si II':  [1.635E1, 1.186E2, 2.556E0, 4.140E0, 1.337E1, 1.191E1, 1.570E0, 6.634E0, 1.272E-1],
        'Si III': [3.349E1, 1.311E2, 1.659E-1, 5.790E-4, 1.474E2, 1.336E1, 8.626E-1, 9.613E1, 6.442E-1],
        'Si IV':  [4.514E1, 1.466E2, 1.288E1, 6.083E0, 1.356E6, 3.353E0, 0, 0, 0],
        'Ca I':   [6.113E0, 3.443E1, 1.278E1, 5.370E5, 3.162E-1, 1.242E1, 4.477E-1, 1.012E-3, 1.851E-2],
        'Ca II':  [1.187E1, 4.090E1, 1.553E1, 1.064E7, 7.790E-1, 2.130E1, 6.453E-1, 2.161E-3, 6.706E-2],
        'Fe I':   [7.902E0, 6.600E1, 5.461E-2, 3.062E-1, 2.671E7, 7.923E0, 2.069E1, 1.382E2, 2.481E-1],
        'Fe II':  [1.619E1, 7.617E1, 1.761E-1, 4.365E3, 6.298E3, 5.204E0, 1.141E1, 9.272E1, 1.075E2],
    }
    return parameters_dict
