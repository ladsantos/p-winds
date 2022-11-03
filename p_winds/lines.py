#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module tabulated information about spectral lines relevant to atmospheric
escape detection.
"""


__all__ = ["he_3_properties", "c_ii_properties", "o_i_properties"]


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


# Line properties of the C II lines available with HST/STIS taken from the NIST
# database https://www.nist.gov/pml/atomic-spectra-database
def c_ii_properties():
    """
    Returns the central wavelengths in air, oscillator strengths and the
    Einstein coefficient of the C II lines in the STIS FUV wavelength range. The
    values were taken from the NIST database:
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

    a_ij_0 (``float``):
        Einstein coefficient of line 0 in unit of 1 / s.

    a_ij_1 (``float``):
        Einstein coefficient of line 1 in unit of 1 / s.

    a_ij_2 (``float``):
        Einstein coefficient of line 2 in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_0 = 1334.5323 * 1E-10
    lambda_1 = 1335.6628 * 1E-10
    lambda_2 = 1335.7079 * 1E-10
    # Oscillator strengths
    f_0 = 1.29e-01
    f_1 = 1.27e-02
    f_2 = 1.15e-01
    # Einstein coefficient in units of s ** (-1)
    a_ij_0 = 2.41e+08
    a_ij_1 = 4.76e+07
    a_ij_2 = 2.88e+08

    return lambda_0, lambda_1, lambda_2, f_0, f_1, f_2, a_ij_0, a_ij_1, a_ij_2


# Line properties of the O I lines available with HST/STIS taken from the NIST
# database https://www.nist.gov/pml/atomic-spectra-database
def o_i_properties():
    """
    Returns the central wavelengths in air, oscillator strengths and the
    Einstein coefficient of the O I lines in the STIS FUV wavelength range. The
    values were taken from the NIST database:
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

    a_ij_0 (``float``):
        Einstein coefficient of line 0 in unit of 1 / s.

    a_ij_1 (``float``):
        Einstein coefficient of line 1 in unit of 1 / s.

    a_ij_2 (``float``):
        Einstein coefficient of line 2 in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_0 = 1302.168 * 1E-10
    lambda_1 = 1304.858 * 1E-10
    lambda_2 = 1306.029 * 1E-10
    # Oscillator strengths
    f_0 = 5.20e-02
    f_1 = 5.18e-02
    f_2 = 5.19e-02
    # Einstein coefficient in units of s ** (-1)
    a_ij_0 = 3.41e+08
    a_ij_1 = 2.03e+08
    a_ij_2 = 6.76e+07

    return lambda_0, lambda_1, lambda_2, f_0, f_1, f_2, a_ij_0, a_ij_1, a_ij_2
