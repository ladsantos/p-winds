#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module tabulated information about spectral lines relevant to atmospheric
escape detection.
"""


__all__ = ["he_3_properties", "c_ii_properties", "o_i_properties",
           "balmer_halpha_properties", "balmer_hbeta_properties",
           "balmer_hgamma_properties"]


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
    lambda_0 : ``float``
        Central wavelength in air of line 0 in unit of m.

    lambda_1 : ``float``
        Central wavelength in air of line 1 in unit of m.

    lambda_2 : ``float``
        Central wavelength in air of line 2 in unit of m.

    f_0 : ``float``
        Oscillator strength of line 0 (unitless).

    f_1 : ``float``
        Oscillator strength of line 1 (unitless).

    f_2 : ``float``
        Oscillator strength of line 2 (unitless).

    a_ij : ``float``
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
    lambda_0 : ``float``
        Central wavelength in air of line 0 in unit of m.

    lambda_1 : ``float``
        Central wavelength in air of line 1 in unit of m.

    lambda_2 : ``float``
        Central wavelength in air of line 2 in unit of m.

    f_0 : ``float``
        Oscillator strength of line 0 (unitless).

    f_1 : ``float``
        Oscillator strength of line 1 (unitless).

    f_2 : ``float``
        Oscillator strength of line 2 (unitless).

    a_ij_0 : ``float``
        Einstein coefficient of line 0 in unit of 1 / s.

    a_ij_1 : ``float``
        Einstein coefficient of line 1 in unit of 1 / s.

    a_ij_2 : ``float``
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
    lambda_0 : ``float``
        Central wavelength in air of line 0 in unit of m.

    lambda_1 : ``float``
        Central wavelength in air of line 1 in unit of m.

    lambda_2 : ``float``
        Central wavelength in air of line 2 in unit of m.

    f_0 : ``float``
        Oscillator strength of line 0 (unitless).

    f_1 : ``float``
        Oscillator strength of line 1 (unitless).

    f_2 : ``float``
        Oscillator strength of line 2 (unitless).

    a_ij_0 : ``float``
        Einstein coefficient of line 0 in unit of 1 / s.

    a_ij_1 : ``float``
        Einstein coefficient of line 1 in unit of 1 / s.

    a_ij_2 : ``float``
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


# Line properties of the Balmer series lines (alpha, beta, gamma) taken from the
# NIST database https://www.nist.gov/pml/atomic-spectra-database
def balmer_halpha_properties():
    """
    Returns the central wavelengths in air, shell number and the Einstein
    coefficient of the Balmer lines. The values were taken from the NIST
    database: https://www.nist.gov/pml/atomic-spectra-database

    Returns
    -------
    lambda_alpha : ``float``
        Central wavelength in air of the H-alpha line in unit of m.

    n_alpha : ``integer``
        Shell number creating the H-alpha line (unitless).
        
    f_alpha : ``float``
        Oscillator strength (unitless).
    
    g_alpha : ``float``
        log10(g2f2n_alpha) taking into account the oscillator strength (f2n)
        and the statistical weight (g2) of the H-alpha line (unitless).

    a_alpha : ``float``
        Einstein coefficient of the H-alpha line in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_alpha = 6562.79 * 1E-10
    
    #shell number
    n_alpha = 3
    
    #oscillator strength
    f_alpha = 6.4108e-01
    
    #statistical weight
    g_alpha = 10**(0.71)
    
    # Einstein coefficient in units of s ** (-1)
    a_alpha = 4.4101e+07

    return lambda_alpha, n_alpha, f_alpha, g_alpha, a_alpha


def balmer_hbeta_properties():
    """
    Returns the central wavelengths in air, shell number and the Einstein
    coefficient of the Balmer lines. The values were taken from the NIST
    database: https://www.nist.gov/pml/atomic-spectra-database

    Returns
    -------
    lambda_beta : ``float``
        Central wavelength in air of the H-beta line in unit of m.

    n_beta : ``integer``
        Shell number creating the H-beta line (unitless).
    
    f_beta : ``float``
        Oscillator strength (unitless).
    
    g_beta : ``float``
        log10(g2f2n_alpha) taking into account the oscillator strength (f2n)
        and the statistical weight (g2) of the H-beta line (unitless).
        
    a_beta : ``float``
        Einstein coefficient of the H-beta line in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_beta = 4861.35 * 1E-10

    # Oscillator strength
    f_beta = 1.1938e-01
    
    #shell number
    n_beta = 4
    
    #statistical weight
    g_beta= 10**(-0.02)
    
    # Einstein coefficient in units of s ** (-1)
    a_beta = 8.4193e+06

    return lambda_beta, n_beta, f_beta, g_beta,  a_beta


def balmer_hgamma_properties():
    """
    Returns the central wavelengths in air, shell number and the Einstein
    coefficient of the Balmer lines. The values were taken from the NIST
    database: https://www.nist.gov/pml/atomic-spectra-database

    Returns
    -------
    lambda_gamma : ``float``
        Central wavelength in air of the H-gamma line in unit of m.

    n_gamma : ``integer``
        Shell number creating the H-gamma line (unitless).
        
    f_gamma : ``float``
        Oscillator strength (unitless).
        
    g_gamma : ``float``
        log10(g2f2n_alpha) taking into account the oscillator strength (f2n)
        and the statistical weight (g2) of the H-gamma line (unitless).

    a_gamma : ``float``
        Einstein coefficient of the H-gamma line in unit of 1 / s.
    """
    # Central wavelengths in units of m
    lambda_gamma = 4340.47 * 1E-10
    
    #shell number
    n_gamma = 5
    
    # Oscillator strength
    f_gamma = 4.4694e-02
    
    #statistical weight
    g_gamma = 10**(-0.447)
    
    # Einstein coefficient in units of s ** (-1)
    a_gamma = 2.5304e+06

    return lambda_gamma, n_gamma, f_gamma, g_gamma, a_gamma
