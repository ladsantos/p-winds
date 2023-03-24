#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute the wind energetics using the method
outlined in Vissapragada et al. (2022).
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, cumtrapz
from scipy.interpolate import interp1d


__all__ = ["calculate_mdot_max", "calculate_epsilon_max", "spec_av_cross",
           "compute_column_densities", "compute_transmission_coefficient",
           "calculate_roche_radius", "calculate_f_xuv", "h_photo_cross",
           "heplus_photo_cross", "helium_photo_cross"]

threshes = {'hydrogen': 13.6 * u.eV, 'helium': 24.58 * u.eV,
            'helium+': 54.4 * u.eV}


def calculate_mdot_max(R_pl, M_pl, M_star, a, r_grid, spectrum, n_h, n_he, 
                       n_he_plus):
    """
    Calculates the maximum mass-loss rate of an outflow given the profiles of
    neutral hydrogen, neutral helium, and ionized helium by Equation (20) of
    Vissapragada et al. (2022).

    Parameters
    ----------
    R_pl : ``astropy.Quantity``
        The planetary radius. An astropy unit (like u.Rjup) must be specified.

    M_pl : ``astropy.Quantity``
        The planetary mass. An astropy unit (like u.Mjup) must be specified.

    M_star : ``astropy.Quantity``
        The stellar mass. An astropy unit (like u.Msun) must be specified.

    a : ``astropy.Quantity``
        The semimajor axis of the planetary orbit. An astropy unit (like u.au)
        must be specified.

    r_grid : ``numpy.ndarray``
        The radius grid for the calculation. An astropy unit (like u.Rjup) must
        be specified for each value on the grid.
  
    spectrum : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes up to
        the wavelength corresponding to the energy to ionize hydrogen (13.6 eV,
        or 911.65 Angstrom). Can be generated using ``tools.make_spectrum_dict``
        or ``tools.generate_muscles_spectrum``. Currently we assume that the
        spectrum does not include lower energies than 13.6 eV.

    n_h : ``numpy.ndarray``
        The neutral hydrogen number density profile for the wind. An astropy
        unit (like u.cm**-3) must be specified for each value on the grid.
 
    n_he : ``numpy.ndarray``
        The neutral helium number density profile for the wind. An astropy
        unit (like u.cm**-3) must be specified for each value on the grid.

    n_he_plus : ``numpy.ndarray``
        The (singly-)ionized helium number density profile for the wind. An
        astropy unit (like u.cm**-3) must be specified for each value on the
        grid.

    Returns
    -------
    md : ``astropy.Quantity``
        Maximum mass-loss rate (in g/s) of the wind assuming it is driven only
        by photoionization.
    """
    R_roche = calculate_roche_radius(R_pl, M_pl, M_star, a)
    eps = calculate_epsilon_max(r_grid, spectrum, n_h, n_he, n_he_plus, 
        R_pl, R_roche)
    K = 1 - 1.5 * (R_pl / R_roche) + 0.5 * (R_pl / R_roche) ** 3
    F_XUV = calculate_f_xuv(spectrum)
    md = 4 * eps * np.pi * R_pl.to(u.cm) ** 3 * F_XUV / (K * c.G * M_pl.to(u.g))
    return md.to(u.g/u.s)


def calculate_epsilon_max(r_grid, spectrum, n_h, n_he, n_he_plus, R_pl,
                          R_roche):
    """
    Calculates the maximum mass-loss efficiency of an outflow given the
    profiles of neutral hydrogen, neutral helium, and ionized helium by
    Equation (19) of Vissapragada et al. (2022).

    Parameters
    ----------
    r_grid : ``numpy.ndarray``
        The radius grid for the calculation. An astropy unit (like u.Rjup) must
        be specified for each value on the grid.
  
    spectrum : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes up to
        the wavelength corresponding to the energy to ionize hydrogen (13.6 eV,
        or 911.65 Angstrom). Can be generated using ``tools.make_spectrum_dict``
        or ``tools.generate_muscles_spectrum``. Currently we assume that the
        spectrum does not include lower energies than 13.6 eV.

    n_h : ``numpy.ndarray``
        The neutral hydrogen number density profile for the wind. An astropy
        unit (like u.cm**-3) must be specified for each value on the grid.
 
    n_he : ``numpy.ndarray``
        The neutral helium number density profile for the wind. An astropy
        unit (like u.cm**-3) must be specified for each value on the grid.

    n_he_plus : ``numpy.ndarray``
        The (singly-)ionized helium number density profile for the wind. An
        astropy unit (like u.cm**-3) must be specified for each value on the
        grid.

    R_pl : ``astropy.Quantity``
        The planetary radius. An astropy unit (like u.Rjup) must be specified.

    R_roche : ``astropy.Quantity``
        The Roche radius. An astropy unit (like u.Rjup) must be specified.

    Returns
    -------
    eps : ``float``
        Maximum mass-loss efficiency of the wind assuming it is driven only
        by photoionization.
    """
    N_h, N_he, N_he_plus = compute_column_densities(r_grid, n_h, n_he, 
        n_he_plus)
    t_coef = compute_transmission_coefficient(spectrum, r_grid,
        N_h, N_he, N_he_plus)

    h_av_cross = spec_av_cross(r_grid, spectrum, t_coef, 'hydrogen')
    he_av_cross = spec_av_cross(r_grid, spectrum, t_coef, 'helium')
    he_plus_av_cross = spec_av_cross(r_grid, spectrum, t_coef, 'helium+')

    h_heating_coef = h_av_cross * n_h
    he_heating_coef = he_av_cross * n_he
    he_plus_heating_coef = he_plus_av_cross * n_he_plus
    total_heating_coef = h_heating_coef + he_heating_coef + he_plus_heating_coef
    
    r = r_grid.to(u.cm)
    integrand = total_heating_coef * r ** 2
    mask = (r >= R_pl) & (r <= R_roche)
    eps = simps(integrand[mask], x=r[mask]) * u.cm ** 2 / (R_pl.to(u.cm) ** 2)
    return eps


def spec_av_cross(r_grid, spectrum, t_coef, species):
    """
    Calculates the heating cross-section for photoionization using Equation (16)
    of Vissapragada et al. (2022).

    Parameters
    ----------
    r_grid : ``numpy.ndarray``
        The radius grid for the calculation. An astropy unit (like u.Rjup) must
        be specified for each value on the grid.
  
    spectrum : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes up to
        the wavelength corresponding to the energy to ionize hydrogen (13.6 eV,
        or 911.65 Angstrom). Can be generated using ``tools.make_spectrum_dict``
        or ``tools.generate_muscles_spectrum``. Currently we assume that the
        spectrum does not include lower energies than 13.6 eV.

    t_coef : ``numpy.ndarray``
        The transmission coefficient profile for the wind as a function of 
        frequency and altitude. In the optically-thin part of the outflow this
        should be very close to 1.
 
    species : ``str``
        The photoionzation target for which we are calculating the heating
        cross-section. Must be one of 'hydrogen', 'helium', or 'helium+'.

    Returns
    -------
    cross : ``astropy.Quantity``
        Heating cross-section in cm ** 2 for the selected species.
    """
    wav_grid = spectrum['wavelength'] * spectrum['wavelength_unit']
    flux_grid = spectrum['flux_lambda'] * spectrum['flux_unit']
    flux_grid = flux_grid.to(u.erg / u.s / u.cm / u.cm / u.Hz,
                             equivalencies=u.spectral_density(wav_grid))
    wavs_hz = wav_grid.to(u.Hz, equivalencies=u.spectral())[::-1]
    flux_grid = flux_grid[::-1]
    xx, yy = np.meshgrid(wavs_hz, r_grid)
    
    threshold = threshes[species]
    crosses = {'hydrogen': h_photo_cross, 'helium': helium_photo_cross,
               'helium+': heplus_photo_cross}
    cross = crosses[species]
    evgrid = xx.to(u.eV, equivalencies=u.spectral())
    eta_grid = 1 - threshold/evgrid
    spec_grid, __ = np.meshgrid(flux_grid, r_grid)
    crossgrid = cross(xx)
    crossgrid[xx.to(u.eV, equivalencies=u.spectral()) < threshold] = \
        0. * u.cm ** 2

    numgrid = eta_grid * spec_grid * crossgrid * t_coef
    numgrid = numgrid.to(u.erg / u.s / u.Hz)
    num = simps(numgrid, x=wavs_hz, axis=-1) * u.erg/u.s

    F_XUV = calculate_f_xuv(spectrum)
    cross = num / F_XUV
    return cross.to(u.cm**2)


def compute_column_densities(r_grid, n_h, n_he, n_he_plus):
    """
    Given the density profiles of H, He, and He+, this function calculates the
    column densities.

    Parameters
    ----------
    r_grid : ``numpy.ndarray``
        The radius grid for the calculation. An astropy unit (like u.Rjup) must
        be specified for each value on the grid.

    n_h : ``numpy.ndarray``
        The neutral hydrogen number density profile for the wind. An astropy
        unit (like u.cm ** -3) must be specified for each value on the grid.
 
    n_he : ``numpy.ndarray``
        The neutral helium number density profile for the wind. An astropy
        unit (like u.cm ** -3) must be specified for each value on the grid.

    n_he_plus : ``numpy.ndarray``
        The (singly-)ionized helium number density profile for the wind. An
        astropy unit (like u.cm ** -3) must be specified for each value on the
        grid.

    Returns
    -------
    column_h : ``numpy.ndarray``
        The neutral hydrogen column density profile in u.cm ** -2.

    column_he : ``numpy.ndarray``
        The neutral helium column density profile in u.cm ** -2.

    column_he_plus : ``numpy.ndarray``
        The ionized helium column density profile in u.cm ** -2.
    """
    # Flip grid to integrate from infty to r
    r_grid_temp = r_grid.to(u.cm).value[::-1]
    n_h_temp = n_h[::-1]
    n_he_temp = n_he[::-1]
    n_he_plus_temp = n_he_plus[::-1]
    
    column_h = cumtrapz(n_h_temp, r_grid_temp, initial=0) * u.cm ** -2
    column_he = cumtrapz(n_he_temp, r_grid_temp, initial=0) * u.cm ** -2
    column_he_plus = cumtrapz(n_he_plus_temp, r_grid_temp,
                              initial=0) * u.cm ** -2
    
    # Flip back
    column_h = -column_h[::-1]
    column_he = -column_he[::-1]
    column_he_plus = -column_he_plus[::-1]

    return column_h, column_he, column_he_plus


def compute_transmission_coefficient(spectrum, r_grid, N_h, N_he, N_he_plus):
    """
    Given the column density profiles of H, He, and He+, this function 
    calculates the transmission coefficient (negative exponential of the optical
    depth) as a function of altitude and frequency. 

    Parameters
    ----------
    r_grid : ``numpy.ndarray``
        The radius grid for the calculation. An astropy unit (like u.Rjup) must
        be specified for each value on the grid.

    spectrum : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes up to
        the wavelength corresponding to the energy to ionize hydrogen (13.6 eV,
        or 911.65 Angstrom). Can be generated using ``tools.make_spectrum_dict``
        or ``tools.generate_muscles_spectrum``. Currently we assume that the
        spectrum does not include lower energies than 13.6 eV.

    N_h : ``numpy.ndarray``
        The neutral hydrogen column density profile. An astropy unit (like
        u.cm ** -2) must be specified for each value on the grid.

    N_he : ``numpy.ndarray``
        The neutral helium column density profile. An astropy unit (like
        u.cm ** -2) must be specified for each value on the grid.

    N_he_plus : ``numpy.ndarray``
        The ionized helium column density profile. An astropy unit (like
        u.cm ** -2) must be specified for each value on the grid.

    Returns
    -------
    t_coef : ``numpy.ndarray``
        The transmission coefficient profile for the wind as a function of 
        frequency and altitude.
    """
    wavs = spectrum['wavelength'] * spectrum['wavelength_unit']
    wavs_hz = wavs.to(u.Hz, equivalencies=u.spectral())[::-1]

    cd_h = interp1d(r_grid.to(u.cm).value, N_h.to(u.cm ** -2).value,
                    kind='cubic')
    cd_he = interp1d(r_grid.to(u.cm).value, N_he.to(u.cm ** -2).value,
                     kind='cubic')
    cd_he_plus = interp1d(r_grid.to(u.cm).value, N_he_plus.to(u.cm**-2).value,
                          kind='cubic')

    xx, yy = np.meshgrid(wavs_hz, r_grid)    
    h_cross = h_photo_cross(xx)
    he_cross = helium_photo_cross(xx) 
    he_plus_cross = heplus_photo_cross(xx)

    h_cross[xx.to(u.eV, equivalencies=u.spectral()) < threshes['hydrogen']] = \
        0. * u.cm ** 2
    he_cross[xx.to(u.eV, equivalencies=u.spectral()) < threshes['helium']] = \
        0. * u.cm ** 2
    he_plus_cross[xx.to(u.eV, equivalencies=u.spectral()) <
                  threshes['helium+']] = 0. * u.cm ** 2

    h_column = cd_h(yy.to(u.cm).value) * u.cm ** -2
    he_column = cd_he(yy.to(u.cm).value) * u.cm ** -2
    he_plus_column = cd_he_plus(yy.to(u.cm).value) * u.cm ** -2

    tau_h = h_cross * h_column
    tau_he = he_cross * he_column
    tau_he_plus = he_plus_cross * he_plus_column

    tau_tot = tau_h + tau_he + tau_he_plus
    transmission_coef = np.exp(-tau_tot)
    
    return transmission_coef


def calculate_roche_radius(R_pl, M_pl, M_star, a):
    """
    Calculates the Roche radius for the planet.

    Parameters
    ----------
    R_pl : ``astropy.Quantity``
        The planetary radius. An astropy unit (like u.Rjup) must be specified.

    M_pl : ``astropy.Quantity``
        The planetary mass. An astropy unit (like u.Mjup) must be specified.

    M_star : ``astropy.Quantity``
        The stellar mass. An astropy unit (like u.Msun) must be specified.

    a : ``astropy.Quantity``
        The semimajor axis of the planetary orbit. An astropy unit (like u.au)
        must be specified.
 
    Returns
    -------
    R_roche : ``astropy.Quantity``
        The Roche radius.
    """
    return (a * (M_pl / (3 * M_star)) ** (1 / 3)).to(u.Rjup)


def calculate_f_xuv(spectrum):
    """
    Calculates the total XUV flux given the spectrum at the planet (Equation
    14 of Vissapragada et al. 2022, where the minimum threshold is 13.6 eV. This
    function currently assumes the spectrum is truncated at 13.6 eV and does not
    include lower energies. 

    Parameters
    ----------
    spectrum : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes up to
        the wavelength corresponding to the energy to ionize hydrogen (13.6 eV,
        or 911.65 Angstrom). Can be generated using ``tools.make_spectrum_dict``
        or ``tools.generate_muscles_spectrum``. Currently we assume that the
        spectrum does not include lower energies than 13.6 eV.

    Returns
    -------
    f_xuv : ``astropy.Quantity``
        The integrated XUV flux.
    """
    wav_grid = spectrum['wavelength'] * spectrum['wavelength_unit']
    flux_grid = spectrum['flux_lambda'] * spectrum['flux_unit']
    flux_grid = flux_grid.to(u.erg / u.s / u.cm / u.cm / u.Hz,
                             equivalencies=u.spectral_density(wav_grid))
    wavs_hz = wav_grid.to(u.Hz, equivalencies=u.spectral())[::-1]
    flux_grid = flux_grid[::-1]
    f_xuv = simps(flux_grid, x=wavs_hz) * u.erg / u.s / u.cm ** 2
    return f_xuv


def h_photo_cross(nu_init):
    """
    Calculates the photoionization cross-section of hydrogen (Equation 17 from
    Vissapragada et al. 2022). Note: potential overlap with 
    ``microphysics.hydrogen_cross_section``.

    Parameters
    ----------
    nu_init : ``numpy.ndarray``
        Frequencies at which to compute the cross-section. An astropy unit
        (like u.Hz) must be specified for each value in the array.

    Returns
    -------
    out : ``numpy.ndarray``
        The cross-section in cm ** 2.
    """
    "nu in Hz -> cross section in units cm^2"
    threshold_energy = threshes['hydrogen']
    threshold_cross = 6.3e-18 * u.cm * u.cm
    
    nu = nu_init.to(u.eV, equivalencies=u.spectral())
    
    eps = np.sqrt(nu / threshold_energy - 1)
    arg1 = threshold_cross*np.exp(4 - 4 * np.arctan(eps) / eps / u.rad)
    arg2 = (1 - np.exp(-2 * np.pi / eps))
    arg3 = (threshold_energy / nu) **4
    
    out = arg1 / arg2 * arg3
    return out


def heplus_photo_cross(nu_init):
    """
    Calculates the photoionization cross-section of ionized helium (Equation 17
    from Vissapragada et al. 2022).

    Parameters
    ----------
    nu_init : ``numpy.ndarray``
        Frequencies at which to compute the cross-section. An astropy unit
        (like u.Hz) must be specified for each value in the array.

    Returns
    -------
    out : ``numpy.ndarray``
        The cross-section in cm**2. 
    """
    "nu in Hz -> cross section in units cm^2"
    threshold_energy = threshes['helium+']
    threshold_cross = 6.3e-18 / 4 * u.cm * u.cm
    
    nu = nu_init.to(u.eV, equivalencies=u.spectral())
    
    eps = np.sqrt(nu/threshold_energy - 1)
    arg1 = threshold_cross * np.exp(4 - 4 * np.arctan(eps) / eps / u.rad)
    arg2 = (1 - np.exp(-2 * np.pi / eps))
    arg3 = (threshold_energy / nu) **4
    
    out = arg1 / arg2 * arg3
    return out


def helium_photo_cross(nu_init):
    """
    Calculates the photoionization cross-section of neutral helium. This is 
    Equation 18 from Vissapragada et al. 2022, which itself is from Yan, 
    Sadeghpour, & Dalgarno (1998, ApJ). Note that this does not overlap with
    the state-resolved cross-sections from the ``microphysics`` module as this
    is a total photoionization cross-section across all states. 

    Parameters
    ----------
    nu_init : ``numpy.ndarray``
        Frequencies at which to compute the cross-section. An astropy unit
        (like u.Hz) must be specified for each value in the array.

    Returns
    -------
    out : ``numpy.ndarray``
        The cross-section in cm**2. 
    """
    # nu in Hz -> cross section in units cm^2
    # Source: Yan, Sadeghpour, & Dalgarno (1998, ApJ)
    threshold_energy = threshes['helium']
    
    nu = nu_init.to(u.eV, equivalencies=u.spectral())
    
    nu_masked = nu
    x = nu_masked / threshold_energy
    e_term = (nu_masked / u.eV / 1000.)**(7 / 2)
    
    coefs = [-4.7416, 14.8200, -30.8678, 37.3584, -23.4585, 5.9133]
    sum_term = 0.
    for i, coef in enumerate(coefs):
        sum_term += coef / x ** ((i + 1) / 2)
    
    num = 733 * u.barn
    out = num.to(u.cm * u.cm)/e_term * (1 + sum_term)

    return out
