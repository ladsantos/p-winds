#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes a cooling rates following equations from Table 3 in Allan
et al. 2023 (https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.4657A/abstract).
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np


__all__ = ["collisional_ionization", "recombination", "collisional_excitation",
           "bremsstrahlung"]


# Cooling rates for collisional ionization
def collisional_ionization(n_e, n_sp, species, temperature):
    """
    General formula for calculating the cooling rate due to collisional
    ionization. Species must be specified.

    Parameters
    ----------
    n_e : ``float``
        Electron number density in units of 1 / cm ** (-3).

    n_sp : ``float``
        Number density of the ``species`` in units of 1 / cm ** (-3).

    species : ``str``
        String that defines which species for which to calculate the cooling
        rate. Supported species are: ``'h_0`` (neutral hydrogen), ``'he_1``
        (helium singlet), ``'he_3`` (helium triplet) and ``'he_ion`` (helium
        ion).

    temperature : ``float``
        Temperature in unit of Kelvin.

    Returns
    -------
    cooling_rate : ``float``
        Cooling rate due to collisional ionization in erg / cm ** (-3) / s.
    """
    term_1 = {'h_0': 1.27e-21,
              'he_1': 9.38E-22,
              'he_3': 6.41E-21,
              'he_ion': 4.95E-22}
    term_2 = {'h_0': -157809.1,
              'he_1': -285335.4,
              'he_3': -55338,
              'he_ion': -631515}
    cooling_rate = (term_1[species] * temperature ** 0.5 *
                    np.exp(term_2[species] / temperature) * n_e * n_sp)
    return cooling_rate


# Cooling rates due to recombination of ions
def recombination(n_e, n_sp, species, temperature):
    """
    General formula for calculating the cooling rate due to recombination.
    Species must be specified.

    Parameters
    ----------
    n_e : ``float``
        Electron number density in units of 1 / cm ** (-3).

    n_sp : ``float``
        Number density of the ``species`` in units of 1 / cm ** (-3).

    species : ``str``
        String that defines which species for which to calculate the cooling
        rate. Supported species are: ``'h_ion`` (singly-ionized hydrogen),
        ``'he_ion`` (singly-ionized helium) and ``'he_++`` (doubly-ionized
        helium).

    temperature : ``float``
        Temperature in unit of Kelvin.

    Returns
    -------
    cooling_rate : ``float``
        Cooling rate due to recombination in erg / cm ** (-3) / s.
    """
    t = temperature
    term = {'h_ion': 8.70E-27 * t ** 0.5 * (t / 1000) ** (-0.2) /
            (1 + (t / 1E6) ** 0.5),
            'he_ion': 1.55E-26 * t ** 0.3647 + 1.24E-13 * t ** (-1.5) *
            np.exp(-4.7E5 / t) * (1 + 0.3 * np.exp(-9.4E4 / t)),
            # he_ion includes dielectronic recombination formula
            'he_++': 3.48E-26 * t ** 0.5 * (t / 1000) ** (-0.2) /
            (1 + (t / 1E6) ** 0.5)}
    cooling_rate = term[species] * n_e * n_sp
    return cooling_rate


# Cooling rate for collisional excitation
def collisional_excitation(n_e, n_sp, species, temperature):
    """
    General formula for calculating the cooling rate due to collisional
    excitation. Species must be specified.

    Parameters
    ----------
    n_e : ``float``
        Electron number density in units of 1 / cm ** (-3).

    n_sp : ``float``
        Number density of the ``species`` in units of 1 / cm ** (-3).

    species : ``str``
        String that defines which species for which to calculate the cooling
        rate. Supported species are: ``'h_0`` (neutral hydrogen),
        ``'he_ion`` (singly-ionized helium) and ``'he_3`` (helium triplet).

    temperature : ``float``
        Temperature in unit of Kelvin.

    Returns
    -------
    cooling_rate : ``float``
        Cooling rate due to collisional excitation in erg / cm ** (-3) / s.
    """
    term_1 = {'h_0': 7.5E-19,
              'he_3': 1.16E-20 * temperature ** 0.5,
              'he_ion': 5.54E-17 * temperature ** (-0.397)}
    term_2 = {'h_0': -118348,
              'he_3': -13179,
              'he_ion': -473638}
    cooling_rate = (term_1[species] * np.exp(term_2[species] / temperature) *
                    n_e * n_sp)
    return cooling_rate


# Cooling rate due to free-free emission
def bremsstrahlung(n_e, n_h_ion, temperature, n_he_ion=0.0, n_he_pp=0.0):
    """
    Calculates the cooling rate due to free-free emission (also known as
    Bremsstrahlung).

    Parameters
    ----------
    n_e : ``float``
        Number density of electrons in units of 1 / cm ** (-3).

    n_h_ion : ``float``
        Number density of ionized hydrogen in units of 1 / cm ** (-3).

    temperature : ``float``
        Temperature in unit of Kelvin.

    n_he_ion : ``float``, optional
        Number density of singly-ionized helium in units of 1 / cm ** (-3).
        Default value is 0.0.

    n_he_pp : ``float``, optional
        Number density of doubly-ionized helium in units of 1 / cm ** (-3).
        Default value is 0.0.

    Returns
    -------
    cooling_rate : ``float``
        Cooling rate due to free-free emission in erg / cm ** (-3) / s.
    """
    cooling_rate = (1.42E-27 * temperature ** 0.5 * (3 / 2) *
                    (n_h_ion + n_he_ion + 4 * n_he_pp) * n_e)
    return cooling_rate
