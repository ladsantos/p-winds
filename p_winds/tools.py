#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from warnings import warn

try:
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
except ModuleNotFoundError:
    warn('`astroquery` is not installed, `fetch_planet_system` cannot be used.',
         Warning)


def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Args:
        array (``numpy.array``): Target array.
        target_value (``float``): Target value.

    Returns:
        index (``int``): Index of the value in ``array`` that is closest to
            ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


def fetch_planet_system(name):
    """
    Use `astroquery` to fetch the planetary and stellar parameters of a given
    exoplanet from the NASA Exoplanet Database.

    Parameters
    ----------
    name (`str`): Name of the planet.

    Returns
    -------
    planet_info (`dict`): Dictionary containing the planetary parameters.

    star_info (`dict): Dictionary containing the stellar parameters.

    """
    try:
        planet_table = NasaExoplanetArchive.query_planet(name)
    except NameError:
        raise RuntimeError('`astroquery` is necessary to run '
                           '`fetch_planet_system.`')

    planet_info = {'radius': planet_table['pl_radj'],
                   'mass': planet_table['pl_bmassj'],
                   'semi_major_axis': planet_table['pl_orbsmax']}
    star_info = {'radius': planet_table['st_rad'],
                 'mass': planet_table['st_mass'],
                 'distance': planet_table['st_dist'],
                 'effective_temperature': planet_table['st_teff']}

    return planet_info, star_info


def make_spectrum_dict(filename, units, path=''):
    """
    Construct a dictionary containing an input spectrum from a text file. The
    input file must have two or more columns, in which the first is the
    wavelength bin center and the second is the flux at 1 au per unit of
    wavelength.

    Parameters
    ----------
    filename (``str``): Name of the file containing the spectrum data.

    units (``dict``): Units of the spectrum. This dictionary must have the
        entries `'wavelength'` and `'flux'`, and the units must be set in
        `astropy.units`. Example:
        ```
        units = {'wavelength': u.angstrom,
                 'flux': u.erg / u.s / u.cm ** 2 / u.angstrom}
        ```

    path (``str``, optional): Path to the spectrum data file.

    Returns
    -------
    spectrum (``dict``): Spectrum dictionary with entries for the wavelength
        and flux, and their units.

    """
    spectrum_table = np.loadtxt(path + filename, usecols=(0, 1))
    spectrum = {'wavelength': spectrum_table[:, 0],
                'flux_1au': spectrum_table[:, 1],
                'wavelength_unit': units['wavelength'],
                'flux_unit': units['flux']}
    return spectrum
