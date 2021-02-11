#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from warnings import warn

try:
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
except ModuleNotFoundError:
    warn('`astroquery` is not installed, `fetch_planet_system` cannot be used.',
         Warning)


__all__ = ["nearest_index", "fetch_planet_system", "make_spectrum_dict"]


def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Parameters
    ----------
    array (``numpy.array``):
        Target array.
    target_value (``float``):
        Target value.

    Returns
    -------
    index (``int``):
        Index of the value in ``array`` that is closest to ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


def fetch_planet_system(name):
    """
    Use ``astroquery`` to fetch the planetary and stellar parameters of a given
    exoplanet from the NASA Exoplanet Database.

    Parameters
    ----------
    name (``str``):
        Name of the planet.

    Returns
    -------
    planet_info (``dict``):
        Dictionary containing the planetary parameters.

    star_info (``dict``):
        Dictionary containing the stellar parameters.

    """
    try:
        planet_table = NasaExoplanetArchive.query_object(name)
    except NameError:
        raise RuntimeError('`astroquery` is necessary to run '
                           '`fetch_planet_system.`')

    planet_info = {'radius': planet_table['pl_radj'],
                   'mass': planet_table['pl_bmassj'],
                   'semi_major_axis': planet_table['pl_orbsmax'],
                   'period': planet_table['pl_orbper']}
    star_info = {'radius': planet_table['st_rad'],
                 'mass': planet_table['st_mass'],
                 'distance': planet_table['st_dist'],
                 'effective_temperature': planet_table['st_teff']}

    # Check if one or more of the retrieved parameters is not "good" and warn
    # the user
    _items = planet_info.items()
    for line in _items:
        if line[1].value < 1E-5:
            warn("Planetary parameter '{}' of {} requires your "
                 "attention.".format(line[0], name), Warning)
    _items = star_info.items()
    for line in _items:
        if line[1].value < 1E-5:
            warn("Stellar parameter '{}' of {} requires your "
                 "attention.".format(line[0], name), Warning)

    return planet_info, star_info


def make_spectrum_dict(filename, units, path='', skiprows=0):
    """
    Construct a dictionary containing an input spectrum from a text file. The
    input file must have two or more columns, in which the first is the
    wavelength or frequency bin center and the second is the flux per bin of
    wavelength or frequency. The code automatically figures out if the input
    spectra are binned in wavelength or frequency based on the units the user
    passes.

    Parameters
    ----------
    filename (``str``):
        Name of the file containing the spectrum data.

    units (``dict``):
        Units of the spectrum. This dictionary must have the entries
        ``'wavelength'`` and ``'flux'``, or ``'frequency'`` and ``'flux'``.
        The units must be set in ``astropy.units``.

    path (``str``, optional):
        Path to the spectrum data file.

    skiprows (``int``, optional):
        Number of rows to skip corresponding to the header of the input text
        file.

    Returns
    -------
    spectrum (``dict``):
        Spectrum dictionary with entries for the wavelength and flux, and their
        units.

    """
    spectrum_table = np.loadtxt(path + filename, usecols=(0, 1),
                                skiprows=skiprows, dtype=float)

    try:
        x_axis = 'wavelength'
        x_axis_unit = units.pop(x_axis)
        y_axis = 'flux_lambda'
    except KeyError:
        x_axis = 'frequency'
        x_axis_unit = units.pop(x_axis)
        y_axis = 'flux_nu'
    y_axis_unit = units.pop('flux')

    spectrum = {x_axis: spectrum_table[:, 0],
                y_axis: spectrum_table[:, 1],
                '{}_unit'.format(x_axis): x_axis_unit,
                'flux_unit': y_axis_unit}
    return spectrum
