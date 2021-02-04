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
    planet_table = NasaExoplanetArchive.query_planet(name)
    planet_info = {'radius': planet_table['pl_radj'],
                   'mass': planet_table['pl_bmassj'],
                   'semi_major_axis': planet_table['pl_orbsmax']}
    star_info = {'radius': planet_table['st_rad'],
                 'mass': planet_table['st_mass'],
                 'distance': planet_table['st_dist'],
                 'effective_temperature': planet_table['st_teff']}

    return planet_info, star_info
