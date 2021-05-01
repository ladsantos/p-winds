#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np


__all__ = ["nearest_index", "make_spectrum_from_file"]


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


def make_spectrum_from_file(filename, units, path='', skiprows=0,
                            scale_flux=1.0):
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

    scale_flux (``float``, optional):
        Scaling factor for flux. Default value is 1.0 (no scaling).

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
                y_axis: spectrum_table[:, 1] * scale_flux,
                '{}_unit'.format(x_axis): x_axis_unit,
                'flux_unit': y_axis_unit}
    return spectrum
