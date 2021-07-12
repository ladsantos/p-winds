#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.io import fits


__all__ = ["nearest_index", "generate_muscles_spectrum",
           "make_spectrum_from_file"]


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


def generate_muscles_spectrum(host_star_name, muscles_dir, semi_major_axis,
                              truncate_wavelength_grid=False):
    """
    Construct a dictionary containing an input spectrum from a MUSCLES spectrum.
    MUSCLES reports spectra as observed at Earth, the code scales this to the
    spectrum received at your planet provided a value for the scaled semi-major
    axis a_rs.

    Parameters
    ----------
    host_star_name (``str``):
        Name of the stellar spectrum you want to use in MUSCLES. Must be one of:
        ['gj176', 'gj436', 'gj551', 'gj581', 'gj667c', 'gj832', 'gj876',
        'gj1214', 'hd40307', 'hd85512', 'hd97658', 'v-eps-eri'].

    muscles_dir (``str``):
        Path to the directory with the MUSCLES data.

    semi_major_axis (``float``):
        Semi-major axis of the planet in units of stellar radii. The code first
        converts the MUSCLES spectrum to what it would be at R_star;
        ``semi_major_axis`` is needed to get the spectrum at the planet.

    truncate_wavelength_grid (``bool``, optional):
        If True, will only return the spectrum with energy > 13.6 eV. This may
        be useful for computational expediency. If False, returns the whole
        spectrum. Default is ``False``.

    Returns
    -------
    spectrum (``dict``):
        Spectrum dictionary with entries for the wavelength and flux, and their
        units.
    """
    # Hard coding some values
    # The stellar radii and distances are taken from NASA Exoplanet Archive.

    thresh = 13.6 * u.eV
    stars = ['gj176', 'gj436', 'gj551', 'gj581', 'gj667c', 'gj832', 'gj876',
             'gj1214', 'hd40307', 'hd85512', 'hd97658', 'v-eps-eri']
    st_rads = np.array([0.46, 0.449, 0.154, 0.3297020, 0.42, 0.45, 0.35, 0.22,
                        0.71, 0.69, 0.74, 0.77]) * u.solRad
    dists = np.array([9.470450, 9.75321, 1.30119, 6.298100, 7.24396, 4.964350,
                      4.67517, 14.6427, 12.9363, 11.2810, 21.5618,
                      3.20260]) * u.pc
    muscles_dists = {starname: dist for starname, dist in zip(stars, dists)}
    muscles_rstars = {starname: st_rad for starname, st_rad in zip(stars,
                                                                   st_rads)}

    # MUSCLES records spectra as observed at earth
    dist = muscles_dists[host_star_name]
    rstar = muscles_rstars[host_star_name]
    conv = float((dist / rstar) ** 2)  # conversion to spectrum at R_star

    # Read the MUSCLES spectrum
    spec = fits.getdata(muscles_dir +
                        f'hlsp_muscles_multi_multi_{host_star_name}_broadband_'
                        f'v22_adapt-const-res-sed.fits', 1)

    if truncate_wavelength_grid:
        mask = spec['WAVELENGTH'] * u.AA < thresh.to(u.AA,
                                                     equivalencies=u.spectral())
    else:
        mask = np.ones(spec.shape, dtype='bool')

    spectrum = {'wavelength': spec['WAVELENGTH'][mask],
                'flux_lambda': spec['FLUX'][mask] * conv *
                semi_major_axis ** (-2),
                'wavelength_unit': u.AA,
                'flux_unit': u.erg / u.s / u.cm / u.cm / u.AA}
    return spectrum


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
