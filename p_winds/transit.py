#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and atmospheres.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as plc
# import scipy.optimize as sp
from scipy.interpolate import interp1d
from PIL import Image, ImageDraw
# from itertools import product

__all__ = ["draw_transit", "column_density"]


# Draw a grid
def draw_transit(planet_to_star_ratio, impact_parameter=0.0, phase=0.0,
                 grid_size=2001):
    """

    Parameters
    ----------
    planet_to_star_ratio (``float``):
        Ratio between the radii of the planet and the star.

    impact_parameter (``float``, optional):
        Transit impact parameter. Default is 0.0.

    phase (``float``, optional):
        Phase of the transit. -0.5, 0.0, and +0.5 correspond to the center of
        planet being located at, respectively, the left limb of the star, the
        center, and the right limb. Default is 0.0.

    grid_size (``int``, optional):
        Size of the transit grid. Default is 2001.

    Returns
    -------
    grid (``numpy.ndarray``):
        Transit grid.
    """
    shape = (grid_size, grid_size)

    # General function to draw a disk
    def _draw_disk(center, radius, value=1.0):
        """

        Parameters
        ----------
        center:
            Coordinates of the center of the disk. The origin is the center of the
            grid.

        radius:
            Radius of the disk in units of grid pixels.

        value:
            Value to be attributed to each pixel inside the disk.

        Returns
        -------

        """
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        disk = np.reshape(np.array(list(image.getdata())), shape) * value
        return disk

    # Draw the host star
    star_radius = grid_size // 2
    star = _draw_disk(center=(star_radius, star_radius), radius=star_radius)
    norm = np.sum(star)  # Normalization factor is the total flux
    # Adding the star to the grid
    grid = star / norm

    # Before drawing the planet, we will need to figure out the exact coordinate
    # of the center of the planet and then draw the cloud
    x_p = grid_size // 2 + int(phase * grid_size)
    y_p = grid_size // 2 + int(impact_parameter * grid_size // 2)

    # It will also be useful to know the matrix r_p containing distances from
    # planet center
    one_d_coords = np.linspace(0, grid_size - 1, grid_size, dtype=int)
    x_s, y_s = np.meshgrid(one_d_coords, one_d_coords)
    planet_centric_coords = np.array([x_s - x_p, y_s - y_p]).T
    r_p = (np.sum(planet_centric_coords ** 2, axis=-1) ** 0.5).T

    # Finally
    planet_radius = star_radius * planet_to_star_ratio
    planet = _draw_disk(center=(x_p, y_p), radius=planet_radius)
    # Adding the planet to the grid, normalized by the stellar flux
    grid -= planet / norm
    # The grid must not have negative values (this may happen if the planet
    # disk falls out of the stellar disk)
    grid = grid.clip(min=0.0)

    return grid


# Calculate a 2-D map of the column densities
def column_density(r, density_profile, sampling=100):
    """

    Parameters
    ----------
    r
    density_profile
    sampling

    Returns
    -------

    """
    # First we need to know how the density behaves with radius. So we create an
    # interpolating function that does that for us
    f = interp1d(r, density_profile, bounds_error=False, fill_value=0.0)

    # Second create a 3-D array containing the cloud and then collapse it into
    # 2-D
    n = sampling
    length = r[-1]
    r_array = np.linspace(-length, length, n)
    xx, yy, zz = np.meshgrid(r_array, r_array, r_array)
    r_map = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5  # 3-D map containing the
    # radial distances
    density_cube = f(r_map)
    density_map = np.sum(density_cube, axis=0)
    return r_map, density_map


# Transmission profile in a given gas cell
def transmission(cell_density, cell_temperature, cell_line_of_sight_speed,
                 absorption_coeff, wavelength_grid, reference_wavelength,
                 particle_mass, line_shape='Doppler'):
    """

    Parameters
    ----------
    cell_density (``float``):
        Cell gas density in 1 / m ** 3

    cell_temperature (``float``):
        Cell gas temperature in K.

    cell_line_of_sight_speed (``float``):
        Cell wind speed in the line of sight in m / s.

    absorption_coeff (``float``):


    wavelength_grid
    reference_wavelength
    particle_mass
    line_shape

    Returns
    -------

    """
    w0 = reference_wavelength * 1E-10  # Reference wavelength in m
    wl_grid = wavelength_grid * 1E-10
    c_speed = 2.99792458e+08  # Speed of light in m / s
    k_B = 1.380649e-23  # Boltzmann's constant in J / K
    nu0 = c_speed / w0  # Reference frequency in Hz
    nu_grid = c_speed / wl_grid
    temp = cell_temperature
    mass = particle_mass
    wind_speed = cell_line_of_sight_speed

    if line_shape == 'Doppler':
        # Calculate turbulence speed for the gas in m / s
        v_turb = (5 * k_B * temp / 3 / mass) ** 0.5
        # Calculate Doppler width in Hz
        alpha_nu = nu0 / c_speed * \
            (2 * k_B * temp / mass + v_turb ** 2) ** 0.5
        # Calculate the Doppler profile
        numerator = (nu_grid - nu0 + (nu_grid / c_speed) * wind_speed) ** 2
        f_nu = 1 / alpha_nu / np.pi ** 0.5 * np.exp(-numerator / alpha_nu ** 2)
    else:
        raise ValueError('This line shape is not implemented.')

    k_nu = absorption_coeff * f_nu * cell_density
    return k_nu
