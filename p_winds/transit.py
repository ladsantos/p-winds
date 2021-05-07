#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and atmospheres.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from PIL import Image, ImageDraw

__all__ = ["draw_transit", "radiative_transfer"]


# Draw a grid
def draw_transit(planet_to_star_ratio, impact_parameter=0.0, phase=0.0,
                 grid_size=1001, density_profile=None, profile_radius=None,
                 planet_physical_radius=None):
    """
    Calculate a normalized transit map. Additionally, calculate a column density
    map around the planet if the user inputs a 1-D volumetric density profile.

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

    density_profile (``numpy.ndarray``, optional):
        1-D profile of volumetric number densities in function of radius. Unit
        has to be 1 / length ** 3, where length is the unit of
        ``planet_physical_radius``. If ``None``, the returned column densities
        will be zero. Default is ``None``.

    profile_radius (``numpy.ndarray``, optional):
        1-D profile of radii in which the densities are sampled. Unit
        has to be the same as ``planet_physical_radius``. Required if you want
        to calculate the map of column densities. Default is ``None``.

    planet_physical_radius (``float``, optional):
        Physical radius of the planet in whatever unit you want to work with.
        Required to calculate the map of column densities. Default is ``None``.

    Returns
    -------
    normalized_intensity_map (``numpy.ndarray``):
        2-D map of intensity normalized in such a way that the sum of the array
        will be 1.0 if the planet is not transiting.

    density_map (``numpy.ndarray``):
        2-D map of column densities in unit of 1 / length ** 2, where length is
        the unit of ``planet_physical_radius``.
    """
    shape = (grid_size, grid_size)

    # General function to draw a disk
    def _draw_disk(center, radius, value=1.0):
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        disk = np.reshape(np.array(list(image.getdata())), shape) * value
        return disk

    # Draw the host star
    star_radius = grid_size // 2
    planet_radius = star_radius * planet_to_star_ratio
    star = _draw_disk(center=(star_radius, star_radius), radius=star_radius)
    norm = np.sum(star)  # Normalization factor is the total intensity
    # Adding the star to the grid
    grid = star / norm

    # Before drawing the planet, we will need to figure out the exact coordinate
    # of the center of the planet and then draw the cloud
    x_p = grid_size // 2 + int(phase * grid_size)
    y_p = grid_size // 2 + int(impact_parameter * grid_size // 2)

    # Add the upper atmosphere if a density profile was input
    if density_profile is not None:
        # We need to know the matrix r_p containing distances from
        # planet center when we draw the extended atmosphere
        one_d_coords = np.linspace(0, grid_size - 1, grid_size, dtype=int)
        x_s, y_s = np.meshgrid(one_d_coords, one_d_coords)
        planet_centric_coords = np.array([x_s - x_p, y_s - y_p]).T
        # We also need to know the physical size of the pixel in the grid
        px_size = planet_physical_radius / planet_radius
        r_p = (np.sum(planet_centric_coords ** 2, axis=-1) ** 0.5).T * px_size
        # Calculate the column densities profile
        column_density = 2 * np.sum(np.array([density_profile,
                                              density_profile]), axis=0)
        # In order to calculate the column density in a given pixel, we need to
        # interpolate from the array above based on the radius map
        f = interp1d(profile_radius, column_density, bounds_error=False,
                     fill_value=0.0)
        density_map = f(r_p)
    else:
        density_map = np.zeros_like(grid)

    # Finally
    planet = _draw_disk(center=(x_p, y_p), radius=planet_radius)
    # Adding the planet to the grid, normalized by the stellar intensity
    grid -= planet / norm
    # The grid must not have negative values (this may happen if the planet
    # disk falls out of the stellar disk)
    normalized_intensity_map = grid.clip(min=0.0)

    return normalized_intensity_map, density_map


# Calculate the radiative transfer
def radiative_transfer(intensity_0, column_density, wavelength_grid,
                       central_wavelength, oscillator_strength,
                       einstein_coefficient, gas_temperature, particle_mass,
                       bulk_los_velocity=0.0, turbulence_speed=0.0):
    """
    Calculate the absorbed intensity profile in a wavelength grid.

    Parameters
    ----------
    intensity_0 (``float`` or ``numpy.ndarray``):
        Original flux intensity originating from a background illuminating
        source. If ``numpy.ndarray``, must have the same shape as
        ``column_density``.

    column_density (``float`` or ``numpy.ndarray``):
        Column density in 1 / m ** 2.

    wavelength_grid (``float`` or ``numpy.ndarray``):
        Wavelengths to calculate the profile in unit of m.

    central_wavelength (``float``):
        Central wavelength of the transition in unit of m.

    oscillator_strength (``float``):
        Oscillator strength of the transition.

    einstein_coefficient (``float``):
        Einstein coefficient of the transition in 1 / s.

    gas_temperature (``float``):
        Gas temperature in K.

    particle_mass (``float``):
        Mass of the particle corresponding to the transition in unit of kg.

    bulk_los_velocity (``float``, optional):
        Bulk velocity of the gas cell in the line of sight in unit of m / s.
        Default is 0.0.

    turbulence_speed (``float``, optional):
        Turbulence speed in m / s. It is added to the Doppler broadening of the
        absorption line. Default is 0.0.

    Returns
    -------
    intensity (``float`` or ``numpy.ndarray``):
        Absorbed intensity profile in function of ``wavelength_grid``.
    """
    w0 = central_wavelength  # Reference wavelength in m
    wl_grid = wavelength_grid
    c_speed = 2.99792458e+08  # Speed of light in m / s
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    nu0 = c_speed / w0  # Reference frequency in Hz
    nu_grid = c_speed / wl_grid
    temp = gas_temperature
    mass = particle_mass
    v_wind = bulk_los_velocity
    v_turb = turbulence_speed

    # Calculate Doppler width
    alpha_nu = nu0 / c_speed * (2 * k_b * temp / mass + v_turb ** 2) ** 0.5
    # Frequency shift due to bulk movement
    delta_nu = -v_wind / c_speed * nu0

    # Cross-section profile based on a Voigt line profile
    def _cross_section(nu, alpha, a_ij, f):
        # Calculate the Lorentzian width; alpha is the Doppler width
        gamma = a_ij / 4 / np.pi
        # Calculate Voigt profile
        phi = voigt_profile(nu, alpha, gamma)
        # Calculate cross-section = pi * e ** 2 / m_e / c * f
        _sigma = 2.6538E+2 * f * phi  # Hard-coded, but it is what it is...
        return _sigma

    # Check if one or more lines as input
    if isinstance(nu0, float):
        # Calculate the cross-section sigma in m ** 2 * Hz
        sigma_nu = _cross_section(nu_grid - nu0 - delta_nu, alpha_nu,
                                  einstein_coefficient, oscillator_strength)
        # The next line is necessary to allow the proper array multiplication
        # later and avoid expensive for-loops
        sigma = np.reshape(sigma_nu, (1, 1, len(sigma_nu)))
    elif isinstance(nu0, np.ndarray):
        # Same here, but for more than one spectral line
        n_lines = len(nu0)
        sigma_nu = np.array([_cross_section(nu_grid - nu0[i] - delta_nu[i],
                             alpha_nu[i], einstein_coefficient[i],
                             oscillator_strength[i]) for i in range(n_lines)])
        sigma_nu = np.sum(sigma_nu, axis=0)
        sigma = np.reshape(sigma_nu, (1, 1, len(sigma_nu)))
    else:
        raise ValueError('``central_wavelength`` must be either ``float`` or'
                         'a 1-dimensional ``numpy.ndarray``.')

    # The extinction is given by intensity_0 * exp(-tau)
    intensity = np.sum(intensity_0 * np.exp(-sigma.T * column_density),
                       axis=(1, 2))
    return intensity
