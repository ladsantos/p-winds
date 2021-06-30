#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and atmospheres.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from flatstar import draw, utils
from warnings import warn


# Show a warning about changes on how this module works now
warn('The transit module has significant syntax changes since version 0.6. '
     'Consult the documentation.', SyntaxWarning)


__all__ = ["draw_transit", "radiative_transfer", "profile_los",
           "optical_depth"]


# Draw a grid
def draw_transit(planet_to_star_ratio, planet_physical_radius,
                 impact_parameter=0.0, phase=0.0, grid_size=100,
                 supersampling=None, resample_method=None,
                 limb_darkening_law=None, ld_coefficient=None):
    """
    Calculate a normalized transit map. Additionally, calculate two-dimensional
    arrays of density and line-of-sight radial velocities around the planet.

    Parameters
    ----------
    planet_to_star_ratio (``float``):
        Ratio between the radii of the planet and the star.

    planet_physical_radius (``float``):
        Physical radius of the planet in whatever unit you want to work with.

    impact_parameter (``float``, optional):
        Transit impact parameter of the planet (not the atmosphere). Default is
        0.0.

    phase (``float``, optional):
        Phase of the transit. -0.5, 0.0, and +0.5 correspond to the center of
        planet being located at, respectively, the left limb of the star, the
        center, and the right limb. Default is 0.0.

    grid_size (``int``, optional):
        Size of the transit grid. Default is 100.

    supersampling (``float`` or ``None``, optional):
        In order to avoid pixels with hard edges, it is useful to first compute
        the transit grid at a high resolution and then downscale it to a
        manageable grid size. Supersampling is the factor by which to increase
        the grid size at first and then downscale to the requested grid size. If
        ``None``, no supersampling is applied. Default is ``None``.

    resample_method (``str`` or ``None``, optional):
        Method by which to resample the image if supersampling is used. If
        ``None``, then fallback to a "box" method. Available methods are
        ``"nearest"``, ``"box"``, ``"bilinear"``, ``"hamming"`` and
        ``"lanczos"`` (the last two are not recommended for research-grade
        results. Default is ``None``.

    limb_darkening_law (``None`` or ``str``, optional):
        String with the name of the limb-darkening law. The options are the same
        currently implemented in the code ``flatstar``: ``'linear'``,
        ``'quadratic'``, ``'square-root'``, ``'log'``, ``'exp'``, ``'s3'``,
        ``'c4'``, or ``None`` (no limb-darkening). Default is ``None``.

    ld_coefficient (``float`` or ``array-like``):
        In case of a linear limb-darkening law, the value of the coefficient
        should be a float. In all other options it should be array-like. Default
        is ``None``.

    Returns
    -------
    normalized_intensity_map (``numpy.ndarray``):
        2-D map of intensity normalized in such a way that the sum of the array
        will be 1.0 if the planet is not transiting.

    transit_depth (``float``):
        Absorption caused by the opaque disk of the planet in the specified
        transit configuration.

    px_size (``float``):
        Physical size of one pixel in the same unit as
        ``planet_physical_radius``.

    r_from_planet (``numpy.ndarray``):
        2-D map of radial distances of each pixel from the center of the planet
        in the same unit as ``planet_physical_radius``.
    """
    if supersampling is not None:
        effective_grid_size = int(round(grid_size * supersampling))
        rescaling = 1 / supersampling
    else:
        effective_grid_size = grid_size
        rescaling = None

    star_grid = draw.star(effective_grid_size,
                          limb_darkening_law=limb_darkening_law,
                          ld_coefficient=ld_coefficient)
    transit_grid = draw.planet_transit(star_grid, planet_to_star_ratio,
                                       impact_parameter, phase,
                                       rescaling_factor=rescaling,
                                       resample_method=resample_method)

    # We will calculate the densities and wind radial velocities of the
    # upper atmosphere. We need to know the matrix r_p containing distances from
    # planet center when we draw the extended atmosphere
    pl_ref = transit_grid.planet_px_coordinates
    planet_centric_r = utils.cylindrical_r(transit_grid.intensity, pl_ref)
    # We also need to know the physical size of the pixel in the grid
    planet_radius = transit_grid.planet_radius_px
    px_size = planet_physical_radius / planet_radius
    r_from_planet = planet_centric_r * px_size

    # Finally
    normalized_intensity_map = transit_grid.intensity
    transit_depth = transit_grid.transit_depth

    return normalized_intensity_map, transit_depth, px_size, r_from_planet


# Calculate the radiative transfer
def radiative_transfer(intensity_0, r_from_planet, px_size, radius_profile,
                       density_profile, velocity_profile, central_wavelength,
                       oscillator_strength, einstein_coefficient,
                       wavelength_grid, gas_temperature, particle_mass,
                       turbulence_speed=0.0, bulk_los_velocity=0.0,
                       method='fast'):
    """
    Calculate the absorbed intensity profile in a wavelength grid.

    Parameters
    ----------
    intensity_0 (``float`` or ``numpy.ndarray``):
        Original flux intensity originating from a background illuminating
        source. If ``numpy.ndarray``, must have the same shape as
        ``r_from_planet``.

    r_from_planet (``numpy.ndarray``):
        2-D map of radial distances of each pixel from the center of the planet
        in the same unit as ``px_size``.

    px_size (``float``):
        Physical size of one pixel in whatever unit you want to work with.

    radius_profile (``numpy.ndarray``):
        1-D profile of radii in which the densities are sampled. Unit has to be
        the same as ``px_size``.

    density_profile (``numpy.ndarray``):
        1-D profile of volumetric number densities in function of radius. Unit
        has to be 1 / length ** 3, where length is the same unit as ``px_size``.

    velocity_profile (``numpy.ndarray``):
        1-D profile of velocities in function of radius. Unit has to be
        m / s.

    central_wavelength (``float`` or ``array_like``):
        Central wavelength of the transition in unit of m. If more than one line
        is to be calculated, then input this parameter as an array-like object.

    oscillator_strength (``float``):
        Oscillator strength of the transition. The format or shape of this input
        parameter needs to be consistent with that of ``central_wavelength``.

    einstein_coefficient (``float``):
        Einstein coefficient of the transition in 1 / s. The format or shape of
        this input parameter needs to be consistent with that of
        ``central_wavelength``.

    wavelength_grid (``numpy.ndarray``):
        Wavelengths to calculate the profile in unit of m.

    gas_temperature (``float``):
        Gas temperature in K.

    particle_mass (``float``):
        Mass of the particle corresponding to the transition in unit of kg.

    turbulence_speed (``float``, optional):
        Turbulence speed in m / s. It is added to the Doppler broadening of the
        absorption line. Default is 0.0.

    bulk_los_velocity (``float``, optional):
        Bulk velocity of the gas cell in the line of sight in unit of m / s.
        Default is 0.0.

    method (``str``, optional):
        Method of calculation for the radiative transfer. There are two options:
        1) ``'formal'``: the formal definition of radiative transfer;
        2) ``'fast'``: assumes that the Parker wind broadening as a Gaussian
        shape. The fast calculation is one order of magnitude faster than the
        formal method. Default is ``'fast'``.

    Returns
    -------
    intensity (``numpy.ndarray``):
        Absorbed intensity profile in function of ``wavelength_grid``.
    """
    # First, calculate the optical depth in function of radial distance from the
    # planet and the wavelength
    optical_depth_profile = optical_depth(
        radius_profile, density_profile, velocity_profile, central_wavelength,
        oscillator_strength, einstein_coefficient, wavelength_grid,
        gas_temperature, particle_mass, px_size, turbulence_speed,
        bulk_los_velocity, method=method
    )

    # Now we interpolate the optical depths to each radius from the planet
    f = interp1d(radius_profile, optical_depth_profile, axis=0,
                 bounds_error=False, fill_value=0.0)
    optical_depth_array = f(r_from_planet)

    # A bit of array-manipulation magic here to allow us to properly broadcast
    intensity_0 = np.reshape(intensity_0, np.shape(intensity_0) + (1,))

    # Finally calculate the radiative transfer
    intensity = np.sum(intensity_0 * np.exp(-optical_depth_array), axis=(0, 1))

    return intensity


# Density and velocity profiles in the line of sight
def profile_los(radius_profile, density_profile, velocity_profile):
    """
    Calculate the profiles of radius and line-of-sight velocities in function of
    sky-projected radial distance from the planet (axis 0) and the line of sight
    direction (axis 1).

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        1-D profile of radii in which the densities are sampled, in whatever
        unit you want to work with.

    density_profile (``numpy.ndarray``):
        1-D profile of volumetric number densities in function of radius, in
        whatever unit you want to work with.

    velocity_profile (``numpy.ndarray``):
        1-D profile of velocities in function of radius, in whatever unit you
        want to work with.

    Returns
    -------
    los_density_r_z (``numpy.ndarray``):
        2-D map of densities in the x- and z-axis distances from the center of
        the planet.

    los_velocity_r_z (``numpy.ndarray``):
        2-D map of line-of-sight velocities in the x- and z-axis distances from
        the center of the planet.
    """
    # Create a two-dimensional array that measures the distance from the planet
    # The second dimension is the line-of-sight direction.
    coordinates = np.array(np.meshgrid(radius_profile, radius_profile))
    distances = np.sum(coordinates ** 2, axis=0) ** 0.5

    # The line-of-sight wind velocity is given by Eq. 6 in Seidel et al.
    # (2020)
    # (https://ui.adsabs.harvard.edu/abs/2020A%26A...633A..86S/abstract)
    x = np.array([radius_profile for i in range(len(radius_profile))]).T
    los_velocity_r_z = np.interp(distances, radius_profile, velocity_profile,
                                 left=0.0, right=0.0) * x / \
        (x ** 2 + radius_profile ** 2) ** 0.5

    # Calculate the line-of-sight density
    los_density_r_z = np.interp(distances, radius_profile, density_profile,
                                left=0.0, right=0.0)

    return los_density_r_z, los_velocity_r_z


# Optical depth in function of cylindrical radius from the planet and
# wavelength. Hold on to your hat because this code is very complex.
def optical_depth(radius_profile, density_profile, velocity_profile,
                  central_wavelength, oscillator_strength,
                  einstein_coefficient, wavelength_grid, gas_temperature,
                  particle_mass, px_size, turbulence_speed=0.0,
                  bulk_los_velocity=0.0, method='fast'):
    """
    Calculate the optical depth in function of cylindrical radius from the
    planet and the wavelength.

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        1-D profile of radii in which the densities are sampled. Unit has to be
        the same as ``px_size``.

    density_profile (``numpy.ndarray``):
        1-D profile of volumetric number densities in function of radius. Unit
        has to be 1 / length ** 3, where length is the same unit as ``px_size``.

    velocity_profile (``numpy.ndarray``):
        1-D profile of velocities in function of radius. Unit has to be
        m / s.

    central_wavelength (``float`` or ``array_like``):
        Central wavelength of the transition in unit of m. If more than one line
        is to be calculated, then input this parameter as an array-like object.

    oscillator_strength (``float``):
        Oscillator strength of the transition. The format or shape of this input
        parameter needs to be consistent with that of ``central_wavelength``.

    einstein_coefficient (``float``):
        Einstein coefficient of the transition in 1 / s. The format or shape of
        this input parameter needs to be consistent with that of
        ``central_wavelength``.

    wavelength_grid (``numpy.ndarray``):
        Wavelengths to calculate the profile in unit of m.

    gas_temperature (``float``):
        Gas temperature in K.

    particle_mass (``float``):
        Mass of the particle corresponding to the transition in unit of kg.

    px_size (``float``):
        Physical size of one pixel in whatever unit you want to work with.

    turbulence_speed (``float``, optional):
        Turbulence speed in m / s. It is added to the Doppler broadening of the
        absorption line. Default is 0.0.

    bulk_los_velocity (``float``, optional):
        Bulk velocity of the gas cell in the line of sight in unit of m / s.
        Default is 0.0.

    method (``str``, optional):
        Method of calculation for the radiative transfer. There are two options:
        1) ``'formal'``: the formal definition of radiative transfer;
        2) ``'fast'``: assumes that the Parker wind broadening as a Gaussian
        shape. The fast calculation is one order of magnitude faster than the
        formal method. Default is ``'fast'``.

    Returns
    -------
    optical_depth_array (``numpy.ndarray``):
        Optical depth in function of radial distance from the center of the
        planet (axis 0) and in function of wavelength (axis 1).
    """
    # Calculate density and velocity profiles in the line of sight
    density_los, velocity_los = profile_los(radius_profile, density_profile,
                                            velocity_profile)

    # Spectral line properties
    w0 = central_wavelength  # Reference wavelength in m
    f = oscillator_strength  # Unitless
    a_ij = einstein_coefficient  # In unit of 1 / s

    # Transform `w0`, `f` and `a_ij` into arrays if they are not already arrays
    if isinstance(w0, float):
        w0 = np.array([w0, ])
        f = np.array([f, ])
        a_ij = np.array([a_ij, ])
    elif isinstance(w0, list):
        w0 = np.array(w0)
        f = np.array(f)
        a_ij = np.array(a_ij)
    elif isinstance(w0, np.ndarray):
        pass
    else:
        raise ValueError('The spectral line properties must be float, list or'
                         ' numpy.ndarray, and their format must be consistent'
                         ' with each other.')

    # Some necessary book-keeping here
    n_lines = len(w0)
    wl_grid = wavelength_grid
    c_speed = 2.99792458e+08  # Speed of light in m / s
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    nu0 = c_speed / w0  # Reference frequency in Hz
    nu_grid_rest = c_speed / wl_grid
    v_turb = turbulence_speed
    v_bulk = bulk_los_velocity
    temp = gas_temperature
    mass = particle_mass

    # Change the shape of `nu_grid_rest` to allow some clever array manipulation
    nu_grid_rest = np.reshape(nu_grid_rest, (len(nu_grid_rest), 1))

    # For each line we calculate the grid of frequency shifts from the central
    # frequency
    delta_nu = (nu_grid_rest - nu0).T

    # Cross-section
    k = 2.654008854574474e-06  # Physical constant in units of m ** 2 * Hz
    sigma = k * f

    # Expand the velocities in the line of sight to include the front of the
    # atmosphere, which comes towards the observer. This part of the code is
    # very hacky, but necessary.
    velocity_los_back = np.copy(velocity_los)
    velocity_los_front = np.flip(-velocity_los, axis=0)
    velocity_los_both = np.concatenate(
        (velocity_los_front, velocity_los_back), axis=0)

    # Do the same for densities
    density_los_back = np.copy(density_los)
    density_los_front = np.flip(density_los, axis=0)
    density_los_both = np.concatenate((density_los_front, density_los_back),
                                      axis=0)
    spatial_shape = np.shape(density_los_both)
    density_multi = np.reshape(density_los_both, spatial_shape + (1,))

    # This is convoluted, but we create a nested function to calculate the
    # optical depth divided by the cross-section (i.e. normalized optical depth)
    def _normalized_optical_depth(delta_nu_grid, nu0_k, a_ij_k, _method=method):
        # Calculate the Lorentzian width of the Voigt profile
        gamma = a_ij_k / 4 / np.pi

        # Formal calculation of optical depth
        if _method == 'formal':
            # Calculate Doppler width of the Voigt profile
            alpha_nu = nu0_k / c_speed * (2 * np.log(2) * k_b * temp / mass +
                                          v_turb ** 2) ** 0.5

            # Calculate the frequency shifts due to wind and bulk motion
            delta_nu_wind = (velocity_los_both + v_bulk) / c_speed * nu0_k
            delta_nu_add = np.reshape(delta_nu_wind, spatial_shape + (1,))

        # Faster calculation of optical depth
        elif _method == 'fast':
            # We assume that the Parker wind broadening has a Gaussian shape.
            # To this end, we take the density-averaged line-of-sight velocity
            # and add it quadratically to the Gaussian broadening term of the
            # Voigt profile.
            parker_wind_broadening = \
                np.sum(velocity_los * density_los) / np.sum(density_los)
            # Calculate Doppler width of the Voigt profile
            alpha_nu = nu0_k / c_speed * (2 * np.log(2) * k_b * temp / mass +
                                          v_turb ** 2 +
                                          parker_wind_broadening ** 2) ** 0.5

            # Frequency shift due to bulk line-of-sight velocity (not to be
            # confused with the Parker wind velocity).
            delta_nu_add = v_bulk / c_speed * nu0_k

        else:
            raise ValueError('The chosen ``method`` is not implemented.')

        # Finally calculate the Voigt profiles
        profiles = voigt_profile(delta_nu_grid + delta_nu_add, alpha_nu,
                                 gamma)

        # Calculate the optical depths divided by the cross section
        opt_depth_over_cross_section_r_nu = \
            np.sum(profiles * density_multi * px_size, axis=0)

        return opt_depth_over_cross_section_r_nu

    # Calculate the optical depth for each line
    optical_depth_list = []
    for i in range(n_lines):
        norm_opt_depth = _normalized_optical_depth(
            delta_nu_grid=delta_nu[i],
            nu0_k=nu0[i],
            a_ij_k=a_ij[i]
        )
        optical_depth_i = norm_opt_depth * sigma[i]
        optical_depth_list.append(optical_depth_i)
    optical_depth_array = np.array(optical_depth_list)
    optical_depth_array = np.sum(optical_depth_array, axis=0)

    return optical_depth_array
