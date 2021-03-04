#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of H in the
upper atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp
from scipy.interpolate import interp1d
from p_winds import parker, tools, microphysics


__all__ = ["radiative_processes", "radiative_processes_mono", "recombination",
           "ion_fraction"]


# Hydrogen photoionization
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom).

    Returns
    -------
    phi (``astropy.Quantity``):
        Ionization rate of hydrogen at null optical depth.

    a_0 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = (c.h * c.c).to(u.erg * u.angstrom).value / wavelength

    # Wavelength corresponding to the energy to ionize H
    wl_break = (c.h * c.c / (13.6 * u.eV)).to(u.angstrom).value

    # Index of the lambda_0 in the wavelength array
    i_break = tools.nearest_index(wavelength, wl_break)

    # Auxiliary definitions
    wavelength_cut = wavelength[:i_break + 1]
    flux_lambda_cut = flux_lambda[:i_break + 1]
    energy_cut = energy[:i_break + 1]

    # Photoionization cross-section in function of wavelength
    a_lambda = microphysics.hydrogen_cross_section(wavelength=wavelength_cut)

    # Flux-averaged photoionization cross-section
    a_0 = simps(flux_lambda_cut * a_lambda, wavelength_cut) / \
        simps(flux_lambda_cut, wavelength_cut) * u.cm ** 2

    # Finally calculate the photoionization rate
    phi = simps(flux_lambda_cut * a_lambda / energy_cut, wavelength_cut) / u.s
    return phi, a_0


# Hydrogen photoionization if you have only a monochromatic channel flux
def radiative_processes_mono(flux_euv):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the monochromatic EUV flux arriving at the planet.

    Parameters
    ----------
    flux_euv (``astropy.Quantity``):
        Monochromatic extreme-ultraviolet (0 - 912 Angstrom) flux arriving at
        the planet.

    Returns
    -------
    phi (``astropy.Quantity``):
        Ionization rate of hydrogen at null optical depth.

    a_0 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen.
    """
    energy = np.logspace(np.log10(13.61), 3, 1000)

    # Photoionization cross-section in function of frequency
    a_nu = microphysics.hydrogen_cross_section(energy=energy)

    # Average cross-section
    a_0 = np.mean(a_nu) * u.cm ** 2

    # Monochromatic ionization rate
    phi = flux_euv.to(u.eV / u.s / u.cm ** 2) * a_0 / np.mean(energy) / u.eV
    return phi, a_0


# Case-B hydrogen recombination
def recombination(temperature):
    """
    Calculates the case-B hydrogen recombination rate for a gas at a certain
    temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    alpha_rec (``astropy.Quantity``):
        Recombination rate of hydrogen.
    """
    alpha_rec = 2.59E-13 * (temperature.to(u.K).value / 1E4) ** (-0.7) * \
        u.cm ** 3 / u.s
    return alpha_rec


# Fraction of ionized hydrogen vs. radius profile
def ion_fraction(radius_profile, planet_radius, temperature, h_he_fraction,
                 mass_loss_rate, planet_mass, average_ion_fraction=0.0,
                 spectrum_at_planet=None, flux_euv=None, velocity=None,
                 density=None, initial_f_ion=0.0, repeat=False, **options_solve_ivp):
    """
    Calculate the fraction of ionized hydrogen in the upper atmosphere in
    function of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        Radius in unit of planetary radii. Not a ``astropy.Quantity``.

    planet_radius (``astropy.Quantity``):
        Planetary radius.

    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    h_he_fraction (``float``):
        H/He fraction of the atmosphere.

    mass_loss_rate (``astropy.Quantity``):
        Mass loss rate of the planet.

    planet_mass (``astropy.Quantity``):
        Planetary mass.

    average_ion_fraction (``float``):
        Average ion fraction in the upper atmosphere.

    spectrum_at_planet (``dict``, optional):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``. If ``None``, then ``flux_euv`` must be
        provided instead. Default is ``None``.

    flux_euv (``astropy.Quantity``, optional):
        Extreme-ultraviolet (0-911.65 Angstrom) flux arriving at the planet.
        If ``None``, then ``spectrum_at_planet`` must be provided instead.
        Default is ``None``.

    velocity (``numpy.ndarray``, optional):
        Velocities of the escaping atmosphere in units of sound speed and in
        function of radius. Providing these values upfront makes the code more
        efficient, but are not strictly necessary. If ``None``, the velocities
        will be calculated  using ``parker.structure()``. Default is ``None``.

    density (``numpy.ndarray``, optional):
        Densities of the upper atmosphere in units of density at the sonic point
        and in function of radius. Providing these values upfront makes the code
        more efficient, but are not strictly necessary. If ``None``, the
        velocities will be calculated  using ``parker.structure()``. Default is
        ``None``.

    initial_f_ion (``float``, optional):
        The initial ionization fraction at the layer near the surface of the
        planet. Default is 0.0, i.e., fully neutral.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``method`` (integration method; default
        is ``'RK45'``), ``atol`` (absolute tolerance; default is 1E-6) or
        ``rtol`` (relative tolerance; default is 1E-3). If you are having
        numerical issues, you may want to decrease the tolerance by a factor of
        10 or 100, or 1000 in extreme cases.

    Returns
    -------
    f_r (``numpy.ndarray``):
        Values of the fraction of ionized hydrogen in function of the radius.

    tau_r (``numpy.ndarray``):
        Values of the optical depth in function of the radius.
    """
    # First calculate the sound speed, radius at the sonic point and the
    # density at the sonic point. They will be useful to change the units of
    # the calculation aiming to avoid numerical overflows
    vs = parker.sound_speed(temperature, h_he_fraction, average_ion_fraction
                            ).to(u.km / u.s)
    rs = parker.radius_sonic_point(planet_mass, vs).to(u.jupiterRad)
    rhos = parker.density_sonic_point(mass_loss_rate, rs, vs).to(
        u.g / u.cm ** 3)

    # Hydrogen recombination rate
    alpha_rec = recombination(temperature)

    # Hydrogen mass
    m_h = c.m_p

    # Photoionization rate at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs
    phi_unit = (vs / rs).to(1 / u.s)
    if spectrum_at_planet is not None:
        phi, a_0 = radiative_processes(spectrum_at_planet)
    elif flux_euv is not None:
        phi, a_0 = radiative_processes_mono(flux_euv)
    else:
        raise ValueError('Either `spectrum_at_planet` or `flux_euv` must be '
                         'provided.')
    phi = (phi / phi_unit).decompose().value

    # Multiplicative factor of Eq. 11 of Oklopcic & Hirata 2018
    k1_unit = 1 / (rhos * rs).to(u.kg / u.cm ** 2)
    k1 = (h_he_fraction * a_0 / (1 + (1 - h_he_fraction) * 4) / m_h) / k1_unit
    k1 = k1.value

    # Multiplicative factor of the second term in the right-hand side of Eq.
    # 13 of Oklopcic & Hirata 2018
    k2_unit = (vs / rs / rhos).to(u.cm ** 3 / u.kg / u.s)
    k2 = h_he_fraction / (1 + (1 - h_he_fraction) * 4) * alpha_rec / m_h
    k2 = (k2 / k2_unit).value

    # The radius in unit of radius at the sonic point
    r = (radius_profile * planet_radius / rs).decompose().value
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([dr[-1],])))

    # The structure of the atmosphere
    if velocity is None or density is None:
        velocity, density = parker.structure(r)
    else:
        pass

    # To start the calculations we need the optical depth, but technically we
    # don't know it yet, because it depends on the ion fraction in the
    # atmosphere, which is what we want to obtain. However, the optical depth
    # depends more strongly on the densities of H than the ion fraction, so a
    # good approximation is to assume the whole atmosphere is neutral at first.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))
    tau_initial = k1 * column_density
    # We do a dirty hack to make tau_initial a callable function so it's easily
    # parsed inside the differential equation solver
    _tau_fun = interp1d(r, tau_initial)

    # Now let's solve the differential eq. 13 of Oklopcic & Hirata 2018
    # The differential equation in function of r
    def _fun(r, f):
        t = _tau_fun(r)
        v, rho = parker.structure(r)
        # In terms 1 and 2 we use the values of k2 and phi from above
        term1 = (1. - f) / v * phi * np.exp(-t)
        term2 = k2 * rho * f ** 2 / v
        df_dr = term1 - term2
        return df_dr

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                    t_eval=r, **options_solve_ivp)
    f_r = sol['y'][0]

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if repeat is True:
        column_density = np.flip(np.cumsum(np.flip(dr * density * (1 - f_r))))
        tau = k1 * column_density
        _tau_fun = interp1d(r, tau)
        sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                        t_eval=r, **options_solve_ivp)
        f_r = sol['y'][0]
    else:
        pass

    return f_r
