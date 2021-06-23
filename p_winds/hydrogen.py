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
    phi (``float``):
        Ionization rate of hydrogen at null optical depth in unit of 1 / s.

    a_0 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in unit of
        cm ** 2.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = (c.h * c.c).to(u.erg * u.angstrom).value / wavelength

    # Wavelength corresponding to the energy to ionize H
    wl_break = 911.65  # angstrom

    # Index of the lambda_0 in the wavelength array
    i_break = tools.nearest_index(wavelength, wl_break)

    # Auxiliary definitions
    wavelength_cut = wavelength[:i_break + 1]
    flux_lambda_cut = flux_lambda[:i_break + 1]
    energy_cut = energy[:i_break + 1]

    # Photoionization cross-section in function of wavelength
    a_lambda = microphysics.hydrogen_cross_section(wavelength=wavelength_cut)

    # Flux-averaged photoionization cross-section
    # Note: For some reason the Simpson's rule implementation of ``scipy`` may
    # yield negative results when the flux varies by a few orders of magnitude
    # at the edges of integration. So we take the absolute values of a_0 and phi
    a_0 = abs(simps(flux_lambda_cut * a_lambda, wavelength_cut) /
              simps(flux_lambda_cut, wavelength_cut))

    # Finally calculate the photoionization rate
    phi = abs(simps(flux_lambda_cut * a_lambda / energy_cut, wavelength_cut))
    return phi, a_0


# Hydrogen photoionization if you have only a monochromatic channel flux
def radiative_processes_mono(flux_euv):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the monochromatic EUV flux arriving at the planet.

    Parameters
    ----------
    flux_euv (``float``):
        Monochromatic extreme-ultraviolet (0 - 912 Angstrom) flux arriving at
        the planet in unit of erg / s / cm ** 2.

    Returns
    -------
    phi (``float``):
        Ionization rate of hydrogen at null optical depth in unit of 1 / s.

    a_0 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in unit of
        cm ** 2.
    """
    energy = np.flip(np.logspace(3, np.log10(13.61), 1000))  # eV

    # Photoionization cross-section in function of frequency
    a_nu = microphysics.hydrogen_cross_section(energy=energy)

    # Average cross-section
    a_0 = np.mean(a_nu)

    # Monochromatic ionization rate
    flux_euv *= 6.24150907E+11  # Convert erg to eV
    phi = flux_euv * np.mean(a_nu / energy)
    return phi, a_0


# Case-B hydrogen recombination
def recombination(temperature):
    """
    Calculates the case-B hydrogen recombination rate for a gas at a certain
    temperature.

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    alpha_rec (``float``):
        Recombination rate of hydrogen in units of cm ** 3 / s.
    """
    alpha_rec = 2.59E-13 * (temperature / 1E4) ** (-0.7)
    return alpha_rec


# Fraction of ionized hydrogen vs. radius profile
def ion_fraction(radius_profile, planet_radius, temperature, h_he_fraction,
                 mass_loss_rate, planet_mass, average_ion_fraction=0.0,
                 spectrum_at_planet=None, flux_euv=None, initial_f_ion=0.0,
                 relax_solution=False, convergence=0.01, max_n_relax=10,
                 **options_solve_ivp):
    """
    Calculate the fraction of ionized hydrogen in the upper atmosphere in
    function of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        Radius in unit of planetary radii.

    planet_radius (``float``):
        Planetary radius in unit of Jupiter radius.

    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    h_he_fraction (``float``):
        H/He fraction of the atmosphere.

    mass_loss_rate (``float``):
        Mass loss rate of the planet in units of g / s.

    planet_mass (``float``):
        Planetary mass in unit of Jupiter mass.

    average_ion_fraction (``float``):
        Average ion fraction in the upper atmosphere.

    spectrum_at_planet (``dict``, optional):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``. If ``None``, then ``flux_euv`` must be
        provided instead. Default is ``None``.

    flux_euv (``float``, optional):
        Extreme-ultraviolet (0-911.65 Angstrom) flux arriving at the planet in
        units of erg / s / cm ** 2. If ``None``, then ``spectrum_at_planet``
        must be provided instead. Default is ``None``.

    initial_f_ion (``float``, optional):
        The initial ionization fraction at the layer near the surface of the
        planet. Default is 0.0, i.e., 100% neutral.

    relax_solution (``bool``, optional):
        The first solution is calculating by initially assuming the entire
        atmosphere is in neutral state. If ``True``, the solution will be
        re-calculated in a loop until it converges to a delta_f of 1%, or for a
        maximum of 10 loops (default parameters). Default is ``False``.

    convergence (``float``, optional):
        Value of delta_f at which to stop the relaxation of the solution for
        ``f_r``. Default is 0.01.

    max_n_relax (``int``, optional):
        Maximum number of loops to perform the relaxation of the solution for
        ``f_r``. Default is 10.

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
    """
    # Hydrogen recombination rate
    alpha_rec = recombination(temperature)

    # Hydrogen mass in g
    m_h = 1.67262192E-24

    # Photoionization rate at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs
    if spectrum_at_planet is not None:
        phi_abs, a_0 = radiative_processes(spectrum_at_planet)
    elif flux_euv is not None:
        phi_abs, a_0 = radiative_processes_mono(flux_euv)
    else:
        raise ValueError('Either `spectrum_at_planet` or `flux_euv` must be '
                         'provided.')

    # Multiplicative factor of Eq. 11 of Oklopcic & Hirata 2018, unit of
    # cm ** 2 / g
    k1_abs = (h_he_fraction * a_0 / (1 + (1 - h_he_fraction) * 4) / m_h)

    # Multiplicative factor of the second term in the right-hand side of Eq.
    # 13 of Oklopcic & Hirata 2018, unit of cm ** 3 / s / g
    k2_abs = h_he_fraction / (1 + (1 - h_he_fraction) * 4) * alpha_rec / m_h

    # In order to avoid numerical overflows, we need to normalize a few key
    # variables. Since the normalization may need to be repeated to relax the
    # solution, we have a function to do it.
    def _normalize(_phi, _k1, _k2, _r, _mean_f_ion):
        # First calculate the sound speed, radius at the sonic point and the
        # density at the sonic point. They will be useful to change the units of
        # the calculation aiming to avoid numerical overflows
        vs = parker.sound_speed(temperature, h_he_fraction, _mean_f_ion)
        rs = parker.radius_sonic_point(planet_mass, vs)
        rhos = parker.density_sonic_point(mass_loss_rate, rs, vs)
        # And now normalize everything
        phi_unit = vs * 1E5 / rs / 7.1492E+09  # 1 / s
        phi_norm = _phi / phi_unit
        k1_unit = 1 / (rhos * rs * 7.1492E+09)  # cm ** 2 / g
        k1_norm = _k1 / k1_unit
        k2_unit = vs * 1E5 / rs / 7.1492E+09 / rhos  # cm ** 3 / g / s
        k2_norm = _k2 / k2_unit
        r_norm = (_r * planet_radius / rs)

        # The differential r will be useful at some point
        dr_norm = np.diff(r_norm)
        dr_norm = np.concatenate((dr_norm, np.array([dr_norm[-1], ])))

        # The structure of the atmosphere
        v_norm, rho_norm = parker.structure(r_norm)

        return phi_norm, k1_norm, k2_norm, r_norm, dr_norm, v_norm, rho_norm

    phi, k1, k2, r, dr, velocity, density = _normalize(
        phi_abs, k1_abs, k2_abs, radius_profile, average_ion_fraction)

    # To start the calculations we need the optical depth, but technically we
    # don't know it yet, because it depends on the ion fraction in the
    # atmosphere, which is what we want to obtain. However, the optical depth
    # depends more strongly on the densities of H than the ion fraction, so a
    # good approximation is to assume the whole atmosphere is neutral at first.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))
    tau_initial = k1 * column_density
    # We do a dirty hack to make tau_initial a callable function so it's easily
    # parsed inside the differential equation solver
    _tau_fun = interp1d(r, tau_initial, fill_value="extrapolate")

    # Now let's solve the differential eq. 13 of Oklopcic & Hirata 2018
    # The differential equation in function of r
    def _fun(_r, _f, _phi, _k1, _k2):
        _t = _tau_fun(np.array([_r, ]))[0]
        _v, _rho = parker.structure(_r)
        # In terms 1 and 2 we use the values of k2 and phi from above
        term1 = (1. - _f) / _v * _phi * np.exp(-_t)
        term2 = _k2 * _rho * _f ** 2 / _v
        df_dr = term1 - term2
        return df_dr

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                    t_eval=r, args=(phi, k1, k2), **options_solve_ivp)
    f_r = sol['y'][0]
    # When `solve_ivp` has problems, it may return an array with different
    # size than `r`. So we raise an exception if this happens
    if len(f_r) != len(r):
        raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                           ' solution.')

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_r_outer_layer = np.copy(f_r)[-1]
            average_ion_fraction = np.mean(np.copy(f_r))

            # We re-normalize key parameters because the newly-calculated f_ion
            # changes the value of the mean molecular weight of the atmosphere
            phi, k1, k2, r, dr, velocity, density = _normalize(
                phi_abs, k1_abs, k2_abs, radius_profile, average_ion_fraction)

            # Re-calculate the column densities
            column_density = np.flip(np.cumsum(np.flip(dr * density *
                                                       (1 - f_r))))
            tau = k1 * column_density
            _tau_fun = interp1d(r, tau, fill_value="extrapolate")

            # And solve it again
            sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                            t_eval=r, args=(phi, k1, k2), **options_solve_ivp)
            f_r = sol['y'][0]

            # Raise an error if the length of `f_r` is different from the length
            # of `r`
            if len(f_r) != len(r):
                raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                                   ' solution.')

            # Calculate the relative change of f_ion in the outer shell of the
            # atmosphere (where we expect the most important change)
            relative_delta_f = abs(f_r[-1] - previous_f_r_outer_layer) \
                / previous_f_r_outer_layer

            # Break the loop if convergence is achieved
            if relative_delta_f < convergence:
                break
            else:
                pass
    else:
        pass
    return f_r
