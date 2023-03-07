#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of O in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp, odeint
from p_winds import tools, microphysics
import warnings


__all__ = ["radiative_processes", "electron_impact_ionization", "recombination",
           "charge_transfer", "ion_fraction"]


# Some hard coding based on the astrophysical literature
_SOLAR_OXYGEN_ABUNDANCE_ = 8.69  # Asplund et al. 2009
_SOLAR_OXYGEN_FRACTION_ = 10 ** (_SOLAR_OXYGEN_ABUNDANCE_ - 12.00)


# Photoionization of O I (neutral)
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of oxygen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        oxygen (13.62 eV, or 910 Angstrom).

    Returns
    -------
    phi_oi : ``float``
        Ionization rate of O I at null optical depth in unit of 1 / s.

    a_oi : ``float``
        Flux-averaged photoionization cross-section of O I in unit of cm ** 2.

    a_h_oi : ``float``
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by O I in unit of cm ** 2.

    a_he : ``float``
        Flux-averaged photoionization cross-section of He I in unit of cm ** 2.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value
    energy_erg = (energy * u.eV).to(u.erg).value

    # Auxiliary definitions
    parameters_dict = microphysics.sigma_properties_v1996()

    energy_threshold_oi = parameters_dict['O I'][0]  # Ionization threshold in
    # eV
    wl_break_oi = 12398.42 / energy_threshold_oi  # O I ionization threshold in
    # angstrom
    wl_break_he = 504  # He ionization threshold in angstrom
    i0 = tools.nearest_index(wavelength, wl_break_he)
    i1 = tools.nearest_index(wavelength, wl_break_oi)
    wavelength_cut_0 = wavelength[:i0]
    flux_lambda_cut_0 = flux_lambda[:i0]
    wavelength_cut_1 = wavelength[:i1]
    flux_lambda_cut_1 = flux_lambda[:i1]
    energy_cut_1 = energy_erg[:i1]

    # Calculate the photoionization cross-section
    a_lambda_oi = microphysics.general_cross_section(wavelength_cut_1,
                                                     species='O I')

    # The flux-averaged photoionization cross-section of O I
    a_oi = abs(simps(flux_lambda_cut_1 * a_lambda_oi, wavelength_cut_1) /
               simps(flux_lambda_cut_1, wavelength_cut_1))

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that O I see.
    a_lambda_h_oi = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_1)
    a_h_oi = abs(simps(flux_lambda_cut_1 * a_lambda_h_oi, wavelength_cut_1) /
                 simps(flux_lambda_cut_1, wavelength_cut_1))

    # Same for the He atoms, but only up to the He ionization threshold
    a_lambda_he = microphysics.helium_total_cross_section(wavelength_cut_0)
    a_he = abs(simps(flux_lambda_cut_0 * a_lambda_he, wavelength_cut_0) /
               simps(flux_lambda_cut_0, wavelength_cut_0))

    # Calculate the photoionization rates
    phi_oi = abs(simps(flux_lambda_cut_1 * a_lambda_oi / energy_cut_1,
                       wavelength_cut_1))

    return phi_oi, a_oi, a_h_oi, a_he


# Ionization rate of O by electron impact
def electron_impact_ionization(electron_temperature):
    """
    Calculates the electron impact ionization rate that consumes neutral O and
    produces singly-ionized O. Based on the formula of Voronov 1997
    (https://ui.adsabs.harvard.edu/abs/1997ADNDT..65....1V/abstract).

    Parameters
    ----------
    electron_temperature : ``float``
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    ionization_rate_oi : ``float``
        Ionization rate of neutral O into singly-ionized O in unit of
        cm ** 3 / s.
    """
    boltzmann_constant = 8.617333262145179e-05  # eV / K
    electron_energy = boltzmann_constant * electron_temperature
    energy_ratio_oi = 11.3 / electron_energy
    ionization_rate_oi = 3.59E-8 * (0.073 + energy_ratio_oi) ** (-1) * \
        energy_ratio_oi ** 0.34 * np.exp(-energy_ratio_oi)
    return ionization_rate_oi


# Recombination of singly-ionized O into neutral O
def recombination(electron_temperature):
    """
    Calculates the rate of recombination of singly-ionized O with an electron to
    produce a neutral O atom. Based on the formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract).

    Parameters
    ----------
    electron_temperature : ``float``
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    alpha_rec_oi  : ``float``
        Recombination rate of O II into O I in units of cm ** 3 / s.
    """
    alpha_rec_oi = 3.25E-12 * (300 / electron_temperature) ** 0.66
    return alpha_rec_oi


# Charge transfer between O and H
def charge_transfer(temperature):
    """
    Calculates the charge exchange rates of O with H nuclei. Based on the
    formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract).

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_oi_hp : ``float``
        Charge transfer rate between neutral O and H+ in units of cm ** 3 / s.

    ct_rate_oii_h : ``float``
        Charge transfer rate between O+ and neutral H in units of cm ** 3 / s.
    """
    # Recombination of O II into O I
    ct_rate_oii_h = 5.66E-10 * (300 / temperature) ** (-0.36) * \
        np.exp(8.6 / temperature)

    # Ionization of O I into O II
    ct_rate_oi_hp = 7.31E-10 * (300 / temperature) ** (-0.23) * \
        np.exp(-226 / temperature)

    return ct_rate_oi_hp, ct_rate_oii_h


# Calculation the number fractions of O II
def ion_fraction(radius_profile, velocity, density, hydrogen_ion_fraction,
                 helium_ion_fraction, planet_radius, temperature, h_fraction,
                 speed_sonic_point, radius_sonic_point, density_sonic_point,
                 spectrum_at_planet, o_fraction=_SOLAR_OXYGEN_FRACTION_,
                 initial_f_o_ion=0.0, relax_solution=False, convergence=0.01,
                 max_n_relax=10, method='Radau', return_rates=False,
                 **options_solve_ivp):
    """
    Calculate the fraction of ionized oxygen in the upper atmosphere in function
    of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile : ``numpy.ndarray``
        Radius in unit of planetary radii.

    velocity : ``numpy.ndarray``
         Velocities sampled at the values of ``radius_profile`` in units of
         sound speed. Similar to the output of ``parker.structure()``.

    density : ``numpy.ndarray``
        Densities sampled at the values of ``radius_profile`` in units of
        density at the sonic point. Similar to the output of
        ``parker.structure()``.

    hydrogen_ion_fraction : ``numpy.ndarray``
        Number fraction of H ion over total H in the upper atmosphere in
        function of radius. Similar to the output of
        ``hydrogen.ion_fraction()``.

    helium_ion_fraction : ``numpy.ndarray``
        Number fraction of He ion over total He in the upper atmosphere in
        function of radius. Similar to the output of
        ``helium.population_fraction()``, but should be ``1 - f_1_r - f_3_r``.

    planet_radius : ``float``
        Planetary radius in unit of Jupiter radius.

    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    h_fraction : ``float``
        Total (ion + neutral) H number fraction of the atmosphere.

    speed_sonic_point : ``float``
        Speed of sound in the outflow in units of km / s.

    radius_sonic_point : ``float``
        Radius of the sonic point in unit of Jupiter radius.

    density_sonic_point : ``float``
        Density at the sonic point in units of g / cm ** 3.

    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        oxygen (13.62 eV, or 910 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``.

    o_fraction : ``float``, optional
        Fraction of total oxygen in the upper atmosphere. Default value assumes
        solar abundance.

    initial_f_o_ion : ``float``, optional
        The initial oxygen ion fraction at the layer near the surface of the
        planet. Default is 0.0, i.e., 100% neutral.

    relax_solution : ``bool``, optional
        The first solution is calculating by initially assuming the entire
        atmosphere is in neutral state. If ``True``, the solution will be
        re-calculated in a loop until it converges to a delta_f of 1%, or for a
        maximum of 10 loops (default parameters). Default is ``False``.

    convergence : ``float``, optional
        Value of delta_f at which to stop the relaxation of the solution for
        ``f_r``. Default is 0.01.

    max_n_relax : ``int``, optional
        Maximum number of loops to perform the relaxation of the solution for
        the ion fractions. Default is 10.

    method : ``str``, optional
        If method is ``'odeint'``, then ``scipy.integrate.odeint()`` is used
        instead of ``scipy.integrate.solve_ivp()`` to calculate the steady-state
        distribution of helium. Any other method will fall back to an option of
        ``solve_ivp()`` methods. For example, if ``method`` is set to
        ``'Radau'``, then use ``solve_ivp(method='Radau')``. Default is
        ``'Radau'``.

    return_rates : ``bool``, optional
        If ``True``, then this function also returns a ``dict`` object
        containing the various reaction rates in function of radius and in units
        of 1 / s. Default is ``False``.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``atol`` (absolute tolerance; default is
        1E-6) or ``rtol`` (relative tolerance; default is 1E-3). If you are
        having numerical issues, you may want to decrease the tolerance by a
        factor of 10 or 100, or 1000 in extreme cases.

    Returns
    -------
    f_oii_r : ``numpy.ndarray``
            Fraction of singly-ionized oxygen in function of radius.

    reaction_rates : ``dict``
        Dictionary containing the reaction rates in function of radius and in
        units of 1 / s. Only returned when ``return_rates`` is set to ``True``.
        Here is a short description of the dict keys:

        * `photoionization`: Photoionization of O I into O II
        * `recombination`: Recombination of O II into O I
        * `e_impact_ionization`: Electron impact ionization of O I into O II
        * `charge_exchange_HII`: Charge exchange between O I and H II
        * `charge_exchange_HI`: Charge exchange between O II and H I
    """
    vs = speed_sonic_point  # km / s
    rs = radius_sonic_point  # jupiterRad
    rhos = density_sonic_point  # g / cm ** 3

    # Recombination rates of C in unit of rs ** 2 * vs
    alpha_rec_unit = ((rs * 7.1492E+09) ** 2 * vs * 1E5)  # cm ** 3 / s
    alpha_rec_oi = recombination(temperature)
    alpha_rec_oi = alpha_rec_oi / alpha_rec_unit

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * (rs * 7.1492E+09) ** 3)  # Converted to g
    m_h = 1.67262192E-24 / m_h_unit

    # Photoionization rates at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs, and the flux-averaged
    # cross-sections in units of rs ** 2
    phi_unit = vs * 1E5 / rs / 7.1492E+09  # 1 / s
    phi_oi, a_oi, a_h_oi, a_he = radiative_processes(spectrum_at_planet)
    phi_oi = phi_oi / phi_unit
    a_oi = a_oi / (rs * 7.1492E+09) ** 2
    a_h_oi = a_h_oi / (rs * 7.1492E+09) ** 2
    a_he = a_he / (rs * 7.1492E+09) ** 2

    # Electron-impact ionization rate for C I in the same unit as the
    # recombination rates
    ionization_rate_oi = electron_impact_ionization(temperature)
    ionization_rate_oi = ionization_rate_oi / alpha_rec_unit

    # Charge transfer rates in the same unit as the recombination rates
    ct_rate_oi_hp, ct_rate_oii_h = charge_transfer(temperature)
    ct_rate_oii_h = ct_rate_oii_h / alpha_rec_unit
    ct_rate_oi_hp = ct_rate_oi_hp / alpha_rec_unit

    # We solve the steady-state ionization balance in a similar way that we do
    # for He

    # The radius in unit of radius at the sonic point
    r = radius_profile * planet_radius / rs
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([dr[-1], ])))

    # With all this setup done, now we need to assume something about the
    # distribution of neutral O in the atmosphere. Let's assume it based on the
    # initial guess input.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))  # Total column
    # density
    column_density_h_0 = np.flip(  # Column density of atomic H only
        np.cumsum(np.flip(dr * density * (1 - hydrogen_ion_fraction))))
    he_fraction = 1 - h_fraction
    column_density_he_0 = np.flip(  # Column density of atomic He only
        np.cumsum(np.flip(dr * density * he_fraction *
                          (1 - helium_ion_fraction))))
    k1 = h_fraction / (h_fraction + 4 * he_fraction + 16 * o_fraction) / m_h
    k2 = he_fraction / (h_fraction + 4 * he_fraction + 16 * o_fraction) / m_h
    k3 = o_fraction / (h_fraction + 4 * he_fraction + 16 * o_fraction) / m_h
    tau_oi_h = k1 * a_h_oi * column_density_h_0
    tau_o_he = k2 * a_he * column_density_he_0
    tau_oi = (1 - initial_f_o_ion) * k3 * a_oi * column_density + tau_oi_h + \
        tau_o_he

    # The differential equation
    def _fun(_r, y, rates=False):
        f_oii = y

        _v = np.interp(_r, r, velocity)
        _rho = np.interp(_r, r, density)
        f_h_ion = np.interp(_r, r, hydrogen_ion_fraction)  # Fraction of H+
        f_he_ion = np.interp(_r, r, helium_ion_fraction)  # Fraction of He+

        # Assume the number density of electrons is equal to the number density
        # of H ions + He ions
        # Since we may run into loss of numerical precision here (big numbers),
        # we manipulate the equations to avoid this problem. It looks a bit
        # messy, but it is necessary
        log_term_1 = np.log(k1) + np.log(_rho)  # H ions
        log_term_2 = np.log(k2) + np.log(_rho)  # He ions
        ionization_rate_oi_n_e = \
            np.exp(log_term_1 + np.log(ionization_rate_oi)) * f_h_ion + \
            np.exp(log_term_2 + np.log(ionization_rate_oi)) * f_he_ion
        ct_rate_oi_hp_n_h_plus = \
            np.exp(log_term_1 + np.log(ct_rate_oi_hp)) * f_h_ion
        alpha_rec_oi_n_e = \
            np.exp(log_term_1 + np.log(alpha_rec_oi)) * f_h_ion + \
            np.exp(log_term_2 + np.log(alpha_rec_oi)) * f_he_ion
        ct_rate_oii_h_n_h0 = \
            np.exp(log_term_1 + np.log(ct_rate_oii_h)) * (1 - f_h_ion)

        # n_e = k1 * _rho * f_h_ion + k2 * _rho * f_he_ion  # Number density of
        # # electrons
        # n_h_plus = k1 * _rho * f_h_ion    # Number density of ionized H
        # n_h0 = k1 * _rho * (1 - f_h_ion)  # Number density of atomic H

        # Terms of dfoii_dr
        t_oi = np.interp(_r, r, tau_oi)
        term1 = (1 - f_oii) * phi_oi * np.exp(-t_oi)  # Photoionization
        term2 = (1 - f_oii) * ionization_rate_oi_n_e  # Electron-impact
        # ionization
        term3 = (1 - f_oii) * ct_rate_oi_hp_n_h_plus  # Charge exchange with
        # H+
        term4 = f_oii * alpha_rec_oi_n_e  # Recombination of O II into O I
        term5 = f_oii * ct_rate_oii_h_n_h0  # Charge exchange of O II with
        # neutral H
        dfoii_dr = (term1 + term2 + term3 - term4 - term5) / _v

        if rates is False:
            return dfoii_dr
        else:
            return np.array([term1, term4, term2, term3, term5]) * phi_unit

    if method == 'odeint':
        # Since 'odeint' yields only warnings when precision is lost or when
        # there is a problem, we transform these warnings into an exception
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                sol = odeint(_fun, y0=initial_f_o_ion, t=r, tfirst=True)
            except Warning:
                raise RuntimeError('The solver ``odeint`` failed to obtain a '
                                   'solution.')
        f_oii_r = np.copy(sol).T[0]
    else:
        # We solve it using `scipy.solve_ivp`
        sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_o_ion, ]),
                        t_eval=r, method=method, **options_solve_ivp)
        f_oii_r = sol['y'][0]
        # When `solve_ivp` has problems, it may return an array with different
        # size than `r`. So we raise an exception if this happens
        if len(f_oii_r) != len(r):
            raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                               ' solution.')

    # High densities can be numerically unstable and produce unphysical values
    # of `f_r`, so we replace negative values with zero and values above 1.0
    # with 1.0
    f_oii_r[f_oii_r < 0] = 1E-15
    f_oii_r[f_oii_r > 1.0] = 1.0

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_oii_r = np.copy(f_oii_r)

            # Re-calculate the column densities
            tau_oi = \
                k3 * a_oi * np.flip(np.cumsum(
                    np.flip(dr * density * (1 - f_oii_r)))) + tau_oi_h + \
                tau_o_he

            # Solve it again
            if method == 'odeint':
                sol = odeint(_fun, y0=initial_f_o_ion, t=r, tfirst=True)
                f_oii_r = np.copy(sol).T[0]
            else:
                sol = solve_ivp(_fun, (r[0], r[-1],),
                                np.array([initial_f_o_ion, ]), t_eval=r,
                                method=method, **options_solve_ivp)
                f_oii_r = sol['y'][0]

            # Replace negative values with zero and values above 1.0 with
            # 1.0
            f_oii_r[f_oii_r < 0] = 1E-15
            f_oii_r[f_oii_r > 1.0] = 1.0

            # Calculate the relative change of f_ion in the outer shell of
            # the atmosphere (where we expect the most important change)
            relative_delta_f_oii = abs(np.sum(f_oii_r - previous_f_oii_r)) \
                / np.sum(previous_f_oii_r)

            # Break the loop if convergence is achieved
            if relative_delta_f_oii < convergence:
                break
            else:
                pass
    else:
        pass

    if return_rates is False:
        return f_oii_r
    else:
        ionization_rate, recombination_rate, e_impact_ion, ch_exchange_hii,\
            ch_exchange_hi = _fun(r, f_oii_r, rates=True)
        reaction_rates = {'photoionization': ionization_rate,
                          'recombination': recombination_rate,
                          'e_impact_ionization': e_impact_ion,
                          'charge_exchange_HII': ch_exchange_hii,
                          'charge_exchange_HI': ch_exchange_hi}
        return f_oii_r, reaction_rates
