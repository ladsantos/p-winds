#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of C in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp, odeint, cumtrapz
from scipy.interpolate import interp1d
from scipy.special import exp1
from p_winds import tools, microphysics
import warnings


__all__ = []


# Some hard coding based on the astrophysical literature
_SOLAR_CARBON_ABUNDANCE_ = 8.43  # Asplund et al. 2009
_SOLAR_CARBON_FRACTION_ = 10 ** (_SOLAR_CARBON_ABUNDANCE_ - 12.00)
_SOLAR_SILICON_ABUNDANCE_ = 7.51  # Asplund et al. 2009
_SOLAR_SILICON_FRACTION_ = 10 ** (_SOLAR_SILICON_ABUNDANCE_ - 12.00)


# Photoionization of C I (neutral) and C II (singly-ionized)
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of carbon at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        carbon (11.26 eV, or 1101 Angstrom).

    Returns
    -------
    phi_ci (``float``):
        Ionization rate of C I at null optical depth in unit of 1 / s.

    phi_cii (``float``):
        Ionization rate of C II at null optical depth in unit of 1 / s.

    a_ci (``float``):
        Flux-averaged photoionization cross-section of C I in unit of cm ** 2.

    a_cii (``float``):
        Flux-averaged photoionization cross-section of C II in unit of cm ** 2.

    a_h_ci (``float``):
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by C I in unit of cm ** 2.

    a_h_cii (``float``):
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by C II in unit of cm ** 2.

    a_he (``float``):
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
    energy_threshold_ci = parameters_dict['C I'][0]  # Ionization threshold in
    # eV
    energy_threshold_cii = parameters_dict['C II'][0]  # Ionization threshold in
    # eV
    wl_break_ci = 12398.42 / energy_threshold_ci  # C I ionization threshold in
    # angstrom
    wl_break_cii = 12398.42 / energy_threshold_cii  # C II ionization threshold
    # in angstrom
    wl_break_he = 504  # He ionization threshold in angstrom
    i0 = tools.nearest_index(wavelength, wl_break_he)
    i1 = tools.nearest_index(wavelength, wl_break_ci)
    i2 = tools.nearest_index(wavelength, wl_break_cii)
    wavelength_cut_0 = wavelength[:i0]
    flux_lambda_cut_0 = flux_lambda[:i0]
    wavelength_cut_1 = wavelength[:i1]
    flux_lambda_cut_1 = flux_lambda[:i1]
    energy_cut_1 = energy_erg[:i1]
    wavelength_cut_2 = wavelength[:i2]
    flux_lambda_cut_2 = flux_lambda[:i2]
    energy_cut_2 = energy_erg[:i2]

    # Calculate the photoionization cross-section
    a_lambda_ci = microphysics.general_cross_section(wavelength_cut_1,
                                                     species='C I')
    a_lambda_cii = microphysics.general_cross_section(wavelength_cut_2,
                                                      species='C II')

    # The flux-averaged photoionization cross-section of C I and C II
    a_ci = abs(simps(flux_lambda_cut_1 * a_lambda_ci, wavelength_cut_1) /
               simps(flux_lambda_cut_1, wavelength_cut_1))
    a_cii = abs(simps(flux_lambda_cut_2 * a_lambda_cii, wavelength_cut_2) /
                simps(flux_lambda_cut_2, wavelength_cut_2))

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that C I and C II see.
    a_lambda_h_ci = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_1)
    a_h_ci = abs(simps(flux_lambda_cut_1 * a_lambda_h_ci, wavelength_cut_1) /
                 simps(flux_lambda_cut_1, wavelength_cut_1))
    a_lambda_h_cii = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_2)
    a_h_cii = abs(simps(flux_lambda_cut_2 * a_lambda_h_cii, wavelength_cut_2) /
                  simps(flux_lambda_cut_2, wavelength_cut_2))

    # Same for the He atoms, but only up to the He ionization threshold
    a_lambda_he = microphysics.helium_total_cross_section(wavelength_cut_0)
    a_he = abs(simps(flux_lambda_cut_0 * a_lambda_he, wavelength_cut_0) /
               simps(flux_lambda_cut_0, wavelength_cut_0))

    # Calculate the photoionization rates
    phi_ci = abs(simps(flux_lambda_cut_1 * a_lambda_ci / energy_cut_1,
                 wavelength_cut_1))
    phi_cii = abs(simps(flux_lambda_cut_2 * a_lambda_cii / energy_cut_2,
                        wavelength_cut_2))

    return phi_ci, phi_cii, a_ci, a_cii, a_h_ci, a_h_cii, a_he


# Ionization rate of C by electron impact
def electron_impact_ionization(electron_temperature):
    """
    Calculates the electron impact ionization rate that consumes neutral C and
    produces singly-ionized C. Based on the formula of Voronov 1997
    (https://ui.adsabs.harvard.edu/abs/1997ADNDT..65....1V/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    ionization_rate_ci (``float``):
        Ionization rate of neutral C into singly-ionized C in unit of
        cm ** 3 / s.

    ionization_rate_cii (``float``):
        Ionization rate of singly-ionized C into doubly-ionized C in unit of
        cm ** 3 / s.
    """
    boltzmann_constant = 8.617333262145179e-05  # eV / K
    electron_energy = boltzmann_constant * electron_temperature
    energy_ratio_ci = 11.3 / electron_energy
    energy_ratio_cii = 24.4 / electron_energy
    ionization_rate_ci = 6.85E-8 * (0.193 + energy_ratio_ci) ** (-1) * \
        energy_ratio_ci ** 0.25 * np.exp(-energy_ratio_ci)
    ionization_rate_cii = 1.86E-8 * (0.286 + energy_ratio_cii) ** (-1) * \
        energy_ratio_cii ** 0.24 * np.exp(-energy_ratio_cii)
    return ionization_rate_ci, ionization_rate_cii


# Recombination of singly-ionized C into neutral C
def recombination(electron_temperature):
    """
    Calculates the rate of recombination of singly-ionized C with an electron to
    produce a neutral C atom. Based on the formulation of Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract). Also
    calculates the recombination of doubly-ionized C with an electron to produce
    a singly-ionized C ion. Based on the formulation of Aldrovandi & Péquignot
    1973 (https://ui.adsabs.harvard.edu/abs/1973A%26A....25..137A/abstract).

    Parameters
    ----------
    electron_temperature (``float``):
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    alpha_rec_ci (``float``):
        Recombination rate of C in units of cm ** 3 / s.
    """
    alpha_rec_ci = 4.67E-12 * (300 / electron_temperature) ** 0.60
    alpha_rec_cii = 2.3E-12 * (1000 / electron_temperature) ** 0.645
    return alpha_rec_ci, alpha_rec_cii


# Charge transfer between C and H, He and Si
def charge_transfer(temperature):
    """
    Calculates the charge exchange rates of C with H, He and Si nuclei. Based on
    the formulation of Stancil et al. 1998
    (https://ui.adsabs.harvard.edu/abs/1998ApJ...502.1006S/abstract),
    Woodall et al. 2007
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract) and
    Glover & Jappsen 2007
    (https://ui.adsabs.harvard.edu/abs/2007ApJ...666....1G/abstract).

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_hp (``float``):
        Charge transfer rate between neutral C and H+ in units of cm ** 3 / s.

    ct_rate_h (``float``):
        Charge transfer rate between C+ and neutral H in units of cm ** 3 / s.

    ct_rate_he (``float``):
        Charge transfer rate between neutral C and He in units of cm ** 3 / s.

    ct_rate_si (``float``):
        Charge transfer rate between C+ and neutral Si in units of cm ** 3 / s.
    """
    ct_rate_hp = 1.31E-15 * (300 / temperature) ** (-0.213)
    ct_rate_h = 6.30E-17 * (300 / temperature) ** (-1.96) * \
        np.exp(-1.7E5 / temperature)
    ct_rate_he = 2.5E-15 * (300 / temperature) ** (-1.597)
    ct_rate_si = 2.1E-9

    return ct_rate_hp, ct_rate_h, ct_rate_he, ct_rate_si


# Excitation of C atoms and ions by electron impact using the formulation of
# Suno & Kato 2006
def electron_impact_excitation(electron_temperature, excitation_energy,
                               statistical_weight, coefficients,
                               forbidden_transition=False):
    """
    Calculate th C ion excitation rates by electron impact following the
    Type 1 formulation of Suno & Kato 2006
    (https://ui.adsabs.harvard.edu/abs/2006ADNDT..92..407S/abstract).

    Parameters
    ----------
    electron_temperature
    excitation_energy
    statistical_weight
    coefficients
    forbidden_transition

    Returns
    -------

    """
    # Some auxiliary definitions
    ka, kb, kc, kd, ke = coefficients
    if forbidden_transition is True:
        ke = 0
    else:
        pass
    electron_energy = 1.380649E-16 * electron_temperature  # erg
    electron_energy_ev = 8.6173333E-5 * electron_temperature  # eV
    y = excitation_energy / electron_energy

    # We use the Type 1 excitation formula (Eq. 10 in Suno & Kato 2006)
    term1 = (ka / y + kc) + kd / 2 * (1 - y)
    term2 = np.exp(y) * exp1(y) * (kb - kc * y + kd / 2 * y ** 2 + ke / y)
    gamma = y * (term1 + term2)
    excitation_rate = 8.010E-8 * np.exp(-y) * gamma / statistical_weight / \
        electron_energy_ev ** 0.5  # cm ** 3 / s

    return excitation_rate


def ion_fraction(radius_profile, velocity, density, hydrogen_ion_fraction,
                 helium_ion_fraction, planet_radius, temperature, h_fraction,
                 speed_sonic_point, radius_sonic_point, density_sonic_point,
                 spectrum_at_planet, c_fraction=_SOLAR_CARBON_FRACTION_,
                 initial_f_c_ion=0.0, relax_solution=False, convergence=0.01,
                 max_n_relax=10, method='odeint', **options_solve_ivp):
    """

    Parameters
    ----------
    radius_profile
    velocity
    density
    hydrogen_ion_fraction
    helium_ion_fraction
    planet_radius
    temperature
    h_fraction
    speed_sonic_point
    radius_sonic_point
    density_sonic_point
    spectrum_at_planet
    c_fraction
    initial_f_c_ion
    relax_solution
    convergence
    max_n_relax
    method
    options_solve_ivp

    Returns
    -------

    """
    vs = speed_sonic_point  # km / s
    rs = radius_sonic_point  # jupiterRad
    rhos = density_sonic_point  # g / cm ** 3

    # Recombination rates of C in unit of rs ** 2 * vs
    alpha_rec_unit = ((rs * 7.1492E+09) ** 2 * vs * 1E5)  # cm ** 3 / s
    alpha_rec_ci, alpha_rec_cii = recombination(temperature)
    alpha_rec_ci = alpha_rec_ci / alpha_rec_unit
    alpha_rec_cii = alpha_rec_cii / alpha_rec_unit

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * (rs * 7.1492E+09) ** 3)  # Converted to g
    m_h = 1.67262192E-24 / m_h_unit

    # Photoionization rates at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs, and the flux-averaged
    # cross-sections in units of rs ** 2
    phi_unit = vs * 1E5 / rs / 7.1492E+09  # 1 / s
    phi_ci, phi_cii, a_ci, a_cii, a_h_ci, a_h_cii, a_he = radiative_processes(
        spectrum_at_planet)
    phi_ci = phi_ci / phi_unit
    a_ci = a_ci / (rs * 7.1492E+09) ** 2
    a_h_ci = a_h_ci / (rs * 7.1492E+09) ** 2
    a_he = a_he / (rs * 7.1492E+09) ** 2

    # Electron-impact ionization rate for C I in the same unit as the
    # recombination rates
    ionization_rate_ci, ionization_rate_cii = \
        electron_impact_ionization(temperature)
    ionization_rate_ci = ionization_rate_ci / alpha_rec_unit
    ionization_rate_cii = ionization_rate_cii / alpha_rec_unit

    # Charge transfer rates in the same unit as the recombination rates
    ct_rate_hp, ct_rate_h, ct_rate_he, ct_rate_si = charge_transfer(temperature)
    ct_rate_h = ct_rate_h / alpha_rec_unit
    ct_rate_hp = ct_rate_hp / alpha_rec_unit
    ct_rate_he = ct_rate_he / alpha_rec_unit
    ct_rate_si = ct_rate_si / alpha_rec_unit
    # ct_rate_h is the conversion of neutral C to C+
    # ct_rate_hp is the conversion of C+ to neutral C
    # ct_rate_he is the conversion of neutral C to C+
    # ct_rate_si is the conversion of C+ to neutral C

    # We solve the steady-state ionization balance in a similar way that we do
    # for He

    # The radius in unit of radius at the sonic point
    r = radius_profile * planet_radius / rs
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([r[-1], ])))
    mock_f_h_ion_r = interp1d(r, hydrogen_ion_fraction,
                              fill_value="extrapolate")
    mock_f_he_ion_r = interp1d(r, helium_ion_fraction,
                              fill_value="extrapolate")
    mock_v_r = interp1d(r, velocity, fill_value="extrapolate")
    mock_rho_r = interp1d(r, density, fill_value="extrapolate")

    # With all this setup done, now we need to assume something about the
    # distribution of neutral C in the atmosphere. Let's assume it based on the
    # initial guess input.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))  # Total column
                                                                # density
    column_density_h_0 = np.flip(  # Column density of atomic H only
        np.cumsum(np.flip(dr * density * (1 - hydrogen_ion_fraction))))
    he_fraction = 1 - h_fraction
    column_density_he_0 = np.flip(  # Column density of atomic He only
        np.cumsum(np.flip(dr * density * he_fraction *
                          (1 - helium_ion_fraction))))
    k1 = h_fraction / (h_fraction + 4 * he_fraction + 6 * c_fraction) / m_h
    k2 = he_fraction / (h_fraction + 4 * he_fraction + 6 * c_fraction) / m_h
    k3 = c_fraction / (h_fraction + 4 * he_fraction + 6 * c_fraction) / m_h
    tau_c_h = k1 * a_h_ci * column_density_h_0
    tau_c_he = k2 * a_he * column_density_he_0
    tau_c_initial = \
        initial_f_c_ion * k3 * a_ci * column_density + tau_c_h + tau_c_he
    # We do a dirty hack to make tau_initial a callable function so it's easily
    # parsed inside the differential equation solver
    _tau_c_fun = interp1d(r, tau_c_initial, fill_value="extrapolate")

    # The differential equation
    def _fun(_r, y):
        f_c = y

        _v = mock_v_r(np.array([_r, ]))[0]
        _rho = mock_rho_r(np.array([_r, ]))[0]
        f_h_ion = mock_f_h_ion_r(np.array([_r, ]))[0]  # Fraction of H+
        f_he_ion = mock_f_he_ion_r(np.array([_r, ]))[0]

        # Assume the number density of electrons is equal to the number density
        # of H ions + He ions
        n_e = k1 * _rho * f_h_ion + k2 * _rho * f_he_ion  # Number density of electrons
        n_h_plus = k1 * _rho * f_h_ion    # Number density of ionized H
        n_h0 = k1 * _rho * (1 - f_h_ion)  # Number density of atomic H
        n_he_plus = k2 * _rho * (1 - f_he_ion)

        # Terms of df_dr
        tau = _tau_c_fun(np.array([_r, ]))[0]
        term1 = (1 - f_c) * phi_ci * np.exp(-tau)  # Photoionization
        term2 = (1 - f_c) * n_e * ionization_rate_ci  # Electron-impact ionization
        term3 = (1 - f_c) * n_h_plus * ct_rate_hp  # Charge exchange with H+
        term4 = (1 - f_c) * n_he_plus * ct_rate_he  # Charge exchange with He+
        term5 = f_c * n_e * alpha_rec  # Recombination
        term6 = f_c * n_h0 * ct_rate_h  # Charge exchange with neutral H
        df_dr = (term1 + term2 + term3 + term4 - term5 - term6) / _v

        return df_dr

    if method == 'odeint':
        # Since 'odeint' yields only warnings when precision is lost or when
        # there is a problem, we transform these warnings into an exception
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                sol = odeint(_fun, y0=initial_f_c_ion, t=r, tfirst=True)
            except Warning:
                raise RuntimeError('The solver ``odeint`` failed to obtain a '
                                   'solution.')
        f_c_r = np.copy(sol).T[0]
    else:
        # We solve it using `scipy.solve_ivp`
        sol = solve_ivp(_fun, (r[0], r[-1],), initial_f_c_ion, t_eval=r,
                        method=method, **options_solve_ivp)
        f_c_r = sol['y'][0]
        # When `solve_ivp` has problems, it may return an array with different
        # size than `r`. So we raise an exception if this happens
        if len(f_c_r) != len(r):
            raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                               ' solution.')

    return f_c_r
