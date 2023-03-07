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
from scipy.integrate import simps, solve_ivp, odeint
from p_winds import tools, microphysics
import warnings


__all__ = ["radiative_processes", "electron_impact_ionization", "recombination",
           "charge_transfer", "ion_fraction"]


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
    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        carbon (11.26 eV, or 1101 Angstrom).

    Returns
    -------
    phi_ci : ``float``
        Ionization rate of C I at null optical depth in unit of 1 / s.

    phi_cii : ``float``
        Ionization rate of C II at null optical depth in unit of 1 / s.

    a_ci : ``float``
        Flux-averaged photoionization cross-section of C I in unit of cm ** 2.

    a_cii : ``float``
        Flux-averaged photoionization cross-section of C II in unit of cm ** 2.

    a_h_ci : ``float``
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by C I in unit of cm ** 2.

    a_h_cii : ``float``
        Flux-averaged photoionization cross-section of H I in the range absorbed
        by C II in unit of cm ** 2.

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
    electron_temperature : ``float``
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    ionization_rate_ci : ``float``
        Ionization rate of neutral C into singly-ionized C in unit of
        cm ** 3 / s.

    ionization_rate_cii : ``float``
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
    electron_temperature : ``float``
        Temperature of the plasma where the electrons are embedded in unit of
        Kelvin.

    Returns
    -------
    alpha_rec_ci : ``float``
        Recombination rate of C II into C I in units of cm ** 3 / s.

    alpha_rec_cii : ``float``
        Recombination rate of C III into C II in units of cm ** 3 / s.
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
    (https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1197W/abstract),
    Glover & Jappsen 2007
    (https://ui.adsabs.harvard.edu/abs/2007ApJ...666....1G/abstract),
    Kingdon & Ferland 1996
    (https://ui.adsabs.harvard.edu/abs/1996ApJS..106..205K/abstract), and
    Brown 1972 (https://ui.adsabs.harvard.edu/abs/1972ApJ...174..511B/abstract).

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_ci_hp : ``float``
        Charge transfer rate between neutral C and H+ in units of cm ** 3 / s.

    ct_rate_cii_h : ``float``
        Charge transfer rate between C+ and neutral H in units of cm ** 3 / s.

    ct_rate_ci_hep : ``float``
        Charge transfer rate between neutral C and He+ in units of cm ** 3 / s.

    ct_rate_cii_sii : ``float``
        Charge transfer rate between C+ and neutral Si in units of cm ** 3 / s.

    ct_rate_ciii_h : ``float``)
        Charge transfer rate between C++ and neutral H in units of cm ** 3 / s.

    ct_rate_ciii_he : ``float``)
        Charge transfer rate between C++ and neutral He in units of cm ** 3 / s.
    """
    # Recombination of C II into C I
    ct_rate_cii_h = 6.30E-17 * (300 / temperature) ** (-1.96) * \
        np.exp(-1.7E5 / temperature)
    ct_rate_cii_sii = 2.1E-9

    # Ionization of C I into C II
    ct_rate_ci_hp = 1.31E-15 * (300 / temperature) ** (-0.213)
    ct_rate_ci_hep = 2.5E-15 * (300 / temperature) ** (-1.597)

    # Recombination of C III into C II
    ct_rate_ciii_h = 1.67E-4 * (temperature / 10000) ** 2.79 * \
        (1 + 304.72 * np.exp(-4.07 * temperature / 10000))
    ct_rate_ciii_he = 1.23E-9  # Very approximated from Brown 1972, but should
    # be good enough for temperatures near 10,000 K

    return ct_rate_ci_hp, ct_rate_cii_h, ct_rate_ci_hep, ct_rate_cii_sii, \
        ct_rate_ciii_h, ct_rate_ciii_he


# Calculation the number fractions of C II and C III
def ion_fraction(radius_profile, velocity, density, hydrogen_ion_fraction,
                 helium_ion_fraction, planet_radius, temperature, h_fraction,
                 speed_sonic_point, radius_sonic_point, density_sonic_point,
                 spectrum_at_planet, c_fraction=_SOLAR_CARBON_FRACTION_,
                 initial_f_c_ion=np.array([0.0, 0.0]), relax_solution=False,
                 convergence=0.01, max_n_relax=10, method='odeint',
                 return_rates=False, **options_solve_ivp):
    """
    Calculates the fractions of singly- and doubly-ionized carbon in the upper
    atmosphere in function of the radius in unit of planetary radius.

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
        carbon (11.26 eV, or 1101 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``.

    c_fraction : ``float``, optional
        Fraction of total carbon in the upper atmosphere. Default value assumes
        solar abundance.

    initial_f_c_ion : ``numpy.ndarray``, optional
        The initial ion fractions are the `y0` of the differential equation to
        be solved. This array has two items: the initial fraction of
        singly-ionized and doubly-ionized carbon in the inner layer of the
        atmosphere. The default value for this parameter is
        ``numpy.array([0.0, 0.0])``, i.e., fully neutral at the inner layer.

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
        distribution of helium. The first seems to be at least twice faster than
        the second in some situations. Any other method will fall back to an
        option of ``solve_ivp()`` methods. For example, if ``method`` is set to
        ``'Radau'``, then use ``solve_ivp(method='Radau')``. Default is
        ``'odeint'``.

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
    f_cii_r : ``numpy.ndarray``
        Fraction of singly-ionized carbon in function of radius.

    f_ciii_r : ``numpy.ndarray``
        Fraction of doubly-ionized carbon in function of radius.

    reaction_rates : ``dict``
        Dictionary containing the reaction rates in function of radius and in
        units of 1 / s. Only returned when ``return_rates`` is set to ``True``.
        Here is a short description of the dict keys:

        * `ionization_CI`: Photoionization of C I into C II
        * `ionization_CII`: Photoionization of C II into C III
        * `recombination_CII`: Recombination of C II into C I
        * `recombination_CIII`: Recombination of C III into C II
        * `e_impact_ion_CI`: Electron impact ionization of C I into C II
        * `e_impact_ion_CII`: Electron impact ionization of C II into C III
        * `charge_exchange_CI_HII`: Charge exchange between C I and H II
        * `charge_exchange_CI_HeII`: Charge exchange between C I and He II
        * `charge_exchange_CII_HI`: Charge exchange between C II and H I
        * `charge_exchange_CIII_HI`: Charge exchange between C III and H I
        * `charge_exchange_CIII_HeI`: Charge exchange between C III and He I
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
    phi_cii = phi_cii / phi_unit
    a_cii = a_cii / (rs * 7.1492E+09) ** 2
    a_h_cii = a_h_cii / (rs * 7.1492E+09) ** 2
    a_he = a_he / (rs * 7.1492E+09) ** 2

    # Electron-impact ionization rate for C I in the same unit as the
    # recombination rates
    ionization_rate_ci, ionization_rate_cii = \
        electron_impact_ionization(temperature)
    ionization_rate_ci = ionization_rate_ci / alpha_rec_unit
    ionization_rate_cii = ionization_rate_cii / alpha_rec_unit

    # Charge transfer rates in the same unit as the recombination rates
    ct_rate_ci_hp, ct_rate_cii_h, ct_rate_ci_hep, ct_rate_cii_sii, \
        ct_rate_ciii_h, ct_rate_ciii_he = charge_transfer(temperature)
    ct_rate_cii_h = ct_rate_cii_h / alpha_rec_unit
    ct_rate_ci_hp = ct_rate_ci_hp / alpha_rec_unit
    ct_rate_ci_hep = ct_rate_ci_hep / alpha_rec_unit
    ct_rate_ciii_h = ct_rate_ciii_h / alpha_rec_unit
    ct_rate_ciii_he = ct_rate_ciii_he / alpha_rec_unit

    # We solve the steady-state ionization balance in a similar way that we do
    # for He

    # The radius in unit of radius at the sonic point
    r = radius_profile * planet_radius / rs
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([dr[-1], ])))

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
    k1 = h_fraction / (h_fraction + 4 * he_fraction + 12 * c_fraction) / m_h
    k2 = he_fraction / (h_fraction + 4 * he_fraction + 12 * c_fraction) / m_h
    k3 = c_fraction / (h_fraction + 4 * he_fraction + 12 * c_fraction) / m_h
    tau_ci_h = k1 * a_h_ci * column_density_h_0
    tau_cii_h = k1 * a_h_cii * column_density_h_0
    tau_c_he = k2 * a_he * column_density_he_0
    tau_ci = \
        (1 - initial_f_c_ion[0] - initial_f_c_ion[1]) * k3 * a_ci * \
        column_density + tau_ci_h + tau_c_he
    tau_cii = \
        initial_f_c_ion[0] * k3 * a_cii * column_density + tau_cii_h + tau_c_he

    # The differential equation
    def _fun(_r, y, rates=False):
        f_cii = y[0]
        f_ciii = y[1]

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
        ionization_rate_ci_n_e = \
            np.exp(log_term_1 + np.log(ionization_rate_ci)) * f_h_ion + \
            np.exp(log_term_2 + np.log(ionization_rate_ci)) * f_he_ion
        ct_rate_ci_hp_n_h_plus = \
            np.exp(log_term_1 + np.log(ct_rate_ci_hp)) * f_h_ion
        ct_rate_ci_hep_n_he_plus = \
            np.exp(log_term_2 + np.log(ct_rate_ci_hep)) * f_he_ion
        alpha_rec_ci_n_e = \
            np.exp(log_term_1 + np.log(alpha_rec_ci)) * f_h_ion + \
            np.exp(log_term_2 + np.log(alpha_rec_ci)) * f_he_ion
        ct_rate_cii_h_n_h0 = \
            np.exp(log_term_1 + np.log(ct_rate_cii_h)) * (1 - f_h_ion)
        alpha_rec_cii_n_e = \
            np.exp(log_term_1 + np.log(alpha_rec_cii)) * f_h_ion + \
            np.exp(log_term_2 + np.log(alpha_rec_cii)) * f_he_ion
        ct_rate_ciii_h_n_h0 = \
            np.exp(log_term_1 + np.log(ct_rate_ciii_h)) * (1 - f_h_ion)
        ct_rate_ciii_he_n_he0 = \
            np.exp(log_term_2 + np.log(ct_rate_ciii_he)) * (1 - f_he_ion)
        ionization_rate_cii_n_e = \
            np.exp(log_term_1 + np.log(ionization_rate_cii)) * f_h_ion + \
            np.exp(log_term_2 + np.log(ionization_rate_cii)) * f_he_ion

        # Terms of dfcii_dr
        t_ci = np.interp(_r, r, tau_ci)
        term11 = (1 - f_cii - f_ciii) * phi_ci * np.exp(-t_ci)  # Photo-
        # ionization
        term12 = (1 - f_cii - f_ciii) * ionization_rate_ci_n_e  # Electron-
        # impact ionization
        term13 = (1 - f_cii - f_ciii) * ct_rate_ci_hp_n_h_plus  # Charge
        # exchange with H+
        term14 = (1 - f_cii - f_ciii) * ct_rate_ci_hep_n_he_plus  # Charge
        # exchange with He+
        term15 = f_cii * alpha_rec_ci_n_e  # Recombination of C II into C I
        term16 = f_cii * ct_rate_cii_h_n_h0  # Charge exchange of C II with
        # neutral H
        term17 = f_ciii * alpha_rec_cii_n_e  # Recombination of C III into
        # C II
        term18 = f_ciii * ct_rate_ciii_h_n_h0  # Charge exchange of C III with
        # neutral H
        term19 = f_ciii * ct_rate_ciii_he_n_he0  # Charge exchange of C III
        # with neutral He

        # Terms of dfciii_dr
        t_cii = np.interp(_r, r, tau_cii)
        term21 = f_cii * phi_cii * np.exp(-t_cii)  # Photoionization
        term22 = f_cii * ionization_rate_cii_n_e  # Electron-impact ionization

        dfcii_dr = (term11 + term12 + term13 + term14 - term15 - term16 +
                    term17 + term18 + term19 - term21 - term22) / _v
        dfciii_dr = (term21 + term22 - term17 - term18 - term19) / _v

        if rates is False:
            return np.array([dfcii_dr, dfciii_dr])
        else:
            return np.array([term11, term21, term15, term17, term12, term22,
                             term13, term14, term16, term18, term19]) * phi_unit

    if method == 'odeint':
        # Since 'odeint' yields only warnings when precision is lost or when
        # there is a problem, we transform these warnings into an exception
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                sol = odeint(_fun, y0=initial_f_c_ion, t=r, tfirst=True, )
            except Warning:
                raise RuntimeError('The solver ``odeint`` failed to obtain a '
                                   'solution.')
        f_cii_r = np.copy(sol).T[0]
        f_ciii_r = np.copy(sol).T[1]
    else:
        # We solve it using `scipy.solve_ivp`
        sol = solve_ivp(_fun, (r[0], r[-1],), initial_f_c_ion, t_eval=r,
                        method=method, **options_solve_ivp)
        f_cii_r = sol['y'][0]
        f_ciii_r = sol['y'][1]
        # When `solve_ivp` has problems, it may return an array with different
        # size than `r`. So we raise an exception if this happens
        if len(f_cii_r) != len(r) or len(f_ciii_r) != len(r):
            raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                               ' solution.')

    # High densities can be numerically unstable and produce unphysical values
    # of `f_r`, so we replace negative values with zero and values above 1.0
    # with 1.0
    f_cii_r[f_cii_r < 0] = 1E-15
    f_ciii_r[f_ciii_r < 0] = 1E-15
    f_cii_r[f_cii_r > 1.0] = 1.0
    f_ciii_r[f_ciii_r > 1.0] = 1.0

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_cii_r = np.copy(f_cii_r)
            previous_f_ciii_r = np.copy(f_ciii_r)

            # Re-calculate the column densities
            tau_ci = \
                k3 * a_ci * np.flip(np.cumsum(
                    np.flip(dr * density * (1 - f_cii_r - f_ciii_r)))) + \
                tau_ci_h + tau_c_he
            tau_cii = k3 * a_cii * np.flip(
                np.cumsum(np.flip(dr * density * f_cii_r))) + tau_cii_h + \
                tau_c_he

            # Solve it again
            if method == 'odeint':
                sol = odeint(_fun, y0=initial_f_c_ion, t=r, tfirst=True)
                f_cii_r = np.copy(sol).T[0]
                f_ciii_r = np.copy(sol).T[1]
            else:
                sol = solve_ivp(_fun, (r[0], r[-1],), initial_f_c_ion,
                                t_eval=r,
                                method=method, **options_solve_ivp)
                f_cii_r = sol['y'][0]
                f_ciii_r = sol['y'][1]

            # Replace negative values with zero and values above 1.0 with
            # 1.0
            f_cii_r[f_cii_r < 0] = 1E-15
            f_ciii_r[f_ciii_r < 0] = 1E-15
            f_cii_r[f_cii_r > 1.0] = 1.0
            f_ciii_r[f_ciii_r > 1.0] = 1.0

            # Calculate the relative change of f_ion in the outer shell of
            # the atmosphere (where we expect the most important change)
            relative_delta_f_cii = abs(np.sum(f_cii_r - previous_f_cii_r)) \
                / np.sum(previous_f_cii_r)
            relative_delta_f_ciii = \
                abs(np.sum(f_ciii_r - previous_f_ciii_r)) / \
                np.sum(previous_f_ciii_r)

            # Break the loop if convergence is achieved
            if relative_delta_f_cii < convergence and \
                    relative_delta_f_ciii < convergence:
                break
            else:
                pass
    else:
        pass

    if return_rates is False:
        return f_cii_r, f_ciii_r
    else:
        ion_rate_ci, ion_rate_cii, recomb_rate_cii, recomb_rate_ciii, \
            e_imp_ion_ci, e_imp_ion_cii, ch_exchange_ci_hii, \
            ch_exchange_ci_heii, ch_exchange_cii_hi, ch_exchange_ciii_hi, \
            ch_exchange_ciii_hei = _fun(r, [f_cii_r, f_ciii_r], rates=True)
        reaction_rates = {'ionization_CI': ion_rate_ci,
                          'ionization_CII': e_imp_ion_cii,
                          'recombination_CII': recomb_rate_cii,
                          'recombination_CIII': recomb_rate_ciii,
                          'e_impact_ion_CI': e_imp_ion_ci,
                          'e_impact_ion_CII': e_imp_ion_cii,
                          'charge_exchange_CI_HII': ch_exchange_ci_hii,
                          'charge_exchange_CI_HeII': ch_exchange_ci_heii,
                          'charge_exchange_CII_HI': ch_exchange_cii_hi,
                          'charge_exchange_CIII_HI': ch_exchange_ciii_hi,
                          'charge_exchange_CIII_HeI': ch_exchange_ciii_hei}
        return f_cii_r, f_ciii_r, reaction_rates
