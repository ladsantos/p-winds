#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of He in the upper
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


__all__ = ["radiative_processes", "radiative_processes_mono", "recombination",
           "recombination_all", "collision", "charge_transfer",
           "population_fraction", "ion_fraction"]


# Helium radiative processes
def radiative_processes(spectrum_at_planet, combined_ionization=False):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        helium (4.8 eV, or 2593 Angstrom).

    Returns
    -------
    phi_1 : ``float``
        Ionization rate of helium singlet at null optical depth in unit of
        1 / s. This is returned if ``combined_ionization`` is set to ``False``.

    phi_3 : ``float``
        Ionization rate of helium triplet at null optical depth in unit of
        1 / s. This is returned if ``combined_ionization`` is set to ``False``.

    a_1 : ``float``
        Flux-averaged photoionization cross-section of helium singlet in unit of
        cm ** 2. This is returned if ``combined_ionization`` is set to
        ``False``.

    a_3 : ``float``
        Flux-averaged photoionization cross-section of helium triplet in unit of
        cm ** 2. This is returned if ``combined_ionization`` is set to
        ``False``.

    a_h_1 : ``float``
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet in unit of cm ** 2. This is returned if
        ``combined_ionization`` is set to ``False``.

    a_h_3 : ``float``
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet in unit of cm ** 2. This is returned if
        ``combined_ionization`` is set to ``False``.

    phi : ``float``
        Ionization rate of helium at null optical depth in unit of 1 / s. This
        is returned if ``combined_ionization`` is set to ``True``.

    a_he : ``float``
        Flux-averaged photoionization cross-section of helium in unit of
        cm ** 2. This is returned if ``combined_ionization`` is set to
        ``True``.

    a_h : ``float``
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium atoms in unit of cm ** 2. This is returned if
        ``combined_ionization`` is set to ``True``.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value
    energy_erg = (energy * u.eV).to(u.erg).value

    # Wavelength corresponding to the energy to ionize He in singlet and triplet
    wl_break_1 = 504  # Angstrom, for He
    wl_break_0 = 911.65  # Angstrom, for H

    # Index of the lambda_1 and lambda_0 in the wavelength array
    i1 = tools.nearest_index(wavelength, wl_break_1)
    i0 = tools.nearest_index(wavelength, wl_break_0)

    # Auxiliary definitions
    wavelength_cut_1 = wavelength[:i1]
    flux_lambda_cut_1 = flux_lambda[:i1]
    energy_cut_1 = energy_erg[:i1]
    wavelength_cut_0 = wavelength[:i0]
    flux_lambda_cut_0 = flux_lambda[:i0]

    # If combined_ionization is False, the code returns the rates for singlet
    # and triplet individually.
    if combined_ionization is False:
        # Photoionization cross-section of He singlet
        a_lambda_1 = microphysics.helium_singlet_cross_section(wavelength_cut_1)

        # Photoionization cross-section of He triplet. Since this is hard-coded
        # at specific wavelengths, we retrieve the wavelength bins from the code
        # itself instead of entering it as input
        wavelength_cut_3, a_lambda_3 = \
            microphysics.helium_triplet_cross_section()
        energy_cut_3 = 1.98644586e-08 / wavelength_cut_3  # Unit of erg
        # Let's interpolate the stellar spectrum to the bins of the cross-
        # section
        flux_lambda_cut_3 = np.interp(wavelength_cut_3, wavelength, flux_lambda)

        # Flux-averaged photoionization cross-sections of He
        # Note: For some reason the Simpson's rule implementation of ``scipy``
        # may yield negative results when the flux varies by a few orders of
        # magnitude at the edges of integration. So we take the absolute values
        # of a_1 and a_3 here.
        a_1 = abs(simps(flux_lambda_cut_1 * a_lambda_1, wavelength_cut_1) /
                  simps(flux_lambda_cut_1, wavelength_cut_1))
        a_3 = abs(simps(flux_lambda_cut_3 * a_lambda_3, wavelength_cut_3) /
                  simps(flux_lambda_cut_3, wavelength_cut_3))

        # The flux-averaged photoionization cross-section of H is also going to
        # be needed because it adds to the optical depth that the He atoms see.
        a_lambda_h_1 = microphysics.hydrogen_cross_section(
            wavelength=wavelength_cut_1)
        a_lambda_h_3 = microphysics.hydrogen_cross_section(
            wavelength=wavelength_cut_0)
        # Contribution to the optical depth seen by He singlet atoms:
        # Note: the same ``scipy.integrate.simps`` behavior may happen here, so
        # again we take the absolute values of a_h_n and phi_n
        a_h_1 = abs(simps(flux_lambda_cut_1 * a_lambda_h_1, wavelength_cut_1) /
                    simps(flux_lambda_cut_1, wavelength_cut_1))
        # Contribution to the optical depth seen by He triplet atoms:
        a_h_3 = abs(simps(flux_lambda_cut_0 * a_lambda_h_3, wavelength_cut_0) /
                    simps(flux_lambda_cut_3, wavelength_cut_3))

        # Calculate the photoionization rates
        phi_1 = abs(simps(flux_lambda_cut_1 * a_lambda_1 / energy_cut_1,
                    wavelength_cut_1))
        phi_3 = abs(simps(flux_lambda_cut_3 * a_lambda_3 / energy_cut_3,
                    wavelength_cut_3))

        return phi_1, phi_3, a_1, a_3, a_h_1, a_h_3

    # Otherwise, the code returns the total ionization rate for all He atoms,
    # independent if they are singlet or triplet
    else:
        # Photoionization cross-section of He
        a_lambda = microphysics.helium_total_cross_section(wavelength_cut_1)

        # Flux-averaged photoionization cross-sections of He
        a_he = abs(simps(flux_lambda_cut_1 * a_lambda, wavelength_cut_1) /
                   simps(flux_lambda_cut_1, wavelength_cut_1))

        # The flux-averaged photoionization cross-section of H is also going to
        # be needed because it adds to the optical depth that the He atoms see.
        a_lambda_h = microphysics.hydrogen_cross_section(
            wavelength=wavelength_cut_1)
        a_h = abs(simps(flux_lambda_cut_1 * a_lambda_h, wavelength_cut_1) /
                    simps(flux_lambda_cut_1, wavelength_cut_1))

        # Calculate the photoionization rates
        phi = abs(simps(flux_lambda_cut_1 * a_lambda / energy_cut_1,
                          wavelength_cut_1))

        return phi, a_he, a_h


# Helium radiative processes if you have only monochromatic fluxes
def radiative_processes_mono(flux_euv, flux_fuv,
                             average_euv_photon_wavelength=242.0,
                             average_fuv_photon_wavelength=2348.0):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    flux_euv : ``float``
        Monochromatic extreme-ultraviolet (0 - 504 Angstrom) flux arriving at
        the planet in units of erg / s / cm ** 2. Attention: notice that this
        ``flux_euv`` is different from the one used for hydrogen, since helium
        ionization happens at a shorter wavelength.

    flux_fuv : ``float``
        Monochromatic far- to middle-ultraviolet (911 - 2593 Angstrom) flux
        arriving at the planet in units of erg / s / cm ** 2.

    average_euv_photon_wavelength : ``float``
        Average wavelength of EUV photons ionizing the He singlet state, in unit
        of Angstrom. Default value is 242 Angstrom. The default value is based
        on a flux-weighted average of the solar spectrum between 0 and 504
        Angstrom.

    average_fuv_photon_wavelength : ``float``
        Average wavelength of FUV-NUV photons ionizing the He triplet state, in
        unit of Angstrom. Default value is 2348 Angstrom. The default value is
        based on a flux-weighted average of the solar spectrum between 911 and
        2593 Angstrom.

    Returns
    -------
    phi_1 : ``float``
        Ionization rate of helium singlet at null optical depth in unit of
        1 / s.

    phi_3 : ``float``
        Ionization rate of helium triplet at null optical depth in unit of
        1 / s.

    a_1 : ``float``
        Flux-averaged photoionization cross-section of helium singlet in unit of
        cm ** 2.

    a_3 : ``float``
        Flux-averaged photoionization cross-section of helium triplet in unit of
        cm ** 2.

    a_h_1 : ``float``
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet in unit of cm ** 2.

    a_h_3 : ``float``
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet in unit of cm ** 2.
    """
    # Average cross-section to ionize helium singlet
    a_1 = microphysics.helium_singlet_cross_section(average_euv_photon_wavelength)

    # The photoionization cross-section of He triplet
    wavelength_3, a_lambda_3 = microphysics.helium_triplet_cross_section()
    # # Average cross-section to ionize helium triplet
    a_3 = np.interp(average_fuv_photon_wavelength, wavelength_3, a_lambda_3)

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the He atoms see.
    # Contribution to the optical depth seen by He singlet atoms:
    # Hydrogen cross-section within the range important to helium singlet
    a_h_1 = 6.3E-18 * (average_euv_photon_wavelength / 13.6) ** (-3)
    # Unit 1 / cm ** 2.
    # Contribution to the optical depth seen by He triplet atoms:
    if average_fuv_photon_wavelength < 911.0:
        a_h_3 = microphysics.hydrogen_cross_section(
            wavelength=average_fuv_photon_wavelength)
    else:
        a_h_3 = 0.0

    # Convert the fluxes from erg to eV and calculate the photoionization rates
    energy_1 = 12398.419843320025 / average_euv_photon_wavelength
    energy_3 = 12398.419843320025 / average_fuv_photon_wavelength
    phi_1 = flux_euv * 6.24150907e+11 * a_1 / energy_1
    phi_3 = flux_fuv * 6.24150907e+11 * a_3 / energy_3

    return phi_1, phi_3, a_1, a_3, a_h_1, a_h_3


# Helium recombination into singlet and triplet atoms
def recombination(temperature):
    """
    Calculates the helium singlet and triplet recombination rates for a gas at
    a certain temperature.

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    alpha_rec_1 : ``float``
        Recombination rate of helium singlet in units of cm ** 3 / s.

    alpha_rec_3 : ``float``
        Recombination rate of helium triplet in units of cm ** 3 / s.
    """
    # The recombination rates come from Benjamin et al. (1999,
    # ADS:1999ApJ...514..307B)
    alpha_rec_1 = 1.54E-13 * (temperature / 1E4) ** (-0.486)
    alpha_rec_3 = 2.10E-13 * (temperature / 1E4) ** (-0.778)
    return alpha_rec_1, alpha_rec_3


# Helium recombination indifferent to singlet and triplet states
def recombination_all(temperature):
    """
    Calculates the helium recombination rates for a gas at a certain
    temperature, with no distinction between singlet and triplet states.

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    alpha_rec : ``float``
        Recombination rate of helium in units of cm ** 3 / s.
    """
    # The recombination rates come from Storey & Hummer 1995
    alpha_rec = 4.6E-12 * (temperature / 300) ** (-0.64)
    return alpha_rec


# Population of helium singlet and triplet through collisions
def collision(temperature):
    """
    Calculates the helium singlet and triplet collisional population rates for
    a gas at a certain temperature.

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    q_13 : ``float``
        Rate of helium transition from singlet (1^1S) to triplet (2^3S) due to
        collisions with free electrons in units of cm ** 3 / s.

    q_31a : ``float``
        Rate of helium transition from triplet (2^3S) to 2^1S due to collisions
        with free electrons in units of cm ** 3 / s.

    q_31b : ``float``
        Rate of helium transition from triplet (2^3S) to 2^1P due to collisions
        with free electrons in units of cm ** 3 / s.

    big_q_he : ``float``
        Rate of charge exchange between helium singlet and ionized hydrogen in
        units of cm ** 3 / s.

    big_q_he_plus : ``float``
        Rate of charge exchange between ionized helium and atomic hydrogen in
        units of cm ** 3 / s.
    """
    # The effective collision strengths are hard-coded from the values provided
    # by Bray et al. (2000, ADS:2000A&AS..146..481B), which are binned to
    # specific temperatures. Thus, we need to interpolate to the specific
    # temperature of our gas. First we parse the tabulated data
    data_array = microphysics.he_collisional_strength()
    tabulated_temp = 10 ** data_array[:, 0]
    tabulated_gamma_13 = data_array[:, 1]
    tabulated_gamma_31a = data_array[:, 2]
    tabulated_gamma_31b = data_array[:, 3]
    # And interpolate to our desired temperature
    gamma_13 = np.interp(temperature, tabulated_temp, tabulated_gamma_13,
                         left=tabulated_gamma_13[0],
                         right=tabulated_gamma_13[-1])
    gamma_31a = np.interp(temperature, tabulated_temp, tabulated_gamma_31a,
                          left=tabulated_gamma_31a[0],
                          right=tabulated_gamma_31a[-1])
    gamma_31b = np.interp(temperature, tabulated_temp, tabulated_gamma_31b,
                          left=tabulated_gamma_31b[0],
                          right=tabulated_gamma_31b[-1])

    # Finally calculate the rates using the equations from Table 2 of Lampón et
    # al. (2020).
    kt = 8.617333262145179e-05 * temperature
    k1 = 2.10E-8 * (13.6 / kt) ** 0.5
    q_13 = k1 * gamma_13 * np.exp(-19.81 / kt)
    q_31a = k1 * gamma_31a / 3 * np.exp(-0.80 / kt)
    q_31b = k1 * gamma_31b / 3 * np.exp(-1.40 / kt)
    big_q_he = 1.75E-11 * (300 / temperature) ** 0.75 * \
        np.exp(-128E3 / temperature)
    big_q_he_plus = 1.25E-15 * (300 / temperature) ** (-0.25)

    return q_13, q_31a, q_31b, big_q_he, big_q_he_plus


# Charge transfer between He and H
def charge_transfer(temperature):
    """
    Calculates the charge exchange rates of He with H nuclei. Based on the
    formulation of Glover & Jappsen et al. 2007.

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    ct_rate_he_hp : ``float``
        Charge transfer rate between neutral He and H+ in units of cm ** 3 / s.

    ct_rate_hep_h : ``float``
        Charge transfer rate between He+ and neutral H in units of cm ** 3 / s.
    """
    # Recombination of He II into He I
    ct_rate_hep_h = 1.25E-15 * (300 / temperature) ** (-0.25)

    # Ionization of He I into He II
    ct_rate_he_hp = 1.75E-11 * (300 / temperature) ** 0.75 * \
        np.exp(-128000 / temperature)

    return ct_rate_he_hp, ct_rate_hep_h


# Fraction of helium in singlet and triplet vs. radius profile
def population_fraction(radius_profile, velocity, density,
                        hydrogen_ion_fraction, planet_radius, temperature,
                        h_fraction, speed_sonic_point, radius_sonic_point,
                        density_sonic_point, spectrum_at_planet=None,
                        flux_euv=None, flux_fuv=None,
                        initial_state=np.array([0.5, 0.5]),
                        relax_solution=False, convergence=0.01, max_n_relax=10,
                        method='odeint', return_rates=False,
                        **options_solve_ivp):
    """
    Calculate the fraction of helium in singlet and triplet state in the upper
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

    spectrum_at_planet : ``dict``, optional
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to populate the
        helium states (4.8 eV, or 2593 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``. If ``None``, then ``flux_euv`` and
        ``flux_fuv`` must be provided instead. Default is ``None``.

    flux_euv : ``float``, optional
        Monochromatic extreme-ultraviolet (0 - 1200 Angstrom) flux arriving at
        the planet in units of erg / s / cm ** 2. If ``None``, then
        ``spectrum_at_planet`` must be provided instead. Default is ``None``.

    flux_fuv : ``float``, optional
        Monochromatic far- to middle-ultraviolet (1200 - 2600 Angstrom) flux
        arriving at the planet in units of erg / s / cm ** 2. If ``None``, then
        ``spectrum_at_planet`` must be provided instead. Default is ``None``.

    initial_state : ``numpy.ndarray``, optional
        The initial state is the `y0` of the differential equation to be solved.
        This array has two items: the initial value of the fractions of singlet
        and triplet state in the inner layer of the atmosphere. The default
        value for this parameter is ``numpy.array([0.5, 0.5])``, i.e., fully
        neutral at the inner layer with 50% in singlet and 50% in triplet
        states.

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
        ``f_r``. Default is 10.

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
    f_1_r : ``numpy.ndarray``
        Fraction of helium in singlet state in function of radius.

    f_3_r : ``numpy.ndarray``
        Fraction of helium in triplet state in function of radius.

    reaction_rates : ``dict``
        Dictionary containing the reaction rates in function of radius and in
        units of 1 / s. Only returned when ``return_rates`` is set to ``True``.
        Here is a short description of the dict keys:

        * `ionization_1`: Photoionization of He singlet atoms
        * `ionization_3`: Photoionization of He triplet atoms
        * `recombination_1`: Recombination of He ions into He singlet
        * `recombination_3`: Recombination of He ions into He triplet
        * `radiative_transition`: Radiative transition of He triplet into singlet
        * `transition_1_to_3`: Transition of He singlet to triplet due to collisions with electrons
        * `transition_3_to_21s`: Transition of He triplet to 2$^1$S due to collisions with electrons
        * `transition_3_to_21p`: Transition of He triplet to 2$^1$P due to collisions with electrons
        * `other_ionization`: Combined rate of associative ionization and Penning ionization
        * `charge_exchange_1`: Charge exchange between helium singlet and ionized hydrogen
        * `charge_exchange_he_ion`: Charge exchange between ionized helium and atomic hydrogen
    """
    vs = speed_sonic_point  # km / s
    rs = radius_sonic_point  # jupiterRad
    rhos = density_sonic_point  # g / cm ** 3

    # Recombination rates of helium singlet and triplet in unit of rs ** 2 * vs
    alpha_rec_unit = ((rs * 7.1492E+09) ** 2 * vs * 1E5)  # cm ** 3 / s
    alpha_rec_1, alpha_rec_3 = recombination(temperature)
    alpha_rec_1 = alpha_rec_1 / alpha_rec_unit
    alpha_rec_3 = alpha_rec_3 / alpha_rec_unit

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * (rs * 7.1492E+09) ** 3)  # Converted to g
    m_h = 1.67262192E-24 / m_h_unit

    # XXX Things start to get very complicated from here, so brace yourself.
    # There are lots of variables to keep track of, since the population of the
    # helium triplet and singlet depend on many processes, including whatever
    # happens with hydrogen as well.

    # Photoionization rates at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs, and the flux-averaged
    # cross-sections in units of rs ** 2
    phi_unit = vs * 1E5 / rs / 7.1492E+09  # 1 / s
    if spectrum_at_planet is not None:
        phi_1, phi_3, a_1, a_3, a_h_1, a_h_3 = radiative_processes(
            spectrum_at_planet)
    elif flux_euv is not None and flux_fuv is not None:
        phi_1, phi_3, a_1, a_3, a_h_1, a_h_3 = radiative_processes_mono(
            flux_euv, flux_fuv)
    else:
        raise ValueError('Either `spectrum_at_planet` must be provided, or '
                         '`flux_euv` and `flux_fuv` must be provided.')
    phi_1 = phi_1 / phi_unit
    phi_3 = phi_3 / phi_unit
    a_1 = a_1 / (rs * 7.1492E+09) ** 2
    a_3 = a_3 / (rs * 7.1492E+09) ** 2
    a_h_1 = a_h_1 / (rs * 7.1492E+09) ** 2
    a_h_3 = a_h_3 / (rs * 7.1492E+09) ** 2

    # Collision-induced transition rates for helium triplet and singlet, in the
    # same unit as the recombination rates
    q_13, q_31a, q_31b, big_q_he, big_q_he_plus = collision(temperature)
    q_13 = q_13 / alpha_rec_unit
    q_31a = q_31a / alpha_rec_unit
    q_31b = q_31b / alpha_rec_unit
    big_q_he = big_q_he / alpha_rec_unit
    big_q_he_plus = big_q_he_plus / alpha_rec_unit

    # Some hard-coding here. The numbers come from Oklopcic & Hirata (2018) and
    # Lampón et al. (2020).
    big_q_31 = 5E-10 / alpha_rec_unit
    big_a_31 = 1.272E-4 / phi_unit

    # Now let's solve the differential eq. 15 of Oklopcic & Hirata 2018

    # The radius in unit of radius at the sonic point
    r = radius_profile * planet_radius / rs
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([dr[-1], ])))

    # With all this setup done, now we need to assume something about the
    # distribution of singlet and triplet helium in the atmosphere. Let's assume
    # it based on the initial guess input.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))  # Total column
                                                                # density
    column_density_h_0 = np.flip(  # Column density of H only
        np.cumsum(np.flip(dr * density * (1 - hydrogen_ion_fraction))))
    he_fraction = 1 - h_fraction
    k1 = h_fraction / (h_fraction + 4 * he_fraction) / m_h
    k2 = he_fraction / (h_fraction + 4 * he_fraction) / m_h
    tau_1_h = k1 * a_h_1 * column_density_h_0
    tau_3_h = k1 * a_h_3 * column_density_h_0
    tau_1 = (initial_state[0] * k2 * a_1 * column_density + tau_1_h)
    tau_3 = (initial_state[1] * k2 * a_3 * column_density + tau_3_h)

    # The differential equation
    def _fun(_r, y, rates=False):
        f_1 = y[0]  # Fraction of helium in singlet
        f_3 = y[1]  # Fraction of helium in triplet

        _v = np.interp(_r, r, velocity)
        _rho = np.interp(_r, r, density)
        f_h_ion = np.interp(_r, r, hydrogen_ion_fraction)  # Fraction of H+

        # Assume the number density of electrons is equal to the number density
        # of H ions
        # Since we may run into loss of numerical precision here (big numbers),
        # we manipulate the equations to avoid this problem. It looks a bit
        # messy, but it is necessary
        alpha_rec_1_n_e = np.exp(
            np.log(k1) + np.log(_rho) + np.log(alpha_rec_1)) * f_h_ion
        alpha_rec_3_n_e = np.exp(
            np.log(k1) + np.log(_rho) + np.log(alpha_rec_3)) * f_h_ion
        q_13_n_e = np.exp(
            np.log(k1) + np.log(_rho) + np.log(q_13)) * f_h_ion
        q_31a_n_e = np.exp(
            np.log(k1) + np.log(_rho) + np.log(q_31a)) * f_h_ion
        q_31b_n_e = np.exp(
            np.log(k1) + np.log(_rho) + np.log(q_31b)) * f_h_ion
        big_q_31_n_h0 = np.exp(
            np.log(k1) + np.log(_rho) + np.log(big_q_31)) * (1 - f_h_ion)
        big_q_he_n_h_plus = np.exp(
            np.log(k1) + np.log(_rho) + np.log(big_q_he)) * f_h_ion
        big_q_he_plus_n_h0 = np.exp(
            np.log(k1) + np.log(_rho) + np.log(big_q_he_plus)) * (1 - f_h_ion)

        # Terms of df1_dr
        term_11 = (1 - f_1 - f_3) * alpha_rec_1_n_e  # Recombination
        term_12 = f_3 * big_a_31  # Radiative transition rate
        t_1 = np.interp(_r, r, tau_1)
        t_3 = np.interp(_r, r, tau_3)
        term_13 = f_1 * phi_1 * np.exp(-t_1)  # Photoionization
        term_14 = f_1 * q_13_n_e  # Transition rate due to collision with e
        term_15 = f_3 * q_31a_n_e  # Transition rate due to collision with e
        term_16 = f_3 * q_31b_n_e  # Transition rate due to collision with e
        term_17 = f_3 * big_q_31_n_h0  # Combined rate of associative
                                         # ionization and Penning ionization
        term_18 = f_1 * big_q_he_n_h_plus  # Charge exchange consuming He
                                             # singlet
        term_19 = (1 - f_1 - f_3) * big_q_he_plus_n_h0  # Charge exchange
                                                          # producing He
                                                          # singlet

        # Terms of df3_dr
        term_31 = (1 - f_1 - f_3) * alpha_rec_3_n_e  # Recombination
        term_33 = f_3 * phi_3 * np.exp(-t_3)  # Photoionization

        # Finally assemble the equations for df3_dr and df3_dr
        df1_dr = (term_11 + term_12 - term_13 - term_14 + term_15 + term_16
                  + term_17 - term_18 + term_19) / _v
        df3_dr = (term_31 - term_12 - term_33 + term_14 - term_15 - term_16
                  - term_17) / _v

        if rates is False:
            return np.array([df1_dr, df3_dr])
        else:
            return np.array([term_13, term_33, term_11, term_31, term_12,
                             term_14, term_15, term_16, term_17, term_18,
                             term_19]) * phi_unit

    if method == 'odeint':
        # Since 'odeint' yields only warnings when precision is lost or when
        # there is a problem, we transform these warnings into an exception
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                sol = odeint(_fun, y0=initial_state, t=r, tfirst=True)
            except Warning:
                raise RuntimeError('The solver ``odeint`` failed to obtain a '
                                   'solution.')
        f_1_r = np.copy(sol).T[0]
        f_3_r = np.copy(sol).T[1]
    else:
        # We solve it using `scipy.solve_ivp`
        sol = solve_ivp(_fun, (r[0], r[-1],), initial_state, t_eval=r,
                        method=method, **options_solve_ivp)
        f_1_r = sol['y'][0]
        f_3_r = sol['y'][1]
        # When `solve_ivp` has problems, it may return an array with different
        # size than `r`. So we raise an exception if this happens
        if len(f_1_r) != len(r) or len(f_3_r) != len(r):
            raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                               ' solution.')

    # High densities can be numerically unstable and produce unphysical values
    # of `f_r`, so we replace negative values with zero and values above 1.0
    # with 1.0
    f_1_r[f_1_r < 0] = 1E-15
    f_3_r[f_3_r < 0] = 1E-15
    f_1_r[f_1_r > 1.0] = 1.0
    f_3_r[f_3_r > 1.0] = 1.0

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_1_r = np.copy(f_1_r)
            previous_f_3_r = np.copy(f_3_r)

            # Re-calculate the column densities
            tau_1 = k2 * a_1 * np.flip(
                np.cumsum(np.flip(dr * density * f_1_r))) + tau_1_h
            tau_3 = k2 * a_3 * np.flip(
                np.cumsum(np.flip(dr * density * f_3_r))) + tau_3_h

            # Solve it again
            if method == 'odeint':
                sol = odeint(_fun, y0=initial_state, t=r, tfirst=True)
                f_1_r = np.copy(sol).T[0]
                f_3_r = np.copy(sol).T[1]
            else:
                sol = solve_ivp(_fun, (r[0], r[-1],), initial_state, t_eval=r,
                                method=method, **options_solve_ivp)
                f_1_r = sol['y'][0]
                f_3_r = sol['y'][1]

            # Replace negative values with zero and values above 1.0 with 1.0
            f_1_r[f_1_r < 0] = 1E-15
            f_3_r[f_3_r < 0] = 1E-15
            f_1_r[f_1_r > 1.0] = 1.0
            f_3_r[f_3_r > 1.0] = 1.0

            # Calculate the relative change of f_ion in the outer shell of the
            # atmosphere (where we expect the most important change)
            relative_delta_f_1 = abs(np.sum(f_1_r - previous_f_1_r)) \
                / np.sum(previous_f_1_r)
            relative_delta_f_3 = abs(np.sum(f_3_r - previous_f_3_r)) \
                / np.sum(previous_f_3_r)

            # Break the loop if convergence is achieved
            if relative_delta_f_1 < convergence and \
                    relative_delta_f_3 < convergence:
                break
            else:
                pass
    else:
        pass

    if return_rates is False:
        return f_1_r, f_3_r
    else:
        ion_rate_1, ion_rate_3, recomb_rate_1, recomb_rate_3, rad_trans_rate, \
            transition_rate_q13, transition_rate_q31a, transition_rate_q31b, \
            other_ionz_rate, ch_exchange_he1, ch_exchange_hep = \
            _fun(r, [f_1_r, f_3_r], rates=True)
        reaction_rates = {'ionization_1': ion_rate_1,
                          'ionization_3': ion_rate_3,
                          'recombination_1': recomb_rate_1,
                          'recombination_3': recomb_rate_3,
                          'radiative_transition': rad_trans_rate,
                          'transition_1_to_3': transition_rate_q13,
                          'transition_3_to_21s': transition_rate_q31a,
                          'transition_3_to_21p': transition_rate_q31b,
                          'other_ionization': other_ionz_rate,
                          'charge_exchange_1': ch_exchange_he1,
                          'charge_exchange_he_ion': ch_exchange_hep}
        return f_1_r, f_3_r, reaction_rates


# Calculate the ion fraction of He
def ion_fraction(radius_profile, velocity, density, hydrogen_ion_fraction,
                 planet_radius, temperature, h_fraction, speed_sonic_point,
                 radius_sonic_point, density_sonic_point, spectrum_at_planet,
                 initial_f_he_ion=0.0, relax_solution=False, convergence=0.01,
                 max_n_relax=10, method='Radau', **options_solve_ivp):
    """
    Sometimes we need to calculate only the fraction of ionized helium and not
    necessarily the triplet and singlet fractions. This function does that,
    which is faster than ``population_fraction()``. The result is in function of
    the radius in unit of planetary radius.

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
        least up to the wavelength corresponding to the energy to ionize helium
        (4.8 eV, or 2593 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``.

    initial_f_he_ion : ``numpy.ndarray``, optional
        The initial helium ion fraction at the layer near the surface of the
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
        ``f_r``. Default is 10.

    method : ``str``, optional
        If method is ``'odeint'``, then ``scipy.integrate.odeint()`` is used
        instead of ``scipy.integrate.solve_ivp()`` to calculate the steady-state
        distribution of helium. The first seems to be at least twice faster than
        the second in some situations. Any other method will fall back to an
        option of ``solve_ivp()`` methods. For example, if ``method`` is set to
        ``'Radau'``, then use ``solve_ivp(method='Radau')``. Default is
        ``'Radau'``.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``atol`` (absolute tolerance; default is
        1E-6) or ``rtol`` (relative tolerance; default is 1E-3). If you are
        having numerical issues, you may want to decrease the tolerance by a
        factor of 10 or 100, or 1000 in extreme cases.

    Returns
    -------
    f_r : ``numpy.ndarray``
        Fraction of ionized helium in function of radius.
    """
    vs = speed_sonic_point  # km / s
    rs = radius_sonic_point  # jupiterRad
    rhos = density_sonic_point  # g / cm ** 3

    # Recombination rates of C in unit of rs ** 2 * vs
    alpha_rec_unit = ((rs * 7.1492E+09) ** 2 * vs * 1E5)  # cm ** 3 / s
    alpha_rec = recombination_all(temperature)
    alpha_rec = alpha_rec / alpha_rec_unit

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * (rs * 7.1492E+09) ** 3)  # Converted to g
    m_h = 1.67262192E-24 / m_h_unit

    # Photoionization rates at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs, and the flux-averaged
    # cross-sections in units of rs ** 2
    phi_unit = vs * 1E5 / rs / 7.1492E+09  # 1 / s
    phi_he, a_he, a_h = radiative_processes(spectrum_at_planet,
                                            combined_ionization=True)
    phi_he = phi_he / phi_unit
    a_he = a_he / (rs * 7.1492E+09) ** 2
    a_h = a_h / (rs * 7.1492E+09) ** 2

    # Charge transfer rates in the same unit as the recombination rates
    ct_rate_he_hp, ct_rate_hep_h = charge_transfer(temperature)
    ct_rate_hep_h = ct_rate_hep_h / alpha_rec_unit
    ct_rate_he_hp = ct_rate_he_hp / alpha_rec_unit

    # We solve the steady-state ionization balance

    # The radius in unit of radius at the sonic point
    r = radius_profile * planet_radius / rs
    dr = np.diff(r)
    dr = np.concatenate((dr, np.array([dr[-1], ])))

    # With all this setup done, now we need to assume something about the
    # distribution of singlet and triplet helium in the atmosphere. Let's assume
    # it based on the initial guess input.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))  # Total column
                                                                # density
    column_density_h_0 = np.flip(  # Column density of H only
        np.cumsum(np.flip(dr * density * (1 - hydrogen_ion_fraction))))
    he_fraction = 1 - h_fraction
    k1 = h_fraction / (h_fraction + 4 * he_fraction) / m_h
    k2 = he_fraction / (h_fraction + 4 * he_fraction) / m_h
    tau_h = k1 * a_h * column_density_h_0
    tau_he = (initial_f_he_ion * k2 * a_he * column_density + tau_h)

    # The differential equation
    def _fun(_r, y):
        f_he_ion = y

        _v = np.interp(_r, r, velocity)
        _rho = np.interp(_r, r, density)
        f_h_ion = np.interp(_r, r, hydrogen_ion_fraction)  # Fraction of H+

        # Assume the number density of electrons is equal to the number density
        # of H ions + He ions
        # Since we may run into loss of numerical precision here (big numbers),
        # we manipulate the equations to avoid this problem. It looks a bit
        # messy, but it is necessary
        log_term_1 = np.log(k1) + np.log(_rho)  # H ions
        log_term_2 = np.log(k2) + np.log(_rho)  # He ions
        ct_rate_he_hp_n_h_plus = \
            np.exp(log_term_1 + np.log(ct_rate_he_hp)) * f_h_ion
        alpha_rec_n_e = \
            np.exp(log_term_1 + np.log(alpha_rec)) * f_h_ion + \
            np.exp(log_term_2 + np.log(alpha_rec)) * f_he_ion
        ct_rate_hep_h_n_h0 = \
            np.exp(log_term_1 + np.log(ct_rate_hep_h)) * (1 - f_h_ion)

        # n_e = k1 * _rho * f_h_ion + k2 * _rho * f_he_ion  # Number density of
        # # electrons
        # n_h_plus = k1 * _rho * f_h_ion    # Number density of ionized H
        # n_h0 = k1 * _rho * (1 - f_h_ion)  # Number density of atomic H

        # Terms of df_dr
        t_he = np.interp(_r, r, tau_he)
        term1 = (1 - f_he_ion) * phi_he * np.exp(-t_he)  # Photoionization
        term2 = (1 - f_he_ion) * ct_rate_he_hp_n_h_plus  # Charge exchange
        # with H+
        term3 = f_he_ion * alpha_rec_n_e  # Recombination of He II into He I
        term4 = f_he_ion * ct_rate_hep_h_n_h0  # Charge exchange of He II with
        # neutral H
        df_dr = (term1 + term2 - term3 - term4) / _v

        return df_dr

    if method == 'odeint':
        # Since 'odeint' yields only warnings when precision is lost or when
        # there is a problem, we transform these warnings into an exception
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                sol = odeint(_fun, y0=initial_f_he_ion, t=r, tfirst=True)
            except Warning:
                raise RuntimeError('The solver ``odeint`` failed to obtain a '
                                   'solution.')
        f_r = np.copy(sol).T[0]
    else:
        # We solve it using `scipy.solve_ivp`
        sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_he_ion, ]),
                        t_eval=r, method=method, **options_solve_ivp)
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
            previous_f_r = np.copy(f_r)

            # Re-calculate the column densities
            tau_he = \
                k2 * a_he * np.flip(np.cumsum(
                    np.flip(dr * density * (1 - f_r)))) + tau_h

            # Solve it again
            if method == 'odeint':
                sol = odeint(_fun, y0=initial_f_he_ion, t=r, tfirst=True)
                f_r = np.copy(sol).T[0]
            else:
                sol = solve_ivp(_fun, (r[0], r[-1],),
                                np.array([initial_f_he_ion, ]), t_eval=r,
                                method=method, **options_solve_ivp)
                f_r = sol['y'][0]

            # Replace negative values with zero and values above 1.0 with
            # 1.0
            f_r[f_r < 0] = 1E-15
            f_r[f_r > 1.0] = 1.0

            # Calculate the relative change of f_ion in the outer shell of
            # the atmosphere (where we expect the most important change)
            relative_delta_f = abs(np.sum(f_r - previous_f_r)) \
                / np.sum(previous_f_r)

            # Break the loop if convergence is achieved
            if relative_delta_f < convergence:
                break
            else:
                pass
    else:
        pass

    return f_r
