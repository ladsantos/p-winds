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
from scipy.interpolate import interp1d
from p_winds import tools, microphysics
import warnings


__all__ = ["radiative_processes", "radiative_processes_mono", "recombination",
           "collision", "population_fraction"]


# Helium radiative processes
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        helium (4.8 eV, or 2593 Angstrom).

    Returns
    -------
    phi_1 (``float``):
        Ionization rate of helium singlet at null optical depth in unit of
        1 / s.

    phi_3 (``float``):
        Ionization rate of helium triplet at null optical depth in unit of
        1 / s.

    a_1 (``float``):
        Flux-averaged photoionization cross-section of helium singlet in unit of
        cm ** 2.

    a_3 (``float``):
        Flux-averaged photoionization cross-section of helium triplet in unit of
        cm ** 2.

    a_h_1 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet in unit of cm ** 2.

    a_h_3 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet in unit of cm ** 2.
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

    # Photoionization cross-section of He singlet
    a_lambda_1 = microphysics.helium_singlet_cross_section(wavelength_cut_1)

    # Photoionization cross-section of He triplet. Since this is hard-coded at
    # specific wavelengths, we retrieve the wavelength bins from the code
    # itself instead of entering it as input
    wavelength_cut_3, a_lambda_3 = microphysics.helium_triplet_cross_section()
    energy_cut_3 = 1.98644586e-08 / wavelength_cut_3  # Unit of erg
    # Let's interpolate the stellar spectrum to the bins of the cross-section
    flux_lambda_cut_3 = np.interp(wavelength_cut_3, wavelength, flux_lambda)

    # Flux-averaged photoionization cross-sections of He
    # Note: For some reason the Simpson's rule implementation of ``scipy`` may
    # yield negative results when the flux varies by a few orders of magnitude
    # at the edges of integration. So we take the absolute values of a_1 and a_3
    # here.
    a_1 = abs(simps(flux_lambda_cut_1 * a_lambda_1, wavelength_cut_1) /
              simps(flux_lambda_cut_1, wavelength_cut_1))
    a_3 = abs(simps(flux_lambda_cut_3 * a_lambda_3, wavelength_cut_3) /
              simps(flux_lambda_cut_3, wavelength_cut_3))

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the He atoms see.
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


# Helium radiative processes if you have only monochromatic fluxes
def radiative_processes_mono(flux_euv, flux_fuv):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    flux_euv (``float``):
        Monochromatic extreme-ultraviolet (0 - 504 Angstrom) flux arriving at
        the planet in units of erg / s / cm ** 2. Attention: notice that this
        ``flux_euv`` is different from the one used for hydrogen, since helium
        ionization happens at a shorter wavelength.

    flux_fuv (``float``):
        Monochromatic far- to middle-ultraviolet (911 - 2593 Angstrom) flux
        arriving at the planet in units of erg / s / cm ** 2.

    Returns
    -------
    phi_1 (``float``):
        Ionization rate of helium singlet at null optical depth in unit of
        1 / s.

    phi_3 (``float``):
        Ionization rate of helium triplet at null optical depth in unit of
        1 / s.

    a_1 (``float``):
        Flux-averaged photoionization cross-section of helium singlet in unit of
        cm ** 2.

    a_3 (``float``):
        Flux-averaged photoionization cross-section of helium triplet in unit of
        cm ** 2.

    a_h_1 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet in unit of cm ** 2.

    a_h_3 (``float``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet in unit of cm ** 2.
    """
    energy_1 = np.logspace(np.log10(24.6), 3, 1000)  # Unit of eV
    wavelength_1 = 12398.419843320025 / energy_1  # Unit of angstrom

    # Hydrogen cross-section within the range important to helium singlet
    a_nu_h_1 = microphysics.hydrogen_cross_section(energy=energy_1)

    # Photoionization cross-section of He singlet
    a_lambda_1 = microphysics.helium_singlet_cross_section(wavelength_1)
    # Average cross-section to ionize helium singlet
    a_1 = np.mean(a_lambda_1)

    # The photoionization cross-section of He triplet
    wavelength_3, a_lambda_3 = microphysics.helium_triplet_cross_section()
    energy_3 = (12398.419843320025 / wavelength_3)  # Unit of eV
    # Average cross-section to ionize helium triplet
    a_3 = np.mean(a_lambda_3)

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the He atoms see.
    # Contribution to the optical depth seen by He singlet atoms:
    a_h_1 = np.mean(a_nu_h_1)
    # Contribution to the optical depth seen by He triplet atoms:
    a_nu_h_3 = microphysics.hydrogen_cross_section(
        energy=energy_3[energy_3 > 13.6])
    a_h_3 = np.mean(a_nu_h_3)

    # Convert the fluxes from erg to eV and calculate the photoionization rates
    phi_1 = flux_euv * 6.24150907e+11 * a_1 / np.mean(energy_1)
    phi_3 = flux_fuv * 6.24150907e+11 * a_3 / np.mean(energy_3)

    return phi_1, phi_3, a_1, a_3, a_h_1, a_h_3


# Helium recombination
def recombination(temperature):
    """
    Calculates the helium singlet and triplet recombination rates for a gas at
    a certain temperature.

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    alpha_rec_1 (``float``):
        Recombination rate of helium singlet in units of cm ** 3 / s.

    alpha_rec_3 (``float``):
        Recombination rate of helium triplet in units of cm ** 3 / s.
    """
    # The recombination rates come from Benjamin et al. (1999,
    # ADS:1999ApJ...514..307B)
    alpha_rec_1 = 1.54E-13 * (temperature / 1E4) ** (-0.486)
    alpha_rec_3 = 2.10E-13 * (temperature / 1E4) ** (-0.778)
    return alpha_rec_1, alpha_rec_3


# Population of helium singlet and triplet through collisions
def collision(temperature):
    """
    Calculates the helium singlet and triplet collisional population rates for
    a gas at a certain temperature.

    Parameters
    ----------
    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    q_13 (``float``):
        Rate of helium transition from singlet (1^1S) to triplet (2^3S) due to
        collisions with free electrons in units of cm ** 3 / s.

    q_31a (``float``):
        Rate of helium transition from triplet (2^3S) to 2^1S due to collisions
        with free electrons in units of cm ** 3 / s.

    q_31b (``float``):
        Rate of helium transition from triplet (2^3S) to 2^1P due to collisions
        with free electrons in units of cm ** 3 / s.

    big_q_he (``float``):
        Rate of charge exchange between helium singlet and ionized hydrogen in
        units of cm ** 3 / s.

    big_q_he_plus (``float``):
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


# Fraction of helium in singlet and triplet vs. radius profile
def population_fraction(radius_profile, velocity, density,
                        hydrogen_ion_fraction, planet_radius, temperature,
                        h_he_fraction, speed_sonic_point, radius_sonic_point,
                        density_sonic_point, spectrum_at_planet=None,
                        flux_euv=None, flux_fuv=None,
                        initial_state=np.array([0.5, 0.5]),
                        relax_solution=False, convergence=0.01, max_n_relax=10,
                        method='odeint', **options_solve_ivp):
    """
    Calculate the fraction of helium in singlet and triplet state in the upper
    atmosphere in function of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile (``numpy.ndarray``):
        Radius in unit of planetary radii.

    velocity (``numpy.ndarray``):
         Velocities sampled at the values of ``radius_profile`` in units of
         sound speed. Similar to the output of ``parker.structure()``.

    density (``numpy.ndarray``):
        Densities sampled at the values of ``radius_profile`` in units of
        density at the sonic point. Similar to the output of
        ``parker.structure()``.

    hydrogen_ion_fraction (``numpy.ndarray``):
        Hydrogen ion fraction in the upper atmosphere in function of radius.
        Similar to the output of ``hydrogen.ion_fraction()``.

    planet_radius (``float``):
        Planetary radius in unit of Jupiter radius.

    temperature (``float``):
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    h_he_fraction (``float``):
        H/He fraction of the atmosphere.

    speed_sonic_point (``float``):
        Speed of sound in the outflow in units of km / s.

    radius_sonic_point (``float``):
        Radius of the sonic point in unit of Jupiter radius.

    density_sonic_point (``float``):
        Density at the sonic point in units of g / cm ** 3.

    spectrum_at_planet (``dict``, optional):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to populate the
        helium states (4.8 eV, or 2593 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``. If ``None``, then ``flux_euv`` and
        ``flux_fuv`` must be provided instead. Default is ``None``.

    flux_euv (``float``, optional):
        Monochromatic extreme-ultraviolet (0 - 1200 Angstrom) flux arriving at
        the planet in units of erg / s / cm ** 2. If ``None``, then
        ``spectrum_at_planet`` must be provided instead. Default is ``None``.

    flux_fuv (``float``, optional):
        Monochromatic far- to middle-ultraviolet (1200 - 2600 Angstrom) flux
        arriving at the planet in units of erg / s / cm ** 2. If ``None``, then
        ``spectrum_at_planet`` must be provided instead. Default is ``None``.

    initial_state (``numpy.ndarray``, optional):
        The initial state is the `y0` of the differential equation to be solved.
        This array has two items: the initial value of the fractions of singlet
        and triplet state in the inner layer of the atmosphere. The default
        value for this parameter is ``numpy.array([0.5, 0.5])``, i.e., fully
        neutral at the inner layer with 50% in singlet and 50% in triplet
        states.

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

    method (``str``, optional):
        If method is ``'odeint'``, then ``scipy.integrate.odeint()`` is used
        instead of ``scipy.integrate.solve_ivp()`` to calculate the steady-state
        distribution of helium. The first seems to be at least twice faster than
        the second in some situations. Any other method will fallback to an
        option of ``solve_ivp()`` methods. For example, if ``method`` is set to
        ``'Radau'``, then use ``solve_ivp(method='Radau')``. Default is
        ``'odeint'``.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``atol`` (absolute tolerance; default is
        1E-6) or ``rtol`` (relative tolerance; default is 1E-3). If you are
        having numerical issues, you may want to decrease the tolerance by a
        factor of 10 or 100, or 1000 in extreme cases.

    Returns
    -------
    f_1_r (``numpy.ndarray``):
        Fraction of helium in singlet state in function of radius.

    f_3_r (``numpy.ndarray``):
        Fraction of helium in triplet state in function of radius.
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
    dr = np.concatenate((dr, np.array([r[-1], ])))

    # The way we solve the differential equation requires us to pass the H ion
    # fraction, densities and velocities at specific values of r, and it can be
    # cumbersome to  parse this inside the callable function _fun(). Instead,
    # let's create a "mock function" that returns the value of v, rho, and
    # f_H_ion in function of r (essentially a scipy.interp1d function)
    mock_f_h_ion_r = interp1d(r, hydrogen_ion_fraction,
                              fill_value="extrapolate")
    mock_v_r = interp1d(r, velocity, fill_value="extrapolate")
    mock_rho_r = interp1d(r, density, fill_value="extrapolate")

    # With all this setup done, now we need to assume something about the
    # distribution of singlet and triplet helium in the atmosphere. Let's assume
    # it based on the initial guess input.
    column_density = np.flip(np.cumsum(np.flip(dr * density)))  # Total column
                                                                # density
    column_density_h_0 = np.flip(  # Column density of H only
        np.cumsum(np.flip(dr * density * (1 - hydrogen_ion_fraction))))
    k1 = h_he_fraction / (h_he_fraction + 4 * (1 - h_he_fraction)) / m_h
    k2 = (1 - h_he_fraction) / (h_he_fraction + 4 * (1 - h_he_fraction)) / m_h
    tau_1_h = k1 * a_h_1 * column_density_h_0
    tau_3_h = k1 * a_h_3 * column_density_h_0
    tau_1_initial = (initial_state[0] * k2 * a_1 * column_density + tau_1_h)
    tau_3_initial = (initial_state[1] * k2 * a_3 * column_density + tau_3_h)
    # We do a dirty hack to make tau_initial a callable function so it's easily
    # parsed inside the differential equation solver
    _tau_1_fun = interp1d(r, tau_1_initial, fill_value="extrapolate")
    _tau_3_fun = interp1d(r, tau_3_initial, fill_value="extrapolate")

    # The differential equation
    def _fun(_r, y):
        f_1 = y[0]  # Fraction of helium in singlet
        f_3 = y[1]  # Fraction of helium in triplet

        _v = mock_v_r(np.array([_r, ]))[0]
        _rho = mock_rho_r(np.array([_r, ]))[0]
        f_h_ion = mock_f_h_ion_r(np.array([_r, ]))[0]  # Fraction of H+

        # Assume the number density of electrons is equal to the number density
        # of H ions
        n_e = k1 * _rho * f_h_ion         # Number density of electrons
        n_h_plus = k1 * _rho * f_h_ion    # Number density of ionized H
        n_h0 = k1 * _rho * (1 - f_h_ion)  # Number density of atomic H

        # Terms of df1_dr
        term_11 = (1 - f_1 - f_3) * n_e * alpha_rec_1  # Recombination
        term_12 = f_3 * big_a_31  # Radiative transition rate
        t_1 = _tau_1_fun(np.array([_r, ]))[0]
        t_3 = _tau_3_fun(np.array([_r, ]))[0]
        term_13 = f_1 * phi_1 * np.exp(-t_1)  # Photoionization
        term_14 = f_1 * n_e * q_13  # Transition rate due to collision with e
        term_15 = f_3 * n_e * q_31a  # Transition rate due to collision with e
        term_16 = f_3 * n_e * q_31b  # Transition rate due to collision with e
        term_17 = f_3 * n_h0 * big_q_31  # Combined rate of associative
                                         # ionization and Penning ionization
        term_18 = f_1 * n_h_plus * big_q_he  # Charge exchange consuming He
                                             # singlet
        term_19 = f_1 * n_h0 * big_q_he_plus  # Charge exchange producing He
                                              # singlet

        # Terms of df3_dr
        term_31 = (1 - f_1 - f_3) * n_e * alpha_rec_3  # Recombination
        term_33 = f_3 * phi_3 * np.exp(-t_3)  # Photoionization

        # Finally assemble the equations for df3_dr and df3_dr
        df1_dr = (term_11 + term_12 - term_13 - term_14 + term_15 + term_16
                  + term_17 - term_18 + term_19) / _v
        df3_dr = (term_31 - term_12 - term_33 + term_14 - term_15 - term_16
                  - term_17) / _v

        return np.array([df1_dr, df3_dr])

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
    f_1_r[f_1_r < 0] = 0.0
    f_3_r[f_3_r < 0] = 0.0
    f_1_r[f_1_r > 1.0] = 1.0
    f_3_r[f_3_r > 1.0] = 1.0

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_1_r_outer_layer = np.copy(f_1_r)[-1]
            previous_f_3_r_outer_layer = np.copy(f_3_r)[-1]

            # Re-calculate the column densities
            tau_1 = k2 * a_1 * np.flip(
                np.cumsum(np.flip(dr * density * f_1_r))) + tau_1_h
            tau_3 = k2 * a_3 * np.flip(
                np.cumsum(np.flip(dr * density * f_3_r))) + tau_3_h
            _tau_1_fun = interp1d(r, tau_1, fill_value="extrapolate")
            _tau_3_fun = interp1d(r, tau_3, fill_value="extrapolate")

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
            f_1_r[f_1_r < 0] = 0.0
            f_3_r[f_3_r < 0] = 0.0
            f_1_r[f_1_r > 1.0] = 1.0
            f_3_r[f_3_r > 1.0] = 1.0

            # Calculate the relative change of f_ion in the outer shell of the
            # atmosphere (where we expect the most important change)
            relative_delta_f_1 = abs(f_1_r[-1] - previous_f_1_r_outer_layer) \
                / previous_f_1_r_outer_layer
            relative_delta_f_3 = abs(f_3_r[-1] - previous_f_3_r_outer_layer) \
                / previous_f_3_r_outer_layer

            # Break the loop if convergence is achieved
            if relative_delta_f_1 < convergence and \
                    relative_delta_f_3 < convergence:
                break
            else:
                pass
    else:
        pass

    return f_1_r, f_3_r
