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
from scipy.integrate import simps, solve_ivp
from scipy.interpolate import interp1d
from p_winds import parker, tools, microphysics


__all__ = ["radiative_processes", "recombination", "collision",
           "population_fraction"]


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
    phi_1 (``astropy.Quantity``):
        Ionization rate of helium singlet at null optical depth.

    phi_3 (``astropy.Quantity``):
        Ionization rate of helium triplet at null optical depth.

    a_1 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium singlet.

    a_3 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium triplet.

    a_h_1 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet.

    a_h_3 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet.
    """
    wavelength = (spectrum_at_planet['wavelength'] *
                  spectrum_at_planet['wavelength_unit']).to(u.angstrom).value
    flux_lambda = (spectrum_at_planet['flux_lambda'] * spectrum_at_planet[
        'flux_unit']).to(u.erg / u.s / u.cm ** 2 / u.angstrom).value
    energy = ((c.h * (c.c / wavelength / u.angstrom).to(u.Hz)).to(u.eV)).value
    energy_erg = (energy * u.eV).to(u.erg).value

    # Wavelength corresponding to the energy to ionize He in singlet and triplet
    wl_break_1 = (c.h * c.c / (24.6 * u.eV)).to(u.angstrom).value  # For He
    wl_break_0 = (c.h * c.c / (13.6 * u.eV)).to(u.angstrom).value  # For H

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
    energy_cut_3 = ((c.h * c.c).to(u.erg * u.angstrom) / wavelength_cut_3).value
    # Let's interpolate the stellar spectrum to the bins of the cross-section
    flux_lambda_cut_3 = np.interp(wavelength_cut_3, wavelength, flux_lambda)

    # Flux-averaged photoionization cross-sections of He
    a_1 = simps(flux_lambda_cut_1 * a_lambda_1, wavelength_cut_1) / \
        simps(flux_lambda_cut_1, wavelength_cut_1) * u.cm ** 2
    a_3 = simps(flux_lambda_cut_3 * a_lambda_3, wavelength_cut_3) / \
        simps(flux_lambda_cut_3, wavelength_cut_3) * u.cm ** 2

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the He atoms see.
    a_lambda_h_1 = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_1)
    a_lambda_h_3 = microphysics.hydrogen_cross_section(
        wavelength=wavelength_cut_0)
    # Contribution to the optical depth seen by He singlet atoms:
    a_h_1 = simps(flux_lambda_cut_1 * a_lambda_h_1, wavelength_cut_1) / \
        simps(flux_lambda_cut_1, wavelength_cut_1) * u.cm ** 2
    # Contribution to the optical depth seen by He triplet atoms:
    a_h_3 = simps(flux_lambda_cut_0 * a_lambda_h_3, wavelength_cut_0) / \
        simps(flux_lambda_cut_3, wavelength_cut_3) * u.cm ** 2

    # Calculate the photoionization rates
    phi_1 = simps(flux_lambda_cut_1 * a_lambda_1 / energy_cut_1,
                  wavelength_cut_1) / u.s
    phi_3 = simps(flux_lambda_cut_3 * a_lambda_3 / energy_cut_3,
                  wavelength_cut_3) / u.s

    return phi_1, phi_3, a_1, a_3, a_h_1, a_h_3


# Helium radiative processes if you have only monochromatic fluxes
def radiative_processes_mono(flux_euv, flux_fuv):
    """
    Calculate the photoionization rate of helium at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    flux_euv (``astropy.Quantity``):
        Monochromatic extreme-ultraviolet (0 - 1200 Angstrom) flux arriving at
        the planet.

    flux_fuv (``astropy.Quantity``):
        Monochromatic far- to middle-ultraviolet (1200 - 2600 Angstrom) flux
        arriving at the planet.

    Returns
    -------
    phi_1 (``astropy.Quantity``):
        Ionization rate of helium singlet at null optical depth.

    phi_3 (``astropy.Quantity``):
        Ionization rate of helium triplet at null optical depth.

    a_1 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium singlet.

    a_3 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of helium triplet.

    a_h_1 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium singlet.

    a_h_3 (``astropy.Quantity``):
        Flux-averaged photoionization cross-section of hydrogen in the range
        absorbed by helium triplet.
    """
    energy_1 = np.logspace(np.log10(24.6), 3, 1000)
    wavelength_1 = (c.h * c.c).to(u.eV * u.angstrom).value / energy_1

    # Hydrogen cross-section within the range important to helium singlet
    a_nu_h_1 = microphysics.hydrogen_cross_section(energy=energy_1)

    # Photoionization cross-section of He singlet
    a_lambda_1 = microphysics.helium_singlet_cross_section(wavelength_1)
    # Average cross-section to ionize helium singlet
    a_1 = np.mean(a_lambda_1) * u.cm ** 2

    # The photoionization cross-section of He triplet
    wavelength_3, a_lambda_3 = microphysics.helium_triplet_cross_section()
    energy_3 = ((c.h * c.c).to(u.erg * u.angstrom) / wavelength_3).value
    # Average cross-section to ionize helium triplet
    a_3 = np.mean(a_lambda_3) * u.cm ** 2

    # The flux-averaged photoionization cross-section of H is also going to be
    # needed because it adds to the optical depth that the He atoms see.
    # Contribution to the optical depth seen by He singlet atoms:
    a_h_1 = np.mean(a_nu_h_1) * u.cm ** 2
    # Contribution to the optical depth seen by He triplet atoms:
    a_nu_h_3 = microphysics.hydrogen_cross_section(energy=energy_3)
    a_h_3 = np.mean(a_nu_h_3) * u.cm ** 2

    # Calculate the photoionization rates
    phi_1 = flux_euv.to(u.eV / u.s / u.cm ** 2) * a_1 / np.mean(energy_1) / u.eV
    phi_3 = flux_fuv.to(u.eV / u.s / u.cm ** 2) * a_3 / np.mean(energy_3) / u.eV

    return phi_1, phi_3, a_1, a_3, a_h_1, a_h_3


# Helium recombination
def recombination(temperature):
    """
    Calculates the helium singlet and triplet recombination rates for a gas at
    a certain temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    alpha_rec_1 (``astropy.Quantity``):
        Recombination rate of helium singlet.

    alpha_rec_3 (``astropy.Quantity``):
        Recombination rate of helium triplet.
    """
    # The recombination rates come from Benjamin et al. (1999,
    # ADS:1999ApJ...514..307B)
    alpha_rec_1 = 1.54E-13 * (temperature.to(u.K).value / 1E4) ** (-0.486) * \
        u.cm ** 3 / u.s
    alpha_rec_3 = 2.10E-13 * (temperature.to(u.K).value / 1E4) ** (-0.778) * \
        u.cm ** 3 / u.s
    return alpha_rec_1, alpha_rec_3


# Population of helium singlet and triplet through collisions
def collision(temperature):
    """
    Calculates the helium singlet and triplet collisional population rates for
    a gas at a certain temperature.

    Parameters
    ----------
    temperature (``astropy.Quantity``):
        Isothermal temperature of the upper atmosphere.

    Returns
    -------
    q_13 (``astropy.Quantity``):
        Rate of helium transition from singlet (1^1S) to triplet (2^3S) due to
        collisions with free electrons.

    q_31a (``astropy.Quantity``):
        Rate of helium transition from triplet (2^3S) to 2^1S due to collisions
        with free electrons.

    q_31b (``astropy.Quantity``):
        Rate of helium transition from triplet (2^3S) to 2^1P due to collisions
        with free electrons.

    big_q_he (``astropy.Quantity``):
        Rate of charge exchange between helium singlet and ionized hydrogen.

    big_q_he_plus (``astropy.Quantity``):
        Rate of charge exchange between ionized helium and atomic hydrogen.
    """
    # The effective collision strengths are hard-coded from the values provided
    # by Bray et al. (2000, ADS:2000A&AS..146..481B), which are binned to
    # specific temperatures. Thus, we need to interpolate to the specific
    # temperature of our gas.
    temperature = temperature.to(u.K).value
    # Parse the tabulated data
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
    kt = c.k_B.to(u.eV / u.K).value * temperature
    k1 = 2.10E-8 * (13.6 / kt) ** 0.5
    q_13 = k1 * gamma_13 * np.exp(-19.81 / kt) * u.cm ** 3 / u.s
    q_31a = k1 * gamma_31a / 3 * np.exp(-0.80 / kt) * u.cm ** 3 / u.s
    q_31b = k1 * gamma_31b / 3 * np.exp(-1.40 / kt) * u.cm ** 3 / u.s
    big_q_he = 1.75E-11 * (300 / temperature) ** 0.75 * \
        np.exp(-128E3 / temperature) * u.cm ** 3 / u.s
    big_q_he_plus = 1.25E-15 * (300 / temperature) ** (-0.25) * u.cm ** 3 / u.s

    return q_13, q_31a, q_31b, big_q_he, big_q_he_plus


# Fraction of helium in singlet and triplet vs. radius profile
def population_fraction(radius_profile, planet_radius, temperature,
                        h_he_fraction, mass_loss_rate, planet_mass,
                        spectrum_at_planet, hydrogen_ion_fraction,
                        initial_state=np.array([0.5, 0.0, 0.0, 0.0]),
                        **options_solve_ivp):
    """
    Calculate the fraction of helium in singlet and triplet state in the upper
    atmosphere in function of the radius in unit of planetary radius. The solver
    is already set to the method ``'Radau'``.

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

    spectrum_at_planet (``dict``):
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to populate the
        helium states (4.8 eV, or 2593 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``.

    hydrogen_ion_fraction (``numpy.ndarray``):
        Hydrogen ion fraction in the upper atmosphere in function of radius (can
        be calculated with ``hydrogen.ion_fraction()``).

    initial_state (``numpy.ndarray``, optional):
        The initial state is the `y0` of the differential equation to be solved.
        This array has two items: the initial value of `f_ion` (ionization
        fraction) and `tau` (optical depth) at the outer layer of the
        atmosphere. The standard value for this parameter is
        ``numpy.array([0.5, 0.0, 0.0, 0.0])``, i.e., 50% ionized at the  surface
        layer and with null optical depth.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``atol`` (absolute tolerance; default is
        1E-6) or ``rtol`` (relative tolerance; default is 1E-3). If you are
        having numerical issues, you may want to decrease the tolerance by a
        factor of 10 or 100, or 1000 in extreme cases. The option ``method`` is
        already set to ``'Radau'`` and cannot be changed.

    Returns
    -------
    f_1_r (``numpy.ndarray``):
        Fraction of helium in singlet state in function of radius.

    f_3_r (``numpy.ndarray``):
        Fraction of helium in triplet state in function of radius.

    tau_1_r (``numpy.ndarray``):
        Optical depth of helium in singlet state in function of radius.

    tau_3_r (``numpy.ndarray``):
        Optical depth of helium in triplet state in function of radius.
    """
    # Calculate sound speed, radius and density at the sonic point
    average_ion_fraction = np.mean(hydrogen_ion_fraction)
    vs = parker.sound_speed(temperature, h_he_fraction, average_ion_fraction
                            ).to(u.km / u.s)
    rs = parker.radius_sonic_point(planet_mass, vs).to(u.jupiterRad)
    rhos = parker.density_sonic_point(mass_loss_rate, rs, vs).to(
        u.g / u.cm ** 3)

    # Recombination rates of helium singlet and triplet in unit of rs ** 2 * vs
    alpha_rec_unit = (rs ** 2 * vs).to(u.cm ** 3 / u.s)
    alpha_rec_1, alpha_rec_3 = recombination(temperature)
    alpha_rec_1 = (alpha_rec_1 / alpha_rec_unit).decompose().value
    alpha_rec_3 = (alpha_rec_3 / alpha_rec_unit).decompose().value

    # Hydrogen mass in unit of rhos * rs ** 3
    m_h_unit = (rhos * rs ** 3).to(u.g)
    m_h = (c.m_p / m_h_unit).decompose().value

    # XXX Things start to get very complicated from here, so brace yourself.
    # There are lots of variables to keep track of, since the population of the
    # helium triplet and singlet depend on many processes, including whatever
    # happens with hydrogen as well.

    # Photoionization rates at null optical depth at the distance of the planet
    # from the host star, in unit of vs / rs, and the flux-averaged
    # cross-sections in units of rs ** 2
    phi_unit = (vs / rs).to(1 / u.s)
    phi_1, phi_3, a_1, a_3, a_h_1, a_h_3 = radiative_processes(
        spectrum_at_planet)
    phi_1 = (phi_1 / phi_unit).decompose().value
    phi_3 = (phi_3 / phi_unit).decompose().value
    a_1 = (a_1 / rs ** 2).decompose().value
    a_3 = (a_3 / rs ** 2).decompose().value
    a_h_1 = (a_h_1 / rs ** 2).decompose().value
    a_h_3 = (a_h_3 / rs ** 2).decompose().value

    # Collision-induced transition rates for helium triplet and singlet, in the
    # same unit as the recombination rates
    q_13, q_31a, q_31b, big_q_he, big_q_he_plus = collision(temperature)
    q_13 = (q_13 / alpha_rec_unit).decompose().value
    q_31a = (q_31a / alpha_rec_unit).decompose().value
    q_31b = (q_31b / alpha_rec_unit).decompose().value
    big_q_he = (big_q_he / alpha_rec_unit).decompose().value
    big_q_he_plus = (big_q_he_plus / alpha_rec_unit).decompose().value

    # Some hard-coding here. The numbers come from Oklopcic & Hirata (2018) and
    # Lampón et al. (2020).
    big_q_31 = (5E-10 * u.cm ** 3 / u.s / alpha_rec_unit).decompose().value
    big_a_31 = (1.272E-4 * u.cm ** 3 / u.s / alpha_rec_unit).decompose().value

    # Now let's solve the differential eq. 15 of Oklopcic & Hirata 2018

    # The radius in unit of radius at the sonic point
    _r = (radius_profile * planet_radius / rs).decompose().value
    _theta = np.flip(1 / _r)

    # The way we solve the differential equation requires us to pass the H ion
    # fraction at specific values of r, and it can be cumbersome to  parse this
    # inside the callable function _fun(). Instead, let's create a "mock
    # function" that returns the value of f_H_ion in function of r (essentially
    # a scipy.interp1d function)
    mock_f_h_ion_theta = interp1d(_theta, hydrogen_ion_fraction)

    # The differential equation
    def _fun(theta, y):
        f_1 = y[0]  # Fraction of helium in singlet
        f_3 = y[1]  # Fraction of helium in triplet
        t_1 = y[2]  # Optical depth for helium singlet
        t_3 = y[3]  # Optical depth for helium triplet
        velocity, rho = parker.structure(1 / theta)
        f_h_ion = mock_f_h_ion_theta(np.array([theta, ]))[0]  # Fraction of H+

        # Assume the number density of electrons is equal to the number density
        # of H ions
        k1 = h_he_fraction / (h_he_fraction + 4 * (1 - h_he_fraction)) / m_h
        k2 = (1 - h_he_fraction) / (h_he_fraction + 4 * (1 - h_he_fraction)) / \
            m_h
        n_e = k1 * rho * f_h_ion         # Number density of electrons
        n_h_plus = k1 * rho * f_h_ion    # Number density of ionized H
        n_h0 = k1 * rho * (1 - f_h_ion)  # Number density of atomic H
        n_he = k2 * rho * h_he_fraction  # Number density of helium nuclei

        # Terms of df1_dr
        term_11 = (1 - f_1 - f_3) * n_e * alpha_rec_1  # Recombination
        term_12 = f_3 * big_a_31  # Radiative transition rate
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

        # Finally assemble the equations for df3_dtheta and df3_dtheta
        df1_dtheta = (term_11 + term_12 - term_13 - term_14 + term_15 + term_16
                      + term_17 - term_18 + term_19) / velocity
        df3_dtheta = (term_31 - term_12 - term_33 + term_14 - term_15 - term_16
                      - term_17) / velocity

        # The other two differential equations in our system are the gradients
        # of the optical depth with 1 / radius, with the contribution of the
        # optical depth of H as well
        dt1_dtheta = a_1 * n_he * f_1 + a_h_1 * n_h0
        dt3_dtheta = a_3 * n_he * f_3 + a_h_3 * n_h0

        return np.array([df1_dtheta, df3_dtheta, dt1_dtheta, dt3_dtheta])

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (_theta[0], _theta[-1],), initial_state,
                    t_eval=_theta, method='Radau', **options_solve_ivp)

    # Finally retrieve the population fraction and optical depth arrays. Since
    # we integrated f and tau from the outside, we have to flip them back to the
    # same order as the radius variable
    f_1_r = np.flip(sol['y'][0])
    f_3_r = np.flip(sol['y'][1])
    tau_1_r = np.flip(sol['y'][2])
    tau_3_r = np.flip(sol['y'][3])

    return f_1_r, f_3_r, tau_1_r, tau_3_r
