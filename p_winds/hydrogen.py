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
from scipy.integrate import simps, solve_ivp, cumtrapz
from p_winds import parker, tools, microphysics


__all__ = ["radiative_processes_exact", "radiative_processes",
           "radiative_processes_mono", "recombination", "ion_fraction"]


# Exact calculation of hydrogen photoionization
def radiative_processes_exact(spectrum_at_planet, r_grid, density, f_h_r,
                              h_fraction, f_he_r=None):
    """
    Calculate the photoionization rate of hydrogen as a function of radius based
    on the EUV spectrum arriving at the planet and the neutral H density
    profile.

    Parameters
    ----------
    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom).
    
    r_grid : ``numpy.ndarray``
        Radius grid for the calculation, in units of cm.

    density : ``numpy.ndarray``
        Total density profile for the atmosphere, in units of g / cm ** 3.

    f_h_r : ``numpy.ndarray`` or ``float``
        H ion fraction profile for the atmosphere.

    h_fraction : ``float``
        Hydrogen number fraction of the outflow.

    f_he_r : ``numpy.ndarray`` or ``float`` or ``None``
        He ion fraction profile for the atmosphere. If ``None``, then assume
        that the profile is the same as ``f_h_r``.

    Returns
    -------
    phi_prime : ``float``
        Ionization rate of hydrogen for each point on r_grid in unit of 1 / s.
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

    # 2d grid of radius and wavelength
    xx, yy = np.meshgrid(wavelength_cut, r_grid)
    # Photoionization cross-section in function of wavelength
    a_lambda = microphysics.hydrogen_cross_section(wavelength=xx)

    # Optical depth to hydrogen photoionization
    m_h = 1.67262192E-24  # Proton mass in unit of g
    r_grid_temp = r_grid[::-1]
    # We assume that the atmosphere is made of only H + He
    he_fraction = 1 - h_fraction
    f_he_to_h = he_fraction / h_fraction
    mu = (1 + 4 * f_he_to_h) / (1 + f_h_r + f_he_to_h)

    n_tot = density / mu / m_h
    n_htot = 1 / (1 + f_h_r + f_he_to_h) * n_tot
    n_h = n_htot * (1 - f_h_r)
    n_hetot = n_htot * f_he_to_h

    if f_he_r is None:
        n_he = n_hetot * (1 - f_h_r)  # Here we assume that the ion fraction of
        # He is the same as H, which may not always be correct
    else:
        n_he = n_hetot * (1 - f_he_r)  # This is more correct

    n_h_temp = n_h[::-1]
    column_h = cumtrapz(n_h_temp, r_grid_temp, initial=0)
    column_density_h = -column_h[::-1]
    tau_rnu = column_density_h[:, None] * a_lambda

    # Optical depth to helium photoionization
    n_he_temp = n_he[::-1]
    column_he = cumtrapz(n_he_temp, r_grid_temp, initial=0)
    column_density_he = -column_he[::-1]
    a_lambda_he = microphysics.helium_total_cross_section(wavelength=xx)
    tau_rnu += column_density_he[:, None] * a_lambda_he

    # Finally calculate the photoionization rate
    phi_prime = abs(simps(flux_lambda_cut * a_lambda / energy_cut *
                    np.exp(-tau_rnu), wavelength_cut, axis=-1))

    return phi_prime


# Stellar flux-average calculation of hydrogen photoionization
def radiative_processes(spectrum_at_planet):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the EUV spectrum arriving at the planet.

    Parameters
    ----------
    spectrum_at_planet : ``dict``
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom).

    Returns
    -------
    phi : ``float``
        Ionization rate of hydrogen at null optical depth in unit of 1 / s.

    a_0 : ``float``
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
def radiative_processes_mono(flux_euv, average_photon_energy=20.):
    """
    Calculate the photoionization rate of hydrogen at null optical depth based
    on the monochromatic EUV flux arriving at the planet.

    Parameters
    ----------
    flux_euv : ``float``
        Monochromatic extreme-ultraviolet (0 - 912 Angstrom) flux arriving at
        the planet in unit of erg / s / cm ** 2.

    average_photon_energy : ``float``, optional
        Average energy of the photons ionizing H in unit of eV. Default is 20 eV
        (as in Murray-Clay et al 2009, Allan & Vidotto 2019).

    Returns
    -------
    phi : ``float``
        Ionization rate of hydrogen at null optical depth in unit of 1 / s.

    a_0 : ``float``
        Flux-averaged photoionization cross-section of hydrogen in unit of
        cm ** 2.
    """
    # Average cross-section
    a_0 = 6.3E-18 * (average_photon_energy / 13.6) ** (-3)  # Unit 1 / cm ** 2.

    # Monochromatic ionization rate
    flux_euv *= 6.24150907E+11  # Convert erg to eV
    phi = flux_euv * a_0 / average_photon_energy
    return phi, a_0


# Case-B hydrogen recombination
def recombination(temperature):
    """
    Calculates the case-B hydrogen recombination rate for a gas at a certain
    temperature.

    Parameters
    ----------
    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    Returns
    -------
    alpha_rec : ``float``
        Recombination rate of hydrogen in units of cm ** 3 / s.
    """
    alpha_rec = 2.59E-13 * (temperature / 1E4) ** (-0.7)
    return alpha_rec


# Fraction of ionized hydrogen vs. radius profile
def ion_fraction(radius_profile, planet_radius, temperature, h_fraction,
                 mass_loss_rate, planet_mass, mean_molecular_weight_0=1.0,
                 star_mass=1.0, semimajor_axis=1.0, spectrum_at_planet=None,
                 flux_euv=None, initial_f_ion=0.0, relax_solution=False,
                 convergence=0.01, max_n_relax=10, exact_phi=False,
                 return_mu=False, return_rates=False, **options_solve_ivp):
    """
    Calculate the fraction of ionized hydrogen in the upper atmosphere in
    function of the radius in unit of planetary radius.

    Parameters
    ----------
    radius_profile : ``numpy.ndarray``
        Radius in unit of planetary radii.

    planet_radius : ``float``
        Planetary radius in unit of Jupiter radius.

    temperature : ``float``
        Isothermal temperature of the upper atmosphere in unit of Kelvin.

    h_fraction : ``float``
        Total (ion + neutral) H number fraction of the atmosphere.

    mass_loss_rate : ``float``
        Mass loss rate of the planet in units of g / s.

    planet_mass : ``float``
        Planetary mass in unit of Jupiter mass.

    mean_molecular_weight_0 : ``float``
        Initial mean molecular weight of the atmosphere in unit of proton mass.
        Default value is 1.0 (100% neutral H). Since its final value depend on
        the H ion fraction itself, the mean molecular weight can be
        self-consistently calculated by setting `relax_solution` to `True`.

    star_mass : ``float``, optional
        Stellar mass in units of M_sun, needed for the tidal gravity
        calculation. Default is 1.

    semimajor_axis : ``float``, optional
        Planetary semimajor axis in units of au, needed for the tidal gravity
        calculation. Default is 1 (so the tidal gravity correction is minimal by
        default).

    spectrum_at_planet : ``dict``, optional
        Spectrum of the host star arriving at the planet covering fluxes at
        least up to the wavelength corresponding to the energy to ionize
        hydrogen (13.6 eV, or 911.65 Angstrom). Can be generated using
        ``tools.make_spectrum_dict``. If ``None``, then ``flux_euv`` must be
        provided instead. Default is ``None``.

    flux_euv : ``float``, optional
        Extreme-ultraviolet (0-911.65 Angstrom) flux arriving at the planet in
        units of erg / s / cm ** 2. If ``None``, then ``spectrum_at_planet``
        must be provided instead. Default is ``None``.

    initial_f_ion : ``float``, optional
        The initial ionization fraction at the layer near the surface of the
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

    exact_phi : ``bool``, optional
        If set to ``True``, the H photoionization is calculated exactly (using
        the ``radiative_processes_exact()`` function). If set to ``False``, then
        calculate it using an approximation with ``radiative_processes()``.
        Default value is ``False``.

    return_mu : ``bool``, optional
        If ``True``, then this function returns a second variable ``mu_bar``,
        which is the self-consistent, density-averaged mean molecular weight of
        the atmosphere. Equivalent to the ``mu_bar`` of Eq. A.3 in Lampón et
        al. 2020.

    return_rates : ``bool``, optional
        If ``True``, then this function also returns a ``dict`` object
        containing the rates of photoionization and recombination in function of
        radius and in units of 1 / s. Default is ``False``.

    **options_solve_ivp:
        Options to be passed to the ``scipy.integrate.solve_ivp()`` solver. You
        may want to change the options ``method`` (integration method; default
        is ``'RK45'``), ``atol`` (absolute tolerance; default is 1E-6) or
        ``rtol`` (relative tolerance; default is 1E-3). If you are having
        numerical issues, you may want to decrease the tolerance by a factor of
        10 or 100, or 1000 in extreme cases.

    Returns
    -------
    f_r : ``numpy.ndarray``
        Values of the fraction of ionized hydrogen in function of the radius.

    mu_bar : ``float``
        Mean molecular weight of the atmosphere, in unit of proton mass,
        averaged across the radial distance using according to the function
        `average_molecular_weight` in the `parker` module. Only returned when
        ``return_mu`` is set to ``True``.

    rates : ``dict``
        Dictionary containing the rates of photoionization and recombination in
        function of radius and in units of 1 / s. Only returned when
        ``return_rates`` is set to ``True``.
    """
    # Hydrogen recombination rate
    alpha_rec = recombination(temperature)

    # Hydrogen mass in g
    m_h = 1.67262192E-24

    # Photoionization rate at null optical depth at the distance of the planet
    # from the host star, in unit of 1 / s.
    vs = parker.sound_speed(temperature, mean_molecular_weight_0)
    rs = parker.radius_sonic_point_tidal(planet_mass, vs, star_mass,
                                         semimajor_axis)
    if exact_phi and spectrum_at_planet is not None:
        rhos = parker.density_sonic_point(mass_loss_rate, rs, vs)
        _, rho_norm = parker.structure_tidal(
            radius_profile * planet_radius / rs, vs, rs, planet_mass,
            star_mass, semimajor_axis)
        f_outer = 0.0  # Assume completely ionized at the top of atmosphere
        phi_abs = radiative_processes_exact(
            spectrum_at_planet,
            (radius_profile * planet_radius * u.Rjup).to(u.cm).value,
            rho_norm * rhos, f_outer, h_fraction)
        a_0 = 0.
    elif spectrum_at_planet is not None:
        phi_abs, a_0 = radiative_processes(spectrum_at_planet)
    elif flux_euv is not None:
        phi_abs, a_0 = radiative_processes_mono(flux_euv)
    else:
        raise ValueError('Either `spectrum_at_planet` or `flux_euv` must be '
                         'provided.')

    # Multiplicative factor of Eq. 11 of Oklopcic & Hirata 2018, unit of
    # cm ** 2 / g
    # We assume that the remaining of the number fraction is pure He
    he_fraction = 1 - h_fraction
    he_h_fraction = he_fraction / h_fraction
    k1_abs = h_fraction * a_0 / (h_fraction + 4 * he_fraction) / m_h

    # Multiplicative factor of the second term in the right-hand side of Eq.
    # 13 of Oklopcic & Hirata 2018, unit of cm ** 3 / s / g
    k2_abs = h_fraction / (h_fraction + 4 * he_fraction) * alpha_rec / m_h

    # In order to avoid numerical overflows, we need to normalize a few key
    # variables. Since the normalization may need to be repeated to relax the
    # solution, we have a function to do it.
    def _normalize(_phi, _k1, _k2, _r, _mu):
        # First calculate the sound speed, radius at the sonic point and the
        # density at the sonic point. They will be useful to change the units of
        # the calculation aiming to avoid numerical overflows
        _vs = parker.sound_speed(temperature, _mu)
        _rs = parker.radius_sonic_point_tidal(planet_mass, _vs, star_mass,
                                              semimajor_axis)
        _rhos = parker.density_sonic_point(mass_loss_rate, _rs, _vs)
        # And now normalize everything
        phi_unit = _vs * 1E5 / _rs / 7.1492E+09  # 1 / s
        phi_norm = _phi / phi_unit
        k1_unit = 1 / (_rhos * _rs * 7.1492E+09)  # cm ** 2 / g
        k1_norm = _k1 / k1_unit
        k2_unit = _vs * 1E5 / _rs / 7.1492E+09 / _rhos  # cm ** 3 / g / s
        k2_norm = _k2 / k2_unit
        r_norm = (_r * planet_radius / _rs)

        # The differential r will be useful at some point
        dr_norm = np.diff(r_norm)
        dr_norm = np.concatenate((dr_norm, np.array([dr_norm[-1], ])))

        # The structure of the atmosphere
        v_norm, rho_norm = parker.structure_tidal(r_norm, _vs, _rs, planet_mass,
                                                  star_mass, semimajor_axis)

        return phi_norm, k1_norm, k2_norm, r_norm, dr_norm, v_norm, rho_norm

    phi, k1, k2, r, dr, velocity, density = _normalize(
        phi_abs, k1_abs, k2_abs, radius_profile, mean_molecular_weight_0)

    if exact_phi is False:
        # To start the calculations we need the optical depth, but technically
        # we don't know it yet, because it depends on the ion fraction in the
        # atmosphere, which is what we want to obtain. However, the optical
        # depth depends more strongly on the densities of H than the ion
        # fraction, so a good first approximation is to assume the whole
        # atmosphere is neutral at first.
        column_density = np.flip(np.cumsum(np.flip(dr * density)))
        tau = k1 * column_density
    else:
        pass

    # Now let's solve the differential eq. 13 of Oklopcic & Hirata 2018
    # The differential equation in function of r
    def _fun(_r, _f, _phi, _k2):
        if exact_phi:
            _phi_prime = np.interp(_r, r, phi)
        else:
            _t = np.interp(_r, r, tau)
            _phi_prime = np.exp(-_t) * _phi

        # The next two lines may need to be substituted by `structure_tidal()`
        # instead of interpolated
        _v = np.interp(_r, r, velocity)
        _rho = np.interp(_r, r, density)

        # In terms 1 and 2 we use the values of k2 and phi from above
        term1 = (1. - _f) / _v * _phi_prime
        term2 = _k2 * _rho * _f ** 2 / _v
        df_dr = term1 - term2

        return df_dr

    # We solve it using `scipy.solve_ivp`
    sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                    t_eval=r, args=(phi, k2,), 
                    **options_solve_ivp)
    f_r = sol['y'][0]

    # When `solve_ivp` has problems, it may return an array with different
    # size than `r`. So we raise an exception if this happens
    if len(f_r) != len(r):
        raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                           ' solution.')

    # Calculate the average mean molecular weight using Eq. A.3 from Lampón et
    # al. 2020
    mu_bar = parker.average_molecular_weight(f_r,
                                             radius_profile * planet_radius,
                                             velocity * vs,
                                             planet_mass, temperature,
                                             he_h_fraction)

    # For the sake of self-consistency, there is the option of repeating the
    # calculation of f_r by updating the optical depth with the new ion
    # fractions.
    if relax_solution is True:
        for i in range(max_n_relax):
            previous_f_r = np.copy(f_r)
            
            if exact_phi:
                # phi_abs will need to be recomputed here with the new density
                # structure
                vs = parker.sound_speed(temperature, mu_bar)
                rs = parker.radius_sonic_point_tidal(planet_mass, vs, star_mass,
                                                     semimajor_axis)
                rhos = parker.density_sonic_point(mass_loss_rate, rs, vs)
                _, rho_norm = parker.structure_tidal(
                    radius_profile * planet_radius / rs, vs, rs, planet_mass,
                    star_mass, semimajor_axis)
                phi_abs = radiative_processes_exact(
                    spectrum_at_planet,
                    (radius_profile * planet_radius * u.Rjup).to(u.cm).value,
                    rho_norm * rhos, f_r, h_fraction)

            # We re-normalize key parameters because the newly-calculated f_ion
            # changes the value of the mean molecular weight of the atmosphere
            phi, k1, k2, r, dr, velocity, density = _normalize(
                phi_abs, k1_abs, k2_abs, radius_profile, mu_bar)

            if exact_phi is False:
                # Re-calculate the column densities
                column_density = np.flip(np.cumsum(np.flip(dr * density *
                                                           (1 - f_r))))
                tau = k1 * column_density
            
            # And solve it again
            sol = solve_ivp(_fun, (r[0], r[-1],), np.array([initial_f_ion, ]),
                            t_eval=r, args=(phi, k2,), 
                            **options_solve_ivp)
            f_r = sol['y'][0]

            # Raise an error if the length of `f_r` is different from the length
            # of `r`
            if len(f_r) != len(r):
                raise RuntimeError('The solver ``solve_ivp`` failed to obtain a'
                                   ' solution.')

            # Here we update the average mean molecular weight
            mu_bar = parker.average_molecular_weight(
                f_r, radius_profile * planet_radius, velocity * vs, planet_mass,
                temperature, he_h_fraction
            )

            # Calculate the relative change of f_ion in the outer shell of the
            # atmosphere (where we expect the most important change)
            # relative_delta_f = abs(f_r[-1] - previous_f_r_outer_layer) \
            #     / previous_f_r_outer_layer
            relative_delta_f = abs(
                np.sum(f_r - previous_f_r) / np.sum(previous_f_r))

            # Break the loop if convergence is achieved
            if relative_delta_f < convergence:
                break
            else:
                pass
    else:
        pass

    # Calculate the final structure and rates of photoionization and
    # recombination in function of radius
    if return_rates is True:
        # Final photoionization rate in unit of 1 / s
        final_phi_prime = radiative_processes_exact(
            spectrum_at_planet,
            (radius_profile * planet_radius * u.Rjup).to(u.cm).value,
            rho_norm * rhos, f_r, h_fraction) * (1 - f_r)
        # Final sound speed in km / s
        final_vs = parker.sound_speed(temperature, mu_bar)
        # Final radius at the sonic point in units of Jupiter radii
        final_rs = parker.radius_sonic_point_tidal(planet_mass, final_vs,
                                                   star_mass, semimajor_axis)
        # Final density at the sonic point in unit of g / cm ** 3
        final_rhos = parker.density_sonic_point(mass_loss_rate, final_rs,
                                                final_vs)
        # Final velocity and density profiles in units of sonic point
        final_r_norm = (radius_profile * planet_radius / final_rs)
        final_v_norm, final_rho_norm = parker.structure_tidal(
            final_r_norm, final_vs, final_rs, planet_mass, star_mass,
            semimajor_axis)
        # Final density profile in g / cm ** 3
        final_rho = final_rho_norm * final_rhos
        # Final recombination rate in 1 / s
        final_alpha_rec = k2_abs * final_rho * f_r ** 2
        rates = {'photoionization': final_phi_prime,
                 'recombination': final_alpha_rec}
    else:
        pass

    if return_mu is True and return_rates is False:
        return f_r, mu_bar
    elif return_mu is False and return_rates is True:
        return f_r, rates
    elif return_mu is True and return_rates is True:
        return f_r, mu_bar, rates
    else:
        return f_r
        
def calc_boltzmann_distribution(T, n, NLTE_scaling = 1.):
    """
    Calculates the distribution of electronic states of the different atomic shells
    of the hydrogen atom via the Boltzmann equation in LTE and NLTE.
    In short, it calculates which fraction of the total number of H atoms are in the shell needed to produce H-alpha, H-beta etc lines.

    Parameters
    ----------
    T: ``float``
        Temperature in K
    
    n : ``integer``
        For which shell number do we calculate the Boltzmann distribution

    NLTE_scaling : ``float``
        Default: 1. (LTE, no scaling)
        If NLTE is assumed, the scaling factor can be a free fitting parameter in the retrieval on the data

    Returns
    -------
    boltzmann_n : ``float``
        Unitless fraction of Hydrogen population in the needed atomic shell for the specific Balmer-line

    """
        
    # Energy of the nth state in the Balmer series as a difference to the ground state
    #ground state energy for hydrogen is 13.6 eV
    #E_n = E1 - E(n) in our case with E(n) = E1/n**2 for the Balmer series
    E_n = 13.6 * (1-1/n**2) * u.eV
    
    # Population distribution via the Boltzmann equation
    #This is the limit case for a large population of atoms as applicable in atmospheres
    #Boltzmann_n = g_n/g_1 * np.exp(-E_n / (const.k_B * T))
    #g_1 in our case is a constant and always 2
    #g_n are the statistical weights of the electronic levels due to their degeneracies
    #to generalise this function for other elements, simply let the user provide these as input, as well as the energy difference
    g_n = 2.*n**2.
    # Partition function for hydrogen
    #is the sum over all quantum states for the atom where the electron could be, thus the sum over g_n*exp(-(E1-En)/(kb*T)
    # in the case of hydrogen any contributions beyond n=2 are negligible
    U_i = 2.*np.exp(-13.6 * u.eV / (c.k_B * T * u.K)) + 8.*np.exp(-13.6 * u.eV / (2**2 * c.k_B * T * u.K))
    
    #The NLTE scaling is defined as NLTE_scaling = Boltzmann_n (NLTE) / Boltzmann_n (LTE)
    #from Barman et al. 2002 and others
    #The scaling factor can be assumed as a constant for thermospheres
    #from Huang et al. 2017, and Garcia Munoz and Schneider (2019)
    
    #calculate fraction of level population
    
    boltzmann_n = NLTE_scaling * (g_n/U_i) * np.exp(-E_n / (c.k_B * T * u.K)).decompose().value
    

    return boltzmann_n

def calc_saha_distribution(T, electron_density):
    """
    Calculate the distribution of ionization states for the Balmer series of hydrogen
    using the Saha equation.

    Parameters
    ----------
    T: ``float``
        Temperature in K
        
    electron_density: ``float``
        the electron density (most likely coming from the background, but tbd). If only one level of ionisation is important then n1 = ne
        The electron density has to have units of (cm**-3)
    
    n : ``integer``
        Number of the atomic shell

    Returns
    -------
    saha_n : ``float``
        Unitless fraction of ionised Hydrogen from total Hydrogen for the specific Balmer-line
    """

    
    # Boltzmann constant in eV/K
#    k_B = c.k_B.to(u.eV / u.K)

    # Ionization potential of hydrogen (eV)
    ionization_potential = 13.6 * u.eV

    # Partition function for hydrogen
    #is the sum over all quantum states for the atom where the electron could be, thus the sum over g_n*exp(-(E1-En)/(kb*T)
    # in the case of hydrogen any contributions beyond n=2 are negligible
    U_i = 0.5 #ZII = 1 and ZI = g1 = 2
    
    #thermal deBroglie wavelength ** (-1)
    debroglie_rev = np.sqrt(2 * np.pi * c.m_e * c.k_B * T * u.K / (c.h**2))

    # Population distribution
    #in reality the equation is not divided by U_i but multliplied by U_i(H II)/U_i(H I)
    #For the Balmer series, H II is the naked proton core, as hydrogen only has one electron. Therefore, U_i(H II) = 1
    saha_n = 2./(electron_density * U_i )* (debroglie_rev)**(3) * np.exp(-ionization_potential / (c.k_B * T * u.K))
  
    return saha_n
    
def relation_boltzmann_saha(T, n, electron_density, NLTE_scaling = 1.):
    """
    Calculates the fraction with relation to the total number of atoms taking into account Boltzmann and Saha

    Parameters
    ----------
    T: ``float``
        Temperature in K
        
    electron_density: ``float``
        the electron density (most likely coming from the background, but tbd). If only one level of ionisation is important then n1 = ne
        The electron density has to have units of (cm**-3)
    
    n : ``integer``
        Number of the atomic shell

    Returns
    -------
    total_fraction : ``float``
        Unitless fraction of available Hydrogen from total Hydrogen for the specific Balmer-line
    """

    boltzmann_frac = calc_boltzmann_distribution(T, n, NLTE_scaling)
    saha_frac = calc_saha_distribution(T, electron_density)
    #relates the boltzmann and saha distribution together via the total number of hydrogen atoms
    total_fraction = boltzmann_frac/((1+boltzmann_frac)*(1+saha_frac)) #1+boltzmann_frac_base+boltzmann_frac)*
    return total_fraction
    
def halpha_scale(T_0,n_e, g_factor, n_shell):
    """
    Calculates the scaling factor of the hydrogen density for the Balmer lines after Wyttenbach et al. 2020 eq. 9

    Parameters
    ----------
    T_0: ``float``
        Temperature in K
        
    n_e: ``float``
        the electron density (most likely coming from the background, but tbd). If only one level of ionisation is important then n1 = ne
        The electron density has to have units of (cm**-3)
        
    g_factor: ''float''
        the g scaling factor of the line in question from lines.py
    
    n_shell : ``integer``
        Number of the atomic shell

    Returns
    -------
        n_H_2n_scale : ``float``
        Unitless fraction to adjust the hydrogen density for the specific Balmer-line
    """

    frac_n = relation_boltzmann_saha(T_0, n_shell, n_e)

    frac_ground = relation_boltzmann_saha(T_0, 2, n_e)
    
    g_n = 2 * n_shell ** 2
    g_2 = 2 * 2 ** 2
    
    n_H_2n_scale = g_shell * (frac_ground/g_2 - frac_n/g_n)

    return n_H_2n_scale


