#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from p_winds import hydrogen, helium, tools, parker


# HD 209458 b
R_pl = (1.39 * u.jupiterRad).value
M_pl = (0.73 * u.jupiterMass).value
m_dot = (5E10 * u.g / u.s).value
T_0 = (9E3 * u.K).value
h_fraction = 0.90
he_fraction = 1 - h_fraction
he_h_fraction = he_fraction / h_fraction
average_f_ion = 0.0
average_mu = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + average_f_ion)

# In the initial state, the fraction of singlet and triplet helium is 1E-6, and
# the optical depths are null
initial_state = np.array([1.0, 0.0])
r = np.logspace(0, np.log10(20), 100)  # Radius in unit of planetary radii
data_test_url = 'https://raw.githubusercontent.com/ladsantos/p-winds/main/data/solar_spectrum_scaled_lambda.dat'


# Let's test if the code is producing reasonable outputs.
def test_population_fraction_spectrum():
    units = {'wavelength': u.angstrom, 'flux': u.erg / u.s / u.cm ** 2 /
                                               u.angstrom}
    spectrum = tools.make_spectrum_from_file(data_test_url, units)

    # First calculate the hydrogen ion fraction
    f_r, mu_bar = hydrogen.ion_fraction(r, R_pl, T_0, h_fraction, m_dot, M_pl,
                                        average_mu, spectrum_at_planet=spectrum,
                                        relax_solution=True, exact_phi=True,
                                        return_mu=True)

    # Calculate the structure
    vs = parker.sound_speed(T_0, mu_bar)  # Speed of sound (km/s, assumed to be
    # constant)
    rs = parker.radius_sonic_point(M_pl,
                                   vs)  # Radius at the sonic point (jupiterRad)
    rhos = parker.density_sonic_point(m_dot, rs,
                                      vs)  # Density at the sonic point (g/cm^3)

    # Some useful arrays for the modeling
    r_array = r * R_pl / rs  # Radius in unit of radius at
    # sonic point
    v_array, rho_array = parker.structure(r_array)

    # Now calculate the population of helium
    f_he_1_odeint, f_he_3_odeint, rates = helium.population_fraction(
        r, v_array, rho_array, f_r,
        R_pl, T_0, h_fraction, vs, rs, rhos, spectrum_at_planet=spectrum,
        initial_state=initial_state, relax_solution=True, return_rates=True)

    # Assert if all values of the fractions are between 0 and 1
    n_neg = len(np.where(f_he_1_odeint < 0)[0]) + \
        len(np.where(f_he_3_odeint < 0)[0])
    n_one = len(np.where(f_he_1_odeint > 1)[0]) + \
        len(np.where(f_he_3_odeint > 1)[0])
    assert n_neg == 0
    assert n_one == 0

    f_he_1_ivp, f_he_3_ivp = helium.population_fraction(
        r, v_array, rho_array, f_r,
        R_pl, T_0, h_fraction, vs, rs, rhos, spectrum_at_planet=spectrum,
        initial_state=initial_state, relax_solution=True, method='Radau',
        atol=1E-8, rtol=1E-8
    )

    # Assert if all values of the fractions are between 0 and 1
    n_neg = len(np.where(f_he_1_ivp < 0)[0]) + \
        len(np.where(f_he_3_ivp < 0)[0])
    n_one = len(np.where(f_he_1_ivp > 1)[0]) + \
        len(np.where(f_he_3_ivp > 1)[0])
    assert n_neg == 0
    assert n_one == 0


# Let's test the code for ionization of He
def test_ion_fraction():
    units = {'wavelength': u.angstrom, 'flux': u.erg / u.s / u.cm ** 2 /
                                               u.angstrom}
    spectrum = tools.make_spectrum_from_file(data_test_url, units)

    # First calculate the hydrogen ion fraction
    f_r, mu_bar = hydrogen.ion_fraction(r, R_pl, T_0, h_fraction, m_dot, M_pl,
                                        average_mu, spectrum_at_planet=spectrum,
                                        relax_solution=True, exact_phi=True,
                                        return_mu=True)

    # Calculate the structure
    vs = parker.sound_speed(T_0, mu_bar)  # Speed of sound (km/s, assumed to be
    # constant)
    rs = parker.radius_sonic_point(M_pl,
                                   vs)  # Radius at the sonic point (jupiterRad)
    rhos = parker.density_sonic_point(m_dot, rs,
                                      vs)  # Density at the sonic point (g/cm^3)

    # Some useful arrays for the modeling
    r_array = r * R_pl / rs  # Radius in unit of radius at
    # sonic point
    v_array, rho_array = parker.structure(r_array)

    # Now calculate the population of helium
    f_ion = helium.ion_fraction(
        r, v_array, rho_array, f_r,
        R_pl, T_0, h_fraction, vs, rs, rhos, spectrum_at_planet=spectrum,
        initial_f_he_ion=0.0, relax_solution=True)

    # Assert if all values of the fractions are between 0 and 1
    n_neg = len(np.where(f_ion < 0)[0])
    n_one = len(np.where(f_ion > 1)[0])
    assert n_neg == 0
    assert n_one == 0
