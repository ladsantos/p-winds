#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from p_winds import hydrogen, helium, tools


# HD 209458 b
R_pl = (1.39 * u.jupiterRad).value
M_pl = (0.73 * u.jupiterMass).value
m_dot = (8E10 * u.g / u.s).value
T_0 = (9E3 * u.K).value
h_he = 0.9
average_f_ion = 0.7

r = np.linspace(1, 15, 500)
atol = 1E-8  # Absolute numerical tolerance for the solver
rtol = 1E-5  # Relative numerical tolerance

# In the initial state, the fraction of singlet and triplet helium is 1E-6, and
# the optical depths are null
initial_state = np.array([1.0, 0.0])


# Let's test if the code is producing reasonable outputs. The ``ion_fraction()``
# function for HD 209458 b should produce a profile with an ion fraction of
# approximately one near the planetary surface, and approximately 4E-4 in the
# outer layers.
def test_population_fraction_spectrum(precision_threshold=1E-4):
    units = {'wavelength': u.angstrom, 'flux': u.erg / u.s / u.cm ** 2 /
                                               u.angstrom}
    spectrum = tools.make_spectrum_from_file(
        '../data/solar_spectrum_scaled_lambda.dat', units)

    # First calculate the hydrogen ion fraction
    f_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl,
                                       average_f_ion,
                                       spectrum_at_planet=spectrum
                                       )

    # Now calculate the population of helium
    f_he_1, f_he_3 = helium.population_fraction(
        r, R_pl, T_0, h_he, m_dot, M_pl, f_r, spectrum_at_planet=spectrum,
        initial_state=initial_state, atol=atol, rtol=rtol, relax_solution=True
        )

    # assert abs(f_he_1[0] - 0.99983) / f_he_1[0] < precision_threshold
    # assert abs(f_he_3[0] - 1.2524E-9) / f_he_3[0] < precision_threshold


# # Now let's test ``ion_fraction()`` with a monochromatic flux instead of
# # spectrum.
# def test_population_fraction_mono(precision_threshold=1E-4):
#     flux_euv = 504 * u.erg / u.s / u.cm ** 2
#     flux_fuv = 1.7E5 * u.erg / u.s / u.cm ** 2
#
#     f_r, tau_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl,
#                                        average_f_ion, flux_euv=flux_euv
#                                        )
#
#     f_he_1, f_he_3, tau_he_1, tau_he_3 = helium.population_fraction(
#         r, R_pl, T_0, h_he, m_dot, M_pl, f_r, flux_euv=flux_euv,
#         flux_fuv=flux_fuv, initial_state=initial_state, atol=atol, rtol=rtol)
#
#     assert abs(f_he_1[0] - 0.9999) / f_he_1[0] < precision_threshold
#     assert abs(f_he_3[0] - 8.6353E-10) / f_he_3[0] < precision_threshold
