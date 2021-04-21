#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from p_winds import hydrogen, helium, tools, parker


# HD 209458 b
R_pl = (1.39 * u.jupiterRad).value
M_pl = (0.73 * u.jupiterMass).value
m_dot = (8E10 * u.g / u.s).value
T_0 = (9E3 * u.K).value
h_he = 0.9
average_f_ion = 0.7

vs = parker.sound_speed(T_0, h_he, average_f_ion)  # Speed of sound (km/s,
                                                   # assumed to be constant)
rs = parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
rhos = parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)

# Some useful arrays for the modeling
r_array = np.linspace(1, 15, 500) * R_pl / rs  # Radius in unit of radius at
                                               # sonic point
v_array, rho_array = parker.structure(r_array)

# In the initial state, the fraction of singlet and triplet helium is 1E-6, and
# the optical depths are null
initial_state = np.array([1.0, 0.0])
r = np.linspace(1, 15, 500)  # Radius in unit of planetary radii


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
    f_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl, average_f_ion,
                                spectrum_at_planet=spectrum,
                                relax_solution=False)

    atol = 1E-8  # Absolute numerical tolerance for the solver
    rtol = 1E-5  # Relative numerical tolerance

    # Now calculate the population of helium
    initial_state = np.array([1.0, 0.0])
    f_he_1, f_he_3 = helium.population_fraction(
        r, v_array, rho_array, f_r,
        R_pl, T_0, h_he, vs, rs, rhos, spectrum,
        initial_state=initial_state, atol=atol, rtol=rtol, relax_solution=False
        )

    assert abs(f_he_1[-1] - 0.0293064) / f_he_1[-1] < precision_threshold
    assert abs(f_he_3[-1] - 6.44021E-8) / f_he_3[-1] < precision_threshold
