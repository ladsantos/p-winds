#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from p_winds import hydrogen, tools


# HD 209458 b
R_pl = (1.39 * u.jupiterRad).value
M_pl = (0.73 * u.jupiterMass).value
m_dot = (8E10 * u.g / u.s).value
T_0 = (9E3 * u.K).value
h_he = 0.9
average_f_ion = 0.7

r = np.linspace(1, 15, 500)


# Let's test if the code is producing reasonable outputs. The ``ion_fraction()``
# function for HD 209458 b should produce a profile with an ion fraction of
# approximately one near the planetary surface, and approximately 4E-4 in the
# outer layers.
def test_ion_fraction_spectrum(precision_threshold=1E-5):
    units = {'wavelength': u.angstrom, 'flux': u.erg / u.s / u.cm ** 2 /
                                               u.angstrom}
    spectrum = tools.make_spectrum_from_file(
        '../data/solar_spectrum_scaled_lambda.dat', units)

    f_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl, average_f_ion,
                                spectrum_at_planet=spectrum,
                                relax_solution=False)
    assert abs((f_r[-1] - 0.99838) / f_r[-1]) < precision_threshold
