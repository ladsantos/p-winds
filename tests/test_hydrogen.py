#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from p_winds import hydrogen, tools


# HD 209458 b
R_pl = 1.39 * u.jupiterRad
M_pl = 0.73 * u.jupiterMass
m_dot = 8E10 * u.g / u.s
T_0 = 9E3 * u.K
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

    f_r, tau_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl,
                                       average_f_ion,
                                       spectrum_at_planet=spectrum
                                       )
    assert abs(f_r[-1] - 1.0) < precision_threshold
    assert abs(f_r[0] - 4.1E-4) < precision_threshold


# Now let's test ``ion_fraction()`` with a monochromatic flux instead of
# spectrum.
def test_ion_fraction_mono(precision_threshold=1E-5):
    flux_euv = 1000 * u.erg / u.s / u.cm ** 2

    r = np.linspace(1, 15, 500)
    f_r, tau_r = hydrogen.ion_fraction(r, R_pl, T_0, h_he, m_dot, M_pl,
                                       average_f_ion, flux_euv=flux_euv,
                                       initial_state=np.array([0.0, 1E3]))
    assert abs(f_r[-1] - 1.0) < precision_threshold
    assert abs(f_r[0] - 4.1E-4) < precision_threshold
