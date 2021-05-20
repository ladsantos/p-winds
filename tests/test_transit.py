#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from p_winds import microphysics, transit


data_test_url = 'https://raw.githubusercontent.com/ladsantos/p-winds/main/data/he_3_profile.dat'


r_pl = 1.39 * 71492000  # Planet radius in m
r = np.loadtxt(data_test_url, usecols=(0,))  # Altitudes in m
n_he_3 = np.loadtxt(data_test_url, usecols=(1,))  # He fraction
planet_to_star_ratio = 0.12086
w0, w1, w2, f0, f1, f2, a_ij = microphysics.he_3_properties()
m_He = 4 * 1.67262192369e-27  # Helium atomic mass in kg
wl = np.linspace(1.0827, 1.0832, 1000) * 1E-6  # Wavelengths in m
v_wind = -2E3  # Line-of-sight wind velocity in m / s
w_array = np.array([w0, w1, w2])
f_array = np.array([f0, f1, f2])
a_array = np.array([a_ij, a_ij, a_ij])
t_0 = 9E3  # Upper-atmospheric temperature in K


# Test both the draw_transit and radiative_transfer functions
def test_draw_transit_and_radiative_transfer(precision_threshold=5E-3):
    flux_map, t_depth, density_map = transit.draw_transit(
        planet_to_star_ratio,
        impact_parameter=0.0,
        phase=0.0,
        supersampling=10,
        density_profile=n_he_3,
        profile_radius=r,
        planet_physical_radius=r_pl,
        grid_size=101
    )
    test_value_0 = flux_map[30, 30]
    test_value_1 = density_map[30, 30]
    assert abs((test_value_0 - 1.249219E-4) / test_value_0) < \
        precision_threshold
    assert abs((test_value_1 - 1.1998E14) / test_value_1) < \
           precision_threshold

    spectrum = transit.radiative_transfer(flux_map, density_map,
                                          wl, w_array, f_array, a_array, t_0,
                                          m_He, v_wind)
    test_value_2 = spectrum[600]
    assert abs((test_value_2 - 0.96657) / test_value_2) < precision_threshold
