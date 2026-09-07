#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from p_winds import lines, transit


data_test_url = 'https://raw.githubusercontent.com/ladsantos/p-winds/main/data/he_3_profile.dat'


r_pl = 1.39 * 71492000  # Planet radius in m
r = np.loadtxt(data_test_url, usecols=(0,))  # Altitudes in m
v = np.loadtxt(data_test_url, usecols=(1,))  # Velocities in m / s
n_he_3 = np.loadtxt(data_test_url, usecols=(2, ))  # He fraction
planet_to_star_ratio = 0.12086
w0, w1, w2, f0, f1, f2, a_ij = lines.he_3_properties()
m_He = 4 * 1.67262192369e-27  # Helium atomic mass in kg
wl = np.linspace(1.0827, 1.0832, 150) * 1E-6  # Wavelengths in m
v_wind = -2E3  # Line-of-sight wind velocity in m / s
w_array = np.array([w0, w1, w2])
f_array = np.array([f0, f1, f2])
a_array = np.array([a_ij, a_ij, a_ij])
t_0 = 9E3  # Upper-atmospheric temperature in K


# Test both the draw_transit and radiative_transfer functions
def test_draw_transit_and_radiative_transfer(precision_threshold=5E-3):
    flux_map, t_depth, r_from_planet = transit.draw_transit(
        planet_to_star_ratio=planet_to_star_ratio,
        planet_physical_radius=r_pl,
        impact_parameter=0.0,
        phase=0.0,
        supersampling=10,
        grid_size=101
    )
    test_value_0 = flux_map[30, 30]
    assert abs((test_value_0 - 1.249219E-4) / test_value_0) < \
        precision_threshold
    assert abs((t_depth - 0.015012130070238605) / t_depth) < \
           precision_threshold

    spectrum = transit.radiative_transfer_2d(flux_map, r_from_planet,
                                             r, n_he_3, v, w_array, f_array,
                                             a_array, wl, t_0, m_He,
                                             bulk_los_velocity=v_wind,
                                             wind_broadening_method='average')
    test_value_2 = spectrum[60]
    assert abs((test_value_2 - 0.97946) / test_value_2) < precision_threshold


@pytest.mark.parametrize('z_grid_size', [0, 1, 23])
def test_average_and_formal_optical_depth_agree_without_wind_broadening(z_grid_size):
    radius = r[::25]
    # With zero wind and uniform temperature, both methods use the same Voigt
    # profile. Empty integrals must also agree, even if the average is undefined.
    args = (radius, n_he_3[::25], np.zeros_like(radius), w_array, f_array,
            a_array, wl[::10], t_0, m_He, z_grid_size)
    with np.errstate(invalid='ignore'):
        average = transit.optical_depth_2d(*args, wind_broadening_method='average')
        formal = transit.optical_depth_2d(*args, wind_broadening_method='formal')
    np.testing.assert_allclose(average, formal, rtol=1e-14, atol=0.0,
                               equal_nan=False)
