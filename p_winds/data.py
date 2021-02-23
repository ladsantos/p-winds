#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains some useful hard-coded data for calculations in the other
modules.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np


# Helium 2^3S differential oscillator strength
def he_2_3_s_oscillator_strength():
    # Wavelength [angstrom]       Differential oscillator strength [ryd^(-1)]
    array = np.array([
        [2593.01,     0.605],
        [2528.27,     0.589],
        [2275.74,     0.537],
        [2023.15,     0.501],
        [1655.63,     0.435],
        [1214.41,     0.247],
        [958.87,     0.1572],
        [792.18,     0.1138],
        [674.86,     0.0780],
        [587.81,     0.0620],
        [520.65,     0.0557],
        [467.27,     0.0461],
        [423.81,     0.0358],
        [387.75,     0.0310],
        [357.34,     0.0325],
        [331.36,     0.0520],
        [271.94,     0.343],
        [271.21,     0.338],
        [256.70,     0.274],
        [243.01,     0.231],
        [230.71,     0.200],
        [219.59,     0.1750],
        [209.49,     0.1537]
    ])
    return array


# Collisional strengths for He
def he_collisional_strength():
    # log(T) [K]    gamma_13    gamma_31a   gamma_31b
    array = np.array([
        [3.75,            6.198E-2,    2.389,       7.965E-1],
        [4.00,            6.458E-2,    2.456,       9.579E-1],
        [4.25,            6.387E-2,    2.275,       1.042],
        [4.50,            6.157E-2,    1.916,       1.015],
        [4.75,            5.832E-2,    1.496,       8.950E-1],
        [5.00,            5.320E-2,    1.111,       7.265E-1],
        [5.25,            4.787E-2,    8.003E-1,    5.516E-1],
        [5.50,            4.018E-2,    5.660E-1,    3.948E-1],
        [5.75,            3.167E-2,    3.944E-1,    2.677E-1],
    ])
    return array
