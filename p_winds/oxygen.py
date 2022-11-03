#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of O in the upper
atmosphere.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simps, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.special import exp1
from p_winds import tools, microphysics
import warnings


__all__ = []


# Some hard coding based on the astrophysical literature
_SOLAR_OXYGEN_ABUNDANCE_ = 8.69  # Asplund et al. 2009
_SOLAR_OXYGEN_FRACTION_ = 10 ** (_SOLAR_OXYGEN_ABUNDANCE_ - 12.00)


# Photoionization of O I (neutral) and O II (singly-ionized)
def radiative_processes(spectrum_at_planet):
    pass


# Recombination of singly-ionized O into neutral O
def recombination(electron_temperature):
    pass


# Calculation the number fractions of O I and O II
def ion_fraction():
    pass
