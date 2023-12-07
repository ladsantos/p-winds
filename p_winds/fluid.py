#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes a planetary outflow model using fluid dynamics.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy.optimize as so
from scipy.integrate import simps, trapz
from astropy import units as u, constants as c
from p_winds import tools

__all__ = {}