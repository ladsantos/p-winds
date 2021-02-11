#! /usr/bin/env python
# -*- coding: utf-8 -*-


import astropy.units as u
from p_winds import tools


# The ``fetch_planet_system`` is prone to warnings or errors because it
# depends on data provided by the NASA Exoplanet Database and a development
# version of ``astroquery``. Let's test by retrieving the data for
# HD 209458 b. The orbital period of the planet is well known to a high
# precision, so let's test for that.
def test_fetch_planet_system(planet_name='HD 209458 b',
                             precision_threshold=1E-6):
    planet_info, star_info = tools.fetch_planet_system(planet_name)
    period = planet_info['period'].to(u.d).value
    assert abs(period - 3.52474859) < precision_threshold
