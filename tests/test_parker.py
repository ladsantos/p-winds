#! /usr/bin/env python
# -*- coding: utf-8 -*-


from p_winds import parker


# The only slightly complicated code in the ``parker`` module is ``structure()``
# so let's test how it is working
# The velocity and density calculated by ``structure()`` should yield values of
# 1.0 at r = 1.0.
def test_structure(r=1.0, precision_threshold=1E-6):
    velocity, density = parker.structure(r)
    assert abs((velocity - 1.0) / velocity) < precision_threshold
    assert abs((density - 1.0) / density) < precision_threshold


# The sound speed for a gas made of 100% atomic hydrogen at 10,000 K should be
# 9.08537273 km /s
def test_sound_speed(temperature=10000, h_he=1.0,
                     precision_threshold=1E-6):
    vs = parker.sound_speed(temperature, h_he)
    assert abs((vs - 9.08537273) / vs) < precision_threshold
