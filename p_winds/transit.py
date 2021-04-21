#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and atmospheres.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as plc
# import scipy.optimize as sp
from PIL import Image, ImageDraw
# from itertools import product

__all__ = ["draw_transit"]


# Draw a grid
def draw_transit(planet_to_star_ratio, impact_parameter=0.0, phase=0.0,
                 grid_size=2001):
    """

    Parameters
    ----------
    planet_to_star_ratio (``float``):
        Ratio between the radii of the planet and the star.

    impact_parameter (``float``, optional):
        Transit impact parameter. Default is 0.0.

    phase (``float``, optional):
        Phase of the transit. -0.5, 0.0, and +0.5 correspond to the center of
        planet being located at, respectively, the left limb of the star, the
        center, and the right limb. Default is 0.0.

    grid_size (``int``, optional):
        Size of the transit grid. Default is 2001.

    Returns
    -------
    grid (``numpy.ndarray``):
        Transit grid.
    """
    shape = (grid_size, grid_size)

    # The grid itself starts as a zeros numpy-array
    # grid = np.zeros(shape, float)

    # General function to draw a disk
    def _draw_disk(center, radius, value=1.0):
        """

        Parameters
        ----------
        center:
            Coordinates of the center of the disk. The origin is the center of the
            grid.

        radius:
            Radius of the disk in units of grid pixels.

        value:
            Value to be attributed to each pixel inside the disk.

        Returns
        -------

        """
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        disk = np.reshape(np.array(list(image.getdata())), shape) * value
        return disk

    # Draw the host star
    star_radius = grid_size // 2
    star = _draw_disk(center=(star_radius, star_radius), radius=star_radius)
    norm = np.sum(star)  # Normalization factor is the total flux
    # Adding the star to the grid
    grid = star / norm

    # Draw the planet
    # First we need to figure out the exact coordinate of the center of the
    # planet
    x_p = grid_size // 2 + int(phase * grid_size)
    y_p = grid_size // 2 + int(impact_parameter * grid_size // 2)
    planet_radius = star_radius * planet_to_star_ratio
    planet = _draw_disk(center=(x_p, y_p), radius=planet_radius)
    # Adding the planet to the grid, normalized by the stellar flux
    grid -= planet / norm
    # The grid must not have negative values (this may happen if the planet
    # disk falls out of the stellar disk)
    grid = grid.clip(min=0.0)

    return grid
