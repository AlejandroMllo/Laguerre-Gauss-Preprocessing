import cmath
import math

import numpy as np


def laguerre_gauss_filter(side_size, omega):
    """
    Computes and returns a matrix containing the Laguerre-Gauss filter.
    """

    scale = (1.j * math.pow(math.pi, 2) * math.pow(omega, 4))
    power_scale = -math.pow(math.pi, 2) * math.pow(omega, 2)

    lg_filter = np.zeros((side_size, side_size), dtype=complex)

    for x in range(side_size):
        x_squared = math.pow(x, 2)
        for y in range(side_size):
            power = cmath.exp(power_scale * (x_squared + math.pow(y, 2)))

            lg_filter[x, y] = scale * complex(x, y) * power

    return lg_filter
