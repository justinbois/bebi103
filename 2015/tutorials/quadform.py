"""
Silly little quadratic formula module.
"""

import numpy as np

def discriminant(a, b, c):
    """
    Discriminant of quadratic polynomial.
    """
    return b**2 - 4 * a * c


def roots(a, b, c):
    """
    Return roots of a quadratic equation.
    """
    delta = discriminant(a, b, c)

    if delta < 0.0:
        raise RuntimeError('Neg. discrim.  We keep it real.')

    root_1 = (-b + np.sqrt(delta)) / (2 * a)
    root_2 = (-b - np.sqrt(delta)) / (2 * a)

    return root_1, root_2
