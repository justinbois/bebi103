"""
Quadratic formula module
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def discriminant(a, b, c):
    """
    Returns the discriminant of a quadratic polynomial
    a * x**2 + b * x + c = 0.    
    """
    return b**2 - 4.0 * a * c


def roots(a, b, c):
    """
    Returns the roots of the quadratic equation
    a * x**2 + b * x + c = 0.
    """ 
    delta = discriminant(a, b, c)
    root_1 = (-b + np.sqrt(delta)) / (2.0 * a)
    root_2 = (-b - np.sqrt(delta)) / (2.0 * a)
    
    return root_1, root_2
