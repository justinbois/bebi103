# As usual, import modules
from __future__ import division, absolute_import, \
                                    print_function, unicode_literals

import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd
import scikits.datasmooth as ds

from brewer2mpl import qualitative
import jb_utils as jb


plt.close('all')

# Define our kernels
def epan_kernel(t):
    """
    Epanechnikov kernel.
    """
    return np.logical_and(t > -1.0, t < 1.0) * 3.0 * (1.0 - t**2) / 4.0

def tri_cube_kernel(t):
    """
    Tri-cube kernel.
    """
    return np.logical_and(t > -1.0, t < 1.0) * (1.0 - abs(t**3))**3

def gauss_kernel(t):
    """
    Gaussian kernel.
    """
    return np.exp(-t**2 / 2.0)
    
    
def nw_kernel_smooth(x_0, x, y, kernel_fun, lam):
    """
    Gives smoothed data at points x_0 using a Nadaraya-Watson kernel 
    estimator.  The data points are given by NumPy arrays x, y.
        
    kernel_fun must be of the form
        kernel_fun(t), 
    where t = |x - x_0| / lam
    
    This is not a fast way to do it, but it simply implemented!
    """
    
    # Function to give estimate of smoothed curve at single point.
    def single_point_estimate(x_0_single):
        """
        Estimate at a single point x_0_single.
        """
        t = np.abs(x_0_single - x) / lam
        return np.dot(kernel_fun(t), y) / kernel_fun(t).sum()
    
    # If we only want an estimate at a single data point
    if np.isscalar(x_0):
        return single_point_estimate(x_0)
    else:  # Get estimate at all points
        y_smooth = np.empty_like(x_0)
        for i in range(len(x_0)):
            y_smooth[i] = single_point_estimate(x_0[i])
        return y_smooth



fname = '../data/weitz_et_al/t5_data/bulk_trace_sustained_oscillation.csv'
df_bulk = pd.read_csv(fname, comment='#')

df_bulk.columns = ['time', 'fl']
df_bulk.time /= 60.0

# plt.plot(df_bulk.time, df_bulk.fl, 'k-', lw=0.5)

bulk_deriv = np.diff(df_bulk.fl[240:400]) / np.diff(df_bulk.time[240:400])
# plt.plot(df_bulk.time[240:399], bulk_deriv, 'k-')

t, fl = df_bulk.time[240:400].values, df_bulk.fl[240:400].values
t_0 = t

lam = 15.0
fl_epan = nw_kernel_smooth(t_0, t, fl, epan_kernel, lam)
fl_tri_cube = nw_kernel_smooth(t_0, t, fl, tri_cube_kernel, lam)
fl_gauss = nw_kernel_smooth(t_0, t, fl, gauss_kernel, lam)

# Plot results
plt.plot(t, fl, 'k-', lw=0.5)
plt.plot(t_0, fl_epan, '-', label='Epan')
plt.plot(t_0, fl_tri_cube, '-', label='Tri-cube')
plt.plot(t_0, fl_gauss, '-', label='Gaussian')
plt.legend(loc='lower right')
























plt.draw()
plt.show()

























