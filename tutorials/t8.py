# As usual, import modules
from __future__ import division, absolute_import, \
                                    print_function, unicode_literals

import os

import numpy as np
import pandas as pd
import scipy.constants
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import cm

# A whole bunch of skimage stuff
import skimage.feature
import skimage.filter
import skimage.io
import skimage.morphology

# Utilities from JB
import jb_utils as jb


plt.close('all')


trap_xyt = jb.XYTStack(directory='../data/lee_et_al/trapped/',
                        conserve_memory=False, dt=0.06, physical_size_x=0.28,
                        physical_size_y=0.28)

peak = np.where(trap_xyt.im(0) == trap_xyt.im(0).max())
peak_i = peak[0][0]
peak_j = peak[1][0]

#im_blur = skimage.filter.gaussian_filter(trap_xyt.im(0), 2.0)
#thresh = im_blur.mean() + 1.0 * im_blur.std()
#
#peaks = skimage.feature.peak_local_max(im_blur, min_distance=6, 
#                                      threshold_abs=thresh)

# Fit symmetric Gaussian to x, y, z data
def fit_gaussian(x, y, z):
    """
    Fits symmetric Gaussian to x, y, z.
    
    Fit func: z = a * exp(-((x - x_0)**2 + (y - y_0)**2) / (2 * sigma**2))
    
    Returns: p = [a, x_0, y_0, sigma]
    """
    
    def sym_gaussian(p):
        """
        Returns a Gaussian function:
        a**2 * exp(-((x - x_0)**2 + (y - y_0)**2) / (2 * sigma**2))
        p = [a, x_0, y_0, sigma]
        """
        a, x_0, y_0, sigma = p
        return a**2 \
                * np.exp(-((x - x_0)**2 + (y - y_0)**2) / (2.0 * sigma**2))
    
    def sym_gaussian_resids(p):
        """Residuals to be sent into leastsq"""
        return z - sym_gaussian(p)
    
    def guess_fit_gaussian():
        """
        return a, x_0, y_0, and sigma based on computing moments of data
        """
        a = z.max()

        # Compute moments
        total = z.sum()
        x_0 = np.dot(x, z) / total
        y_0 = np.dot(y, z) / total

        # Approximate sigmas
        sigma_x = np.dot(x**2, z) / total
        sigma_y = np.dot(y**2, z) / total
        sigma = np.sqrt(sigma_x * sigma_y)
        
        # Return guess
        return (a, x_0, y_0, sigma)

    # Get guess
    p0 = guess_fit_gaussian()
    
    # Perform optimization using nonlinear least squares
    popt, junk_output, info_dict, mesg, ier = \
            scipy.optimize.leastsq(sym_gaussian_resids, p0, full_output=True)
    
    # Check to make sure leastsq was successful.  If not, return centroid
    # estimate.
    if ier in (1, 2, 3, 4):
        return (popt[0]**2, popt[1], popt[2], popt[3])
    else:
        return p0

def bead_position_pix(im, selem):
    """
    Determines the position of bead in image in units of pixels with
    subpixel accuracy.
    """
    # The x, y coordinates of pixels are nonzero values in selem
    y, x = np.nonzero(selem)
    x = x - selem.shape[1] // 2
    y = y - selem.shape[0] // 2
    
    # Find the center of the bead to pixel accuracy
    peak_flat_ind = np.argmax(im)
    peak_j = peak_flat_ind % im.shape[0]
    peak_i = (peak_flat_ind - peak_j) // im.shape[1]
    
    # Define local neighborhood
    irange = (peak_i - selem.shape[0] // 2, peak_i + selem.shape[0] // 2 + 1)
    jrange = (peak_j - selem.shape[1] // 2, peak_j + selem.shape[1] // 2 + 1)
    
    # Get values of the image in local neighborhood
    z = im[irange[0]:irange[1], jrange[0]:jrange[1]][selem.astype(np.bool)]
    
    # Fit Gaussian
    a, j_subpix, i_subpix, sigma = fit_gaussian(x, y, z)
    
    # Return x-y position
    return np.array([peak_i + i_subpix, peak_j + j_subpix])

# Finds the bead position in each frame
selem = skimage.morphology.square(3)
centers = []
for i in range(trap_xyt.size_t):
    centers.append(bead_position_pix(trap_xyt.im(i), selem))
centers = np.array(centers)

# Get displacements
x = centers[:,1] - centers[:,1].mean()
y = centers[:,0] - centers[:,0].mean()

# Get x and y in units of microns
x_micron = x * trap_xyt.physical_size_x
y_micron = y * trap_xyt.physical_size_y

# Get MSD
msd_x = (x_micron**2).mean()
msd_y = (y_micron**2).mean()

# Compute kT
kT = scipy.constants.k * (22.0 + 273.15) * 1e18

k_x = kT / msd_x
k_y = kT / msd_y

plt.plot(trap_xyt.t, x, 'b-')
plt.plot(trap_xyt.t, y, 'g-', lw=0.5)


















plt.draw()
plt.show()





