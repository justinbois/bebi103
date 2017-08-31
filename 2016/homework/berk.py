import glob
import os
import math
import numpy as np
import pandas as pd
import scipy.signal
import scipy.special
import skimage
import skimage.io
import emcee
import bebi103


# Load in TIFF stack
fname = '../data/goehring_FRAP_data/PH_138_A.tif'
ic = skimage.io.ImageCollection(fname, conserve_memory=False)
# Now, let's perform some exploratory data analysis on the FRAP images. We'll plot a few images to get a feel for the ROI:
# How long is it?
print('There are {0:d} frames.'.format(len(ic)))

# The directory containing daytime data
data_dir = '../data/goehring_FRAP_data'

# Glob string for images
im_glob = os.path.join(data_dir, '*.tif')

# Get list of files in directory
im_list = glob.glob(im_glob)

ic = skimage.io.ImageCollection(im_glob, conserve_memory=True)
print(len(ic))
print(ic[20].shape)
#test = ic[20]/numpy.max()
#skimage.io.imshow(ic[20]) #+ 0 * 149]) # The shape of the bleaching is very clear starting from the 21st frame. (?)
verts = [(16.991726866271648, 9.4876724371770536), (65.043057562209043, 10.481837899851612), (63.717503611976291, 58.201780108230835), (15.334784428480702, 55.21928372020713)]
roi, roi_bbox, roi_box = bebi103.verts_to_roi(verts, *ic[0].shape)
print(verts)

fps = 1 / 0.188
t = np.arange(0, len(ic)) / fps

# Set up NumPy array to store total pixel intensity
total_int = np.empty(len(t))

lists = []
for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    lists.append(ic[20 + 149 * (i - 1) : 149 * i - 1])

new_t = np.arange(0, len(lists[0])) / fps
#for series in range(8):
    # Look through and compute total intensity in the ROI
for i, im in enumerate(ic):
    if i % 10 == 0:
        print(i)
    total_int[i] = ic[i][roi_bbox].sum()

verts = [(16.991726866271648, 9.4876724371770536), (65.043057562209043, 10.481837899851612), (63.717503611976291, 58.201780108230835), (15.334784428480702, 55.21928372020713)]
roi, roi_bbox, roi_box = bebi103.verts_to_roi(verts, *ic[0].shape)
total_int = np.empty(len(t))
for i, im in enumerate(ic):
    total_int[i] = ic[i][roi_bbox].sum()

new_total_int = total_int[20 + 149 * 0 : 148 + 149 * 0]
norm_total_int = new_total_int / total_int[0]

def theoretical_intensity(p, t, d_x, d_y):
    """
    Theoretical model for normalized intensity
    """
    d, k_off, f_f, f_b = p

    return f_f * (1 - f_b * (4 * np.exp(-k_off * t) / (d_x * d_y)) * phi_x(d_x, d, t) * phi_y(d_y, d, t))

def phi_x(d_x, d, t):
    return d_x / 2 * scipy.special.erf(d_x / np.sqrt(4 * d * t)) - np.sqrt(d * t / math.pi) * (1 - np.exp(-d_x ** 2 / (4 * d * t)))

def phi_y(d_y, d, t):
    return d_y / 2 * scipy.special.erf(d_y / np.sqrt(4 * d * t)) - np.sqrt(d * t / math.pi) * (1 - np.exp(-d_y ** 2 / (4 * d * t)))


def log_post(p, t, norm_total_int, d_x, d_y):

    # Unpack parameters
    d, k_off, f_f, f_b = p

    int_theor = theoretical_intensity(p, t, d_x, d_y)
    return -len(t) / 2 * np.log(np.sum((norm_total_int - int_theor)**2))

n_dim = 4        # number of parameters in the model (r and p)
n_walkers = 50   # number of MCMC walkers
n_burn = 10    # "burn-in" period to let chains stabilize
n_steps = 10   # number of MCMC steps to take after burn-in
np.random.seed(42)
# p0[i,j] is the starting point for walk i along variable j.
p0 = np.empty((n_walkers, n_dim))
p0[:,0] = np.random.uniform(0, 1, n_walkers)
p0[:,1] = np.random.uniform(0, 0.1, n_walkers)
p0[:,2] =  np.random.uniform(0, 1, n_walkers)
p0[:,3] =  np.random.uniform(0, 1, n_walkers)

sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                args=(new_t, norm_total_int, 50, 51), threads=2)
# Do burn-in
pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)
# Sample again, starting from end burn-in state
_ = sampler.run_mcmc(pos, n_steps)

# Conver sampler output to DataFrame
df_mcmc = bebi103.sampler_to_dataframe(sampler, columns=['d', 'k_off', 'f_f', 'f_b'])

# Take a look
df_mcmc.head()
