# As usual, import modules
from __future__ import division, absolute_import, \
                                    print_function, unicode_literals

# This is a core Python module for communicating with the operating system
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# A whole bunch of skimage stuff
import skimage.feature
import skimage.filter
import skimage.filter.rank
import skimage.io
import skimage.morphology
import skimage.restoration
import skimage.segmentation

# And some useful scipy.ndimage stuff
import scipy.ndimage

# Utilities from JB
import jb_utils as jb


plt.close('all')



image_dir = '../data/park_et_al/'
im_p_file = image_dir + 'snaps001-001-p.tif'
im_r_file = image_dir + 'snaps001-001-r.tif'
im_c_file = image_dir + 'snaps001-001-c.tif'
im_y_file = image_dir + 'snaps001-001-y.tif'

ims = [skimage.io.imread(im_p_file),
skimage.io.imread(im_r_file),
skimage.io.imread(im_c_file),
skimage.io.imread(im_y_file)]

im_names = ['phase', 'RFP', 'CFP', 'YFP']

im_sp = [(0,0), (0,1), (1,0), (1,1)]

#fig, ax = plt.subplots(2, 2, figsize=(8,7))
#for i in range(4):
#    ax[im_sp[i]].imshow(ims[i], cmap=cm.gray)
#    ax[im_sp[i]].set_title(im_names[i])

hist, bins = 4*[None], 4*[None]
for i in range(4):
    hist[i], bins[i] = skimage.exposure.histogram(ims[i])

#fig, ax = plt.subplots(2, 2, figsize=(9,8))
#for i in range(4):
#    ax[im_sp[i]].fill_between(bins[i], np.log10(hist[i]), lw=0.5, alpha=0.5)
#    ax[im_sp[i]].set_title(im_names[i])
##    ax[im_sp[i]].set_yscale('log')


im = np.copy(ims[0])
# plt.imshow(im, cmap=cm.RdBu_r)

im_float = (im - im.min()) / (im.max() - im.min())

im_bg = skimage.filter.gaussian_filter(im_float, 50.0)
im_no_bg = im_float - im_bg

#fig, ax = plt.subplots(1, 2, figsize=(8,5))
#ax[0].imshow(im_float, cmap=cm.RdBu_r)
#ax[1].imshow(im_no_bg, cmap=cm.RdBu_r)

im_filt_gauss = skimage.filter.gaussian_filter(im_float, 5.0)
#im_filt_tv = skimage.restoration.denoise_tv_chambolle(im_float, 0.1)

#fig, ax = plt.subplots(1, 2, figsize=(8,5))
# ax[0].imshow(im[75:175, 175:275], cmap=cm.RdBu_r)
#ax[1].imshow(im_filt_tv[75:175, 175:275], cmap=cm.RdBu_r)

# Introduce salt and pepper noise
#np.random.seed(42)
#noise = np.random.random(im.shape)
#im_snp = np.copy(im)
#im_snp[noise > 0.96] = im.max()
#im_snp[noise < 0.04] = im.min()

# plt.imshow(im_snp, cmap=cm.gray)

#selem = skimage.morphology.disk(5)
#im_filt_median = skimage.filter.rank.median(im, selem)
#
#fig, ax = plt.subplots(1, 2, figsize=(8,5))
#ax[0].imshow(im_filt_gauss, cmap=cm.gray)
#ax[1].imshow(im_filt_median, cmap=cm.gray)

#selem = skimage.morphology.disk(25)
#
#im_mean = skimage.filter.rank.mean(im, selem)
#
#im_bw = im < 0.85 * im_mean
#
#fig, ax = plt.subplots(1, 2, figsize=(8,5))
#ax[0].imshow(im[75:175, 175:275], cmap=cm.gray)
#ax[1].imshow(im_bw[75:175, 175:275], cmap=cm.gray)


# Get the RFP channel
im = np.copy(ims[1])

# Display the image
#fig, ax = plt.subplots(1, 2, figsize=(8,5))
#ax[0].imshow(im, cmap=cm.RdBu_r)
#ax[1].imshow(im[100:300, 850:1050], cmap=cm.RdBu_r)

#def our_thresh(im, selem, white_true=True, k_range=(0.5, 1.5), min_size=100):
#    """
#    Threshold image as described above.  Morphological mean filter is 
#    applied using selem.
#    """
#    
#    # Determine comparison operator
#    if white_true:
#        compare = np.greater
#        sign = -1
#    else:
#        compare = np.less
#        sign = 1
#    
#    # Do the mean filter
#    im_mean = skimage.filter.rank.mean(im, selem)
#
#    # Compute number of pixels in binary image as a function of k
#    k = np.linspace(k_range[0], k_range[1], 100)
#    n_pix = np.empty_like(k)
#    for i in range(len(k)):
#        n_pix[i] = compare(im, k[i] * im_mean).sum() 
#
#    # Compute rough second derivative
#    dn_pix_dk2 = np.diff(np.diff(n_pix))
#
#    # Find index of maximal second derivative
#    max_ind = np.argmax(sign * dn_pix_dk2)
#
#    # Use this index to set k
#    k_opt = k[max_ind - sign * 2]
#
#    # Threshold with this k
#    im_bw = compare(im, k_opt * im_mean)
#
#    # Remove all the small objects
#    im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=min_size)
#   
#    return im_bw, k_opt
#
## Make the structuring element 50 pixel radius disk
#selem = skimage.morphology.disk(50)
#
## Threshhold based on mean filter
#im_bw, k = our_thresh(im, selem, white_true=True, min_size=400)
#
## Clear border
#im_bw = skimage.segmentation.clear_border(im_bw)
#
## Show image
#fig, ax = plt.subplots(1, 2, figsize=(9,5))
#ax[0].imshow(im_bw, cmap=cm.gray)
#ax[1].imshow(im_bw[100:300, 850:1050], cmap=cm.gray);
#

im_float = (im - im.min()) / (im.max() - im.min())

im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, 2.0)

def zero_crossing_filter(im, thresh):
    """
    Returns image with 1 if there is a zero crossing and 0 otherwise.
    
    thresh is the the minimal value of the gradient, as computed by Sobel
    filter, at crossing to count as a crossing.
    """
    # Square structuring element
    selem = skimage.morphology.square(3)
    
    # Do max filter and min filter
    im_max = scipy.ndimage.filters.maximum_filter(im, footprint=selem)
    im_min = scipy.ndimage.filters.minimum_filter(im, footprint=selem)
    
    # Compute gradients using Sobel filter
    im_grad = skimage.filter.sobel(im)
    
    # Return edges
    return (((im >= 0) & (im_min < 0)) | ((im <= 0) & (im_max > 0))) \
                & (im_grad >= thresh)


im_edge = zero_crossing_filter(im_LoG, 0.001)
im_edge = skimage.morphology.skeletonize(im_edge)
im_bw = scipy.ndimage.morphology.binary_fill_holes(im_edge)
im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=200)
im_bw = skimage.segmentation.clear_border(im_bw, buffer_size=5)

#fig, ax = plt.subplots(1, 2, figsize=(9,5))
#ax[0].imshow(im_LoG[100:300, 850:1050], cmap=cm.RdBu_r)
#ax[1].imshow(im_bw[100:300, 850:1050], cmap=cm.gray);

im_labeled, n_labels = skimage.measure.label(im_bw, background=0,
        return_num=True, neighbors=4)
       
# Load other images
image_dir = '../data/park_et_al/'
im_p_file = image_dir + 'snaps001-001-p.tif'
im_c_file = image_dir + 'snaps001-001-c.tif'
im_y_file = image_dir + 'snaps001-001-y.tif'

# Load images
im_p = skimage.io.imread(im_p_file)
im_c = skimage.io.imread(im_c_file)
im_y = skimage.io.imread(im_y_file)

# Upsample other images (2 means 2x as big, order=0 means no interpolation)
im_p = scipy.ndimage.zoom(im_p, 2, order=0)
im_c = scipy.ndimage.zoom(im_c, 2, order=0)
im_y = scipy.ndimage.zoom(im_y, 2, order=0)        

# Compute props
im_c_props = skimage.measure.regionprops(im_labeled, intensity_image=im_c)
im_y_props = skimage.measure.regionprops(im_labeled, intensity_image=im_y)






plt.draw()
plt.show()

















