# As usual, import modules
from __future__ import division, absolute_import,  print_function

import os
import random

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage.io
import skimage.filter

# Utilities from JB
import jb_utils as jb

# Module to do Costes localization
import costes



plt.close('all')


data_dir = '../data/ramesh_et_al'
fname_g = os.path.join(data_dir, 'mamB-antiHA-antiCOX4-488-1.ome.tiff')
fname_r = os.path.join(data_dir, 'mamB-antiHA-antiCOX4-543-1.ome.tiff')
fname_b = os.path.join(data_dir, 'mamB-antiHA-antiCOX4-DAPI-1.ome.tiff')

# Load three channels
im_r = skimage.io.imread(fname_r)
im_g = skimage.io.imread(fname_g)
im_b = skimage.io.imread(fname_b)


def im_merge(im_cyan, im_magenta, im_yellow=None):
    """
    Merge channels to make RGB image that has cyan, magenta, and yellow.
    """
    im_cyan_scaled = (im_cyan - im_cyan.min()) \
                            / (im_cyan.max() - im_cyan.min())
    im_magenta_scaled = (im_magenta - im_magenta.min()) \
                            / (im_magenta.max() - im_magenta.min())
    
    if im_yellow is None:
        im_yellow_scaled = np.zeros_like(im_cyan)
    else:
        im_yellow_scaled = (im_yellow - im_yellow.min()) \
                                / (im_yellow.max() - im_yellow.min())

    # Convert images to RGB with magenta, cyan, and yellow channels
    im_cyan_scaled_rgb = np.dstack((im_cyan_scaled, 
                                    np.zeros_like(im_cyan_scaled), 
                                    im_cyan_scaled))
    im_magenta_scaled_rgb = np.dstack((np.zeros_like(im_magenta_scaled), 
                                       im_magenta_scaled, 
                                       im_magenta_scaled))
    im_yellow_scaled_rgb = np.dstack((im_yellow_scaled, 
                                      im_yellow_scaled, 
                                      np.zeros_like(im_yellow_scaled)))

    # Merge together
    merged_image = im_cyan_scaled_rgb + im_magenta_scaled_rgb \
                        + im_yellow_scaled_rgb

    # Scale each channel to be between zero and 1
    merged_image[:,:,0] /= merged_image[:,:,0].max()
    merged_image[:,:,1] /= merged_image[:,:,1].max()
    merged_image[:,:,2] /= merged_image[:,:,2].max()
    
    return merged_image


# Make merged image
#merged_image = im_merge(im_g, im_r, im_b)
#
#fig, ax = plt.subplots(2, 2)
#ax[0,0].imshow(im_r, cmap=cm.gray)
#ax[0,1].imshow(im_g, cmap=cm.gray)
#ax[1,0].imshow(im_b, cmap=cm.gray)
#ax[1,1].imshow(merged_image)


# plt.plot(im_r.ravel(), im_g.ravel(), 'k.', alpha=0.2)

r, p = scipy.stats.pearsonr(im_r.ravel(), im_g.ravel())


def mirror_edges(im, psf_width, roi=None):
    """
    Given a 2D image im, pads the boundaries by mirroring so that the
    dimensions of the image are a multiple of psf_width.
    
    Also pads an "image" containing the ROI if necessary.
    """
    
    # How much we need to pad
    pad_i = psf_width - (im.shape[0] % psf_width)
    pad_j = psf_width - (im.shape[1] % psf_width)

    # Get widths
    pad_top = pad_i // 2
    pad_bottom = pad_i - pad_top
    pad_left = pad_j // 2
    pad_right = pad_j - pad_left

    # Do the padding
    im_pad = np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='reflect')

    if roi is None:
        return im_pad
    else:
        roi_pad = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='reflect')
        return im_pad, roi_pad

def im_to_blocks(im, width, roi=None, roi_method='all'):
    """
    Converts image to list of blocks.  If roi is not None, only keeps blocks
    that are within an ROI.  The test for being within and ROI is either
    'all' (all pixels much be in ROI) or 'any' (a block is in the ROI if
    any of its pixels are in the ROI).
    """
    # Initialize ROI
    if roi is None:
        roi = np.ones_like(im)

    # Specify method for determining if in ROI or not
    if roi_method == 'all':
        roi_test = np.all
    else:
        roi_test = np.any

    # Construct list of blocks
    blocks = []
    for i in range(0, im.shape[0], width):
        for j in range(0, im.shape[1], width):
            if roi_test(roi[i:i+width, j:j+width]):
                blocks.append(im[i:i+width, j:j+width])

    return blocks

psf_width = 3
im_r_mirror = mirror_edges(im_r, psf_width)
im_g_mirror = mirror_edges(im_g, psf_width)

blocks_r = im_to_blocks(im_r_mirror, psf_width)
blocks_g = im_to_blocks(im_g_mirror, psf_width)
blocks_g_flat = np.array(blocks_g).flatten()


r_unscr, p = scipy.stats.pearsonr(np.array(blocks_r).ravel(), blocks_g_flat)

def scrambled_r(n_scramble, blocks_1, blocks_2_flat):
    """
    Scrambles blocks_1 n_scramble times and returns the Pearson r values.
    """
    r_scr = np.empty(n_scramble)
    for i in range(n_scramble):
        random.shuffle(blocks_1)
        r, p = scipy.stats.pearsonr(np.array(blocks_1).ravel(), 
                                    blocks_2_flat)
        r_scr[i] = r
    return r_scr

## Do the scamblin'!
#n_scramble = 200
#r_scr = scrambled_r(n_scramble, blocks_r, blocks_g_flat)


# Threshold the imge
#thresh_r = skimage.filter.threshold_yen(im_r)
#thresh_g = skimage.filter.threshold_yen(im_g)
#
## Make ROI
#roi = (im_r > thresh_r) | (im_g > thresh_g)
#roi = skimage.morphology.remove_small_objects(roi, min_size=psf_width**2)
#
#roi_mirror = mirror_edges(roi, psf_width)
#
## Get blocks of red and green channels
#blocks_r = im_to_blocks(im_r_mirror, psf_width, roi=roi_mirror)
#blocks_g = im_to_blocks(im_g_mirror, psf_width, roi=roi_mirror)
#
## Store blocks as flattened array in case we don't want to scramble them
#blocks_g_flat = np.array(blocks_g).flatten()

# Get r value for unscrambled images
# r_unscr, p = scipy.stats.pearsonr(np.array(blocks_r).ravel(), blocks_g_flat)

# Do the scamblin'!
# n_scramble = 200
# r_scr = scrambled_r(n_scramble, blocks_r, blocks_g_flat)

# Compute how sure we are that it's colocalized
# p_coloc = (r_scr < r_unscr).sum() / n_scramble

data_dir = '../data/ramesh_et_al'

# Define file names
fname_g = os.path.join(data_dir, 'mamP-488-antiHA-3.ome.tiff')
fname_r = os.path.join(data_dir, 'mamP-650-antiHA-3.ome.tiff')
fname_b = os.path.join(data_dir, 'mamP-DAPI-antiHA-3.ome.tiff')

# Load three channels
im_r = skimage.io.imread(fname_r)
im_g = skimage.io.imread(fname_g)
im_b = skimage.io.imread(fname_b)

# Display the images
merged_image = im_merge(im_g, im_r, im_b)

# Show the images
figure, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0,0].imshow(im_r, cmap=cm.gray)
ax[0,1].imshow(im_g, cmap=cm.gray)
ax[1,0].imshow(im_b, cmap=cm.gray)
ax[1,1].imshow(merged_image)

ax[0,0].set_title('anit-Calnexin')
ax[0,1].set_title('anti-HA')
ax[1,0].set_title('DAPI')
ax[1,1].set_title('merged')


# Compute ROI
thresh_r = skimage.filter.threshold_yen(im_r)
thresh_g = skimage.filter.threshold_yen(im_g)

# Make ROI
roi = (im_r > thresh_r) | (im_g > thresh_g)
roi = skimage.morphology.remove_small_objects(roi, min_size=psf_width**2)

coloc = costes.costes_coloc(im_r, im_g, n_scramble=200, psf_width=3,
                            do_manders=True, roi=roi, roi_method='all')

























plt.draw()
plt.show()