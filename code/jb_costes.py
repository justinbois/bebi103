from __future__ import division

import random

import numpy as np
import scipy.ndimage
import scipy.odr
import scipy.stats

import skimage.transform


# #####################
class CostesColocalization(object):
    """
    Generic class just to store attributes
    """
    def __init__(self, **kw):
        self.__dict__ = kw

        
# #####################
def costes_coloc(im_1, im_2, psf_width=3, n_scramble=10, thresh_r=0.0,
                 roi=None, roi_method='all', do_manders=True):
    """
    Costes colocalization.
    """

    # Make mirrored boundaries in preparation for scrambling
    im_1_mirror = mirror_edges(im_1, psf_width)
    im_2_mirror = mirror_edges(im_2, psf_width)

    # Set up ROI
    if roi is None:
        roi = np.ones_like(im_1, dtype='bool')

    # Rename images to be sliced ROI
    im_1 = im_1[roi]
    im_2 = im_2[roi]

    # Mirror ROI at edges
    roi_mirror = mirror_edges(roi, psf_width)
    
    # Compute the blocks that we'll scramble
    blocks_1 = im_to_blocks(im_1_mirror, psf_width, roi_mirror, roi_method)
    blocks_2 = im_to_blocks(im_2_mirror, psf_width, roi_mirror, roi_method)
    blocks_2_flat = np.array(blocks_2).flatten()

    # Compute the Pearson coefficient
    pearson_r, p = scipy.stats.pearsonr(np.array(blocks_1).ravel(),
                                        blocks_2_flat)

    # Do image scrambling and r calculations
    r_scr = np.empty(n_scramble)
    for i in range(n_scramble):
        random.shuffle(blocks_1)
        r, p = scipy.stats.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)
        r_scr[i] = r

    # Compute percent chance of coloc
    p_coloc = (r_scr < pearson_r).sum() / n_scramble


    # Now do work to compute adjusted Manders's coefficients
    if do_manders:
        # Get the linear relationship between im_2 and im_1
        a, b = odr_linear(im_1.ravel(), im_2.ravel())

        # Perform threshold calculation
        thresh_1 = find_thresh(im_1, im_2, a, b, thresh_r=thresh_r)
        thresh_2 = a * thresh_1 + b

        # Compute Costes's update to the Manders's coefficients
        M_1 = im_1[im_1 > thresh_1].sum() / im_1.sum()
        M_2 = im_2[im_2 > thresh_2].sum() / im_2.sum()

        # Toss results into class for returning
        return CostesColocalization(
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=thresh_r,
            thresh_1=thresh_1, thresh_2=thresh_2, a=a, b=b, M_1=M_1,
            M_2=M_2, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)
    else:
        return CostesColocalization(
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=None,
            thresh_1=None, thresh_2=None, a=None, b=None, M_1=None,
            M_2=None, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)
    
    
# #####################
def odr_linear(x, y, intercept=None, beta0=None):
    """
    Performs orthogonal linear regression on x, y data.  Fixes the intercept
    if intercept is not None.

    beta0 is the guess at the slope and intercept, respectively
    """

    def linear_fun(p, x):
        return p[0] * x + p[1]

    def linear_fun_fixed(p, x):
        return p[0] * x + intercept

    # Set the model to be used for the ODR fitting
    if intercept is None:
        model = scipy.odr.Model(linear_fun)
        if beta0 is None:
            beta0 = (0.0, 1.0)
    else:
        model = scipy.odr.Model(linear_fun_fixed)
        if beta0 is None:
            beta0 = (1.0,)

    # Make a Data instance
    data = scipy.odr.Data(x, y)

    # Instantiate ODR
    odr = scipy.odr.ODR(data, model, beta0=beta0)

    # Perform ODR fit
    try:
        result = odr.run()
    except scipy.odr.odr_error:
        raise scipy.odr.odr_error('ORD failed.')
        
    return result.beta


# ####################
def find_thresh(im_1, im_2, a, b, thresh_r=0.0):
    """
    Reduces threshold value until Pearson correlation goes below thresh_r.
    """
    if im_1.dtype not in [np.uint16, np.uint8]:
        incr = (im_1.max() - im_1.min()) / 256.0
    else:
        incr = 1
    
    thresh_max = im_1.max()
    thresh_min = im_1.min()
    thresh = thresh_max
    r = pearsonr(thresh, im_1, im_2, a, b)
    min_r = r
    min_thresh = thresh
    while thresh > thresh_min and r > thresh_r:
        thresh -= incr
        r = pearsonr(thresh, im_1, im_2, a, b)
        if min_r > r:
            min_r = r
            min_thresh = thresh

    if thresh == thresh_min:
        thresh = min_thresh

    return thresh


# ####################    
def pearsonr(thresh, im_1, im_2, a, b):
    """
    Returns the Pearson correlation coefficient for given threshold
    value.
    """
    inds = (im_1 <= thresh) & (im_2 <= a * thresh + b)
    r, p = scipy.stats.pearsonr(im_1[inds], im_2[inds])
    return r


# ####################    
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
        

# ####################
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
         

# ####################
def blocks_to_im(blocks, im_shape):
    """
    Converts list of blocks to image.

    Only works if blocks were created without ROI.
    """
    im = np.empty(im_shape)
    width = blocks[0].shape[0]
    k = 0
    for i in range(0, im.shape[0], width):
        for j in range(0, im.shape[1], width):
            im[i:i+width, j:j+width] = blocks[k]
            k += 1
    return im
        
