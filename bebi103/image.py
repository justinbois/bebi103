import numpy as np
import scipy.odr

import numba

import skimage.io
import skimage.measure

from matplotlib import path

class SimpleImageCollection(object):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : string or list
        If string, uses glob to generate list of files containing
        images. If list, this is the list of files containing images.
    load_func : callable, default skimage.io.imread
        Function to be called to load images.
    conserve_memory : bool, default True
        If True, do not load all images into RAM. If False, load
        all into a list.

    Returns
    -------
    ic : SimpleImageCollection instance
        ic[n] gives image n of the image collection.

    Notes
    -----
    .. Any keyword arguments except those listed above are passed into
    load_func as kwargs.
    .. This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """

    def __init__(self, load_pattern, load_func=skimage.io.imread,
                 conserve_memory=True, **load_func_kwargs):
        if isinstance(load_pattern, str):
            self.fnames = glob.glob(load_pattern)
        else:
            self.fnames = load_pattern

        self.conserve_memory = conserve_memory

        if self.conserve_memory:
            self.load_func = load_func
            self.kwargs = load_func_kwargs
        else:
            self.ims = [load_func(f, **load_func_kwargs) for f in self.fnames]



    def __getitem__(self, n):
        """
        Return selected image.
        """
        if self.conserve_memory:
            return self.load_func(self.fnames[n], **self.load_func_kwargs)
        else:
            return self.ims[n]


def simple_image_collection(im_glob, load_func=skimage.io.imread,
                            conserve_memory=True, **load_func_kwargs):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : string or list
        If string, uses glob to generate list of files containing
        images. If list, this is the list of files containing images.
    load_func : callable, default skimage.io.imread
        Function to be called to load images.
    conserve_memory : bool, default True
        If True, do not load all images into RAM. If False, load
        all into a list.

    Returns
    -------
    ic : SimpleImageCollection instance
        ic[n] gives image n of the image collection.

    Notes
    -----
    .. Any keyword arguments except those listed above are passed into
    load_func as kwargs.
    .. This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """
    return SimpleImageCollection(im_glob, load_func=load_func,
                                 conserve_memory=conserve_memory,
                                 **load_func_kwargs)


def verts_to_roi(verts, size_i, size_j):
    """
    Converts list of vertices to an ROI and ROI bounding box

    Parameters
    ----------
    verts : array_like, shape (n_verts, 2)
        List of vertices of a polygon with no crossing lines.  The units
        describing the positions of the vertices are interpixel spacing.
    size_i : int
        Number of pixels in the i-direction (number of rows) in
        the image
    size_j : int
        Number of pixels in the j-direction (number of columns) in
        the image

    Returns
    -------
    roi : array_like, Boolean, shape (size_i, size_j)
        roi[i,j] is True if pixel (i,j) is in the ROI.
        roi[i,j] is False otherwise
    roi_bbox : tuple of slice objects
        To get a subimage with the bounding box of the ROI, use
        im[roi_bbox].
    roi_box : array_like, shape is size of bounding box or ROI
        A mask for the ROI with the same dimension as the bounding
        box.  The indexing starts at zero at the upper right corner
        of the box.
    """

    # Make list of all points in the image in units of pixels
    i = np.arange(size_i)
    j = np.arange(size_j)
    ii, jj = np.meshgrid(j, i)
    pts = np.array(list(zip(ii.ravel(), jj.ravel())))

    # Make a path object from vertices
    p = path.Path(verts)

    # Get list of points that are in roi
    in_roi = p.contains_points(pts)

    # Convert it to an image
    roi = in_roi.reshape((size_i, size_j)).astype(np.bool)

    # Get bounding box of ROI
    regions = skimage.measure.regionprops(roi.astype(np.int))
    bbox = regions[0].bbox
    roi_bbox = np.s_[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1]

    # Get ROI mask for just within bounding box
    roi_box = roi[roi_bbox]

    # Return boolean in same shape as image
    return (roi, roi_bbox, roi_box)


class _CostesColocalization(object):
    """
    Generic class just to store attributes.
    """
    def __init__(self, **kw):
        self.__dict__ = kw


def costes_coloc(im_1, im_2, psf_width=3, n_scramble=1000, thresh_r=0.0,
                 roi=None, roi_method='all', do_manders=True):
    """
    Perform Costes colocalization analysis on a pair of images.

    Parameters
    ----------
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    psf_width: int, default 3
        Width, in pixels of the point spread function.
    n_scramble: int, default 1000
        Number of strambled image comparisons to do to get statistics.
    thresh_r: float, default 0.0
        Threshold Pearson r value to be considered colocalized.
    roi: array_like, dtype bool, default None
        Boolean image the same shape as `im_1` and `im_2` that
        is True for pixels within the ROI.
    roi_method: str, default 'all'
        If 'all', all pixels of a given subimage must be within
        the ROI for the subimage itself to be considered part
        of the ROI.  If 'any', if any one pixel is within the ROI,
        the subimage is considered part of the ROI.
    do_manders: bool, default True
        If True, compute the Manders coefficients.

    Returns
    -------
    output: A CostesColocalization instance.
        The CostesColocalization instance has the following attributes.
            im_1, im_2, psf_width, n_scramble, thresh_r, roi,
                roi_method: As in the input parameters.
            a: slope of the regression line I_2 = a * I_1 + b
            b: intercept of regression line I_2 = a * I_1 + b
            M_1: Manders coefficient for image 1
            M_2: Manders coefficient for image 2
            pearson_r: Pearson coerrelaction coefficient of the pixels
                in the two images.
            p_coloc: The probability of colocalization being present
                in the two images.
    """

    # Make float mirrored boundaries in preparation for scrambling
    im_1_mirror = mirror_edges(im_1, psf_width).astype(float)
    im_2_mirror = mirror_edges(im_2, psf_width).astype(float)

    # Set up ROI
    if roi is None:
        roi = np.ones_like(im_1, dtype='bool')

    # Rename images to be sliced ROI and convert to float
    im_1 = im_1[roi].astype(float)
    im_2 = im_2[roi].astype(float)

    # Mirror ROI at edges
    roi_mirror = mirror_edges(roi, psf_width)

    # Compute the blocks that we'll scramble
    blocks_1 = im_to_blocks(im_1_mirror, psf_width, roi_mirror, roi_method)
    blocks_2 = im_to_blocks(im_2_mirror, psf_width, roi_mirror, roi_method)

    # Compute the Pearson coefficient
    pearson_r = _pearson_r(blocks_1.ravel(), blocks_2.ravel())

    # Do image scrambling and r calculations
    r_scr = scrambled_r(blocks_1, blocks_2, n=n_scramble)

    # Compute percent chance of coloc
    p_coloc = (r_scr < pearson_r).sum() / n_scramble

    # Now do work to compute adjusted Manders's coefficients
    if do_manders:
        # Get the linear relationship between im_2 and im_1
        a, b = _odr_linear(im_1.ravel(), im_2.ravel())

        # Perform threshold calculation
        thresh_1 = _find_thresh(im_1, im_2, a, b, thresh_r=thresh_r)
        thresh_2 = a * thresh_1 + b

        # Compute Costes's update to the Manders's coefficients
        inds = (im_1 > thresh_1) & (im_2 > thresh_2)
        M_1 = im_1[inds].sum() / im_1.sum()
        M_2 = im_2[inds].sum() / im_2.sum()

        # Toss results into class for returning
        return _CostesColocalization(
            im_1=im_1, im_2=im_2, roi=roi, roi_method=roi_method,
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=thresh_r,
            thresh_1=thresh_1, thresh_2=thresh_2, a=a, b=b, M_1=M_1,
            M_2=M_2, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)
    else:
        return _CostesColocalization(
            im_1=im_1, im_2=im_2, roi=roi, roi_method=roi_method,
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=None,
            thresh_1=None, thresh_2=None, a=None, b=None, M_1=None,
            M_2=None, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)


@numba.jit(nopython=True)
def _pearson_r(x, y):
    """
    Compute the Pearson correlation coefficient between two samples.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.
        
    Returns
    -------
    output : float
        The Pearson correlation coefficient between `data_1`
        and `data_2`.
    """
    return (np.mean(x*y) - np.mean(x) * np.mean(y)) / np.std(x) / np.std(y)


@numba.jit(nopython=True)
def scrambled_r(blocks_1, blocks_2, n=200):
    """
    Scrambles blocks_1 n_scramble times and returns the Pearson r values.

    Parameters
    ----------
    blocks_1 : n x n_p x n_p Numpy array
        First index corresponds to block ID, and second and third
        indices have pixel values in blocks.
    blocks_2 : n x n_p x n_p Numpy array
        First index corresponds to block ID, and second and third
        indices have pixel values in blocks.        
    n : int
        Number of scrambled Pearson r values to compute.

    Returns
    -------
    output : ndarray
        Numpy array with Pearson r values computed by scrambling the
        first set of blocks.
    """
    # Indicies of blocks
    block_inds = np.arange(blocks_1.shape[0])
    
    # Flatten blocks 2
    blocks_2_flat = blocks_2.flatten()
    
    r_scr = np.empty(n)
    for i in range(n):
        np.random.shuffle(block_inds)
        r = _pearson_r(blocks_1[block_inds].ravel(), blocks_2_flat)
        r_scr[i] = r
    return r_scr



def _odr_linear(x, y, intercept=None, beta0=None):
    """
    Performs orthogonal linear regression on x, y data.

    Parameters
    ----------
    x: array_like
        x-data, 1D array.  Must be the same lengths as `y`.
    y: array_like
        y-data, 1D array.  Must be the same lengths as `x`.
    intercept: float, default None
        If not None, fixes the intercept.
    beta0: array_like, shape (2,)
        Guess at the slope and intercept, respectively.

    Returns
    -------
    output: ndarray, shape (2,)
        Array containing slope and intercept of ODR line.
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


def _find_thresh(im_1, im_2, a, b, thresh_r=0.0):
    """
    Find the threshold pixel intensity of `im_1` where
    the Pearson correlation between the images goes below `thresh_r`.

    Parameters
    ----------
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    a: float
        Slope of the ORD regression of `im_2` vs. `im_1`.
    b: float
        Intercept of the ORD regression of `im_2` vs. `im_1`.
    thresh_r: float, default 0.0
        Threshold Pearson correlation

    Returns
    -------
    output: int or float
        The threshold pixel intensity for colocalization
        (see notes below).

    Notes
    -----
    ..  To determine which pixels are colocalized in two images, we
        do the following:
            1. Perform a regression based on all points of to give
               I_2 = a * I_1 + b.
            2. Define T = I_1.max().
            3. Compute the Pearson r value considering all pixels with
               I_1 < T and I_2 < a * T + b.
            4. If r <= thresh_r decrement T and goto 3.  Otherwise,
               save $T_1 = T$ and $T_2 = a * T + b.
            5. Pixels with I_2 > T_2 and I_1 > T_1 are colocalized.
        This function returns T.
    """
    if im_1.dtype not in [np.uint16, np.uint8]:
        incr = (im_1.max() - im_1.min()) / 256.0
    else:
        incr = 1

    thresh_max = im_1.max()
    thresh_min = im_1.min()
    thresh = thresh_max
    r = _pearsonr_below_thresh(thresh, im_1, im_2, a, b)
    min_r = r
    min_thresh = thresh
    while thresh > thresh_min and r > thresh_r:
        thresh -= incr
        r = _pearsonr_below_thresh(thresh, im_1, im_2, a, b)
        if min_r > r:
            min_r = r
            min_thresh = thresh

    if thresh == thresh_min:
        thresh = min_thresh

    return thresh


def _pearsonr_below_thresh(thresh, im_1, im_2, a, b):
    """
    The Pearson r between two images for pixel values below
    threshold.

    Parameters
    ----------
    thresh: float or int
        The threshold value of pixel intensities to consider for
        `im_1`.
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    a: float
        Slope of the ORD regression of `im_2` vs. `im_1`.
    b: float
        Intercept of the ORD regression of `im_2` vs. `im_1`.
    """
    inds = (im_1 <= thresh) | (im_2 <= a * thresh + b)
    r = _pearson_r(im_1[inds], im_2[inds])
    return r


def mirror_edges(im, psf_width):
    """
    Given a 2D image pads the boundaries by mirroring so that the
    dimensions of the image are multiples for the width of the
    point spread function.

    Parameters
    ----------
    im: array_like
        Image to mirror edges
    psf_width: int
        The width, in pixels, of the point spread function

    Returns
    -------
    output: array_like
        Image with mirrored edges
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
    return np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right)),
                  mode='reflect')


def im_to_blocks(im, width, roi=None, roi_method='all'):
    """
    Converts image to list of square subimages called "blocks."

    Parameters
    ----------
    im: array_like
        Image to convert to a list of blocks.
    width: int
        Width of square blocks in units of pixels.
    roi: array_like, dtype bool, default None
        Boolean image the same shape as `im_1` and `im_2` that
        is True for pixels within the ROI.
    roi_method: str, default 'all'
        If 'all', all pixels of a given subimage must be within
        the ROI for the subimage itself to be considered part
        of the ROI.  If 'any', if any one pixel is within the ROI,
        the subimage is considered part of the ROI.

    Returns
    -------
    output: list of ndarrays
        Each entry is a `width` by `width` NumPy array containing
        a block.
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
    return np.array([im[i:i + width, j:j + width]
                        for i in range(0, im.shape[0], width)
                            for j in range(0, im.shape[1], width)
                                if roi_test(roi[i:i + width, j:j + width])])
