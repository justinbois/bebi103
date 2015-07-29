"""
Set of utilities written by Justin Bois for use in BE/Bi 103 (2014
edition) and beyond.
"""
from __future__ import division, print_function, \
                                absolute_import, unicode_literals

import os
import glob
import warnings

import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np

try:
    import numdifftools as nd
except:
    warnings.warn('Unable to import numdifftools.  hess_nd unavailable.',
                  ImportWarning)

try: 
    import pywt
except:
    warnings.warn('Unable to import PyWavelets. visushrink will not work.',
                  ImportWarning)

try:
    import skimage.io
except:
    warnings.warn('Unable to import skimage.  ' \
                  + 'Image processing utils will not work.', ImportWarning)
    
try:
    from PIL import Image
except:
    warnings.warn('Unable to import PIL via Pillow.  ' \
                  + 'Image processing utils will not work.', ImportWarning)

try:
    import skimage.io
    import skimage.measure
except:
    warnings.warn('Unable to import skimage. '\
                  + 'Image processing utils will not work.', ImportWarning)
        
    

# ###############################################
# FOLLOWING ARE UTILITIES DATA SMOOTHING
# ###############################################
# ############################
def epan_kernel(t):
    """
    Epanechnikov kernel.
    """
    return np.logical_and(t > -1.0, t < 1.0) * 3.0 * (1.0 - t**2) / 4.0

# ############################
def tri_cube_kernel(t):
    """
    Tri-cube kernel.
    """
    return np.logical_and(t > -1.0, t < 1.0) * (1.0 - abs(t**3))**3

# ############################
def gauss_kernel(t):
    """
    Gaussian kernel.
    """
    return np.exp(-t**2 / 2.0)

# ############################
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



# ###############################################
# FOLLOWING ARE UTILITIES NUMERICAL DIFFENTIATION
# ###############################################
# ###############################################
def hess_nd(f, x, args):
    """
    Compute the Hessian of a scalar-valued function f.

    Parameters
    ----------
    f : function
        The function to be differentiated to compute the Hessian. f must be 
        of the form f(x, *args) and must return a real scalar.
    x : array_like, shape (n,)
        The arguments of f that are differentiated.  I.e., hessian[i,j] is
        d^2f / d x[i] d x[j].
    args : tuple
        Other arguments to be passed to f.

    Returns
    -------
    hessian : array_like, shape(n, n)
        The Hessian of f(x).  hessian[i,j] = d^2f / d x[i] d x[j].

    Notes
    -----
    .. This is a convenience function for use with numdifftools.
    """

    def new_f(y):
        return f(y, *args)

    hess_fun = nd.Hessian(new_f)
    return hess_fun(x)


# ###############################################
# FOLLOWING ARE UTILITIES FOR MCMC PARSING
# ###############################################

# ##########################################################
def extract_1d_hist(trace, i=0, nbins=100, density=True):
    """
    Extract a 1D histogram of counter
    """
    if len(trace.shape) == 1:
        trace = trace.reshape((len(trace), 1))

    # Obtain histogram
    count, bins = np.histogram(trace[:,i], bins=nbins, density=density)
   
    # Make the bins into the bin centers, not the edges
    x = (bins[:-1] + bins[1:]) / 2.0

    return count, x


# ##########################################################
def extract_2d_hist(trace, i=0, j=1, nbins=100, density=True, meshgrid=False):
    """
    Extract a 2D histogram of counter
    """
    # Obtain histogram
    count, x_bins, y_bins = np.histogram2d(trace[:,i], trace[:,j], bins=nbins, 
                                           normed=density)
   
    # Make the bins into the bin centers, not the edges
    x = (x_bins[:-1] + x_bins[1:]) / 2.0
    y = (y_bins[:-1] + y_bins[1:]) / 2.0

    # Make mesh grid out of x_bins and y_bins
    if meshgrid:
        y, x = np.meshgrid(x, y)

    return count.transpose(), x, y


# ##########################################################
def norm_cumsum_2d(trace, i=0, j=1, nbins=100, density=True, meshgrid=False):
    """
    Returns the 1.0 - the normalized cumulative sum, normalized such
    that the maximum term in the cumulative sum is unity.  I.e., an
    isocontour on this surface at level alpha encompases a fraction
    alpha of the total probability.
    """

    # Compute the histogram
    count, x, y = extract_2d_hist(trace, i=i, j=j, nbins=nbins, density=False, 
                                  meshgrid=meshgrid)
    # Remember the shape
    shape = count.shape
    count = count.ravel()

    # Inverse sort the histogram
    isort = np.argsort(count)[::-1]
    unsort = np.argsort(isort)
    
    # Compute the cumulative sum and normalize
    count_cumsum = count[isort].cumsum()
    count_cumsum /= count_cumsum[-1]

    # Normalized, reshaped cumulative sum
    return count_cumsum[unsort].reshape(shape), x, y


# ##########################################################
def hpd(trace, mass_frac) :
    """
    Returns HPD interval containing mass_frac fraction of the total
    proability for an MCMC trace of a single variable given by trace.
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n)
    
    # Get width (in units of data) 
    # of all intervals containing n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])


# ###############################################
# FOLLOWING ARE UTILITIES FOR DATA SMOOTHING
# ###############################################

# #######################################################
def visushrink(data, threshold_factor=1.0, thresh_method='hard', 
               wavelet='sym15', level=None):
    """
    Smooth data using the VisuShrink method, described in Donoho and
    Johnstone, Biometrika, Vol. 81, No. 3 (Aug., 1994), pp. 425-455.
    """
    # Check inputs
    if wavelet not in pywt.wavelist():
        raise ValueError('Invalid wavelet.')

    if level is not None and level < 2:
        raise ValueError('level must be >= 2')

    if thresh_method == 'soft':
        thresh = pywt.thresholding.soft
    elif thresh_method == 'hard':
        thresh = pywt.thresholding.hard
    else:
        raise ValueError('Invalid thresh_method.')

    # Compute wavelet coefficients
    data_wt = pywt.wavedec(data, wavelet, level=level)

    # Approximate noise in wavelet coefficients at finest level
    sigma = np.median(abs(data_wt[-1])) / 0.6745

    # Determine universal threshold using VisuShrink
    lam = threshold_factor * sigma * np.sqrt(2.0 * np.log(len(data)))

    # Threshold each level
    for j in range(len(data_wt)):
        # Perform hard threshold
        data_wt[j] = thresh(data_wt[j], lam)

    # Compute the inverse wavelet transform
    denoised_data = pywt.waverec(data_wt, wavelet)

    return denoised_data[:len(data)]


# ###############################################
# FOLLOWING ARE UTILITIES FOR IMAGE PROCESSING
# ###############################################
# ########################################################################### #

# A dictionary containing PIL image modes and NumPy data types
PIL_to_np_dtypes = {'I': np.int32,
                    'I;8': np.uint8,
                    'I;8S': np.uint8,
                    'I;16': np.uint16,
                    'I;16S': np.uint16,
                    'I;16B': np.uint16,
                    'I;16BS': np.uint16,
                    'I;16N': np.uint16,
                    'I;16NS': np.uint16,
                    'I;32': np.uint32,
                    'I;32S': np.uint32,
                    'I;32B': np.uint32,
                    'I;32BS': np.uint32,
                    'I;32N': np.uint32,
                    'I;32NS': np.uint32,
                    'P': np.uint8,
                    'F': np.float32,
                    'F;8': np.float32,
                    'F;8S': np.float32,
                    'F;16': np.float32,
                    'F;16S': np.float32,
                    'F;16B': np.float32,
                    'F;16BS': np.float32,
                    'F;16N': np.float32,
                    'F;16NS': np.float32,
                    'F;32': np.float32,
                    'F;32S': np.float32,
                    'F;32B': np.float32,
                    'F;32BS': np.float32,
                    'F;32N': np.float32,
                    'F;32NS': np.float32,
                    'F;32F': np.float32,
                    'F;32BF': np.float32,
                    'F;32NF': np.float32,
                    'F;64F': np.float64,
                    'F;64BF': np.float64,
                    'F;64NF': np.float64,
                    'RGB': np.uint8,
                    'RGB;16B': np.uint8,
                    'BGR': np.uint8,
                    'BGR;16': np.uint16}


# ############################
def verts_to_roi(verts, size_x, size_y):
    """
    Converts list of vertices to an ROI and ROI bounding box

    Parameters
    ----------
    verts : array_like, shape (n_verts, 2)
        List of vertices of a polygon with no crossing lines.  The units
        describing the positions of the vertices are pixels.
    size_x : int
        Number of pixels in the x-direction in the image
    size_=y : int
        Number of pixels in the =y-direction in the image

    Returns
    -------
    roi : array_like, Boolean, shape (size_y, size_x)
        roi[i,j] is True if pixel (i,j) is in the ROI.
        roi[i,j] is False otherwise
    roi_bbox : array_like, shape (4,)
        roi_bbox = (min_y, min_x, max_y, max_x), the positions
        of the upper left and lower right vertices of the bounding
        box of the ROI.  To extract the square image bounding the ROI,
        use:
            im_roi = im[roi_bbox[0]:roi_bbox[2]+1, roi_bbox[1]:roi_bbox[3]+1]
    roi_box: array_like, shape is size of bounding box or ROI
        A mask for the ROI with the same dimension as the bounding 
        box.  The indexing starts at zero at the upper right corner 
        of the box.
    """

    # Make list of all points in the image in units of pixels
    x = np.arange(size_x)
    y = np.arange(size_y)
    xx, yy = np.meshgrid(x, y)
    pts = np.array(list(zip(xx.ravel(), yy.ravel())))

    # Make a path object from vertices
    p = path.Path(verts)

    # Get list of points that are in roi
    in_roi = p.contains_points(pts)

    # Convert it to an image
    roi = in_roi.reshape((size_y, size_x)).astype(np.bool)

    # Get bounding box of ROI
    regions = skimage.measure.regionprops(roi)
    roi_bbox = regions[0].bbox

    # Get ROI mask for just within bounding box
    roi_box = roi[roi_bbox[0]:roi_bbox[2]+1, roi_bbox[1]:roi_bbox[3]+1]

    # Return boolean in same shape as image
    return (roi, roi_bbox, roi_box)



# ############################
def n_frames(im_PIL):
    """
    Returns the number of frames in a PIL image.
    """
    i = 0
    while True or i > 1e8:
        try:
            im_PIL.seek(i)
            i += 1
        except EOFError:
            return im_PIL.tell()

# ############################
class XYTStack(object):
    """
    This is a class to hold image data for a stack of images.  Each
    image has x-y data and the stack is through time.
    """

    # ###########################
    def __init__(self, fname=None, directory=None, has='',
                 physical_size_x=None, physical_size_y=None, dt=None,
                 t=None, time_units='s', length_units='um', roi=None,
                 conserve_memory=False, one_channel=False, **kw):

        """
        Load images into an XYTStack instance.

        Parameters
        ----------
        fname : string
            Name of file containing a TIFF stack of images.  Ignored
            if directory is not None.
        directory : string
            Path to directory containing individual images to be stored
            in XYTStack.  Ignored is fname is not None.
        has : string
            Only files in directory with this string are considered.
            Ignored if fname is not None.
        physical_size_x : float
            Interpixel distance in x-direction in units of length_units
        physical_size_y : float
            Interpixel distance in y-direction in units of length_units
        dt : float
            Time difference between frames in units of time_units. 
            Assumed the same for all frames.  Ignored if t is not None.
        t : array_like shape (n_frames,)
            t[i] is the time point corresponding to frame i.  Units of 
            time are given by time_units
        time_units : string
            Units of time.
        length_units : string
            Units of length
        roi : list of tuples
            Each entry in the list of tuples has three arrays that
            characterize an ROI.
                Entry 1: Mask for ROI with the same dimension as whole image.
                         An entry is True if a pixel is in the ROI and
                         False otherwise.
                Entry 2: The bounding box of the ROI, 
                         (min_y, min_x, max_y, max_x), the positions
                         of the upper left and lower right vertices of the 
                         bounding box of the ROI.
                Entry 3: A mask for the ROI with the same dimension as the
                         bounding box.  The indexing starts at zero at the
                         upper right corner of the box.
        conserve_memory : Boolean
            If True, images are not loaded into RAM, but are read in as
            needed.  This will be slower than if they are loaded into RAM
            and converted into NumPy arrays.
        one_channel : Boolean
            For RGB or BRG images, consider only the first channel.

        Returns
        -------
        Does not return anything, but instantiates an XYTStack class with
        images and metadata.
        """

        # Must give either fname or directory, not both
        if fname is not None and directory is not None:
            raise ValueError('Must give EITHER fname OR directory.')
        
        # Populate object with kwargs
        self.__dict__ = kw
        self.physical_size_x = physical_size_x
        self.physical_size_y = physical_size_y
        self.time_units = time_units
        self.length_units = length_units
        self.conserve_memory = conserve_memory
        self.one_channel = one_channel

        # If it's a single file (TIFF stack), load images
        if directory is not None:
            # Make sure directory has trailing slash
            if directory[-1] == '/':
                self.directory = directory
            else:
                self.directory = directory + '/'

            # Get all files in directory that have string
            file_list = glob.glob(self.directory + '*' + has + '*')
            file_list.sort()

            # Number of time points
            self.size_t = len(file_list)
        
            # Get data type and image size
            im_0 = skimage.io.imread(file_list[0])
            self.data_type = im_0.dtype

            # Get shape
            self.size_y, self.size_x = im_0.shape[:2]
            
            # Either read in all images or make a list of files
            if conserve_memory:
                self.im_file_list = file_list
            else:
                # Store images in list of images
                self.images = []

                # Loop through and append image list
                for filename in file_list:
                    self.images.append(skimage.io.imread(filename))

        # If we read from a single file
        if fname is not None:
            # Open using PIL
            self.im_PIL = Image.open(fname)

            # Get image size (PIL goes width then height)
            self.size_x, self.size_y = self.im_PIL.size
            
            # Find out how many frames there are
            self.size_t = n_frames(self.im_PIL)

            # Get data type
            try:
                self.data_type = PIL_to_np_dtypes[self.im_PIL.mode]
            except KeyError:
                warnings.warn('Unrecognized image data type.  Using uint16.',
                              UserWarning)
                self.data_type = np.uint16

            # If we're not conserving memory, load and store all images
            if not conserve_memory:
                self.images = []

                # Go through image stack 
                for i in range(self.size_t):
                    # Get frame and pull out data
                    self.im_PIL.seek(i)
                    im_data = np.array(self.im_PIL.getdata(), self.data_type)

                    # Check to see if its multichannel
                    if len(im_data.shape) > 1:
                        if one_channel:
                            im_data = im_data[:,0].reshape(
                                (self.size_y, self.size_x))
                        else:
                            im_data = im_data.reshape(
                                (self.size_y, self.size_x, im_data.shape[1]))
                    else:
                        im_data = im_data.reshape((self.size_y, self.size_x))

                    # Store image in images list
                    self.images.append(im_data)

        # Save the time points
        if t is not None:
            if len(t) == self.size_t:
                self.t = t
                self.dt = None
            else:
                raise ValueError('len(t) must equal number of images.')
        elif dt is not None:
            self.t = np.linspace(0.0, dt * (self.size_t - 1), self.size_t)
            self.dt = dt
        else:
            self.t = np.arange(self.size_t)
            self.dt = 1.0

        # Get the maximum possible pixel value based on data type
        if not hasattr(self, 'max_pixel'):
            if self.data_type == np.uint8:
                self.max_pixel = 255
            elif self.data_type == np.uint16:
                self.max_pixel = 65535
            elif self.data_type == np.uint32:
                self.max_pixel = 4294967295
            else:  # float
                self.max_pixel = 1.0

    # ###########################
    def im(self, i, roi=None):
        """
        Returns image for frame i as a NumPy array.
        """
        
        # Check to make sure input is ok
        if i < 0:
            raise ValueError('Can''t use negative indexing.')
        elif i >= self.size_t:
            raise ValueError(
                'Index i too big, only have %d images.' % self.size_t)
        
        # Use the appropriate method to fetch the image
        if not self.conserve_memory:
            out_im = self.images[i]
            if self.one_channel and len(out_im.shape) == 3:
                out_im = out_im[:,:,0]
        elif hasattr(self, 'im_file_list'):
            out_im = skiamge.io.imread(self.im_file_list[i])
            if self.one_channel and len(out_im.shape) == 3:
                out_im = out_im[:,:,0]
        else:  # Seek in TIFF stack
            # Get frame and pull out data
            self.im_PIL.seek(i)
            im_data = np.array(self.im_PIL.getdata(), self.data_type)

            # Check to see if it's multichannel
            if len(im_data.shape) > 1:
                if self.one_channel:
                    im_data = im_data[:,0].reshape(
                        (self.size_y, self.size_x))
                else:
                    im_data = im_data.reshape(
                        (self.size_y, self.size_x, im_data.shape[1]))
            else:
                im_data = im_data.reshape((self.size_y, self.size_x))
            out_im = im_data

        # Only return bounding box of ROI
        if roi is not None:
            bbox = self.roi[roi][1]
            return out_im[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        else:
            return out_im

    # ########################
    def show_movie(self, start_frame=0, end_frame=None, skip=1):
        """
        Shows a movie of the images.
        """

        # Have to use a hack, described here:
        # http://stackoverflow.com/a/21116525/1224002, to get it to
        # work with Canopy.

        def anim_fun(start_frame, end_frame, skip):
            # Set up figure and set axis bounds
            fig = plt.figure()

            # Get end frame
            if end_frame is None:
                end_frame = self.size_t - 1

            ims = []
            for i in range(start_frame, end_frame+1, skip):
                im = plt.imshow(self.im(i), cmap=cm.gray)
                ims.append([im])

            # call the animator.
            anim = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

            # Return animation
            return anim

        # Call animation function
        ani = anim_fun(start_frame, end_frame, skip)

        # Show the movie
        plt.show()
# ########################################################################### #


# ############################
def rename_files(directory, prefix, suffix, overwrite=False):
    """
    For all files in file_list with names of the form prefix%dsuffix,
    renames the files to be of the form prefix%08dsuffix.

    If overwrite is True, will overwrite already existing files.
    """
    # Get length of file prefix and suffix
    len_prefix = len(prefix)
    len_suffix = len(suffix)
    
    # Get the list of files in the directory
    file_list = os.listdir(directory)
    
    # Make sure directory has trailing slash
    if directory[-1] != '/':
        directory += '/'

    # Go through each file and rename it if necessary
    for fname in file_list:
        # Check to make sure it matches template
        if len(fname) > len_prefix + len_suffix \
                and fname[:len_prefix] == prefix \
                and fname[-len_suffix:] == suffix:
            
            # Extract number
            n = int(fname[len_prefix:-len_suffix])
            
            # Create new filename with leading zeros
            new_fname = '%s%08d%s' % (prefix, n, suffix)
            
            # Make sure there isn't already something with that name
            if new_fname in file_list and not overwrite:
                raise RuntimeError('File %s already exists.' % new_fname)
            else:
                os.rename(directory + fname, directory + new_fname)
