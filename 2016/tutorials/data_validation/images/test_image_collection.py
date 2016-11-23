import glob
import re

import pytest

import numpy as np
import skimage.io


fnames = glob.glob('*.tif')
pattern = re.compile('im_[0-9]{6}.tif')
bitdepth = 12

def get_proper_frames(fnames, pattern):
    """Get all properly named frames in directory."""
    frames = []
    for fname in fnames:
        if re.fullmatch(pattern, fname) is not None:
            frames.append(int(fname[3:-4]))
    return frames


def test_file_names():
    """
    Ensure all TIFF files follow naming convention.
    """
    for fname in fnames:
        assert re.fullmatch(pattern, fname) is not None, \
                            fname + ' does not match pattern.'


def test_dropped_frames():
    """
    Check for dropped frames
    """
    # Get all proper file names (Should be all of them)
    frames = get_proper_frames(fnames, pattern)

    # Look for skipped frames
    actual_frames = set(frames)
    max_frame = max(actual_frames)
    desired_frames = set(np.arange(max_frame+1))
    skipped_frames = desired_frames - actual_frames

    assert not skipped_frames, 'Missing frames: ' + str(skipped_frames)


def test_exposure():
    """
    Check for frames with overexposure
    """
    # Get all proper file names (Should be all of them)
    frames = get_proper_frames(fnames, pattern)

    # Look for unexposed and overexposed frames
    unexposed = []
    overexposed = []
    for frame in frames:
        fname = 'im_{0:06d}.tif'.format(frame)
        im = skimage.io.imread(fname)
        if im.max() == 0:
            unexposed.append(frame)
        if im.max() == 2**bitdepth-1:
            overexposed.append(frame)

    assert not unexposed and not overexposed, \
            'unexposed: ' + str(unexposed) \
                + '  overexposed: ' + str(overexposed)
