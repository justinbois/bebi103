import numpy as np
import skimage.io

def make_random_im(height=128, width=128, max_val=4095):
    """
    Make random image.
    """
    return np.random.randint(0, max_val, size=(height, width)).astype(np.uint16)

for i in range(300):
    im = make_random_im()
    fname = 'im_%06d.tif' % i

    if i in [176, 214, 177]:
        im[np.random.randint(0, 128), np.random.randint(0, 128)] = 4095

    if i != 56:
        skimage.io.imsave(fname, im)
