def zero_crossing_filter(im, thresh):
    """
    Returns image with 1 if there is a zero crossing and 0 otherwise.
    
    thresh is the the minimal value of the gradient, as computed by Sobel
    filter, at crossing to count as a crossing.
    """
    
    def zero_cross(p):
        # Check to see if it's a crossing
        if (p[4] > 0 and p.min() < 0) or (p[4] < 0 and p.max() > 0):
            pass
        else:
            return False

        # Compute gradient
        grad = np.sqrt((p[6] + 2 * p[7] + p[8] - p[0] - 2 * p[1] - p[2])**2 
                    + (p[2] + 2 * p[5] + p[8] - p[0] - 2 * p[3] - p[6])**2)
        
        # Check to make sure gradient is big enough
        if grad < thresh:
            return False
        else:
            return True
    
    return scipy.ndimage.filters.generic_filter(
            im, zero_cross, footprint=np.ones((3,3)))
