def image_bin(img, size_bin):
    """
    function to bin a 2d image
    
    Args:
    _____________
        img: numpy.array
            image to be binned
        size_bin: int
            size of binning
    Return:
    _____________
        img_new: numpy.array
            binned image
    """
    if  not isinstance(size_bin, int):
        raise TypeError('size_bin must be int.')
    
    if img.shape[-1] % size_bin == 0:        
        return img.reshape(-1,  img.shape[1]//size_bin, size_bin, img.shape[2]//size_bin, size_bin).sum(axis = (2, 4))
    else:
        raise ValueError('size_bin is not correct.')
