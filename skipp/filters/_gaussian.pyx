import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

cdef extern from "src/gaussian.c":
    int  GaussianFilter(int index,
                        void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue)


def _getIppBorderType(mode):
    """ Convert an extension mode to the corresponding IPP's IppiBorderType integer code.
    """
    # 'nearest' -----> IPP's ippBorderRepl
    if mode == 'nearest':
        return 1
    # 'wrap' --------> IPP's ippBorderWrap
    elif mode == 'wrap':
        return 2
    # 'mirror' ------> IPP's ippBorderMirror
    elif mode == 'mirror':
        return 3
    # 'reflect' -----> IPP's ippBorderMirrorR
    elif mode == 'reflect':
        return 4
    # IPP's ippBorderDefault
    elif mode == 'default':
        return 5
    # 'constant' ----> IPP's ippBorderConst
    elif mode == 'constant':
        return 6
    # IPP's ippBorderTransp
    elif mode == 'transp':
        return 7
    else:
        raise RuntimeError('boundary mode not supported')


# from _ni_support.py scipy/ndimage/_ni_support.py
def _get_output(output, input):
    shape = input.shape
    if output is None:
        # as in skimage gaussian filter logic
        # Integer arrays are converted to float.
        output = np.zeros(shape, dtype=np.float32)
    elif type(output) in [type(type), type(np.zeros((4,)).dtype)]:
        output = np.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


# needed more correct version (guest_spatial_dim skimage)
def _get_number_of_channels(image):
    if image.ndim == 2:
        channels = 1    # single (grayscale)
    elif image.ndim == 3 and image.shape[-1] == 3:
        channels = 3   # 3 channels
    else:
        ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)
    return channels


def _get_gaussian_filter_func_index(dtype, int numChannels):
    if(numChannels == 1):
        if(dtype == np.uint8):
            return 0
        elif(dtype == np.uint16):
            return 1
        elif(dtype == np.int16):
            return 2
        elif(dtype == np.float32):
            return 3
        else:
            raise ValueError("Currently not supported")
    elif(numChannels == 3):
        if(dtype == np.uint8):
            return 4
        elif(dtype == np.uint16):
            return 5
        elif(dtype == np.int16):
            return 6
        elif(dtype == np.float32):
            return 7
        else:
            raise ValueError("Currently not supported")
    else:
        raise ValueError("Currently not supported")


# from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py
# convert_to_float
def convert_to_float(image, preserve_range):
    """Convert input image to double image with the appropriate range.
    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    image : ndarray
        Transformed version of the input.
    """
    if preserve_range:
        return image.astype(np.float32)
    # TODO add img_as_float32
    else:
        raise ValueError("Currently not supported")


cpdef gaussian(image, sigma=1.0, output=None, mode='nearest', cval=0,
               multichannel=None, preserve_range=False, truncate=4.0):
    # TODO
    # use numpy.require to provid type that satisfies requirements.
    # image = convert_to_float(image)

    # TODO
    # add warnings for multichannel

    cdef cnp.ndarray destination = _get_output(output, image)

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)

    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    cdef int kernelSize = int(tr * sd + 0.5) * 2 - 1

    cdef void * cyimage
    cdef void * cydestination

    # TODO
    # use IPP's ippiFilterGaussian_<> ---> platform-aware functions
    # int kernelSize --> cnp.uint64_t or ctypedef unsigned long
    cdef int img_width
    cdef int img_height
    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    cdef int ippBorderType = _getIppBorderType(mode)

    # If dtype is set, array is copied only if dtype does not match
    image = np.asarray(image, dtype=destination.dtype)
    # needed more correct way. Warning: conversion from 'npy_intp'
    # to 'int', possible loss of data
    img_width = image.shape[0]
    img_height = image.shape[1]
    stepsize = image.strides[0]
    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    cdef int index = _get_gaussian_filter_func_index(destination.dtype,
                                                     numChannels)
    cdef int ippStatusIndex = 0  # OK
    ippStatusIndex = GaussianFilter(index,
                                    cyimage,
                                    cydestination,
                                    img_width,
                                    img_height,
                                    numChannels,
                                    sigma,
                                    kernelSize,
                                    stepsize,
                                    ippBorderType,
                                    ippBorderValue)
    # ippStatusIndex: ipp error handler will be added
    return destination
