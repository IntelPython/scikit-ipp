import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

cdef extern from "gaussian.c":
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


# from _ni_support.py scipy/ndimage/_ni_support.py :_extend_mode_to_code
cdef int _getIppBorderType(mode):
    """ Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 1
    elif mode == 'wrap':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')

# from _ni_support.py scipy/ndimage/_ni_support.py
def _get_output(output, input, shape=None):
    if shape is None:
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

# maybe cdef needed
# needed more correct version (guest_spatial_dim skimage)
cdef int _get_number_of_channels(image):
    if image.ndim == 2:
        channels = 1    # single (grayscale)
    elif image.ndim == 3:
        channels = image.shape[-1]   # RGB
    else:
        raise ValueError('invalid axis')
    return channels

cdef int _get_gaussian_filter_func_index(dtype, int numChannels):
    if(numChannels == 1):
        if(dtype == np.uint8):
            return 0
        elif(numChannels ==np.uint16):
            return 1
        elif(numChannels ==np.int16):
            return 2
        elif(numChannels ==np.float32):
            return 3
        else:
            # ~~~change case
            raise ValueError("Currently not supported")           
    elif(numChannels == 1):
        if(dtype == np.uint8):
            return 4
        elif(numChannels ==np.uint16):
            return 5
        elif(numChannels ==np.int16):
            return 6
        elif(numChannels ==np.float32):
            return 7
        else:
            # ~~~change case
            raise ValueError("Currently not supported")
    else:
        raise ValueError("Currently not supported")


def gaussian(cnp.ndarray image, float sigma=1.0, output=None, mode='nearest', 
                cval=0, multichannel=None, preserve_range=False, float truncate=4.0):
    image = np.asarray(image)
    if not image.flags.f_contiguous:
        image = np.ascontiguousarray(image)
    destination = _get_output(output, image)

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    kernelSize = int(tr * sd + 0.5) * 2 - 1
    cdef void* cyimage
    cdef void* cydestination

    cdef int img_width
    cdef int img_height
    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    cdef int ippBorderType = _getIppBorderType(mode)



    # If dtype is set, array is copied only if dtype does not match
    image = np.asarray(image, dtype=destination.dtype)
    # needed more correct way. Warning: conversion from 'npy_intp' to 'int', possible loss of data
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
