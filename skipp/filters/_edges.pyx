import numpy as np
from dtype import img_as_float32
cimport numpy as cnp
cimport cython

cnp.import_array()


cdef extern from "src/edges.c":
    int PrewittFilterFLOAT32(void * pA_srcDst,
                             void * pB_srcDst,
                             int stepsize,
                             int img_width,
                             int img_height)


cdef int _get_number_of_channels(image):
    if image.ndim == 2:
        channels = 1    # single (grayscale)
    elif image.ndim == 3:
        channels = image.shape[-1]   # RGB
    else:
        raise ValueError('invalid axis')
    return channels


def prewitt(image, mask=None):
    """
    currently like skimage implementation
    """

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')
    
    out = np.sqrt(prewitt_h(image, mask) ** 2 + prewitt_v(image, mask) ** 2)
    out /= np.sqrt(2)
    return out

def prewitt_proto(image, mask=None):
    """
    Prototype of the prewitt filter, implemented with ipp primitives.

    Return the square root of the sum of squares of the horizontal
    and vertical Prewitt transforms. The edge magnitude depends slightly
    on edge directions, since the approximation of the gradient operator by
    the Prewitt operator is not completely rotation invariant. For a better
    rotation invariance, the Scharr operator should be used. The Sobel operator
    has a better rotation invariance than the Prewitt operator, but a worse
    rotation invariance than the Scharr operator.

    """
    cdef cnp.ndarray A = prewitt_h(image)
    cdef cnp.ndarray B = prewitt_v(image)

    cdef int img_width = int(A.shape[0])
    cdef int img_height = int(A.shape[1])
    cdef int stepsize = int(A.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(A)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(B)

    cdef int ippStatusIndex = 0  # OK 
    ippStatusIndex = PrewittFilterFLOAT32(cyimage,
                                          cydestination,
                                          stepsize,
                                          img_width,
                                          img_height)
    # ippStatusIndex: ipp error handler will be added
    return B
