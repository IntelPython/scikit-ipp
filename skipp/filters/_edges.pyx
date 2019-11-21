import numpy as np
from dtype import img_as_float32
cimport numpy as cnp
cimport cython

cnp.import_array()

cdef extern from "ippbase.h":
    ctypedef unsigned char  Ipp8u
    ctypedef unsigned short Ipp16u
    ctypedef unsigned int   Ipp32u
    ctypedef signed char    Ipp8s
    ctypedef signed short   Ipp16s
    ctypedef signed int     Ipp32s
    ctypedef float          Ipp32f
    # ctypedef IPP_INT64    Ipp64s
    # ctypedef IPP_UINT64   Ipp64u
    ctypedef double         pp64f


cdef extern from "src/edges.c":
    int PrewittFilterFLOAT32(void * pA_srcDst,
                             void * pB_srcDst,
                             int stepsize,
                             int img_width,
                             int img_height)

    int PrewittFilterHorizonFLOAT32(void * pSRC,
                                    int srcStep,
                                    void * pDST,
                                    int dstStep,
                                    int img_width,
                                    int img_height,
                                    int maskSize,
                                    int borderType,
                                    Ipp32f borderValue)

    int PrewittFilterVertFLOAT32(void * pSRC,
                                 int srcStep,
                                 void * pDST,
                                 int dstStep,
                                 int img_width,
                                 int img_height,
                                 int maskSize,
                                 int borderType,
                                 Ipp32f borderValue)

    int SobelFilterFLOAT32(void * pSRC,
                           int srcStep,
                           void * pDST,
                           int dstStep,
                           int img_width,
                           int img_height,
                           int maskSize,   # 33 or 55
                           int normType,   # 2 or 4 in skimage supports l2 only
                           int borderType,  # IppiBorderType
                           Ipp32f borderValue)

    int SobelFilterHorizonFLOAT32(void * pSRC,
                                  int srcStep,
                                  void * pDST,
                                  int dstStep,
                                  int img_width,
                                  int img_height,
                                  int maskSize,
                                  int borderType,
                                  Ipp32f borderValue)

    int SobelFilterVertFLOAT32(void * pSRC,
                               int srcStep,
                               void * pDST,
                               int dstStep,
                               int img_width,
                               int img_height,
                               int maskSize,
                               int borderType,
                               Ipp32f borderValue)

    int SobelFilterCrossFLOAT32(void * pSRC,
                                int srcStep,
                                void * pDST,
                                int dstStep,
                                int img_width,
                                int img_height,
                                int maskSize,
                                int borderType,
                                Ipp32f borderValue)

cdef int _getIPPNormType(normType):
    if normType == 'l1':
      return 2
    elif normType == 'l2':
      return 4
    else:
      raise RuntimeError('norm type not supported')


cdef int _get_number_of_channels(image):
    if image.ndim == 2:
        channels = 1    # single (grayscale)
    elif image.ndim == 3:
        channels = image.shape[-1]   # RGB
    else:
        raise ValueError('invalid axis')
    return channels

# ipp binary_erosion will be added for mask mode
def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        raise RuntimeError('mask mode not supported')


def sobel(cnp.ndarray image, mask=None, normType='l2'):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])
    cdef int normtype = _getIPPNormType(normType)

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 
    ippStatusIndex = SobelFilterFLOAT32(cyimage,
                                        stepsize,
                                        cydestination,
                                        stepsize,
                                        img_width,
                                        img_height,
                                        33, # mask size
                                        normtype, # l2 norm default
                                        1, # bordervalue reflect
                                        0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)



def sobel_h(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 

    ippStatusIndex = SobelFilterHorizonFLOAT32(cyimage,
                                               stepsize,
                                               cydestination,
                                               stepsize,
                                               img_width,
                                               img_height,
                                               33, # mask size
                                               1, # bordervalue reflect
                                               0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel_v(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 

    ippStatusIndex = SobelFilterVertFLOAT32(cyimage,
                                            stepsize,
                                            cydestination,
                                            stepsize,
                                            img_width,
                                            img_height,
                                            33, # mask size
                                            1, # bordervalue reflect
                                            0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel_c(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 

    ippStatusIndex = SobelFilterCrossFLOAT32(cyimage,
                                            stepsize,
                                            cydestination,
                                            stepsize,
                                            img_width,
                                            img_height,
                                            33, # mask size
                                            1, # bordervalue reflect
                                            0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


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


def prewitt_h(image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 
    ippStatusIndex = PrewittFilterHorizonFLOAT32(cyimage,
                                                 stepsize,
                                                 cydestination,
                                                 stepsize,
                                                 img_width,
                                                 img_height,
                                                 33, # mask size
                                                 1, # bordervalue reflect
                                                 0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)

def prewitt_v(image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    image = img_as_float32(image)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    if _get_number_of_channels(image) is not 1:
        raise ValueError('invalid axis')

    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK 
    ippStatusIndex = PrewittFilterVertFLOAT32(cyimage,
                                             stepsize,
                                             cydestination,
                                             stepsize,
                                             img_width,
                                             img_height,
                                             33, # mask size
                                             1, # bordervalue reflect
                                             0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)
