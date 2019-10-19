import numpy as np
# from dtype import img_as_float32
from cpython.exc cimport PyErr_SetString
from cpython.exc cimport PyErr_Occurred
from cpython.ref cimport PyObject
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

# gaussian
cdef extern from "src/gaussian.c":
    int  GaussianFilter(int input_index,
                        int output_index,
                        void * pInput,
                        void * pOutput,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int ippBorderType,
                        float ippBorderValue)

# edges
cdef extern from "src/edges.c":
    int FilterBorderFLOAT32(void * pSRC,
                            int srcStep,
                            void * pDST,
                            int dstStep,
                            int img_width,
                            int img_height,
                            int borderType)

    int LaplaceFilterFLOAT32(void * pSRC,
                             int srcStep,
                             void * pDST,
                             int dstStep,
                             int img_width,
                             int img_height,
                             int maskSize,
                             int borderType,
                             Ipp32f borderValue)

    int LaplaceFilterFLOAT32RGB(void * pSRC,
                                int srcStep,
                                void * pDST,
                                int dstStep,
                                int img_width,
                                int img_height,
                                int maskSize,
                                int borderType,
                                Ipp32f borderValue)

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


cdef extern from "src/dtypes.c":
    int convert(int index1,
                int index2,
                void * pSrc,
                void * pDst,
                int numChannels,
                int img_width,
                int img_height)


cdef extern from "ipptypes.h":
    ctypedef int IppStatus

cdef extern from "ipptypes.h":
    ctypedef enum IppiBorderType:
        ippBorderRepl = 1
        ippBorderWrap = 2
        ippBorderMirror = 3    # left border: 012... -> 21012...
        ippBorderMirrorR = 4  # left border: 012... -> 210012...
        ippBorderDefault = 5
        ippBorderConst = 6
        ippBorderTransp = 7


cdef extern from "ippcore.h":
    const char * ippGetStatusString(IppStatus stsCode)

# >>> utiles module
"""
ctypedef struct dtype_meta:
    cdef int IppDataType
    cdef int ippNumpyTableIndex
"""

cdef int __ipp_equalent_number_for_numpy(cnp.ndarray image):
    # TODO
    # return int -> ctypedef enum elems
    # change str ->
    cdef str kind = image.dtype.kind
    cdef int elemSize = image.dtype.itemsize
    if kind == str('u'):
        if elemSize == 1:
            # Ipp8u
            return 0
        elif elemSize == 2:
            # Ipp16u
            return 2
        elif elemSize == 4:
            # Ipp32u
            return 4
        elif elemSize == 8:
            # Ipp64u
            return 6
        else:
            # ippUndef
            return -1
    elif kind == str('i'):
        if elemSize == 1:
            # Ipp8s
            return 1
        elif elemSize == 2:
            # Ipp16s
            return 3
        elif elemSize == 4:
            # Ipp32s
            return 5
        elif elemSize == 8:
            # Ipp64s
            return 7
        else:
            # ippUndef
            return -1
    elif kind == str('f'):
        if elemSize == 4:
            # Ipp32f
            return 8
        elif elemSize == 8:
            # Ipp64f
            return 9
        else:
            # ippUndef
            return -1
    else:
        # ippUndef
        return -1


cdef int __get_IppBorderType(str mode):
    """ Convert an extension mode to the corresponding IPP's IppiBorderType integer code.
    """
    # add border types defenitions from ipptypes.h
    cdef int borderType
    # 'nearest' -----> IPP's ippBorderRepl
    if mode == 'nearest':
        borderType = 1
    # 'wrap' --------> IPP's ippBorderWrap
    elif mode == 'wrap':
        borderType = 2
    # 'mirror' ------> IPP's ippBorderMirror
    elif mode == 'mirror':
        borderType = 3
    # 'reflect' -----> IPP's ippBorderMirrorR
    elif mode == 'reflect':
        borderType = 4
    # IPP's ippBorderDefault
    elif mode == 'default':
        borderType = 5
    # 'constant' ----> IPP's ippBorderConst
    elif mode == 'constant':
        borderType = 6
    # IPP's ippBorderTransp
    elif mode == 'transp':
        borderType = 7
    else:
        # Undef boundary mode
        borderType = -1
    return borderType

# needed more correct version (guest_spatial_dim skimage)
cdef PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)

# TODO
# update
cdef int _get_number_of_channels(cnp.ndarray image):
    cdef int numChannels
    cdef int image_ndim = image.ndim
    cdef int shape_last_value = image.shape[-1]
    if(image.ndim == 2):
        numChannels = 1
        # TODO image.shape[-1]
    elif(image.ndim == 3) & (image.shape[3] == 3):
        numChannels = 3
    else:
        numChannels = -1
    return numChannels


cdef __convert(cnp.ndarray source, cnp.ndarray destination, int numChannels, int index1, int index2):
    cdef int ippStatusIndex = 0
    cdef int img_width = source.shape[0]
    cdef int img_height = source.shape[1]

    # TODO change to platform aware integer
    cdef int stepsize = source.strides[0]

    cdef void * cysource
    cdef void * cydestination

    cdef int py_array_type = cnp.PyArray_TYPE(source)

    cysource = <void*> cnp.PyArray_DATA(source)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    ippStatusIndex = convert(index1,
                             index2,
                             cysource,
                             cydestination,
                             numChannels,
                             img_width,
                             img_height)
    __get_ipp_error(ippStatusIndex)


# from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py
def convert_to_float(image, preserve_range):
    """Convert input image to double image with the appropriate range.
    """
    if preserve_range:
        return image.astype(np.float32)
    # TODO add img_as_float32
    else:
        raise ValueError("Currently not supported")
# <<< utiles module


# >>> gaussian filter module
cdef __pass_ipp_gaussian(cnp.ndarray source, cnp.ndarray destination, int source_index, int destination_index,
                         int numChannels, float sigma, float truncate, int ippBorderType, float ippBorderValue):

    cdef int ippStatusIndex = 0   # OK

    cdef void * cysource
    cdef void * cydestination

    cysource = <void*> cnp.PyArray_DATA(source)
    cydestination = <void*> cnp.PyArray_DATA(destination)
    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    # ~ use double cast
    cdef int kernelSize = int(truncate * sigma + 0.5) * 2 - 1

    # needed more correct way. Warning: conversion from 'npy_intp'
    # to 'int', possible loss of data
    cdef int img_width = source.shape[0]
    cdef int img_height = source.shape[1]

    # pass to IPP the source and destination arrays
    ippStatusIndex = GaussianFilter(source_index,
                                    destination_index,
                                    cysource,
                                    cydestination,
                                    img_width,
                                    img_height,
                                    numChannels,
                                    sigma,
                                    kernelSize,
                                    ippBorderType,
                                    ippBorderValue)
    __get_ipp_error(ippStatusIndex)

cpdef gaussian(image, sigma=1.0, output=None, mode='nearest', cval=0,
               multichannel=None, preserve_range=False, truncate=4.0):
    """
    Parameters
    ----------
    image :
    sigma :
    output :
    mode :
    cval :
    multichannel :
    preserve_range :
    truncate :
    """
    # TODO
    # add warnings for multichannel

    # TODO
    # def test_dimensiona_error_gaussian():

    # TODO
    # get input require
    # TODO module with numpy.require to provid type that satisfies requirements.

    # get output
    shape = image.shape
    if output is None:
        output_dtype = image.dtype.name
        output = np.empty_like(image, dtype=output_dtype, order='C')
    # TODO
    elif type(output) in [type(type), type(np.zeros((4,)).dtype)]:
        output_dtype = output
        output = np.zeros(shape, dtype=output_dtype)
    elif isinstance(output, np.ndarray):
        output_dtype = output.dtype
        # TODO
        # module with numpy.require to provid type that satisfies requirements.
    else:
        raise ValueError("Incorrect output value")

    # TODO
    # add correct preserve range module
    # image = convert_to_float(image)

    # TODO
    # add case when sigma is zero -> IPP funcs raises error

    # TODO
    # add case when dtype is np.int64, np.uint64

    # TODO
    # in separate cdef func
    cdef int numChannels
    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    cdef int ippBorderType = __get_IppBorderType(mode)
    if(ippBorderType == -1):
        raise ValueError("Boundary mode not supported")

    cdef int image_index = __ipp_equalent_number_for_numpy(image)
    if(image_index == -1):
        raise ValueError("Undefined ipp data type")
    cdef int output_index = __ipp_equalent_number_for_numpy(output)
    if(output_index == -1):
        raise ValueError("Undefined ipp data type")

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)
    __pass_ipp_gaussian(image, output, image_index, output_index, numChannels, sd, tr, ippBorderType, ippBorderValue)
    return output
# <<< gaussian filter module


# >>> edges module
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


def laplace(image, ksize=3, mask=None):
    """
    //
    //                               -1 -1 -1
    //              Laplace (3x3)    -1  8 -1
    //                               -1 -1 -1

    """
    # image = img_as_float32(image)
    cdef int numChannels = _get_number_of_channels(image)
    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK

    if(numChannels == 1):
        ippStatusIndex = LaplaceFilterFLOAT32(cyimage,
                                              stepsize,
                                              cydestination,
                                              stepsize,
                                              img_width,
                                              img_height,
                                              33,  # mask size
                                              1,   # bordervalue reflect
                                              0)
    elif(numChannels == 3):
        ippStatusIndex = LaplaceFilterFLOAT32RGB(cyimage,
                                                 stepsize,
                                                 cydestination,
                                                 stepsize,
                                                 img_width,
                                                 img_height,
                                                 33,  # mask size
                                                 1,   # bordervalue reflect
                                                 0)
    else:
        raise ValueError("Currently not supported")
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def laplace1(image, ksize=3, mask=None):
    """
    //
    //                                0 -1  0
    //              Laplace (3x3)    -1  4 -1
    //                                0 -1  0

    """
    # image = img_as_float32(image)
    cdef int numChannels = _get_number_of_channels(image)
    destination = np.zeros_like(image, dtype=np.float32, order='c')

    cdef int img_width = int(image.shape[0])
    cdef int img_height = int(image.shape[1])
    cdef int stepsize = int(image.strides[0])

    cdef void * cyimage = <void * > cnp.PyArray_DATA(image)
    cdef void * cydestination = <void * > cnp.PyArray_DATA(destination)
    cdef int ippStatusIndex = 0  # OK

    if(numChannels == 1):
        ippStatusIndex = FilterBorderFLOAT32(cyimage,
                                             stepsize,
                                             cydestination,
                                             stepsize,
                                             img_width,
                                             img_height,
                                             1)  # bordervalue reflect
    elif(numChannels == 3):
        raise ValueError("Currently not supported")
    else:
        raise ValueError("Currently not supported")
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel(cnp.ndarray image, mask=None, normType='l2'):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    # image = img_as_float32(image)

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
                                        33,        # mask size
                                        normtype,  # l2 norm default
                                        1,         # bordervalue reflect
                                        0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel_h(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    # image = img_as_float32(image)

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
                                               33,   # mask size
                                               1,    # bordervalue reflect
                                               0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel_v(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    # image = img_as_float32(image)

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
                                            33,    # mask size
                                            1,     # bordervalue reflect
                                            0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def sobel_c(cnp.ndarray image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    # image = img_as_float32(image)

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
                                             33,    # mask size
                                             1,     # bordervalue reflect
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
    # image = img_as_float32(image)

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
                                                 33,    # mask size
                                                 1,     # bordervalue reflect
                                                 0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)


def prewitt_v(image, mask=None):
    # currently doesnt use `mask`
    # image = np.asarray(image, dtype=np.float32)

    # curerntly uses skimage's utils.img_as_float
    # image = img_as_float32(image)

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
                                              33,    # mask size
                                              1,     # bordervalue reflect
                                              0)
    # ippStatusIndex: ipp error handler will be added
    return _mask_filter_result(destination, mask)
# <<< edges module


# >>> median filter module
cdef extern from "src/median.c":
    int MedianFilter_32f_C1_3D(void * pSRC,
                               void * pDST,
                               int img_width,
                               int img_height,
                               int img_depth,
                               int mask_width,
                               int mask_height,
                               int mask_depth,
                               int borderType)

    int MedianFilterFLOAT32(void * pSRC,
                            int stepSize,
                            void * pDST,
                            int img_width,
                            int img_height,
                            int mask_width,
                            int mask_height,
                            int borderType)  # const float * pBorderValue) <-----~~

__all__ = ['median', 'median_1']


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


def median(image, selem=None, out=None, mask=None, shift_x=False,
           shift_y=False, mode='nearest', cval=0.0, behavior='ipp'):
    if(behavior != 'ipp'):
        raise ValueError("There is only 'ipp' behavior is allowed")

    destination = _get_output(out, image)

    # warn about selem and mask

    cdef void * cyimage
    cdef void * cydestination

    cdef int img_width
    cdef int img_height
    cdef int img_depth = 1
    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    # raise error if numChannels is not 1
    cdef int ippBorderType = __get_IppBorderType(mode)

    # for ippMaskSize struct
    # median uses only selem's parametr
    cdef int selem_width = selem.shape[0]
    cdef int selem_height = selem.shape[1]
    cdef int selem_depth = 1

    # needed more correct way. Warning: conversion from 'npy_intp' to 'int', possible loss of data
    img_width = image.shape[0]
    img_height = image.shape[1]
    stepsize = image.strides[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    cdef int ippStatusIndex = 0  # OK

    ippStatusIndex = MedianFilter_32f_C1_3D(cyimage,
                                            cydestination,
                                            img_width,
                                            img_height,
                                            img_depth,
                                            selem_width,
                                            selem_height,
                                            selem_depth,
                                            ippBorderType)
    __get_ipp_error(ippStatusIndex)
    return destination


def median_1(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False, mode='nearest', cval=0.0, behavior='ipp'):
    if(behavior != 'ipp'):
        raise ValueError("There is only 'ipp' behavior is allowed")

    destination = _get_output(out, image)

    # warn about selem and mask

    cdef void * cyimage
    cdef void * cydestination

    cdef int img_width
    cdef int img_height

    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    # raise error if numChannels is not 1
    cdef int ippBorderType = __get_IppBorderType(mode)

    cdef int selem_width = selem.shape[0]
    cdef int selem_height = selem.shape[1]

    img_width = image.shape[0]
    img_height = image.shape[1]
    stepsize = image.strides[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    cdef int ippStatusIndex = 0  # OK

    ippStatusIndex = MedianFilterFLOAT32(cyimage,
                                         stepsize,
                                         cydestination,
                                         img_width,
                                         img_height,
                                         selem_width,
                                         selem_height,
                                         ippBorderType)
    __get_ipp_error(ippStatusIndex)
    return destination
# <<< median filter module


# >>> for tests
def _get_cy__convert(source, destination, index1, index2):
    # __convert(cnp.ndarray source, cnp.ndarray destination, int index)
    # for the tests
    cdef int numChannels = _get_number_of_channels(source)
    if(numChannels == -1):
        ValueError("Expected 2D array with 1 or 3 channels, got %iD." % source.ndim)
    return __convert(source, destination, numChannels, index1, index2)


def _get_cy__ipp_equalent_number_for_numpy(image):
    # cdef int __ipp_equalent_number_fornumpy(cnp.ndarray image):
    # for tests
    return __ipp_equalent_number_for_numpy(image)


def _get_cy__get_IppBorderType(mode):
    # cdef int __get_IppBorderType(str mode)
    return __get_IppBorderType(mode)


def _get_cy__get_number_of_channels(image):
    # cdef _get_number_of_channels(cnp.ndarray image):
    return _get_number_of_channels(image)
# <<< for tests
