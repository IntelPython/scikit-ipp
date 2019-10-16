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


cdef extern from "ippcore.h":
    const char * ippGetStatusString(IppStatus stsCode)

# >>> utiles module
"""
ctypedef struct dtype_meta:
    cdef int IppDataType
    cdef int ippNumpyTableIndex
"""

cdef int __ipp_equalent_number_for_numpy(cnp.ndarray image):
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
        # TODO: set exception behavior
        PyErr_SetString(ValueError, "boundary mode not supported")

# needed more correct version (guest_spatial_dim skimage)
cdef PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)


cdef int _get_number_of_channels(cnp.ndarray image):
    cdef int numChannels
    if image.ndim == 2:
        numChannels = 1    # single (grayscale)
    elif image.ndim == 3 and image.shape[-1] == 3:
        numChannels = 3   # 3 channels
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
IPP_GAUSSIAN_SUPPORTED_DTYPES = [np.uint8, np.uint16, np.int16, np.float32]


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


cdef __pass_ipp_gaussian(cnp.ndarray source, cnp.ndarray destination, float sigma, float truncate,
                         int ippBorderType, float ippBorderValue):

    cdef int index  # index for _get_gaussian_filter_func_index
    cdef int ippStatusIndex = 0

    cdef void * cysource
    cdef void * cydestination

    cysource = <void*> cnp.PyArray_DATA(source)
    cydestination = <void*> cnp.PyArray_DATA(destination)
    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    cdef int kernelSize = int(truncate * sigma + 0.5) * 2 - 1

    cdef int numChannels = _get_number_of_channels(source)

    # needed more correct way. Warning: conversion from 'npy_intp'
    # to 'int', possible loss of data
    cdef int img_width = source.shape[0]
    cdef int img_height = source.shape[1]

    # TODO change to platform aware integer
    cdef int stepsize = source.strides[0]
    # pass to IPP the source and destination arrays
    index = _get_gaussian_filter_func_index(destination.dtype, numChannels)
    # ~~~ delete number of channels from here
    ippStatusIndex = GaussianFilter(index,
                                    cysource,
                                    cydestination,
                                    img_width,
                                    img_height,
                                    numChannels,
                                    sigma,
                                    kernelSize,
                                    stepsize,
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
    # check input
    # is input correct array
    # use numpy.require to provid type that satisfies requirements.
    # image = convert_to_float(image)

    # TODO
    # add correct preserve range module

    shape = image.shape

    input_dtype = image.dtype

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)

    if output is None:
        output_dtype = None
    elif isinstance(output, np.dtype):
        output_dtype = output
        output = np.zeros(shape, dtype=output_dtype)
    elif isinstance(output, np.ndarray):
        output_dtype = output.dtype
    else:
        raise ValueError("not correct output value or ~~~")

    # check if zero sigma
    # ~~~~~~~~
    # if sigma == 0:
    #    pass

    cdef int numChannels = _get_number_of_channels(image)
    if(numChannels == -1):
        ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    cdef int ippBorderType = __get_IppBorderType(mode)

    cdef int index1
    cdef int index2

    if output_dtype is None:
        if input_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
            # create output as input dtype
            output = np.zeros(shape, dtype=input_dtype)

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)

        elif input_dtype == np.int8:
            # convert input to np.uint8 ---> converted copy of input

            # incorrect convert
            # image = image.astype(dtype=np.uint8, order='C', copy=True)

            ipp_src = np.zeros(shape, dtype=np.uint8)

            index1 = __ipp_equalent_number_for_numpy(image)
            if index1 == -1:
                raise ValueError("Unsupported numpy dtype")

            index2 = __ipp_equalent_number_for_numpy(ipp_src)
            if index2 == -1:
                raise ValueError("Unsupported numpy dtype")

            __convert(image, ipp_src, numChannels, index1, index2)
            # __convert_8s_8u(image, ipp_src)

            # create output as np.uint8
            ipp_dst = np.zeros(shape, dtype=np.uint8)

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(ipp_src, ipp_dst, sd, tr, ippBorderType, ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del ipp_src

            # convert destination to np.int8
            output = np.zeros(shape, dtype=input_dtype)
            # __convert_8u_8s(ipp_dst, output)

            index1 = __ipp_equalent_number_for_numpy(ipp_dst)
            if index1 == -1:
                raise ValueError("Unsupported numpy dtype")
            index2 = __ipp_equalent_number_for_numpy(output)
            if index2 == -1:
                raise ValueError("Unsupported numpy dtype")
            __convert(ipp_dst, output, numChannels, index1, index2)

        else:
            # convert input to np.float32 ---> converted copy of input
            image = image.astype(dtype=np.float32, order='C', copy=True)

            # create output as np.float32 ---> converted copy of output
            output = np.zeros(shape, dtype=np.float32)

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del image

            # convert destination to np.int8 ---> output
            output = output.astype(dtype=input_dtype, order='C')
    elif output_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
        if input_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
            if output_dtype == input_dtype:
                # pass to IPP the source and destination arrays
                __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
            else:
                # convert input to output dtype
                image = image.astype(dtype=output_dtype, order='C', copy=True)

                # pass to IPP the source and destination arrays
                __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
        else:
            # convert input to outputdtype and
            image = image.astype(dtype=output_dtype, order='C', copy=True)
            # converted input is source, output is destination
            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
    elif output_dtype is np.int8 and input_dtype is np.uint8:
        # convert output to np.uint8 ---> converted copy of output
        # converted copy of output is destination
        # input is source
        # pass to IPP the source and destination arrays
        # convert destination and save in output array (
        # or create copy of destination in output dtype and copy all  in output)
        # return output
        raise RuntimeError("currently not implemented 5")
    else:
        if input_dtype == np.float32:
            # input is source
            # convert output to np.float32---> converted copy of output
            # converted copy of output is destination
            # pass to IPP source and destination
            # convert destination and save in output array (
            # or create copy of destination in output dtype and copy all  in output)
            # return output
            raise RuntimeError("currently not implemented 6")
        else:
            # convert input to np.float32 ---> converted copy of input
            # converted copy of input is source
            # convert output to np.float32---> converted copy of output
            # converted copy of output is destination
            # pass to IPP source and destination
            # delete source (copy of input) ---> free copy of input from mem
            # convert destination and save in output array (
            # or create copy of destination in output dtype and copy all  in output)
            # return output
            raise RuntimeError("currently not implemented 7")

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
