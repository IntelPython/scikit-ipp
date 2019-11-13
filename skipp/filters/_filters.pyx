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
    int GaussianFilter(IppDataTypeIndex input_index,
                       IppDataTypeIndex output_index,
                       void * pInput,
                       void * pOutput,
                       int img_width,
                       int img_height,
                       int numChannels,
                       float sigma,
                       int kernelSize,
                       IppiBorderType ippBorderType,
                       float ippBorderValue,
                       preserve_range_flag preserve_range)


cdef extern from "src/dtypes.c":
    int image_ScaleC(IppDataTypeIndex src_index,
                     IppDataTypeIndex dst_index,
                     void * pSrc,
                     void * pDst,
                     int numChannels,
                     int img_width,
                     int img_height,
                     preserve_range_flag preserve_range);


cdef extern from "src/dtypes.h":
    ctypedef enum IppDataTypeIndex:
        ipp8u_index = 0
        ipp8s_index = 1
        ipp16u_index = 2
        ipp16s_index = 3
        ipp32u_index = 4
        ipp32s_index = 5
        ipp64u_index = 6
        ipp64s_index = 7
        ipp32f_index = 8
        ipp64f_index = 9
        ippUndef_index = -1

    ctypedef enum preserve_range_flag:
        preserve_range_false = 0
        preserve_range_true = 1
        preserve_range_true_for_small_bitsize_src = 2


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
cdef IppDataTypeIndex __ipp_equalent_number_for_numpy(cnp.ndarray image):
    cdef str kind = image.dtype.kind
    cdef int elemSize = image.dtype.itemsize
    if kind == str('u'):
        if elemSize == 1:
            # Ipp8u
            return ipp8u_index
        elif elemSize == 2:
            # Ipp16u
            return ipp16u_index
        elif elemSize == 4:
            # Ipp32u
            return ipp32u_index
        elif elemSize == 8:
            # Ipp64u
            return ipp64u_index
        else:
            # ippUndef
            return ippUndef_index
    elif kind == str('i'):
        if elemSize == 1:
            # Ipp8s
            return ipp8s_index
        elif elemSize == 2:
            # Ipp16s
            return ipp16s_index
        elif elemSize == 4:
            # Ipp32s
            return ipp32s_index
        elif elemSize == 8:
            # Ipp64s
            return ipp64s_index
        else:
            # ippUndef
            return ippUndef_index
    elif kind == str('f'):
        if elemSize == 4:
            # Ipp32f
            return ipp32f_index
        elif elemSize == 8:
            # Ipp64f
            return ipp64f_index
        else:
            # ippUndef
            return ippUndef_index
    else:
        # ippUndef
        return ippUndef_index


def __get_IppBorderType(str mode):
    """ Convert an extension mode to the corresponding IPP's IppiBorderType integer code.
    """
    cdef IppiBorderType borderType
    # 'nearest' -----> IPP's ippBorderRepl
    if mode == 'nearest':
        borderType = ippBorderRepl
    # 'wrap' --------> IPP's ippBorderWrap
    elif mode == 'wrap':
        borderType = ippBorderWrap
    # 'mirror' ------> IPP's ippBorderMirror
    elif mode == 'mirror':
        borderType = ippBorderMirror
    # 'reflect' -----> IPP's ippBorderMirrorR
    elif mode == 'reflect':
        borderType = ippBorderMirrorR
    # IPP's ippBorderDefault
    elif mode == 'default':
        borderType = ippBorderDefault
    # 'constant' ----> IPP's ippBorderConst
    elif mode == 'constant':
        borderType = ippBorderConst
    # IPP's ippBorderTransp
    elif mode == 'transp':
        borderType = ippBorderTransp
    else:
        # Undef boundary mode
        return -1
    return borderType

def __get_numChannels(image):
    cdef int numChannels
    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

def __get_output(image, output):
    # TODO
    # get output require
    # module with numpy.require to provid type that satisfies requirements.
    shape = image.shape
    if output is None:
        output_dtype = image.dtype.name
        output = np.empty_like(image, dtype=output_dtype, order='C')
    elif type(output) in [type(type), type(np.zeros((4,)).dtype)]:
        output_dtype = output
        output = np.zeros(shape, dtype=output_dtype)
    elif isinstance(output, np.ndarray):
        if output.shape != shape:
            raise RuntimeError("output shape not correct")
        # output_dtype = output.dtype
        # TODO
        # module with numpy.require to provid type that satisfies requirements.
    else:
        raise ValueError("Incorrect output value")
    return output

# needed more correct version (guest_spatial_dim skimage)
cdef PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)
# <<< utiles module

# >>> gaussian filter module
cdef __pass_ipp_gaussian(cnp.ndarray source,
                         cnp.ndarray destination,
                         IppDataTypeIndex source_index,
                         IppDataTypeIndex destination_index,
                         int numChannels,
                         float sigma,
                         float truncate,
                         IppiBorderType ippBorderType,
                         float ippBorderValue,
                         preserve_range_flag preserve_range):

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
    cdef int img_width = source.shape[1]
    cdef int img_height = source.shape[0]

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
                                    ippBorderValue,
                                    preserve_range)
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

    # ~~~~
    # cdef int numChannels = int(__get_numChannels(image))
    cdef int numChannels
    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    output = __get_output(image, output)

    if sigma == 0.0:
        output[...] = image[...]
        return output

    cdef IppiBorderType ippBorderType = __get_IppBorderType(mode)
    if(ippBorderType == -1):
        raise ValueError("Boundary mode not supported")

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)

    cdef preserve_range_flag preserve_Range = preserve_range_false

    if preserve_range:
        preserve_Range = preserve_range_true

    cdef IppDataTypeIndex image_index = __ipp_equalent_number_for_numpy(image)
    if(image_index == ippUndef_index):
        raise ValueError("Undefined ipp data type")
    elif(image_index == ipp64u_index or image_index == ipp64s_index):   # input image is np.uint64 or np.int64
        # make a np.float32 copy
        image = image.astype(np.float32, order='C')
        image_index = ipp32f_index


    cdef IppDataTypeIndex output_index = __ipp_equalent_number_for_numpy(output)
    if(output_index == ippUndef_index):
        raise ValueError("Undefined ipp data type")
    elif(output_index == ipp64u_index or output_index == ipp64s_index):  # output image is np.uint64 or np.int64
        # TODO
        # add case when dtype is np.int64, np.uint64
        raise ValueError("output 64 bit is currently not supported")
    else:
        __pass_ipp_gaussian(image, output, image_index, output_index, numChannels, sd,
                            tr, ippBorderType, ippBorderValue, preserve_Range)
    return output
# <<< gaussian filter module

# >>> for tests
def _get_cy__ipp_equalent_number_for_numpy(image):
    # cdef int __ipp_equalent_number_fornumpy(cnp.ndarray image):
    # for tests
    return __ipp_equalent_number_for_numpy(image)


def _get_cy__get_IppBorderType(mode):
    # cdef int __get_IppBorderType(str mode)
    return __get_IppBorderType(mode)
# <<< for tests
