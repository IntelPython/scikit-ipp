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

# needed more correct version (guest_spatial_dim skimage)
cdef PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)


cdef __img_as_float(cnp.ndarray source, cnp.ndarray destination, int numChannels, int input_index,
                    int output_index):
    cdef int ippStatusIndex = 0
    cdef int img_width = source.shape[1]
    cdef int img_height = source.shape[0]

    cdef void * cysource
    cdef void * cydestination

    cysource = <void*> cnp.PyArray_DATA(source)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    ippStatusIndex = image_ScaleC(input_index, output_index, cysource, cydestination,
                                                  numChannels, img_width, img_height)
    __get_ipp_error(ippStatusIndex)


cpdef img_as_float(image):
    # TODO
    # in separate cdef func
    cdef int numChannels
    if(image.ndim == 1):
        numChannels = 1
    elif(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    cdef int image_index = __ipp_equalent_number_for_numpy(image)

    if(image_index == -1):
        raise ValueError("Undefined ipp data type")
    # TODO
    elif(image_index == ipp64u_index):   # if input image np.uint64
        raise ValueError("image int 64 bit int is currently not supported")
    # TODO
    elif(image_index == ipp64s_index):   # if input image np.int64
        raise ValueError("image int 64 bit int is currently not supported")

    output = np.zeros_like(image, dtype=np.float64)
    # cdef float_type_index = 1   # for covertToFloatTable
    __img_as_float(image, output, numChannels, image_index, ipp64f_index)

    return output
