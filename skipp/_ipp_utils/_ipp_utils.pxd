import numpy as np
from cpython.exc cimport PyErr_SetString
from cpython.exc cimport PyErr_Occurred
from cpython.ref cimport PyObject
cimport numpy as cnp
cimport cython
cimport _ippi as ippi

cnp.import_array()

cdef int UndefValue = 432

cdef inline ippi.IppDataType __get_ipp_data_type(cnp.ndarray image):
    """
    Get equalent IppDataType for provided numpy array
    """
    cdef str kind = image.dtype.kind
    cdef int elemSize = image.dtype.itemsize
    if kind == str('b'):
        if elemSize == 1:
            # Ipp1u
            return ippi.ipp1u
    if kind == str('u'):
        if elemSize == 1:
            # Ipp8u
            return ippi.ipp8u
        elif elemSize == 2:
            # Ipp16u
            return ippi.ipp16u
        elif elemSize == 4:
            # Ipp32u
            return ippi.ipp32u
        elif elemSize == 8:
            # Ipp64u
            return ippi.ipp64u
        else:
            # ippUndef
            return ippi.ippUndef
    elif kind == str('i'):
        if elemSize == 1:
            # Ipp8s
            return ippi.ipp8s
        elif elemSize == 2:
            # Ipp16s
            return ippi.ipp16s
        elif elemSize == 4:
            # Ipp32s
            return ippi.ipp32s
        elif elemSize == 8:
            # Ipp64s
            return ippi.ipp64s
        else:
            # ippUndef
            return ippi.ippUndef
    elif kind == str('f'):
        if elemSize == 4:
            # Ipp32f
            return ippi.ipp32f
        elif elemSize == 8:
            # Ipp64f
            return ippi.ipp64f
        else:
            # ippUndef
            return ippi.ippUndef
    else:
        # ippUndef
        return ippi.ippUndef


cdef inline __get_IppBorderType(str mode):
    """ Convert an extension mode to the corresponding IPP's IppiBorderType integer code.
    """
    cdef ippi.IppiBorderType borderType
    # 'nearest' -----> IPP's ippBorderRepl
    if mode == 'nearest':
        borderType = ippi.ippBorderRepl
    # 'wrap' --------> IPP's ippBorderWrap
    elif mode == 'wrap':
        borderType = ippi.ippBorderWrap
    # 'mirror' ------> IPP's ippBorderMirror
    elif mode == 'mirror':
        borderType = ippi.ippBorderMirror
    # 'reflect' -----> IPP's ippBorderMirrorR
    elif mode == 'reflect':
        borderType = ippi.ippBorderMirrorR
    # IPP's ippBorderDefault
    elif mode == 'default':
        borderType = ippi.ippBorderDefault
    # 'constant' ----> IPP's ippBorderConst
    elif mode == 'constant':
        borderType = ippi.ippBorderConst
    # IPP's ippBorderTransp
    elif mode == 'transp':
        borderType = ippi.ippBorderTransp
    else:
        # Undef boundary mode
        return UndefValue
    return borderType


cdef inline __get_IppiInterpolationType(order):
    """ Convert a given `order` number to the Intel IPP's IppiInterpolationType enum value.
        The order of interpolation in `scikit-image`. The order has to be in the range 0-5:
            0: Nearest-neighbor    -->   ippNearest
            1: Bi-linear (default) -->   ippLinear
            2: Bi-quadratic        -->   TODO
            3: Bi-cubic            -->   ippCubic
            4: Bi-quartic          -->   TODO
            5: Bi-quintic          -->   TODO
    """
    cdef ippi.IppiInterpolationType interpolation
    # 0: Nearest-neighbor    -->   ippNearest
    if order == 0:
        interpolation = ippi.ippNearest
    # 1: Bi-linear (default) -->   ippLinear
    elif order == 1:
        interpolation = ippi.ippLinear
    # 1: Bi-cubic (default) -->   ippCubic
    elif order == 3:
        interpolation = ippi.ippCubic
    # Undef order
    else:
        return UndefValue
    return interpolation


# needed more correct version (guest_spatial_dim skimage)
cdef inline PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippi.ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)
# <<< utils module