# ******************************************************************************
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************

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
    """ Convert an extension mode to the corresponding
    Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) IppiBorderType integer code.
    As is in scikit-image filters module
    """
    cdef ippi.IppiBorderType borderType

    # 'nearest' -----> ippBorderRepl
    if mode == 'nearest':
        borderType = ippi.ippBorderRepl

    # 'constant' ----> ippBorderConst
    elif mode == 'constant':
        borderType = ippi.ippBorderConst

    # 'mirror' ------> ippBorderMirror
    elif mode == 'mirror':
        borderType = ippi.ippBorderMirror

    # 'reflect' -----> ippBorderMirrorR
    elif mode == 'reflect':
        borderType = ippi.ippBorderMirrorR

    # 'wrap' --------> ippBorderWrap
    elif mode == 'wrap':
        borderType = ippi.ippBorderWrap

    # Intel(R) IPP ippBorderDefault
    elif mode == 'default':
        borderType = ippi.ippBorderDefault

    # Intel(R) IPP ippBorderTransp
    elif mode == 'transp':
        borderType = ippi.ippBorderTransp

    else:
        # Undef boundary mode
        return UndefValue
    return borderType


cdef inline __get_numpy_pad_IppBorderType(str mode):
    """ Convert an extension mode to the corresponding
    Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) IppiBorderType integer code.
    Modes as is in numpy.pad
    """
    cdef ippi.IppiBorderType borderType

    # 'constant' ----> ippBorderConst
    if mode == 'constant':
        borderType = ippi.ippBorderConst

    # 'edge' -----> ippBorderRepl
    elif mode == 'edge':
        borderType = ippi.ippBorderRepl

    # Intel(R) IPP ippBorderTransp
    elif mode == 'transp':
        borderType = ippi.ippBorderTransp

    # 'reflect' ------> ippBorderMirror
    elif mode == 'reflect':
        borderType = ippi.ippBorderMirror

    # 'symmetric' -----> ippBorderMirrorR
    elif mode == 'symmetric':
        borderType = ippi.ippBorderMirrorR

    # 'wrap' --------> ippBorderWrap
    elif mode == 'wrap':
        borderType = ippi.ippBorderWrap

    # Intel(R) IPP ippBorderDefault
    elif mode == 'default':
        borderType = ippi.ippBorderDefault

    else:
        # Undef boundary mode
        return UndefValue
    return borderType


cdef inline __get_IppiInterpolationType(order):
    """ Convert a given `order` number to the Intel(R) IPP
    IppiInterpolationType enum value.
    The order of interpolation as is in `scikit-image` (0-5).
    The order has to be in the range 0-7:
        0: Nearest-neighbor    -->   ippNearest
        1: Bi-linear (default) -->   ippLinear
        2: Bi-quadratic        -->   TODO
        3: Bi-cubic            -->   ippCubic
        4: Bi-quartic          -->   TODO
        5: Bi-quintic          -->   TODO
        6: Lanczos             -->   ippLanczos
        7: Super               -->   ippSuper
    """
    cdef ippi.IppiInterpolationType interpolation
    # 0: Nearest-neighbor    -->   ippNearest
    if order == 0:
        interpolation = ippi.ippNearest
    # 1: Bi-linear (default) -->   ippLinear
    elif order == 1:
        interpolation = ippi.ippLinear
    # 1: Bi-cubic            -->   ippCubic
    elif order == 3:
        interpolation = ippi.ippCubic
    # 6: Lanczos             -->   ippLanczos
    elif order == 6:
        interpolation = ippi.ippLanczos
    # 7: Super               -->   ippSuper
    elif order == 7:
        interpolation = ippi.ippSuper
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
