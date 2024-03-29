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

from __future__ import absolute_import
include "../_ipp_utils/_ipp_utils.pxd"
include "../_ipp_utils/_ippi.pxd"

import numpy as np
from cpython.exc cimport PyErr_SetString
from cpython.exc cimport PyErr_Occurred
from cpython.ref cimport PyObject
cimport numpy as cnp
cimport cython
# cimport _ippi as ippi.
cimport _morphology as morph

cnp.import_array()


cpdef dilation(image, selem=None, out=None, shift_x=False, shift_y=False):
    """
    dilation(image, selem=None, out=None, shift_x=False, shift_y=False)

    Return greyscale morphological dilation of an image.

    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels
    in the neighborhood centered at (i,j). Dilation enlarges bright regions
    and shrinks dark regions.

    The function has `skimage.morphology.dilation` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Image array.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarray, optional
        The array to store the result of the morphology. If None, is
        passed, a new array will be allocated.
        Should be the same type and shape as `image`.
    shift_x, shift_y : bool, optional
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    dilated : array, same shape and type as `image`
        The result of the morphological dilation.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiDilateBorder_<mod> on the backend,
    that performs dilation of an image, see: `DilateBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - The `selem` should be only `uint8` dtype.
    - The `image` and `out` should be the same data type
    - Currently `dilation` function supports `image` of the following types:
      - one channel image: `uint8`, `uint16`, `int16`, `float32`
      - three channel image: `uint8`, `float32`
      - four channel image: `uint8`, `float32`
    - Currently `out`, `shift_x`, `shift_y` are not processed.
    - If `selem` is None:
      `scikit-ipp` creates directly ndarray with shape (3, 3) like a
      `skimage.morphology.selem.diamond(radius=1)` for
      2D grayscale images and ndarray with shape (3, 3, 3) like a
      (`skimage.morphology.selem.diamond(radius=1) for each channel).

    Examples
    --------
    >>> # Dilation enlarges bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> from skipp.morphology import dilation
    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> dilation(bright_pixel, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """

    cdef int ippStatusIndex = 0  # OK
    cdef morph.ippiMorphologyFunction ippiFunc = morph.IppiDilateBorder

    cdef void * cyimage
    cdef void * cydestination
    cdef void * cyselem
    cdef IppDataType ipp_src_datatype
    cdef IppDataType selem_datatype

    cdef int selem_width
    cdef int selem_height
    cdef int img_width
    cdef int img_height
    cdef int numChannels

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    # TODO
    # add _get_output// check output dtype
    out = np.empty_like(image, dtype=image.dtype, order='C')

    if(image.ndim == 2 or image.ndim == 1):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 1D array or 2D array with 1 or 3 channels, got %iD." % image.ndim)

    cdef IppiBorderType ippBorderType = ippBorderRepl

    # TODO
    # will be romoved
    cdef float ippBorderValue = float(0)

    if selem is None:
        if image.ndim == 1:
            selem = np.ones((3, 1), dtype=np.uint8, order='C')
            selem_width = selem.shape[0]
            selem_height = 1
        elif image.ndim == 2:
            selem = np.asarray([0, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8, order='C').reshape((3, 3))
            selem_width = selem.shape[1]
            selem_height = selem.shape[0]
        elif image.ndim == 3:
            selem = np.asarray([0, 1, 0, 1, 1, 1, 0, 1, 0,
                                0, 1, 0, 1, 1, 1, 0, 1, 0,
                                0, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8, order='C').reshape((3, 3, 3))
            selem_width = selem.shape[1]
            selem_height = selem.shape[0]
        else:
            raise ValueError("Expected image with 1, 2 or 3 ndim, got %iD." % image.ndim)
    else:
        selem_datatype = __get_ipp_data_type(selem)
        if(selem_datatype != ipp8u):
            raise ValueError("Selem data type not supported")
        selem_width = selem.shape[1]
        selem_height = selem.shape[0]

    if(image.ndim == 1):
        img_width = image.shape[0]
        img_height = 1
    else:
        img_width = image.shape[1]
        img_height = image.shape[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(out)
    cyselem = <void*> cnp.PyArray_DATA(selem)

    ippStatusIndex = morph.own_Morphology(ipp_src_datatype,
                                          ippiFunc,
                                          cyimage,
                                          cydestination,
                                          img_width,
                                          img_height,
                                          numChannels,
                                          cyselem,
                                          selem_width,
                                          selem_height,
                                          ippBorderType,
                                          ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return out


cpdef erosion(image, selem=None, out=None, shift_x=False, shift_y=False):
    """
    erosion(image, selem=None, out=None, shift_x=False, shift_y=False)

    Return greyscale morphological erosion of an image.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j). Erosion shrinks bright regions and
    enlarges dark regions.

    The function has `skimage.morphology.erosion` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Image array.
    selem : ndarray, optional
        The neighborhood expressed as an array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarrays, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    shift_x, shift_y : bool, optional
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    eroded : array, same shape and type as `image`
        The result of the morphological erosion.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiErodeBorder_<mod> on the backend,
    that performs dilation of an image, see: `ErodeBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - The `selem` should be only `uint8` dtype.
    - The `image` and `out` should be the same data type
    - Currently `dilation` function supports `image` of the following types:
      - one channel image: `uint8`, `uint16`, `int16`, `float32`
      - three channel image: `uint8`, `float32`
      - four channel image: `uint8`, `float32`
    - Currently `out`, `shift_x`, `shift_y` are not processed.
    - If `selem` is None:
      `scikit-ipp` creates directly ndarray with shape (3, 3) like a
      `skimage.morphology.selem.diamond(radius=1)` for
      2D grayscale images and ndarray with shape (3, 3, 3) like a
      (`skimage.morphology.selem.diamond(radius=1) for each channel).

    Examples
    --------
    >>> # Erosion shrinks bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> from skipp.morphology import erosion
    >>> bright_square = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> erosion(bright_square, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """
    cdef int ippStatusIndex = 0  # OK
    cdef morph.ippiMorphologyFunction ippiFunc = morph.IppiErodeBorder

    cdef void * cyimage
    cdef void * cydestination
    cdef void * cyselem
    cdef IppDataType ipp_src_datatype
    cdef IppDataType selem_datatype

    cdef int selem_width
    cdef int selem_height
    cdef int img_width
    cdef int img_height
    cdef int numChannels

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    #  TODO
    # add _get_output// check output dtype
    out = np.empty_like(image, dtype=image.dtype, order='C')

    if(image.ndim == 2 or image.ndim == 1):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 1D array or 2D array with 1 or 3 channels, got %iD." % image.ndim)

    cdef IppiBorderType ippBorderType = ippBorderRepl

    # TODO
    # will be romoved
    cdef float ippBorderValue = float(0)

    if selem is None:
        if image.ndim == 1:
            selem = np.ones((3, 1), dtype=np.uint8, order='C')
            selem_width = selem.shape[0]
            selem_height = 1
        elif image.ndim == 2:
            selem = np.asarray([0, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8, order='C').reshape((3, 3))
            selem_width = selem.shape[1]
            selem_height = selem.shape[0]
        elif image.ndim == 3:
            selem = np.asarray([0, 1, 0, 1, 1, 1, 0, 1, 0,
                                0, 1, 0, 1, 1, 1, 0, 1, 0,
                                0, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8, order='C').reshape((3, 3, 3))
            selem_width = selem.shape[1]
            selem_height = selem.shape[0]
        else:
            raise ValueError("Expected image with 1, 2 or 3 ndim, got %iD." % image.ndim)
    else:
        selem_datatype = __get_ipp_data_type(selem)
        if(selem_datatype != ipp8u):
            raise ValueError("Selem data type not supported")
        selem_width = selem.shape[1]
        selem_height = selem.shape[0]

    if(image.ndim == 1):
        img_width = image.shape[0]
        img_height = 1
    else:
        img_width = image.shape[1]
        img_height = image.shape[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(out)
    cyselem = <void*> cnp.PyArray_DATA(selem)

    ippStatusIndex = morph.own_Morphology(ipp_src_datatype,
                                          ippiFunc,
                                          cyimage,
                                          cydestination,
                                          img_width,
                                          img_height,
                                          numChannels,
                                          cyselem,
                                          selem_width,
                                          selem_height,
                                          ippBorderType,
                                          ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return out
