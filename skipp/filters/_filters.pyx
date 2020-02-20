from __future__ import absolute_import
include "../_ipp_utils/_ipp_utils.pxd"
include "../_ipp_utils/_ippi.pxd"

import numpy as np
# from dtype import img_as_float32
from cpython.exc cimport PyErr_SetString
from cpython.exc cimport PyErr_Occurred
from cpython.ref cimport PyObject
cimport numpy as cnp
cimport cython

from _filters cimport own_FilterGaussian as FilterGaussian
from _filters cimport own_FilterMedian as FilterMedian
from _filters cimport own_FilterLaplace as FilterLaplace
from _filters cimport own_FilterEdge as FilterEdge
from _filters cimport own_FilterPrewitt as FilterPrewitt
from _filters cimport own_EdgeFilterKernel
from _filters cimport own_filterSobelVert
from _filters cimport own_filterSobelHoriz
from _filters cimport own_filterSobel
from _filters cimport own_filterPrewittVert
from _filters cimport own_filterPrewittHoriz
from _filters cimport own_filterPrewitt


cimport _filters

cnp.import_array()


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


cpdef gaussian(image, sigma=1.0, output=None, mode='nearest', cval=0,
               multichannel=None, preserve_range=False, truncate=4.0):
    # TODO
    # add warnings for multichannel

    # TODO
    # def test_dimensiona_error_gaussian():

    # TODO
    # get input require
    # TODO module with numpy.require to provid type that satisfies requirements.
    # TODO
    # add to Notes
    # * enabled preserve_range, output
    # * use scikit-image's img_as_float32 before for non-float images before

    cdef int ippStatusIndex = 0  # OK

    cdef void * cyimage
    cdef void * cydestination
    cdef int img_width
    cdef int img_height
    cdef int numChannels
    cdef IppDataType ipp_src_datatype

    cdef float sd
    cdef float tr
    cdef float ippBorderValue
    cdef int kernelSize

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    #if output is None:
    #    output = np.empty_like(image, dtype=image.dtype, order='C')
    #elif not np.issubdtype(output.dtype, np.floating):
    #    raise ValueError("Provided output data type is not float")
    # TODO
    # add getting output as is in scikit-image
    # implement __get_output like in scipy.ndimage
    output = np.empty_like(image, dtype=image.dtype, order='C')

    if sigma == 0.0:
        output[...] = image[...]
        return output
    cdef IppiBorderType ippBorderType = __get_IppBorderType(mode)
    if(ippBorderType == UndefValue):
        raise ValueError("Boundary mode not supported")

    sd = float(sigma)
    tr = float(truncate)
    ippBorderValue = float(cval)

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(output)
    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    # ~ use double cast
    kernelSize = int(truncate * sigma + 0.5) * 2 - 1

    # needed more correct way. Warning: conversion from 'npy_intp'
    # to 'int', possible loss of data
    img_width = image.shape[1]
    img_height = image.shape[0]

    # pass to IPP the source and destination arrays
    ippStatusIndex = FilterGaussian(ipp_src_datatype,
                                    cyimage,
                                    cydestination,
                                    img_width,
                                    img_height,
                                    numChannels,
                                    sd,
                                    kernelSize,
                                    ippBorderType,
                                    ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return output

cpdef median(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False, mode='nearest', cval=0.0, behavior='ipp'):
    """
    # TODO
    # add documentation
    Note: scikit-image's median filter requiers the `image`, that must be a 2-dimensional array
    scikit-ipp could processing also multichannel image
    scikit-ipp uses only recantagle shape masks with ones
    * if mask size is egen ipp raises RuntimeError: ippStsMaskSizeErr: Invalid mask size
    * behavior disabled default is `ipp`
    """
    # TODO
    # get input require
    # TODO module with numpy.require to provide type that satisfies requirements.

    # TODO
    # add documentation
    cdef int ippStatusIndex = 0  # OK

    cdef void * cyimage
    cdef void * cydestination
    cdef IppDataType ipp_src_datatype
    cdef IppiBorderType ippBorderType
    cdef int selem_width
    cdef int selem_height
    cdef int img_width
    cdef int img_height
    cdef float ippBorderValue = float(cval)

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    ippBorderType = __get_IppBorderType(mode)
    if(ippBorderType == UndefValue):
        raise ValueError("Boundary mode not supported")

    cdef int numChannels
    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    # TODO
    # add _get_output
    out = np.empty_like(image, dtype=image.dtype, order='C')

    # TODO
    # case when selem is shape or None
    selem_width = selem.shape[1]
    selem_height = selem.shape[0]

    img_width = image.shape[1]
    img_height = image.shape[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(out)

    # pass to IPP the source and destination arrays
    ippStatusIndex = FilterMedian(ipp_src_datatype,
                                  cyimage,
                                  cydestination,
                                  img_width,
                                  img_height,
                                  numChannels,
                                  selem_width,
                                  selem_height,
                                  ippBorderType,
                                  ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return out


cpdef laplace(image, ksize=3, mask=None):
    """
    # TODO
    # add documentation
    Notes
    -----

    """
    # TODO
    # investigate ksize

    # TODO
    # add _get_output

    # currently converting int images into float not supported
    # That is why funcs is waiting converted by img_as_float32 images
    # for further processing
    # if image dtype is not numpy.float32, ValueError will be raised
    # call before img_as_float32 if it is needed

    cdef int ippStatusIndex = 0  # OK

    cdef void * cyimage
    cdef void * cydestination
    cdef int img_width
    cdef int img_height
    cdef int numChannels
    cdef IppiBorderType ippBorderType = ippBorderRepl
    cdef float ippBorderValue = 0.0
    cdef IppDataType ipp_src_datatype

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    if(ipp_src_datatype == ipp32f):
        output = np.empty_like(image, dtype=image.dtype, order='C')
    else:
        # TODO
        raise ValueError("currently not supported")
        # output = np.empty_like(image, dtype=image.float64, order='C')

    img_width = image.shape[1]
    img_height = image.shape[0]
    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(output)

    ippStatusIndex = FilterLaplace(ipp_src_datatype,
                                   cyimage,
                                   cydestination,
                                   img_width,
                                   img_height,
                                   numChannels,
                                   ippBorderType,
                                   ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    # TODO
    # implemented maskfilter IPP
    return _mask_filter_result(output, mask)

# >>> edge filter module
cpdef __edge(image, mask=None, edgeKernel=own_filterSobel):
    """Find the edge magnitude using the Sobel transform.
    # TODO
    # add documentation
    """
    # TODO
    # add _get_output
    cdef int ippStatusIndex = 0  # OK

    cdef void * cyimage
    cdef void * cydestination
    cdef IppDataType ipp_src_datatype
    cdef int img_width
    cdef int img_height
    cdef int numChannels

    # ~~~~ remove this func
    # check_nD(image, 2)
    if(image.ndim == 2):
        numChannels = 1
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    if(ipp_src_datatype == ipp32f):
        output = np.empty_like(image, dtype=image.dtype, order='C')
    else:
        # TODO
        raise ValueError("currently not supported")
        # output = np.empty_like(image, dtype=image.float64, order='C')

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(output)

    img_width = image.shape[1]
    img_height = image.shape[0]
    # TODO
    # in ipp wrapper
    # image = img_as_float(image)
    if (edgeKernel == own_filterPrewitt):
        ippStatusIndex = FilterPrewitt(edgeKernel,
                                       ipp_src_datatype,
                                       ipp_src_datatype,
                                       cyimage,
                                       cydestination,
                                       img_width,
                                       img_height,
                                       numChannels)
    else:
        ippStatusIndex = FilterEdge(edgeKernel,
                                    ipp_src_datatype,
                                    ipp_src_datatype,
                                    cyimage,
                                    cydestination,
                                    img_width,
                                    img_height,
                                    numChannels)
    __get_ipp_error(ippStatusIndex)
    return _mask_filter_result(output, mask)


cpdef sobel(image, mask=None):
    """Find the edge magnitude using the Sobel transform.
    # TODO
    # add documentation
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobel
    return __edge(image, mask, sobelKernel)


cpdef sobel_h(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.
    # TODO
    # add documentation
    We use the following kernel::
      1   2   1
      0   0   0
     -1  -2  -1
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobelHoriz
    return __edge(image, mask, sobelKernel)


cpdef sobel_v(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.
    # TODO
    # add documentation
    -----
    We use the following kernel::
      1   0  -1
      2   0  -2
      1   0  -1
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobelVert
    return __edge(image, mask, sobelKernel)


cpdef prewitt(image, mask=None):
    """Find the edge magnitude using the Prewitt transform.
    # TODO
    # add documentation
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewitt
    return __edge(image, mask, sobelKernel)


cpdef prewitt_h(image, mask=None):
    """Find the horizontal edges of an image using the Prewitt transform.
    # TODO
    # add documentation
    We use the following kernel::
      1   1   1
      0   0   0
     -1  -1  -1
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewittHoriz
    return __edge(image, mask, sobelKernel)


cpdef prewitt_v(image, mask=None):
    """Find the vertical edges of an image using the Prewitt transform.
    # TODO
    # add documentation
    -----
    We use the following kernel::
      1   0  -1
      1   0  -1
      1   0  -1
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewittVert
    return __edge(image, mask, sobelKernel)
