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
    """ Gaussian filter.
    The function has `skimage.filters.gaussian` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : array-like
        Input image (grayscale or color) to filter.
    sigma : scalar, optional
        Standard deviation for Gaussian kernel.
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    multichannel : bool, optional (default: None)
    preserve_range : bool, optional
    truncate : float, optional
        Truncate the filter at this many standard deviations.

    Returns
    -------
    filtered_image : ndarray
        the filtered image array

    Notes
    -----
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterGaussianBorder_<mod> on the backend,
    that performs Gaussian filtering of an image with user-defined borders,
    see: `FilterGaussianBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - The `image` should be an array of numpy.float32 dtype.
    - Currently `output`, `multichannel` and `preserve_range` are not processed.

    Examples
    --------
    >>> a = np.zeros((3, 3), dtype=np.float32)
    >>> a[1, 1] = 1
    >>> a
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>> gaussian(a, sigma=0.4)  # mild smoothing
    array([[0.00163118, 0.03712554, 0.00163118],
           [0.03712554, 0.844973  , 0.03712554],
           [0.00163118, 0.03712554, 0.00163118]], dtype=float32)
    >>> gaussian(a, sigma=1)  # more smoothing
    array([[0.05858153, 0.09658462, 0.05858153],
           [0.09658462, 0.15924111, 0.09658462],
           [0.05858153, 0.09658462, 0.05858153]], dtype=float32)
    >>> # Several modes are possible for handling boundaries
    >>> gaussian(a, sigma=1, mode='nearest')
    array([[0.05858153, 0.09658462, 0.05858153],
           [0.09658462, 0.15924111, 0.09658462],
           [0.05858153, 0.09658462, 0.05858153]], dtype=float32)
    """
    # TODO
    # add warnings for multichannel

    # TODO
    # def test_dimensiona_error_gaussian():

    # TODO
    # get input require

    # TODO
    # module with numpy.require to provid type that satisfies requirements.

    # TODO
    # add to Notes
    # * enabled preserve_range, output

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
    """ Median filter.

    Return local median of an image.
    The function has `skimage.filters.median` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : array-like
        Input image.
    selem : ndarray, optional
        If ``behavior=='ipp'``, ``selem`` is a 2-D array of 1's and 0's.
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'constant', 'nearest'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'.
        Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0
    behavior : {'ipp'}, optional

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterMedianBorder_<mod> on the backend,
    that performs median filtering of an image with user-defined borders,
    see: `FilterMedianBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - The `image` should be a numpy array with uint8, uint16, int16 or float32
      dtype.
    - Currently `out` and `behavior` are not processed.
      The `behavior` disabled due to only one option existence, default is `ipp`
      The `mask`, `shift_x` and `shift_y` are not processed - will be deprecated
      since `scikit-image` v0.17.
    - `skimage.filters.median` requiers the `image`, that should be a 2-dimensional
      array. `skipp.filters.median` can also processing also multichannel (3-channel)
      images.
    - `scikit-ipp` supports only recantagle shape `selem` with ones.
    - Indicates an error if `selem` shape has a field with a zero, negative
      or even value.

    Examples
    --------
    >>> from skimage import data
    >>> import numpy as np
    >>> from skipp.filters import median
    >>> img = data.camera()
    >>> mask = np.ones((5,5), dtype=np.uint8, order='C')
    >>> med = median(img, mask)
    """
    # TODO
    # get input require

    # TODO
    # module with numpy.require to provide type that satisfies requirements.

    # TODO
    # add documentation
    cdef int ippStatusIndex = 0  # OK

    cdef void * cyimage
    cdef void * cydestination
    cdef IppDataType ipp_src_datatype
    cdef IppiBorderType ippBorderType
    cdef int selem_width = 3
    cdef int selem_height = 3
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

    if selem is not None:
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
    """Find the edges of an image using the Laplace operator.
    The function has `skimage.filters.laplace` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Image to process.
    ksize : int, optional
        Define the size of the discrete Laplacian operator such that it
        will have a size of (ksize,) * image.ndim.
    mask : ndarray, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The Laplace edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterBorder_<mod> on the backend,
    that filters an image using a rectangular filter with coeffs
    (Laplace (3x3))::
        | 0  -1   0 |
        |-1   4  -1 |
        | 0  -1   0 |
    for implementing laplace filtering as is in `scikit-image`,
    see: `FilterBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.

    Examples
    --------
    >>> from skimage import data
    >>> import numpy as np
    >>> from skipp.filters import laplace
    >>> img = data.camera().astype(np.float32)
    >>> lap = laplace(img)
    """
    # TODO
    # add _get_output

    # TODO
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
    # implement by using Intel(R) IPP
    return _mask_filter_result(output, mask)


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
    """Find edges in an image using the Sobel filter.
    The function has `skimage.filters.sobel` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array of float32
        The Sobel edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterSobel_<mod> on the backend,
    see: `FilterSobel` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.

    - `skimage.filters.sobel` is a wrapper on ``scipy.ndimage``
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterSobel_<mode> doesn't supports this border type.

    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.
    The 3x3 convolution kernel used in the horizontal and vertical Sobels is
    an approximation of the gradient of the image (with some slight blurring
    since 9 pixels are used to compute the gradient at a given pixel). As an
    approximation of the gradient, the Sobel operator is not completely
    rotation-invariant.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import sobel
    >>> camera = data.camera().astype(np.float32)
    >>> edges = sobel(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobel
    return __edge(image, mask, sobelKernel)


cpdef sobel_h(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.
    The function has `skimage.filters.sobel_h` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Sobel edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterSobelHorizBorder_<mod> on the backend,
    see: `FilterSobelHorizBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    We use the following kernel::
        | 1   2   1 |
        | 0   0   0 |
        |-1  -2  -1 |

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.
    - `skimage.filters.sobel_h` is a wrapper on `scipy.ndimage`
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterSobelHorizBorder_<mode> doesn't supports this border type.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import sobel_h
    >>> camera = data.camera().astype(np.float32)
    >>> edges = sobel_h(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobelHoriz
    return __edge(image, mask, sobelKernel)


cpdef sobel_v(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.
    The function has `skimage.filters.sobel_v` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Sobel edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterSobelVertBorder_<mod> on the backend,
    see: `FilterSobelVertBorder` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    We use the following kernel::
        | 1   0  -1 |
        | 2   0  -2 |
        | 1   0  -1 |

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.

    - `skimage.filters.sobel_v` is a wrapper on `scipy.ndimage`
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterSobelVertBorder_<mode> doesn't supports this border type.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import sobel_v
    >>> camera = data.camera().astype(np.float32)
    >>> edges = sobel_v(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterSobelVert
    return __edge(image, mask, sobelKernel)


cpdef prewitt(image, mask=None):
    """Find the edge magnitude using the Prewitt transform.
    The function has `skimage.filters.prewitt` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterPrewittVertBorder_<mod> and
    ippiFilterPrewittHorizBorder_<mod> on the backend
    see: `FilterPrewittHorizBorder`, `FilterPrewittVertBorder`
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    Return the square root of the sum of squares of the horizontal
    and vertical Prewitt transforms. The edge magnitude depends slightly
    on edge directions, since the approximation of the gradient operator by
    the Prewitt operator is not completely rotation invariant.

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.
    - `skimage.filters.prewitt` is a wrapper on `scipy.ndimage`
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterPrewittVertBorder_<mode> and  ippiFilterPrewittHorizBorder_<mode>
      don't support this border type.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import prewitt
    >>> camera = data.camera().astype(np.float32)
    >>> edges = prewitt(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewitt
    return __edge(image, mask, sobelKernel)


cpdef prewitt_h(image, mask=None):
    """Find the horizontal edges of an image using the Prewitt transform.
    The function has `skimage.filters.prewitt_h` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterPrewittHorizBorder_<mod> on the backend
    see: `FilterPrewittHorizBorder`
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    We use the following kernel::
        | 1   1   1 |
        | 0   0   0 |
        |-1  -1  -1 |

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.
    - `skimage.filters.prewitt_h` is a wrapper on `scipy.ndimage`
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterPrewittHorizBorder_<mode> doesn't support this border type.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import prewitt_h
    >>> camera = data.camera().astype(np.float32)
    >>> edges = prewitt_h(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewittHoriz
    return __edge(image, mask, sobelKernel)


cpdef prewitt_v(image, mask=None):
    """Find the vertical edges of an image using the Prewitt transform.
    The function has `skimage.filters.prewitt_v` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiFilterPrewittVertBorder_<mod> on the backend
    see: `FilterPrewittVertBorder`
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    We use the following kernel::
        | 1   0  -1 |
        | 1   0  -1 |
        | 1   0  -1 |

    - Currently converting integer images into float32 is not supported.
    - The `image` should be an array of numpy.float32 dtype.

    - `skimage.filters.prewitt_v` is a wrapper on `scipy.ndimage`
      `convolve` func. `convolve` uses `reflect` border mode.
      `reflect` border mode is equivalent of Intel(R) IPP ippBorderMirrorR border
      type. ippiFilterPrewittVertBorder_<mode> doesn't support this border type.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.filters import prewitt_v
    >>> camera = data.camera().astype(np.float32)
    >>> edges = prewitt_v(camera)
    """
    # TODO
    # add _get_output
    cdef own_EdgeFilterKernel sobelKernel = own_filterPrewittVert
    return __edge(image, mask, sobelKernel)
