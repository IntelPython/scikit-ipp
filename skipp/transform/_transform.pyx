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

# cython: profile=True
from __future__ import absolute_import
include "../_ipp_utils/_ipp_utils.pxd"
include "../_ipp_utils/_ippi.pxd"

import numpy as np
cimport numpy as cnp
import math
cimport _transform as transform
cnp.import_array()


class AffineTransform(object):
    """2D affine transformation.

    The same interface as is for `skimage.transform.AffineTransform`,
    see: https://scikit-image.org/

    Has the following form::

        X = a0*x + a1*y + a2 =
          = sx*x*cos(rotation) - sy*y*sin(rotation + shear) + a2

        Y = b0*x + b1*y + b2 =
          = sx*x*sin(rotation) + sy*y*cos(rotation + shear) + b2

    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::

        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    Parameters
    ----------
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    scale : (sx, sy) as array, list or tuple, optional
        Scale factors.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
    shear : float, optional
        Shear angle in counter-clockwise direction as radians.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters.

    Attributes
    ----------
    params : (3, 3) array
        Homogeneous transformation matrix.
    """

    _coeffs = range(6)

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            self.params = np.array([
                [sx * math.cos(rotation), -sy * math.sin(rotation + shear), 0],
                [sx * math.sin(rotation),  sy * math.cos(rotation + shear), 0],
                [                      0,                                0, 1]
            ], order='C')
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)

    def __add__(self, other):
        """Combine this transformation with another.
        """
        if type(self) == type(other):
            tform = self.__class__
            return tform(other.params @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing "
                            "types.")

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def _inv_matrix(self):
        inv = np.linalg.inv(self.params)
        return inv

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        src = np.transpose(src)
        matrix = np.transpose(matrix)
        dst = src @ matrix

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :2] /= dst[:, 2:3]

        return dst[:, :2]


    def inverse(self, coords):
        """Apply inverse transformation.
        Parameters
        ----------
        coords : (N, 2) array
            Destination coordinates.
        Returns
        -------
        coords : (N, 2) array
            Source coordinates.
        """
        inverted_matrix = self._inv_matrix()
        return self._apply_mat(coords, inverted_matrix)


    def scale(self):
        sx = math.sqrt(self.params[0, 0] ** 2 + self.params[1, 0] ** 2)
        sy = math.sqrt(self.params[0, 1] ** 2 + self.params[1, 1] ** 2)
        return sx, sy

    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[0, 0])

    def shear(self):
        beta = math.atan2(- self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    def translation(self):
        return self.params[0:2, 2]


cpdef warp(image, inverse_map, map_args={}, output_shape=None, order=1,
           mode='constant', cval=0., clip=True, preserve_range=False):
    """Warp an image according to a given coordinate transformation.

    The function has `skimage.transform.warp` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.
        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.
         - For 2-D images, you can directly pass a transformation object,
           e.g. `skipp.transform.AffineTransform`.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g. `skipp.transform.AffineTransform.params`.
        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix.
        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: linear (default)
         - 2: Bi-quadratic [not supported]
         - 3: cubic
         - 4: Bi-quartic [not supported]
         - 5: Bi-quintic [not supported]
    mode : {'constant', 'edge', 'transp'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values.

    Returns
    -------
    warped : ndarray
        The warped input image, same type as `image`.

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiWarpAffineLinear_<mod>,  ippiWarpAffineNearest_<mod>
    and ippiWarpAffineCubic_<mod> on the backend, that performs
    warp affine transformation of an image using the linear,
    nearest neighbor or cubic interpolation method, see: `WarpAffineLinear`,
    `WarpAffineCubic`, `WarpAffineNearest` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - `transform.warp` function gets an upper bound on the number of threads
      that could be used, and takes it for improving performance of warping
      by using OpenMP with parallelization. It divides the processing of the
      destination (output) image into tiles in the number of available
      threads, for processing each tile separately on each thread. The number
      of tiles corresponds to the number of available threads. If it is
      possible it uses multiple threads for procesing all tiles
      simultaneously.
    - Currently `warp` function supports `image` of the following types
      for one, three and four channel images:
        `uint8`, `uint16`, `int16`, `float32`, `float64`
    - Currently modes don't match the behaviour of `numpy.pad`.
    - Currently `map_args`, `clip`, `preserve_range` are not processed.
    - ``scikit-image`` uses Catmull-Rom spline (0.0, 0.5). In `scikit-ipp`
      the same method was implemented. [1]

    References
    ----------
    .. [1] Don P. Mitchell, Arun N. Netravali. Reconstruction Filters in Computer Graphics.
           Computer Graphics, Volume 22, Number 4, AT&T Bell Laboratories, Murray Hill,
           New Jersey, August 1988.

    Examples
    --------
    >>> from skipp.transform import warp
    >>> from skimage import data
    >>> image = data.camera()
    >>> # The following image warps are all equal but differ substantially in
    >>> # execution time. The image is shifted to the bottom.
    >>> # Use a transformation matrix to warp an image (fast):
    >>> matrix = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> # Use a geometric transform to warp an image (fast):
    >>> from skipp.transform import AffineTransform
    >>> tform = AffineTransform(translation=(0, -10))
    >>> warped = warp(image, tform)
    """
    cdef int ippStatusIndex = 0  # OK
    cdef void * cyimage
    cdef void * cydestination
    cdef double * cy_coeffs

    if isinstance(inverse_map, AffineTransform):
        cy_coeffs = <double * > cnp.PyArray_DATA(inverse_map.params[:2])
    elif isinstance(inverse_map, np.ndarray):
        if inverse_map.shape == (3, 3):
            inverse_map = inverse_map.astype(np.double)
            cy_coeffs = <double*> cnp.PyArray_DATA(inverse_map[:2])
        elif inverse_map.shape == (2, 3):
            inverse_map = inverse_map.astype(np.double)
            cy_coeffs = <double*> cnp.PyArray_DATA(inverse_map)
        else:
            raise ValueError("Invalid shape of transformation matrix.")
    else:
        # TODO:
        # change the raised error msg
        raise ValueError("inverse_map type not supported")

    # TODO
    # check if inverse_map C-order, if ndarray
    #cdef cnp.ndarray inverse_map_matrix =  np.zeros((3, 3), dtype = np.double, order='C')
    #cy_inverse_map_matrix = <double*> cnp.PyArray_DATA(inverse_map_matrix)

    cdef IppDataType ipp_src_datatype

    cdef int img_width
    cdef int img_height

    cdef int dst_width
    cdef int dst_height

    cdef int numChannels

    cdef IppiInterpolationType interpolation

    # TODO
    # check after
    cdef double ippBorderValue = cval
    
    cdef IppiWarpDirection direction = ippWarpBackward

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    interpolation = __get_IppiInterpolationType(order)
    if(interpolation == UndefValue):
        raise ValueError("Undef `order` or not supported")

    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    # matches with `numpy.pad`
    # {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
    cdef IppiBorderType ippBorderType = __get_numpy_pad_IppBorderType(mode)
    if(ippBorderType == UndefValue):
        raise ValueError("Boundary mode not supported")

    # TODO
    # more safe initialization
    img_width = image.shape[1]
    img_height = image.shape[0]

    # TODO
    # more safe check of output_shape integers
    if output_shape is not None:
        dst_width = output_shape[1]
        dst_height = output_shape[0]
        if numChannels == 1:
            output = np.empty((dst_height, dst_width), dtype=image.dtype,
                                   order='C')
        else:
            output = np.empty((dst_height, dst_width, numChannels),
                                   dtype=image.dtype, order='C')
    else:
        dst_width = img_width
        dst_height = img_height
        output = np.empty_like(image, dtype=image.dtype, order='C')

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(output)

    ippStatusIndex = transform.own_Warp(ipp_src_datatype,
                                        cyimage,
                                        cydestination,
                                        img_width,
                                        img_height,
                                        dst_width,
                                        dst_height,
                                        numChannels,
                                        cy_coeffs,
                                        interpolation,
                                        direction,
                                        ippBorderType,
                                        ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return output


cpdef rotate(image, angle, resize=False, center=None, order=1, mode='constant',
             cval=0, clip=True, preserve_range=False):
    """Rotate image by a certain angle around its center.

    The function has `skimage.transform.rotate` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize : bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.
    center : iterable of length 2
        The rotation center. If ``center=None``, the image is rotated around
        its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
        note that this parameter is (cols, rows), contrary to normal skimage
        ordering.

    Returns
    -------
    rotated : ndarray
        Rotated version of the 2D input image.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to be
        in the range 0-5:
         - 0: Nearest-neighbor
         - 1: linear (default)
         - 2: Bi-quadratic [not supported]
         - 3: cubic
         - 4: Bi-quartic [not supported]
         - 5: Bi-quintic [not supported]
    mode : {'edge', 'constant', 'transp'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
    preserve_range : bool, optional
        Whether to keep the original range of values.

    Notes
    --------
    This function uses `skipp.transform.warp` on the backend, and
    `skipp.transform.warp` in turn uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs: ippiWarpAffineLinear_<mod>,  ippiWarpAffineNearest_<mod>
    and ippiWarpAffineCubic_<mod> on the backend, that performs
    warp affine transformation of an image using the linear,
    nearest neighbor or cubic interpolation method, see: `WarpAffineLinear`,
    `WarpAffineCubic`, `WarpAffineNearest` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - Currently `rotate` function supports `image` of the following types
      for one, three and four channel images:
        `uint8`, `uint16`, `int16`, `float32`, `float64`
    - Currently modes don't match the behaviour of `numpy.pad`.
    - Currently `clip`, `preserve_range` are not processed.
    - ``scikit-image`` uses Catmull-Rom spline (0.0, 0.5). In `scikit-ipp` the same
      method was implemented. [1]
    - Modificated version of `skimage.transform.rotate` (v0.17). `scikit-ipp` uses
      `skipp.transform.AffineTransform` instead of `skimage.transform.SimilarityTransform` 

    References
    ----------
    .. [1] Don P. Mitchell, Arun N. Netravali. Reconstruction Filters in Computer Graphics.
           Computer Graphics, Volume 22, Number 4, AT&T Bell Laboratories, Murray Hill,
           New Jersey, August 1988.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.transform import rotate
    >>> image = data.camera()
    >>> rotate(image, 2).shape
    (512, 512)
    >>> rotate(image, 2, resize=True).shape
    (530, 530)
    >>> rotate(image, 90, resize=True).shape
    (512, 512)
    """
    rows, cols = image.shape[0], image.shape[1]

    if center is None:
        center = np.array((cols, rows)) / 2. - 0.5
    else:
        center = np.asarray(center)
    tform1 = AffineTransform(translation=center)
    tform2 = AffineTransform(rotation=np.deg2rad(angle))
    tform3 = AffineTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = AffineTransform(translation=translation)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


cpdef resize(image, output_shape, order=1, mode='edge', cval=0, clip=True,
             preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None,
             num_lobes=3):
    """Resize image to match a certain size.

    Performs interpolation to up-size or down-size 2D images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts.

    The function has `skimage.transform.resize` like signature,
    see: https://scikit-image.org/

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-7:
         - 0: Nearest-neighbor
         - 1: Linear (default)
         - 2: Bi-quadratic [not supported]
         - 3: Cubic
         - 4: Bi-quartic [not supported]
         - 5: Bi-quintic [not supported]
         - 6: Lanczos
         - 7: Super
    mode : {'constant', 'edge', 'transp'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
    preserve_range : bool, optional
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior to
        down-scaling. It is crucial to filter when down-sampling the image to
        avoid aliasing artifacts.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
    num_lobes : int, optional
        Used in conjunction with order 6 (Lanczos). The parameter for
        specifying Lanczos (2 or 3) filters. Options:
         - 2: 2-lobed Lanczos 4x4
         - 3: 3-lobed Lanczos 6x6

    Notes
    --------
    This function uses Intel(R) Integrated Performance Primitives
    (Intel(R) IPP) funcs on the backend: ippiResizeNearest_<mod>,
    ippiResizeLinear_<mod>, ippiResizeCubic_<mod>, ippiResizeLanczos_<mod>,
    ippiResizeSuper_<mod> that changes an image size using nearest neighbor,
    linear, cubic, Lanczos or super interpolation method, and
    ippiResizeAntialiasing_<mod>, that changes an image size using using the
    linear and cubic interpolation
    method with antialiasing, see: `ResizeNearest`, `ResizeLinear`,
    `ResizeCubic`, `ResizeLanczos`, `ResizeSuper`,`ResizeAntialiasing` on
    https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/

    - `transform.resize` function gets an upper bound on the number of threads
      that could be used, and takes it for improving performance of warping
      by using OpenMP with parallelization. It divides the processing of the
      destination (output) image into tiles in the number of available
      threads, for processing each tile separately on each thread. The number
      of tiles corresponds to the number of available threads. If it is
      possible it uses multiple threads for procesing all tiles
      simultaneously.
    - Currently `resize` function supports `image` of the following types
      for one, three and four channel images:
        `uint8`, `uint16`, `int16`, `float32`.
    - Currently modes don't match the behaviour of `numpy.pad`.
    - Currently `clip`, `preserve_range` and `anti_aliasing_sigma` are not
      processed.
    - if `antialiasing` is `True`, supported interpolation methods are
      `linear` and `cubic`.
    - if `antialiasing` is `False`, supported interpolation methods are
      `nearest`, `linear`, `cubic`, `Lanczos` and `super`.
    - if `antialiasing` is `True`, supported boundary `mode` is `edge`.
    - if `antialiasing` is `False`, supported boundary `mode` is `edge`.
    - The Lanczos interpolation (order=6) have the following filter sizes:
      2-lobed Lanczos 4x4, 3-lobed Lanczos 6x6.
    - Indicates an error (RuntimeError: ippStsSizeErr: Incorrect value for
      data size) in the following cases:
      - If the source image size is less than the filter size of the chosen
        interpolation method (except ippSuper).
      - If one of the specified dimensions of the source image is less than
        the corresponding dimension of the destination image (for ippSuper
        method only).
      - If the width or height of the source or destination image is
        negative.

    Examples
    --------
    >>> from skimage import data
    >>> from skipp.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100)).shape
    (100, 100)
    """
    cdef int ippStatusIndex = 0  # OK
    cdef void * cyimage
    cdef void * cydestination

    cdef IppDataType ipp_src_datatype

    cdef int img_width
    cdef int img_height

    cdef int dst_width
    cdef int dst_height

    cdef int numChannels

    cdef IppiInterpolationType interpolation

    cdef Ipp32u antialiasing

    # TODO
    # more safe initialization
    cdef Ipp32u numLobes = num_lobes # for Lanczos interpolation

    # TODO
    # check after
    cdef double ippBorderValue = cval

    ipp_src_datatype = __get_ipp_data_type(image)
    if(ipp_src_datatype == ippUndef):
        raise ValueError("Image data type not supported")

    interpolation = __get_IppiInterpolationType(order)
    if(interpolation == UndefValue):
        raise ValueError("Undef `order` or not supported")

    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    # matchs with `numpy.pad`
    # {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
    cdef IppiBorderType ippBorderType = __get_numpy_pad_IppBorderType(mode)
    if(ippBorderType == UndefValue):
        raise ValueError("Boundary mode not supported")

    if anti_aliasing:
        antialiasing = <Ipp32u>1
    else:
        antialiasing = <Ipp32u>0

    # TODO
    # more safe initialization
    img_width = image.shape[1]
    img_height = image.shape[0]

    dst_width = output_shape[1]
    dst_height = output_shape[0]

    if numChannels == 1:
        output = np.empty((dst_height, dst_width), dtype=image.dtype,
                               order='C')
    else:
        output = np.empty((dst_height, dst_width, numChannels),
                               dtype=image.dtype, order='C')

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(output)

    ippStatusIndex = transform.own_Resize(ipp_src_datatype,
                                          cyimage,
                                          cydestination,
                                          img_width,
                                          img_height,
                                          dst_width,
                                          dst_height,
                                          numChannels,
                                          antialiasing,
                                          numLobes,
                                          interpolation,
                                          ippBorderType,
                                          ippBorderValue)
    __get_ipp_error(ippStatusIndex)
    return output
