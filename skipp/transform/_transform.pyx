from __future__ import absolute_import
include "../_ipp_utils/_ipp_utils.pxd"
include "../_ipp_utils/_ippi.pxd"

import numpy as np
cimport numpy as cnp
cimport _transform as transform
cnp.import_array()


cpdef warp(image, inverse_map, map_args={}, output_shape=None, order=1,
           mode='constant', cval=0., clip=True, preserve_range=False):
    """Warp an image according to a given coordinate transformation.
    # TODO - update docstrings
    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.
        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.
         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.
        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.
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
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    warped : double ndarray
        The warped input image.
    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.
    Examples
    --------
    >>> from skimage.transform import warp
    >>> from skimage import data
    >>> image = data.camera()
    The following image warps are all equal but differ substantially in
    execution time. The image is shifted to the bottom.
    Use a geometric transform to warp an image (fast):
    >>> from skimage.transform import SimilarityTransform
    >>> tform = SimilarityTransform(translation=(0, -10))
    >>> warped = warp(image, tform)
    Use a callable (slow):
    >>> def shift_down(xy):
    ...     xy[:, 1] -= 10
    ...     return xy
    >>> warped = warp(image, shift_down)
    Use a transformation matrix to warp an image (fast):
    >>> matrix = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> from skimage.transform import ProjectiveTransform
    >>> warped = warp(image, ProjectiveTransform(matrix=matrix))
    You can also use the inverse of a geometric transformation (fast):
    >>> warped = warp(image, tform.inverse)
    For N-D images you can pass a coordinate array, that specifies the
    coordinates in the input image for every element in the output image. E.g.
    if you want to rescale a 3-D cube, you can do:
    >>> cube_shape = np.array([30, 30, 30])
    >>> cube = np.random.rand(*cube_shape)
    Setup the coordinate array, that defines the scaling:
    >>> scale = 0.1
    >>> output_shape = (scale * cube_shape).astype(int)
    >>> coords0, coords1, coords2 = np.mgrid[:output_shape[0],
    ...                    :output_shape[1], :output_shape[2]]
    >>> coords = np.array([coords0, coords1, coords2])
    Assume that the cube contains spatial data, where the first array element
    center is at coordinate (0.5, 0.5, 0.5) in real space, i.e. we have to
    account for this extra offset when scaling the image:
    >>> coords = (coords + 0.5) / scale - 0.5
    >>> warped = warp(cube, coords)
    """
    cdef int ippStatusIndex = 0  # OK
    cdef void * cyimage
    cdef void * cydestination
    cdef double * cy_coeffs
    cdef double * cy_inverse_map_matrix  # ndarray: ``(3, 3)`` homogeneous transformation matrix

    cdef cnp.ndarray coeffs =  np.zeros((2, 3),dtype = np.double, order='C')
    # ~~~~ currently
    #cy_coeffs = <double*> cnp.PyArray_DATA(coeffs)
    cy_coeffs = <double*> cnp.PyArray_DATA(inverse_map)

    # TODO
    # check if inverse_map C-order, if ndarray
    cdef cnp.ndarray inverse_map_matrix =  np.zeros((3, 3),dtype = np.double, order='C')
    cy_inverse_map_matrix = <double*> cnp.PyArray_DATA(inverse_map_matrix)

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
    
    cdef IppiWarpDirection direction = ippWarpForward

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

    cdef IppiBorderType ippBorderType = __get_IppBorderType(mode)
    if(ippBorderType == UndefValue):
        raise ValueError("Boundary mode not supported")

    # TODO
    # more safete initialization
    img_width = image.shape[1]
    img_height = image.shape[0]

    if output_shape:
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

    ippStatusIndex = transform.ippi_Warp(ipp_src_datatype,
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
    # TODO
    Parameters
    ----------
    image : ndarray
        Input image.
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
        # TODO
        in scikit-ipp:
            img_width/2 -0.5
            img_height/2 0.5
    Returns
    -------
    rotated : ndarray
        Rotated version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to be
        in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    # TODO
    # Notes
    # * `preserve_range` disabled. Call ``skimage.img_as_float32`` before using
    #   skipp.tranform.rotate. This func doesn't convert provided image to float
    #   types and `preserve_range` feature is not implemented
    # * supported borders
    # * supported interpolations
    # * supported data types
    # * supported dimentions
    # * disabled params
    # * Bi-cubic
    # Examples
    # * examples with image_as_float (for `preserve_range`)
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import rotate
    >>> image = data.camera()
    >>> rotate(image, 2).shape
    (512, 512)
    >>> rotate(image, 2, resize=True).shape
    (530, 530)
    >>> rotate(image, 90, resize=True).shape
    (512, 512)
    """
    cdef int ippStatusIndex = 0  # OK

    cdef int img_width
    cdef int img_height

    cdef int dst_width
    cdef int dst_height

    cdef double cy_xCenter
    cdef double cy_yCenter

    cdef int numChannels
    cdef double cy_angle = angle

    cdef double * cy_coeffs
    cdef double * cy_inverse_map_matrix # ndarray: ``(3, 3)`` homogeneous transformation matrix

    cdef cnp.ndarray coeffs =  np.zeros((2, 3),dtype = np.double, order='C')
    cy_coeffs = <double*> cnp.PyArray_DATA(coeffs)

    cdef cnp.ndarray inverse_map_matrix =  np.zeros((3, 3), dtype = np.double, order='C')
    cy_inverse_map_matrix = <double*> cnp.PyArray_DATA(inverse_map_matrix)

    if(image.ndim == 2):
        numChannels = 1
    elif(image.ndim == 3) & (image.shape[-1] == 3):
        numChannels = 3
    else:
        raise ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)

    # TODO
    # more safety initialization
    img_width = image.shape[1]
    img_height = image.shape[0]

    if center is None:
        cy_xCenter = img_width/2.0 - 0.5
        cy_yCenter = img_height/2.0 - 0.5
    else:
        cy_xCenter = center[0]
        cy_yCenter = center[1]

    # TODO
    # correct coeffs, if `resize` is `True`
    # TODO
    # move ippi_RotateCoeffs, by using `warp` function's `map_args` param 
    ippStatusIndex = transform.ippi_RotateCoeffs(cy_angle, cy_xCenter,
                                                 cy_yCenter, cy_coeffs)
    __get_ipp_error(ippStatusIndex)

    ippStatusIndex = transform.ippi_GetAffineDstSize(img_width, img_height, &dst_width,
                                                     &dst_height, cy_coeffs)
    __get_ipp_error(ippStatusIndex)

    # TODO
    # add _get_output// check output dtype
    if resize:
        if numChannels ==1:
            output_shape = (dst_height, dst_width)
        else:
            output_shape = (dst_height, dst_width, numChannels)
    else:
        output_shape = None
    # TODO
    # enable `map_args`
    #~~~~~~ currently `coeffs` instead of `inverse_map_matrix`
    return warp(image, inverse_map=coeffs, map_args={}, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)