from __future__ import absolute_import
include "../_ipp_utils/_ipp_utils.pxd"
include "../_ipp_utils/_ippi.pxd"

import numpy as np
cimport numpy as cnp
cimport _transform as transform
cnp.import_array()


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
    cdef void * cyimage
    cdef void * cydestination
    cdef double * cy_coeffs

    cdef cnp.ndarray coeffs =  np.zeros((2, 3),dtype = np.double, order='C')
    cy_coeffs = <double*> cnp.PyArray_DATA(coeffs)

    cdef IppDataType ipp_src_datatype

    cdef int img_width
    cdef int img_height

    cdef int dst_width
    cdef int dst_height

    cdef int numChannels

    cdef IppiInterpolationType interpolation

    cdef double cy_xCenter
    cdef double cy_yCenter
    cdef double cy_angle = angle
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
        output = np.empty(output_shape, dtype=image.dtype, order='C')
    else:
        output = np.empty_like(image, dtype=image.dtype, order='C')

    dst_width = output.shape[1]
    dst_height = output.shape[0]

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
