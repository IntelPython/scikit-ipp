import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

cdef extern from "src/gaussian.c":
    int  GaussianFilter(int index,
                        void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue)

IPP_GAUSSIAN_SUPPORTED_DTYPES = [np.uint8, np.uint16, np.int16, np.float32]


def _getIppBorderType(mode):
    """ Convert an extension mode to the corresponding IPP's IppiBorderType integer code.
    """
    # 'nearest' -----> IPP's ippBorderRepl
    if mode == 'nearest':
        return 1
    # 'wrap' --------> IPP's ippBorderWrap
    elif mode == 'wrap':
        return 2
    # 'mirror' ------> IPP's ippBorderMirror
    elif mode == 'mirror':
        return 3
    # 'reflect' -----> IPP's ippBorderMirrorR
    elif mode == 'reflect':
        return 4
    # IPP's ippBorderDefault
    elif mode == 'default':
        return 5
    # 'constant' ----> IPP's ippBorderConst
    elif mode == 'constant':
        return 6
    # IPP's ippBorderTransp
    elif mode == 'transp':
        return 7
    else:
        raise RuntimeError('boundary mode not supported')


# needed more correct version (guest_spatial_dim skimage)
def _get_number_of_channels(image):
    if image.ndim == 2:
        channels = 1    # single (grayscale)
    elif image.ndim == 3 and image.shape[-1] == 3:
        channels = 3   # 3 channels
    else:
        ValueError("Expected 2D array with 1 or 3 channels, got %iD." % image.ndim)
    return channels


def _get_gaussian_filter_func_index(dtype, int numChannels):
    if(numChannels == 1):
        if(dtype == np.uint8):
            return 0
        elif(dtype == np.uint16):
            return 1
        elif(dtype == np.int16):
            return 2
        elif(dtype == np.float32):
            return 3
        else:
            raise ValueError("Currently not supported")
    elif(numChannels == 3):
        if(dtype == np.uint8):
            return 4
        elif(dtype == np.uint16):
            return 5
        elif(dtype == np.int16):
            return 6
        elif(dtype == np.float32):
            return 7
        else:
            raise ValueError("Currently not supported")
    else:
        raise ValueError("Currently not supported")


# from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py
# convert_to_float
def convert_to_float(image, preserve_range):
    """Convert input image to double image with the appropriate range.
    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    image : ndarray
        Transformed version of the input.
    """
    if preserve_range:
        return image.astype(np.float32)
    # TODO add img_as_float32
    else:
        raise ValueError("Currently not supported")


cpdef gaussian(image, sigma=1.0, output=None, mode='nearest', cval=0,
                 multichannel=None, preserve_range=False, truncate=4.0):
    # TODO
    # use numpy.require to provid type that satisfies requirements.
    # image = convert_to_float(image)

    # TODO
    # add warnings for multichannel

    # TODO
    # check input
    # is input correct array

    shape = image.shape

    input_dtype = image.dtype

    cdef cnp.ndarray destination

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)

    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    cdef int kernelSize = int(tr * sd + 0.5) * 2 - 1

    cdef void * cysource
    cdef void * cydestination

    cdef int index  # index for _get_gaussian_filter_func_index

    # TODO
    # use IPP's ippiFilterGaussian_<> ---> platform-aware functions
    # int kernelSize --> cnp.uint64_t or ctypedef unsigned long
    cdef int img_width
    cdef int img_height
    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    cdef int ippBorderType = _getIppBorderType(mode)

    cdef int ippStatusIndex = 0  # OK

    if output is None:
        output_dtype = None
    elif isinstance(output, np.dtype):
        output_dtype = output
        output = np.zeros(shape, dtype=output_dtype)
    elif isinstance(output, np.ndarray):
        output_dtype = output.dtype
    else:
        raise ValueError("not correct output value or ~~~")

    if output_dtype is None:
        if input_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
            # create output as input dtype
            output = np.zeros(shape, dtype=input_dtype)

            # input is source
            # output is destination
            cysource = <void*> cnp.PyArray_DATA(image)
            cydestination = <void*> cnp.PyArray_DATA(output)

            # needed more correct way. Warning: conversion from 'npy_intp'
            # to 'int', possible loss of data
            img_width = image.shape[0]
            img_height = image.shape[1]
            stepsize = image.strides[0]
            # pass to IPP the source and destination arrays
            index = _get_gaussian_filter_func_index(output.dtype, numChannels)
            ippStatusIndex = GaussianFilter(index,
                                            cysource,
                                            cydestination,
                                            img_width,
                                            img_height,
                                            numChannels,
                                            sigma,
                                            kernelSize,
                                            stepsize,
                                            ippBorderType,
                                            ippBorderValue)

        elif input_dtype == np.int8:
            # convert input to np.uint8 ---> converted copy of input
            image = image.astype(dtype=np.uint8, order='C', copy=True)

            # converted copy of input is source
            cysource = <void*> cnp.PyArray_DATA(image)

            # create output as np.uint8
            output = np.zeros(shape, dtype=np.uint8)

            # needed more correct way. Warning: conversion from 'npy_intp'
            # to 'int', possible loss of data
            img_width = image.shape[0]
            img_height = image.shape[1]
            stepsize = image.strides[0]

            # output is destination
            cydestination = <void*> cnp.PyArray_DATA(output)

            # pass to IPP the source and destination arrays
            index = _get_gaussian_filter_func_index(output.dtype, numChannels)
            ippStatusIndex = GaussianFilter(index,
                                            cysource,
                                            cydestination,
                                            img_width,
                                            img_height,
                                            numChannels,
                                            sigma,
                                            kernelSize,
                                            stepsize,
                                            ippBorderType,
                                            ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del image

            # convert destination to np.int8
            output = output.astype(dtype=np.int8, order='C')
        else:
            # convert input to np.float32 ---> converted copy of input
            image = image.astype(dtype=np.float32, order='C', copy=True)

            # converted copy of input is source
            cysource = <void*> cnp.PyArray_DATA(image)

            # create output as np.float32 ---> converted copy of output
            output = np.zeros(shape, dtype=np.float32)

            # needed more correct way. Warning: conversion from 'npy_intp'
            # to 'int', possible loss of data
            img_width = image.shape[0]
            img_height = image.shape[1]
            stepsize = image.strides[0]

            # output is destination
            cydestination = <void*> cnp.PyArray_DATA(output)

            # pass to IPP the source and destination arrays
            index = _get_gaussian_filter_func_index(output.dtype, numChannels)
            ippStatusIndex = GaussianFilter(index,
                                            cysource,
                                            cydestination,
                                            img_width,
                                            img_height,
                                            numChannels,
                                            sigma,
                                            kernelSize,
                                            stepsize,
                                            ippBorderType,
                                            ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del image

            # convert destination to np.int8 ---> output
            output = output.astype(dtype=input_dtype, order='C')
    elif output_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
        if input_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
            if output_dtype == input_dtype:
                # input is source, output is destination
                cysource = <void*> cnp.PyArray_DATA(image)
                cydestination = <void*> cnp.PyArray_DATA(output)

                # needed more correct way. Warning: conversion from 'npy_intp'
                # to 'int', possible loss of data
                img_width = image.shape[0]
                img_height = image.shape[1]
                stepsize = image.strides[0]
                # pass to IPP the source and destination arrays
                index = _get_gaussian_filter_func_index(output.dtype, numChannels)
                ippStatusIndex = GaussianFilter(index,
                                                cysource,
                                                cydestination,
                                                img_width,
                                                img_height,
                                                numChannels,
                                                sigma,
                                                kernelSize,
                                                stepsize,
                                                ippBorderType,
                                                ippBorderValue)
            else:
                # convert input to output dtype and
                # converted input is source, output is destination
                # pass to IPP the source and destination arrays
                # return output
                raise RuntimeError("currently not implemented 3")
        else:
            # convert input to outputdtype and
            # converted input is source, output is destination
            # pass to IPP the source and destination arrays
            # return output
            raise RuntimeError("currently not implemented 4")
    elif output_dtype is np.int8 and input_dtype is np.uint8:
        # convert output to np.uint8 ---> converted copy of output
        # converted copy of output is destination
        # input is source
        # pass to IPP the source and destination arrays
        # convert destination and save in output array (
        # or create copy of destination in output dtype and copy all  in output)
        # return output
        raise RuntimeError("currently not implemented 5")
    else:
        if input_dtype != np.float32:
            # input is source
            # convert output to np.float32---> converted copy of output
            # converted copy of output is destination
            # pass to IPP source and destination
            # convert destination and save in output array (
            # or create copy of destination in output dtype and copy all  in output)
            # return output
            raise RuntimeError("currently not implemented 6")
        else:
            # convert input to np.float32 ---> converted copy of input
            # converted copy of input is source
            # convert output to np.float32---> converted copy of output
            # converted copy of output is destination
            # pass to IPP source and destination
            # delete source (copy of input) ---> free copy of input from mem
            # convert destination and save in output array (
            # or create copy of destination in output dtype and copy all  in output)
            # return output
            raise RuntimeError("currently not implemented 7")

    # ippStatusIndex: ipp error handler will be added
    return output
