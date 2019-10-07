import numpy as np
from cpython.exc cimport PyErr_SetString
from cpython.exc cimport PyErr_Occurred
from cpython.ref cimport PyObject
cimport numpy as cnp
cimport cython

cnp.import_array()

IPP_GAUSSIAN_SUPPORTED_DTYPES = [np.uint8, np.uint16, np.int16, np.float32]

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

cdef extern from "ipptypes.h":
    ctypedef int IppStatus


cdef extern from "ippcore.h":
    const char * ippGetStatusString(IppStatus stsCode)


cdef int __get_IppBorderType(str mode):
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
        # TODO: set exception behavior
        PyErr_SetString(ValueError, "boundary mode not supported")


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
def convert_to_float(image, preserve_range):
    """Convert input image to double image with the appropriate range.
    """
    if preserve_range:
        return image.astype(np.float32)
    # TODO add img_as_float32
    else:
        raise ValueError("Currently not supported")


cdef PyObject * __get_ipp_error(int ippStatusIndex) except *:
    cdef const char * status_string
    if ippStatusIndex != int(0):
        status_string = ippGetStatusString(ippStatusIndex)
        PyErr_SetString(RuntimeError, status_string)


cdef __pass_ipp_gaussian(cnp.ndarray source, cnp.ndarray destination, float sigma, float truncate,
                         int ippBorderType, float ippBorderValue):

    cdef int index  # index for _get_gaussian_filter_func_index

    cdef void * cysource
    cdef void * cydestination

    cysource = <void*> cnp.PyArray_DATA(source)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    # TODO
    # check the equation that provides the kernelSize
    # make the radius of the filter equal to truncate standard deviations
    # as is in SciPy
    cdef int kernelSize = int(truncate * sigma + 0.5) * 2 - 1

    cdef int numChannels = _get_number_of_channels(source)

    # needed more correct way. Warning: conversion from 'npy_intp'
    # to 'int', possible loss of data
    cdef int img_width = source.shape[0]
    cdef int img_height = source.shape[1]

    # TODO change to platform aware integer
    cdef int stepsize = source.strides[0]
    # pass to IPP the source and destination arrays
    index = _get_gaussian_filter_func_index(destination.dtype, numChannels)
    # ~~~ delete number of channels from here
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
    __get_ipp_error(ippStatusIndex)

cpdef gaussian(image, sigma=1.0, output=None, mode='nearest', cval=0,
               multichannel=None, preserve_range=False, truncate=4.0):
    """
    Parameters
    ----------
    image :
    sigma :
    output :
    mode :
    cval :
    multichannel :
    preserve_range :
    truncate :
    """
    # TODO
    # add warnings for multichannel

    # TODO
    # check input
    # is input correct array
    # use numpy.require to provid type that satisfies requirements.
    # image = convert_to_float(image)

    # TODO
    # add correct preserve range module

    shape = image.shape

    input_dtype = image.dtype

    cdef float sd = float(sigma)
    cdef float tr = float(truncate)
    cdef float ippBorderValue = float(cval)

    cdef int ippBorderType = __get_IppBorderType(mode)

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

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)

        elif input_dtype == np.int8:
            # convert input to np.uint8 ---> converted copy of input
            image = image.astype(dtype=np.uint8, order='C', copy=True)

            # create output as np.uint8
            output = np.zeros(shape, dtype=np.uint8)

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del image

            # convert destination to np.int8
            output = output.astype(dtype=np.int8, order='C')
        else:
            # convert input to np.float32 ---> converted copy of input
            image = image.astype(dtype=np.float32, order='C', copy=True)

            # create output as np.float32 ---> converted copy of output
            output = np.zeros(shape, dtype=np.float32)

            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)

            # delete source (copy of input) ---> free copy of input from mem
            del image

            # convert destination to np.int8 ---> output
            output = output.astype(dtype=input_dtype, order='C')
    elif output_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
        if input_dtype in IPP_GAUSSIAN_SUPPORTED_DTYPES:
            if output_dtype == input_dtype:
                # pass to IPP the source and destination arrays
                __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
            else:
                # convert input to output dtype
                image = image.astype(dtype=output_dtype, order='C', copy=True)

                # pass to IPP the source and destination arrays
                __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
        else:
            # convert input to outputdtype and
            image = image.astype(dtype=output_dtype, order='C', copy=True)
            # converted input is source, output is destination
            # pass to IPP the source and destination arrays
            __pass_ipp_gaussian(image, output, sd, tr, ippBorderType, ippBorderValue)
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
        if input_dtype == np.float32:
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

    return output
