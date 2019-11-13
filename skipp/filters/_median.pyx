import numpy as np
from dtype import img_as_float32
cimport numpy as cnp
cimport cython

cnp.import_array()
cdef extern from "src/median.c":
    int MedianFilter_32f_C1_3D(void * pSRC,
                               void * pDST,
                               int img_width,
                               int img_height,
                               int img_depth,
                               int mask_width,
                               int mask_height,
                               int mask_depth,
                               int borderType)

    int MedianFilterFLOAT32(void * pSRC,
                            int stepSize,
                            void * pDST,
                            int img_width,
                            int img_height,
                            int mask_width,
                            int mask_height,
                            int borderType)  # const float * pBorderValue) <-----~~

__all__ = ['median', 'median_1']


# from _ni_support.py scipy/ndimage/_ni_support.py
def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if output is None:
        # as in skimage gaussian filter logic
        # Integer arrays are converted to float.
        output = np.zeros(shape, dtype=np.float32)
    elif type(output) in [type(type), type(np.zeros((4,)).dtype)]:
        output = np.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


def median(image, selem=None, out=None, mask=None, shift_x=False,
           shift_y=False, mode='nearest', cval=0.0, behavior='ipp'):
    if(behavior != 'ipp'):
        raise ValueError("There is only 'ipp' behavior is allowed")

    destination = _get_output(out, image)

    # warn about selem and mask

    cdef void * cyimage
    cdef void * cydestination

    cdef int img_width
    cdef int img_height
    cdef int img_depth = 1
    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    # raise error if numChannels is not 1
    cdef int ippBorderType = __get_IppBorderType(mode)

    # for ippMaskSize struct
    # median uses only selem's parametr
    cdef int selem_width = selem.shape[0]
    cdef int selem_height = selem.shape[1]
    cdef int selem_depth = 1

    # needed more correct way. Warning: conversion from 'npy_intp' to 'int', possible loss of data
    img_width = image.shape[0]
    img_height = image.shape[1]
    stepsize = image.strides[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    cdef int ippStatusIndex = 0  # OK

    ippStatusIndex = MedianFilter_32f_C1_3D(cyimage,
                                            cydestination,
                                            img_width,
                                            img_height,
                                            img_depth,
                                            selem_width,
                                            selem_height,
                                            selem_depth,
                                            ippBorderType)
    __get_ipp_error(ippStatusIndex)
    return destination


def median_1(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False, mode='nearest', cval=0.0, behavior='ipp'):
    if(behavior != 'ipp'):
        raise ValueError("There is only 'ipp' behavior is allowed")

    destination = _get_output(out, image)

    # warn about selem and mask

    cdef void * cyimage
    cdef void * cydestination

    cdef int img_width
    cdef int img_height

    cdef int stepsize

    cdef int numChannels = _get_number_of_channels(image)
    # raise error if numChannels is not 1
    cdef int ippBorderType = __get_IppBorderType(mode)

    cdef int selem_width = selem.shape[0]
    cdef int selem_height = selem.shape[1]

    img_width = image.shape[0]
    img_height = image.shape[1]
    stepsize = image.strides[0]

    cyimage = <void*> cnp.PyArray_DATA(image)
    cydestination = <void*> cnp.PyArray_DATA(destination)

    cdef int ippStatusIndex = 0  # OK

    ippStatusIndex = MedianFilterFLOAT32(cyimage,
                                         stepsize,
                                         cydestination,
                                         img_width,
                                         img_height,
                                         selem_width,
                                         selem_height,
                                         ippBorderType)
    __get_ipp_error(ippStatusIndex)
    return destination
# <<< median filter module