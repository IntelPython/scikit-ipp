from __future__ import absolute_import
include "../_ipp_utils/_ippi.pxd"

cdef extern from "morphology.h":
    ctypedef enum ippiMorphologyFunction:
        IppiErodeBorder
        IppiDilateBorder


cdef extern from "morphology.h":
    IppStatus ippiMorphology(IppDataType datatype,
                             ippiMorphologyFunction ippiFunc,
                             void * pSrc,
                             void * pDst,
                             int img_width,
                             int img_height,
                             int numChannels,
                             void * pMask,
                             int mask_width,
                             int mask_height,
                             IppiBorderType ippBorderType,
                             float ippBorderValue)
