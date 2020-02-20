from __future__ import absolute_import
include "../_ipp_utils/_ippi.pxd"


cdef extern from "own_filters.h":
    ctypedef enum own_EdgeFilterKernel:
        own_filterSobelVert = 0
        own_filterSobelHoriz = 1
        own_filterSobel = 2
        own_filterPrewittVert = 3
        own_filterPrewittHoriz = 4
        own_filterPrewitt = 5 

    IppStatus own_FilterGaussian(IppDataType ippDataType,
                                 void * pSrc,
                                 void * pDst,
                                 int img_width,
                                 int img_height,
                                 int numChannels,
                                 float sigma_,
                                 int kernelSize,
                                 IppiBorderType ippBorderType,
                                 float ippBorderValue)

    IppStatus own_FilterMedian(IppDataType ippDataType,
                               void * pSrc,
                               void * pDst,
                               int img_width,
                               int img_height,
                               int numChannels,
                               int mask_width,
                               int mask_height,
                               IppiBorderType ippBorderType,
                               float ippBorderValue)

    IppStatus own_FilterLaplace(IppDataType ippDataType,
                                void * pSrc,
                                void * pDst,
                                int img_width,
                                int img_height,
                                int numChannels,
                                IppiBorderType ippBorderType,
                                float ippBorderValue)

    IppStatus own_FilterEdge(own_EdgeFilterKernel edgeKernel,
                             IppDataType ippSrcDataType,
                             IppDataType ippDstDataType,
                             void * pSrc,
                             void * pDst,
                             int img_width,
                             int img_height,
                             int numChannels)

    IppStatus own_FilterPrewitt(own_EdgeFilterKernel edgeKernel,
                                IppDataType ippSrcDataType,
                                IppDataType ippDstDataType,
                                void * pSrc,
                                void * pDst,
                                int img_width,
                                int img_height,
                                int numChannels)
