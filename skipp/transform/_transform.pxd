from __future__ import absolute_import
include "../_ipp_utils/_ippi.pxd"

cdef extern from "own_warp.h":
    IppStatus own_Warp(IppDataType ippDataType,
                       void * pSrc,
                       void * pDst,
                       int img_width,
                       int img_height,
                       int dst_width,
                       int dst_height,
                       int numChannels,
                       double * coeffs,
                       IppiInterpolationType interpolation,
                       IppiWarpDirection direction,
                       IppiBorderType ippBorderType,
                       double ippBorderValue)

    IppStatus own_RotateCoeffs(double angle,
                                double xCenter,
                                double yCenter,
                                double * coeffs)

    IppStatus own_GetAffineDstSize(int img_width,
                                   int img_height,
                                   int * dst_width,
                                   int * dst_height,
                                   double * coeffs)


cdef extern from "own_resize.h":
    IppStatus own_Resize(IppDataType ippDataType,
                         void * pSrc,
                         void * pDst,
                         int img_width,
                         int img_height,
                         int dst_width,
                         int dst_height,
                         int numChannels,
                         Ipp32u antialiasing,
                         IppiInterpolationType interpolation,
                         IppiBorderType ippBorderType,
                         double ippBorderValue)
