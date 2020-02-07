from __future__ import absolute_import
include "../_ipp_utils/_ippi.pxd"

cdef extern from "warp.h":
    IppStatus ippi_Warp(IppDataType ippDataType,
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

    IppStatus ippi_RotateCoeffs(double angle,
                                double xCenter,
                                double yCenter,
                                double * coeffs)

    IppStatus ippi_GetAffineDstSize(int img_width,
                                    int img_height,
                                    int * dst_width,
                                    int * dst_height,
                                    double * coeffs)
