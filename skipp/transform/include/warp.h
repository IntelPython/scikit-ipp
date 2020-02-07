#ifndef WARP_H
#define WARP_H
#include <stddef.h>
#include "ipp.h"
#include "utils.h"

IppStatus
ippiWarpAffineCubic(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
);

IppStatus
ippiWarpAffineNearest(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
);

IppStatus
ippiWarpAffineLinear(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
);

IppStatus
ippi_RotateCoeffs(
    double angle,
    double xCenter,
    double yCenter,
    double *coeffs);

IppStatus
ippi_GetAffineDstSize(
    int img_width,
    int img_height,
    int * dst_width,
    int * dst_height,
    double * coeffs);

IppStatus
ippi_Warp(
    IppDataType ippDataType,
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
    double ippBorderValue);
#endif
