#ifndef WARP_H
#define WARP_H
#include <stddef.h>
#include "ipp.h"
#include "utils.h"

/**
* _ippiWarpAffine_interpolation:
* Adapter for:  Intel(R) IPP's ippiWarpAffine<interpolation>_<ipp_type>_<channels>,
* where <interpolation> is `Nearest`, `Linear` or `Cubic`,
* <ipp_type> is `8u`, `16u`, `16s`, `32f` or `64f`,
* <channels> is C1R, C3R or C4R.
*/
IppStatus
_ippiWarpAffine_interpolation(
    IppDataType ippDataType,
    IppiInterpolationType interpolation,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer);

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
