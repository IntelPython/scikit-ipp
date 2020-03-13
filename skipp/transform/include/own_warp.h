
////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for image transformations, that uses Intel(R) IPP.
//
////////////////////////////////////////////////////////////////////////////////////////
#ifndef WARP_H
#define WARP_H
#include <stddef.h>
#include "ipp.h"
#include "_ipp_wr.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_RotateCoeffs
//
//    own_RotateCoeffs uses Intel(R) IPP's ippiGetRotateShift and
//    ippiGetRotateTransform functions for getting affine coefficients for the rotation
//    transform. ippiGetRotateShift, computes shift values for rotation of an image
//    around the specified center. ippiGetRotateTransform computes the affine
//    coefficients for the rotation transform.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_RotateCoeffs(
    double angle,
    double xCenter,
    double yCenter,
    double *coeffs);

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_GetAffineDstSize
//
//    own_GetAffineDstSize uses Intel(R) IPP's ippiGetAffineBound for computing size
//    destination image for the provided coeffs for the affine transformations.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_GetAffineDstSize(
    int img_width,
    int img_height,
    int * dst_width,
    int * dst_height,
    double * coeffs);

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Warp
//
//    own_Warp uses Intel(R) IPP's funcstions for implementing image warp
//    transformations
//
//    TODO: complete the description.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Warp(
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
#endif // WARP_H
