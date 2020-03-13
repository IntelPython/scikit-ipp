#ifndef RESIZE_H
#define RESIZE_H
#include <stddef.h>
#include "ipp.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Resize
//
//          Changes an image size using the different interpolation methods.
//          own_Resize uses Intel(R) IPP functions on the backend for implementing
//          resizing of image as is in scikit-image.
//
//    Note: currently ippBorderValue is disabled. TODO: needs implementing.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Resize(
    IppDataType ippDataType,
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
    double ippBorderValue);
#endif
