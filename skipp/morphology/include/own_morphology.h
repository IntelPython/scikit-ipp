
////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for morphology transfomations of image, that uses
//    Intel(R) IPP.
//
////////////////////////////////////////////////////////////////////////////////////////
#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H
#include <stddef.h>
#include "ipp.h"
#include "_ipp_wr.h"
#include "utils.h"

typedef enum {
    IppiErodeBorder,
    IppiDilateBorder
} ippiMorphologyFunction;

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Morphology
//
//        scikit-ipp's own image processing functions that perform morphological
//        operations on images.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Morphology(
    IppDataType datatype,
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
    float ippBorderValue);
#endif
