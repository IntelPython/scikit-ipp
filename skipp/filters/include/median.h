#ifndef MEDIAN_H
#define MEDIAN_H
#include <stddef.h>
#include "ipp.h"
#include "dtypes.h"

int
MedianFilter_32f_C1_3D(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int img_depth,
    int mask_width,
    int mask_height,
    int mask_depth,
    int borderType);

int
ippiFilterMedianBorder(
    IppDataTypeIndex ipp_src_dst_index,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    int mask_width,
    int mask_height,
    IppiBorderType ippBorderType,
    float ippBorderValue);
#endif
