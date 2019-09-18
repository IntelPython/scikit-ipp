#include "ipp.h"
#ifndef MEDIAN_H
#define MEDIAN_H

int
MedianFilterFLOAT32_3D(void * pSRC,
                       int srcStep,
                       void * pDST,
                       int dstStep,
                       int img_width,
                       int img_height,
                       int img_depth,
                       int kSize,
                       IpprBorderType borderType,
                       const Ipp32f * pBorderValue);

int
MedianFilterUINT8_3D(void * pSRC,
                     int srcStep,
                     void * pDST,
                     int dstStep,
                     int img_width,
                     int img_height,
                     int img_depth,
                     int kSize,
                     IpprBorderType borderType,
                     const Ipp8u * pBorderValue);

#endif
