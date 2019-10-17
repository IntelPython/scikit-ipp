#include "ipp.h"
#ifndef MEDIAN_H
#define MEDIAN_H

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
MedianFilterFLOAT32(void * pSRC,
                    int stepSize,
                    void * pDST,
                    int img_width,
                    int img_height,
                    int mask_width,
                    int mask_height,
                    int borderType); // const float * pBorderValue) <-----~~
#endif
