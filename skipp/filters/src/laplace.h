#ifndef LAPLACE_H
#define LAPLACE_H
#include "ipp.h"
#include "dtypes.h"
#include "borderfilter.h"

//                           -1 -1 -1
//          Laplace (3x3)    -1  8 -1
//                           -1 -1 -1

int
ippiFilterLaplaceBorder(
    IppDataTypeIndex ipp_src_dst_index,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue);

//                            0 -1  0
//          Laplace (3x3)    -1  4 -1
//                            0 -1  0

int
LaplaceFilter(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiBorderType ippBorderType,
    float ippBorderValue);
#endif