#ifndef LAPLACE_H
#define LAPLACE_H
#include "ipp.h"
#include "dtypes.h"

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
#endif
