#ifndef SOBEL_H
#define SOBEL_H
#include "ipp.h"
#include "dtypes.h"

int
ippiFilterSobel(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppNormType normType,
    IppiMaskSize maskSize,
    IppiBorderType ippBorderType,
    float ippBorderValue);

int
ippiFilterSobelHorizBorder(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue);

int
ippiFilterSobelVertBorder(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue);

int
FilterSobelHoriz(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels);

int
FilterSobelVert(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels);
#endif