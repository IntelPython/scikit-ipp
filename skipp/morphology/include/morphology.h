#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H
#include <stddef.h>
#include "ipp.h"
#include "utils.h"

typedef enum {
    IppiErodeBorder,
    IppiDilateBorder
} ippiMorphologyFunction;

IppStatus
ippiMorphologyBorderGetSize(
    IppDataType datatype,
    IppiSize roiSize,
    IppiSize maskSize,
    int numChannels,
    int * pSpecSize,
    int * pBufferSize);

IppStatus
ippiMorphologyBorderInit(
    IppDataType datatype,
    int numChannels,
    IppiSize roiSize,
    const Ipp8u * pMask,
    IppiSize maskSize,
    IppiMorphState* pSpec,
    Ipp8u* pBuffer);

IppStatus
ippiErodeBorder(
    IppDataType datatype,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    const IppiMorphState * pSpec,
    Ipp8u* pBuffer);

IppStatus
ippiDilateBorder(
    IppDataType datatype,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    const IppiMorphState * pSpec,
    Ipp8u* pBuffer);

IppStatus
ippiMorphology(
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
