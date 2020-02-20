#ifndef _IPP_WR_H
#define _IPP_WR_H
#include <stddef.h>
#include "ipp.h"

/**************************************************************************************
 * filters module
 * funcs ...
 **************************************************************************************/

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterBorderInit_<mode> functions,
// where <mode> is: 16s, 32f or 64f
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterBorderInit(
    IppDataType ippKernelDataType,
    void * pKernel,
    IppiSize kernelSize, 
    int divisor,
    IppDataType ippImageDataType,
    int numChannels,
    IppRoundMode roundMode,
    IppiFilterBorderSpec * pSpec);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterBorder_<mode> functions,
// where <mode> is: 8u_C1R, 16u_C1R, 16s_C1R, 32f_C1R, 64f_C1R,
// 8u_C3R, 16u_C3R, 16s_C3R, 32f_C3R, 8u_C4R, 16u_C4R, 16s_C4R,
// or 32f_C4R,
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterBorder(
    IppDataType ippDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    IppiFilterBorderSpec * pSpec,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterGaussianBorder_<mode> functions,
// where <mode> is: 8u_C1R, 16u_C1R, 16s_C1R, 32f_C1R, 8u_C3R,
// 16u_C3R, 16s_C3R or 32f_C3R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterGaussianBorder(
    IppDataType ippDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    float ippBorderValue,
    IppFilterGaussianSpec* pSpec,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterMedianBorder_<mode> functions,
// where <mode> is: 8u_C1R, 16u_C1R, 16s_C1R, 32f_C1R, 8u_C3R,
// 16u_C3R or 16s_C3R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterMedianBorder(
    IppDataType ippDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize dstRoiSize,
    int numChannels,
    IppiSize maskSize, 
    IppiBorderType ippBorderType, 
    float ippbordervalue, 
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterSobel_<mode> functions,
// where <mode> is: 8u16s_C1R, 16s32f_C1R, 16u32f_C1R or 32f_C1R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterSobel(
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiMaskSize maskSize,
    IppNormType normType,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterSobelHorizBorder_<mode> functions,
// where <mode> is: 8u16s_C1R, 16s_C1R or 32f_C1R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterSobelHorizBorder(
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiMaskSize maskSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterSobelVertBorder_<mode> functions,
// where <mode> is: 8u16s_C1R, 16s_C1R or 32f_C1R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterSobelVertBorder(
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiMaskSize maskSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterPrewittHorizBorder_<mode> functions,
// where <mode> is: 8u16s_C1R, 16s_C1R or 32f_C1R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterPrewittHorizBorder(
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiMaskSize maskSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    Ipp8u * pBuffer);

////////////////////////////////////////////////////////////////////////////////////////
//
// Adapter for ippiFilterPrewittVertBorder_<mode> functions,
// where <mode> is: 8u16s_C1R, 16s_C1R or 32f_C1R.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
ippiFilterPrewittVertBorder(
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    int numChannels,
    IppiMaskSize maskSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    Ipp8u * pBuffer);
#endif  // _IPP_WR_H
