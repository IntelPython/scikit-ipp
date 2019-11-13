#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include "ipp.h"
#include "dtypes.h"

#define IPP_GAUSSIAN_SUPPORTED_DTYPES  0x2C2    // 1011000010
#define IPP_GAUSSIAN_SUPPORTED_TYPES_NUMBER 4   // 8u, 16u, 16s, 32f

typedef enum {
  ippiFilterGaussianBorder_8u = 0,
  ippiFilterGaussianBorder_16u = 1,
  ippiFilterGaussianBorder_16s = 2,
  ippiFilterGaussianBorder_32f = 3,
  undef = -1
} ippiFilterGaussianBorder_jump_table_index;

typedef
IppStatus(*ippiFilterGaussianBorder_C1R)(
  void * pSrc,
  int srcStep,
  void * pDst,
  int dstStep,
  IppiSize roiSize,
  float borderValue,  // TODO: unsafe cast -> check
  IppFilterGaussianSpec * pSpec,
  Ipp8u * pBuffer);

typedef
IppStatus(*ippiFilterGaussianBorder_C3R)(
  void * pSrc,
  int srcStep,
  void * pDst,
  int dstStep,
  IppiSize roiSize,
  void * borderValue, // TODO: unsafe cast -> check
  IppFilterGaussianSpec * pSpec,
  Ipp8u * pBuffer);

int
ippiFilterGaussianBorder(
  IppDataTypeIndex ipp_src_dst_index,
  void * pSrc,
  void * pDst,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  IppiBorderType ippBorderType,
  float ippBorderValue);

IppStatus
get_borderValue_C3(
  IppDataTypeIndex ipp_src_dst_index,
  void * borderValue_C3,
  float ippBorderValue);

int
GaussianFilter(
  IppDataTypeIndex input_index,
  IppDataTypeIndex output_index,
  void * pInput,
  void * pOutput,
  int img_width,
  int img_height,
  int numChannels,
  float sigma,
  int kernelSize,
  IppiBorderType ippBorderType,
  float ippBorderValue,
  preserve_range_flag preserve_range);
#endif
