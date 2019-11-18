#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include "ipp.h"
#include "dtypes.h"

#define IPP_GAUSSIAN_SUPPORTED_DTYPES  0x2C2    // 1011000010

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
