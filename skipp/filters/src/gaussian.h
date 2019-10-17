#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include "ipp.h"
#include "dtypes.h"

#define IPP_GAUSSIAN_SUPPORTED_DTYPES  0x2C2 // 1011000010

typedef enum {
  gaussianFilter_Ipp8u = 0,
  gaussianFilter_Ipp16u = 1,
  gaussianFilter_Ipp16s = 2,
  gaussianFilter_Ipp32f = 3
} IppGaussianFilterIndex;

int
GaussianFilter_Ipp8u(
  void * pSRC,
  void * pDST,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);

int
GaussianFilter_Ipp16u(
  void * pSRC,
  void * pDST,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);

int
GaussianFilter_Ipp16s(
  void * pSRC,
  void * pDST,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);

int
GaussianFilter_Ipp32f(
  void * pSRC,
  void * pDST,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);

int
GaussianFilter(
  int input_index,
  int output_index,
  void * pInput,
  void * pOutput,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);

typedef
int(*gaussianFuncHandler)(
  void * pSRC,
  void * pDST,
  int img_width,
  int img_height,
  int numChannels,
  float sigma_,
  int kernelSize,
  int ippBorderType,
  float ippBorderValue);
#endif
