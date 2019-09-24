#include "ipp.h"
#ifndef GAUSSIAN_H
#define GAUSSIAN_H

int
GaussianFilterIpp8u_C1(void * pSRC,
                       void * pDST,
                       int img_width,
                       int img_height,
                       int numChannels,
                       float sigma_,
                       int kernelSize,
                       int stepSize,
                       int ippBorderType,
                       float ippBorderValue);

int
GaussianFilterIpp8u_C3(void * pSRC,
                       void * pDST,
                       int img_width,
                       int img_height,
                       int numChannels,
                       float sigma_,
                       int kernelSize,
                       int stepSize,
                       int ippBorderType,
                       float ippBorderValue);

int
GaussianFilterIpp16u_C1(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int
GaussianFilterIpp16u_C3(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int
GaussianFilterIpp16s_C1(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int
GaussianFilterIpp16s_C3(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int
GaussianFilterIpp32f_C1(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int
GaussianFilterIpp32f_C3(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

typedef 
int(*funcHandler)(void *pSRC,
                  void * pDST, 
                  int img_width, 
                  int img_height,
                  int numChannels, 
                  float sigma_, 
                  int kernelSize, 
                  int stepSize, 
                  int ippBorderType, 
                  float ippBorderValue);


int GaussianFilter(int index,
                   void * pSRC,
                   void * pDST,
                   int img_width,
                   int img_height,
                   int numChannels,
                   float sigma_,
                   int kernelSize,
                   int stepSize,
                   int ippBorderType,
                   float ippBorderValue);

#endif
