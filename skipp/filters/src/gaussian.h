#include "ipp.h"
#ifndef GAUSSIAN_H
#define GAUSSIAN_H

int GaussianFilterUINT8(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int GaussianFilterUINT8RGB(void * pSRC,
                           void * pDST,
                           int img_width,
                           int img_height,
                           int numChannels,
                           float sigma_,
                           int kernelSize,
                           int stepSize,
                           int ippBorderType,
                           float ippBorderValue);

int GaussianFilterUINT16(void * pSRC,
                         void * pDST,
                         int img_width,
                         int img_height,
                         int numChannels,
                         float sigma_,
                         int kernelSize,
                         int stepSize,
                         int ippBorderType,
                         float ippBorderValue);

int GaussianFilterUINT16RGB(void * pSRC,
                            void * pDST,
                            int img_width,
                            int img_height,
                            int numChannels,
                            float sigma_,
                            int kernelSize,
                            int stepSize,
                            int ippBorderType,
                            float ippBorderValue);

int GaussianFilterINT16(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue);

int GaussianFilterINT16RGB(void * pSRC,
                           void * pDST,
                           int img_width,
                           int img_height,
                           int numChannels,
                           float sigma_,
                           int kernelSize,
                           int stepSize,
                           int ippBorderType,
                           float ippBorderValue);

int GaussianFilterFLOAT32(void * pSRC,
                          void * pDST,
                          int img_width,
                          int img_height,
                          int numChannels,
                          float sigma_,
                          int kernelSize,
                          int stepSize,
                          int ippBorderType,
                          float ippBorderValue);

int GaussianFilterFLOAT32RGB(void * pSRC,
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
