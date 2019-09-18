#include "ipp.h"
#ifndef EDGES_H
#define EDGES_H

int
FilterBorderFLOAT32(void * pSRC,
                    int srcStep,
                    void * pDST,
                    int dstStep,
                    int img_width,
                    int img_height,
                    int borderType);

int
FilterBorderFLOAT32RGB(void * pSRC,
                       int srcStep,
                       void * pDST,
                       int dstStep,
                       int img_width,
                       int img_height,
                       int borderType);

int
LaplaceFilterFLOAT32(void * pSRC,
                     int srcStep,
                     void * pDST,
                     int dstStep,
                     int img_width,
                     int img_height,
                     int maskSize,
                     int borderType,
                     Ipp32f borderValue);

int
LaplaceFilterFLOAT32RGB(void * pSRC,
                        int srcStep,
                        void * pDST,
                        int dstStep,
                        int img_width,
                        int img_height,
                        int maskSize,
                        int borderType,
                        Ipp32f borderValue);

int
PrewittFilterFLOAT32(void * pA_srcDst,
                     void * pB_srcDst,
                     int stepsize,
                     int img_width,
                     int img_height);

int
PrewittFilterHorizonFLOAT32(void * pSRC,
                            int srcStep,
                            void * pDST,
                            int dstStep,
                            int img_width,
                            int img_height,
                            int maskSize,
                            int borderType,
                            Ipp32f borderValue);

int
PrewittFilterVertFLOAT32(void * pSRC,
                         int srcStep,
                         void * pDST,
                         int dstStep,
                         int img_width,
                         int img_height,
                         int maskSize,
                         int borderType,
                         Ipp32f borderValue);

int 
SobelFilterFLOAT32(void * pSRC,
                   int srcStep,
                   void * pDST,
                   int dstStep,
                   int img_width,
                   int img_height,
                   int maskSize,
                   int normType,
                   int borderType,
                   Ipp32f borderValue);

int
SobelFilterHorizonFLOAT32(void * pSRC,
                          int srcStep,
                          void * pDST,
                          int dstStep,
                          int img_width,
                          int img_height,
                          int maskSize,
                          int borderType,
                          Ipp32f borderValue);

int
SobelFilterVertFLOAT32(void * pSRC,
                       int srcStep,
                       void * pDST,
                       int dstStep,
                       int img_width,
                       int img_height,
                       int maskSize,
                       int borderType,
                       Ipp32f borderValue);

int
SobelFilterCrossFLOAT32(void * pSRC,
                        int srcStep,
                        void * pDST,
                        int dstStep,
                        int img_width,
                        int img_height,
                        int maskSize,
                        int borderType,
                        Ipp32f borderValue);


#endif