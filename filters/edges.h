#include "ipp.h"
#ifndef EDGES_H
#define EDGES_H

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