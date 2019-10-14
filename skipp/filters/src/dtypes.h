#include "ipp.h"
#ifndef DTYPES_H
#define DTYPES_H

int
image_UINT8_as_float32(void * pSrc,
                       int srcStep,
                       void * pDst,
                       int dstStep,
                       int img_width,
                       int img_height);

int
image_INT8_as_float32(void * pSrc,
                      int srcStep,
                      void * pDst,
                      int dstStep,
                      int img_width,
                      int img_height);

int
image_INT16_as_float32(void * pSrc,
                       int srcStep,
                       void * pDst,
                       int dstStep,
                       int img_width,
                       int img_height);

int
image_UINT16_as_float32(void * pSrc,
                        int srcStep,
                        void * pDst,
                        int dstStep,
                        int img_width,
                        int img_height);

int
image_8s_as_8u_C1(void * pSrc,
                  int srcStep,
                  void * pDst,
                  int dstStep,
                  int img_width,
                  int img_height);

int
image_8u_as_8s_C1(void * pSrc,
                  int srcStep,
                  void * pDst,
                  int dstStep,
                  int img_width,
                  int img_height);

typedef 
int(*covertHandler)(void * pSrc,
                    int srcStep,
                    void * pDst,
                    int dstStep,
                    int img_width,
                    int img_height);
#endif
