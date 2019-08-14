#include "edges.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; 

int
PrewittFilterHorizonFLOAT32(void * pSRC,
                            int srcStep,
                            void * pDST,
                            int dstStep,
                            int img_width,
                            int img_height,
                            int maskSize,
                            int borderType,
                            Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterPrewittHorizBorderGetBufferSize(roiSize,
                                                         maskSize,
                                                         ipp32f,
                                                         ipp32f,
                                                         1,
                                                         &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterPrewittHorizBorder_32f_C1R(pSrc1,
                                                    srcStep,
                                                    pDst1,
                                                    dstStep,
                                                    roiSize,
                                                    maskSize,
                                                    borderType,
                                                    borderValue,
                                                    pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status; 
};

int
PrewittFilterVertFLOAT32(void * pSRC,
                         int srcStep,
                         void * pDST,
                         int dstStep,
                         int img_width,
                         int img_height,
                         int maskSize,
                         int borderType,
                         Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterPrewittVertBorderGetBufferSize(roiSize,
                                                        maskSize,
                                                        ipp32f,
                                                        ipp32f,
                                                        1,
                                                        &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterPrewittVertBorder_32f_C1R(pSrc1,
                                                   srcStep,
                                                   pDst1,
                                                   dstStep,
                                                   roiSize,
                                                   maskSize,
                                                   borderType,
                                                   borderValue,
                                                   pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status; 
};

int 
SobelFilterFLOAT32(void * pSRC,
                   int srcStep,  
                   void * pDST,
                   int dstStep,
                   int img_width,
                   int img_height,
                   int maskSize,    // IppiMaskSize
                   int normType,    // IppNormType
                   int borderType,  // IppiBorderType
                   Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;

    check_sts( ippiFilterSobelGetBufferSize(roiSize,
                                            maskSize,
                                            normType,
                                            ipp32f,
                                            ipp32f,
                                            1,        // number of channels
                                            &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterSobel_32f_C1R(pSrc1,
                                       srcStep,
                                       pDst1,
                                       dstStep,
                                       roiSize,
                                       maskSize,
                                       normType,
                                       borderType,
                                       33,
                                       pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status; 
};


int
SobelFilterHorizonFLOAT32(void * pSRC,
                          int srcStep,
                          void * pDST,
                          int dstStep,
                          int img_width,
                          int img_height,
                          int maskSize,
                          int borderType,
                          Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterSobelHorizBorderGetBufferSize(roiSize,
                                                       maskSize,
                                                       ipp32f,
                                                       ipp32f,
                                                       1,
                                                       &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterSobelHorizBorder_32f_C1R(pSrc1,
                                                  srcStep,
                                                  pDst1,
                                                  dstStep,
                                                  roiSize,
                                                  maskSize,
                                                  borderType,
                                                  borderValue,
                                                  pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
};

int
SobelFilterVertFLOAT32(void * pSRC,
                          int srcStep,
                          void * pDST,
                          int dstStep,
                          int img_width,
                          int img_height,
                          int maskSize,
                          int borderType,
                          Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterSobelVertBorderGetBufferSize(roiSize,
                                                       maskSize,
                                                       ipp32f,
                                                       ipp32f,
                                                       1,
                                                       &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterSobelVertBorder_32f_C1R(pSrc1,
                                                  srcStep,
                                                  pDst1,
                                                  dstStep,
                                                  roiSize,
                                                  maskSize,
                                                  borderType,
                                                  borderValue,
                                                  pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
}

int
SobelFilterCrossFLOAT32(void * pSRC,
                        int srcStep,
                        void * pDST,
                        int dstStep,
                        int img_width,
                        int img_height,
                        int maskSize,
                        int borderType,
                        Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterSobelCrossGetBufferSize_32f_C1R(roiSize,
                                                         maskSize,
                                                         &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterSobelCrossBorder_32f_C1R(pSrc1,
                                                  srcStep,
                                                  pDst1,
                                                  dstStep,
                                                  roiSize,
                                                  maskSize,
                                                  borderType,
                                                  borderValue,
                                                  pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
}
