#include "edges.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;


//
//                                0 -1  0
//              Laplace (3x3)    -1  4 -1
//                                0 -1  0

const Ipp32f kernelLaplace[3 * 3] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };

int
FilterBorderFLOAT32(void * pSRC,
                    int srcStep,
                    void * pDST,
                    int dstStep,
                    int img_width,
                    int img_height,
                    int borderType)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc = NULL, *pDst = NULL;     // Pointers to source and 
                                           // destination images
    IppiSize roiSize = { img_width, img_height }; // Size of source and 
                                                  // destination ROI in pixels
    IppiSize  kernelSize = { 3, 3 };
    Ipp8u * pBuffer = NULL;                // Pointer to the work buffer
    IppiFilterBorderSpec* pSpec = NULL;    // context structure
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    Ipp32f borderValue = 0;
    int numChannels = 1;
    pSrc = (Ipp32f *) pSRC;
    pDst = (Ipp32f *) pDST;

    check_sts(status = ippiFilterBorderGetSize(kernelSize,
                                               roiSize,
                                               ipp32f,
                                               ipp32f,
                                               numChannels,
                                               &iSpecSize,
                                               &iTmpBufSize))

    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    check_sts(status = ippiFilterBorderInit_32f(kernelLaplace,
                                                kernelSize,
                                                ipp32f,
                                                numChannels,
                                                ippRndNear,
                                                pSpec))

    check_sts(status = ippiFilterBorder_32f_C1R(pSrc,
                                                srcStep,
                                                pDst,
                                                dstStep,
                                                roiSize,
                                                borderType,
                                                &borderValue,
                                                pSpec,
                                                pBuffer))
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
};

int
FilterBorderFLOAT32RGB(void * pSRC,
                       int srcStep,
                       void * pDST,
                       int dstStep,
                       int img_width,
                       int img_height,
                       int borderType)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc = NULL, *pDst = NULL;     // Pointers to source
                                           // and destination images
    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    IppiSize  kernelSize = { 3, 3 };
    Ipp8u * pBuffer = NULL;                // Pointer to the work buffer
    IppiFilterBorderSpec * pSpec = NULL;    // context structure
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    Ipp32f borderValue[] = {0, 0, 0};
    int numChannels = 3;
    pSrc = (Ipp32f *) pSRC;
    pDst = (Ipp32f *) pDST;

    check_sts(status = ippiFilterBorderGetSize(kernelSize,
                                               roiSize,
                                               ipp32f,
                                               ipp32f,
                                               numChannels,
                                               &iSpecSize,
                                               &iTmpBufSize))

    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    check_sts(status = ippiFilterBorderInit_32f(kernelLaplace,
                                                kernelSize,
                                                ipp32f,
                                                numChannels,
                                                ippRndNear,
                                                pSpec))

    check_sts(status = ippiFilterBorder_32f_C3R(pSrc,
                                                srcStep,
                                                pDst,
                                                dstStep,
                                                roiSize,
                                                borderType,
                                                &borderValue,
                                                pSpec,
                                                pBuffer))
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

//
//                               -1 -1 -1
//              Laplace (3x3)    -1  8 -1
//                               -1 -1 -1

int
LaplaceFilterFLOAT32(void * pSRC,
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
    Ipp32f * pSrc = NULL, * pDst = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    Ipp8u *pBuffer = NULL;                   // Pointer to the work buffer
    int bufferSize;
    pSrc = (Ipp32f *) pSRC;
    pDst = (Ipp32f *) pDST;

    check_sts( status = ippiFilterLaplaceBorderGetBufferSize(roiSize,
                                                             maskSize,
                                                             ipp32f,
                                                             ipp32f,
                                                             1,
                                                             &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    check_sts( ippiFilterLaplaceBorder_32f_C1R(pSrc,
                                               srcStep,
                                               pDst,
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
LaplaceFilterFLOAT32RGB(void * pSRC,
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
    Ipp32f * pSrc = NULL, * pDst = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    Ipp8u *pBuffer = NULL;                   // Pointer to the work buffer
    int bufferSize;
    
    Ipp32f bordervalue[] = {0,
                            0,
                            0};

//    Ipp32f bordervalue[] = {ippBorderValue,
//                            ippBorderValue,
//                            ippBorderValue};
    pSrc = (Ipp32f *) pSRC;
    pDst = (Ipp32f *) pDST;

    check_sts( status = ippiFilterLaplaceBorderGetBufferSize(roiSize,
                                                             maskSize,
                                                             ipp32f,
                                                             ipp32f,
                                                             3,
                                                             &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    };
    check_sts( ippiFilterLaplaceBorder_32f_C3R(pSrc,
                                               srcStep,
                                               pDst,
                                               dstStep,
                                               roiSize,
                                               maskSize,
                                               borderType,
                                               bordervalue,
                                               pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
};

int
PrewittFilterFLOAT32(void * pA_srcDst,
                     void * pB_srcDst,
                     int stepsize,
                     int img_width,
                     int img_height)
{
    // computes the square root of the sum of squares of the horizontal
    // sqrt(A**2 + B**2)/sqrt(2) and vertical Prewitt transforms.
    IppStatus status = ippStsNoErr;
    Ipp32f* pAsrcDst = NULL, *pBsrcDst = NULL;     // Pointers to source-destination
                                                   // images for IPP in-place funcs
    IppiSize roiSize = { img_width, img_height };  // Size of source-destination
                                                   // ROI in pixels
    pAsrcDst = (Ipp32f *) pA_srcDst;  // A
    pBsrcDst = (Ipp32f *) pB_srcDst;  // B pointer to the image array, that contains
                                      // results

    int aSrcDstStep = stepsize;
    int bSrcDstStep = stepsize;

    check_sts( ippiSqr_32f_C1IR(pAsrcDst, aSrcDstStep, roiSize) );
    check_sts( ippiSqr_32f_C1IR(pBsrcDst, bSrcDstStep, roiSize) );
    check_sts( ippiAdd_32f_C1IR(pAsrcDst, aSrcDstStep, pBsrcDst, bSrcDstStep, roiSize) );
    check_sts( ippiSqrt_32f_C1IR(pBsrcDst, bSrcDstStep, roiSize) );

    Ipp32f sqrt2 = (Ipp32f)IPP_SQRT2;
   
    check_sts( ippiDivC_32f_C1IR(sqrt2, pBsrcDst, bSrcDstStep, roiSize) );
EXIT_FUNC
    return (int)status; 
};

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
    Ipp8u * pBuffer = NULL;
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
