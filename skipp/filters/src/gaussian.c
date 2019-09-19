#include "gaussian.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine 

int
GaussianFilterUINT8(void * pSRC,
                    void * pDST,
                    int img_width,
                    int img_height,
                    int numChannels,
                    float sigma_,
                    int kernelSize,
                    int stepSize,
                    int ippBorderType, 
                    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the
                                                         //source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp8u borderValue = (Ipp8u) ippBorderValue;
    //// border values i
    pSrc1 = (Ipp8u *) pSRC;
    pDst1 = (Ipp8u *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp8u, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_8u_C1R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterUINT8RGB(void * pSRC,
                       void * pDST,
                       int img_width,
                       int img_height,
                       int numChannels,
                       float sigma_,
                       int kernelSize,
                       int stepSize,
                       int ippBorderType,
                       float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images 
    int srcStep = stepSize, dstStep = stepSize;          //Steps, in bytes, through the 
                                                         // source/destination images 
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels 
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
    IppFilterGaussianSpec* pSpec = NULL;   // context structure
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp8u borderValue[] = {0, 0, 0};
    pSrc1 = (Ipp8u *) pSRC;
    pDst1 = (Ipp8u *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp8u, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_8u_C3R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterUINT16(void * pSRC,
                     void * pDST,
                     int img_width,
                     int img_height,
                     int numChannels,
                     float sigma_,
                     int kernelSize,
                     int stepSize,
                     int ippBorderType,
                     float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp16u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the 
                                                         // source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
    IppFilterGaussianSpec* pSpec = NULL;   // context structure
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp16u borderValue = (Ipp16u)ippBorderValue;
    pSrc1 = (Ipp16u *) pSRC;
    pDst1 = (Ipp16u *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp16u, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_16u_C1R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterINT16(void * pSRC,
                    void * pDST,
                    int img_width,
                    int img_height,
                    int numChannels,
                    float sigma_,
                    int kernelSize,
                    int stepSize,
                    int ippBorderType,
                    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp16s* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through 
                                                         // the source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
    IppiBorderType borderType = ippBorderType;
    Ipp16s borderValue = (Ipp16s)ippBorderValue;
    pSrc1 = (Ipp16s *) pSRC;
    pDst1 = (Ipp16s *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16s,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp16s, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_16s_C1R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterUINT16RGB(void * pSRC,
                        void * pDST,
                        int img_width,
                        int img_height,
                        int numChannels,
                        float sigma_,
                        int kernelSize,
                        int stepSize,
                        int ippBorderType,
                        float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp16u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the 
                                                         // source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp16u borderValue[] = {0, 0, 0};
    pSrc1 = (Ipp16u *) pSRC;
    pDst1 = (Ipp16u *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp16u, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_16u_C3R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterINT16RGB(void * pSRC,
                       void * pDST,
                       int img_width,
                       int img_height,
                       int numChannels,
                       float sigma_,
                       int kernelSize,
                       int stepSize,
                       int ippBorderType,
                       float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp16s* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the
                                                         // source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
    IppiBorderType borderType = ippBorderType;
    Ipp16s borderValue[] = {0, 0, 0};
    pSrc1 = (Ipp16s *) pSRC;
    pDst1 = (Ipp16s *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16s,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp16s, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_16s_C3R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterFLOAT32(void * pSRC,
                      void * pDST,
                      int img_width,
                      int img_height,
                      int numChannels,
                      float sigma_,
                      int kernelSize,
                      int stepSize,
                      int ippBorderType,
                      float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the 
                                                         //source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
    IppiBorderType borderType = ippBorderType;
    Ipp32f borderValue = (Ipp32f)ippBorderValue;
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp32f,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp32f, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_32f_C1R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilterFLOAT32RGB(void * pSRC,
                         void * pDST,
                         int img_width,
                         int img_height,
                         int numChannels,
                         float sigma_,
                         int kernelSize,
                         int stepSize,
                         int ippBorderType,
                         float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
    int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through 
                                                         //the source/destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
    IppiBorderType borderType = ippBorderType;
    Ipp32f borderValue[] = {0, 0, 0};
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    check_sts( status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp32f,
        numChannels, &iSpecSize, &iTmpBufSize) );
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if(NULL == pSpec)
        check_sts( status = ippStsMemAllocErr);
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if(NULL == pBuffer)
        check_sts( status = ippStsMemAllocErr);
    check_sts( status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        borderType, ipp32f, numChannels, pSpec, pBuffer) );
    check_sts( status = ippiFilterGaussianBorder_32f_C3R(pSrc1, srcStep, pDst1, dstStep,
        roiSize, borderValue, pSpec, pBuffer));

EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

static funcHandler
gaussianFilterTable[] = { GaussianFilterUINT8,
                          GaussianFilterUINT16,
                          GaussianFilterINT16,
                          GaussianFilterFLOAT32,
                          GaussianFilterUINT8RGB,
                          GaussianFilterUINT16RGB,
                          GaussianFilterINT16RGB,
                          GaussianFilterFLOAT32RGB,
                        };

int  
GaussianFilter(int index,
               void * pSRC,
               void * pDST,
               int img_width,
               int img_height,
               int numChannels,
               float sigma_,
               int kernelSize,
               int stepSize,
               int ippBorderType,
               float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    return status = gaussianFilterTable[index](pSRC,
                                               pDST,
                                               img_width,
                                               img_height,
                                               numChannels,
                                               sigma_,
                                               kernelSize,
                                               stepSize,
                                               ippBorderType,
                                               ippBorderValue);
}