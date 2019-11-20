#include "borderfilter.h"

#define EXIT_LINE exitLine:                // Label for Exit
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

int
ippiFilterBorder(
	IppDataTypeIndex ipp_src_dst_index,
	IppDataTypeIndex border_dtype_index,
	void * pSrc,
	void * pDst,
	void * pKernel,
	int img_width,
	int img_height,
	int kernel_width,
	int kernel_height,
	int numChannels,
	IppRoundMode roundMode,
	IppiBorderType ippBorderType,
	float ippBorderValue)
{
	IppStatus status = ippStsNoErr;
	IppiSize roiSize = { img_width, img_height }; // Size of source and
												  // destination ROI in pixels
	IppiSize  kernelSize = { kernel_width, kernel_height };

	Ipp8u * pBuffer = NULL;                 // Pointer to the work buffer
	IppiFilterBorderSpec * pSpec = NULL;    // context structure
	int iTmpBufSize = 0;                    // Common work buffer size
	int iSpecSize = 0;                      // Common work buffer size
	int srcStep;                            // Steps, in bytes, through the
	int dstStep;
	int sizeof_src_dst;
	IppDataType ippDataType;

	status = sizeof_ipp_dtype_by_index(&sizeof_src_dst, ipp_src_dst_index);
	check_sts(status);

	srcStep = numChannels * img_width * sizeof_src_dst;
	dstStep = srcStep;

	status = ipp_type_index_as_IppDataType(&ippDataType, ipp_src_dst_index);
	check_sts(status);

	status = ippiFilterBorderGetSize(kernelSize, roiSize, ippDataType, ippDataType,
		                             numChannels, &iSpecSize, &iTmpBufSize);
	check_sts(status);
	pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
	if (NULL == pSpec)
	{
		check_sts(status = ippStsMemAllocErr);
	};
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	if (NULL == pBuffer)
	{
		check_sts(status = ippStsMemAllocErr);
	};

	switch (border_dtype_index)
	{
	case ipp16s_index:
	{
		status = ippiFilterBorderInit_16s(pKernel, kernelSize, 4, ippDataType, numChannels, roundMode, pSpec);
		break;
	}
	case ipp32f_index:
	{
		status = ippiFilterBorderInit_32f(pKernel, kernelSize, ippDataType, numChannels,roundMode, pSpec);
		break;
	}
	case ipp64f_index:
	{
		status = ippiFilterBorderInit_64f(pKernel, kernelSize, ippDataType, numChannels, roundMode, pSpec);
		break;
	}
	default:
	{
		status = ippStsErr;
	}
	}
	if (numChannels == 1)
	{
		switch (ipp_src_dst_index)
		{
		case ipp8u_index:
		{
			Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
			status = ippiFilterBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				                             &ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp16u_index:
		{
			Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
			status = ippiFilterBorder_16u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				&ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp16s_index:
		{
			Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
			status = ippiFilterBorder_16s_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				&ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp32f_index:
		{
			Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
			status = ippiFilterBorder_32f_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				&ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp64f_index:
		{
			Ipp64f ippbordervalue = (Ipp64f)ippBorderValue;
			status = ippiFilterBorder_64f_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				&ippbordervalue, pSpec, pBuffer);
			break;
		}
		default:
		{
			status = ippStsErr;
		}
		}
	}
	else if (numChannels == 3)
	{
		switch (ipp_src_dst_index)
		{
		case ipp8u_index:
		{
			Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
			status = ippiFilterBorder_8u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp16u_index:
		{
			Ipp16u ippbordervalue[3] = { (Ipp16u)ippBorderValue, (Ipp16u)ippBorderValue , (Ipp16u)ippBorderValue };
			status = ippiFilterBorder_16u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp16s_index:
		{
			Ipp16s ippbordervalue[3] = { (Ipp16s)ippBorderValue, (Ipp16s)ippBorderValue , (Ipp16s)ippBorderValue };
			status = ippiFilterBorder_16s_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				ippbordervalue, pSpec, pBuffer);
			break;
		}
		case ipp32f_index:
		{
			Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
			status = ippiFilterBorder_32f_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
				ippbordervalue, pSpec, pBuffer);
			break;
		}
		default:
		{
			status = ippStsErr;
		}
		}
	}
	else
	{
		status = ippStsErr;
	}
	check_sts(status);
EXIT_LINE
	ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}
