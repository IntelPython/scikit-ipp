#ifndef BORDERFILTER_H
#define BORDERFILTER_H
#include <stddef.h>
#include "ipp.h"
#include "dtypes.h"

int
ippiFilterBorder(
	IppDataTypeIndex ipp_src_dst_index,
	IppDataTypeIndex border_index,
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
	float ippBorderValue);
#endif