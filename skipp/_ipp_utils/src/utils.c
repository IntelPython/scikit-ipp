#include "utils.h"

IppStatus
get_sizeof(
    IppDataType datatype,
    int * sizeof_datatype)
{
    IppStatus status = ippStsNoErr;
    switch (datatype)
    {
    case ipp1u:
    {
        *sizeof_datatype = sizeof(Ipp8u);
        break;
    }
    case ipp8u:
    {
        *sizeof_datatype = sizeof(Ipp8u);
        break;
    }
    case ipp8s:
    {
        *sizeof_datatype = sizeof(Ipp8s);
        break;
    }
    case ipp8sc:
    {
        *sizeof_datatype = sizeof(Ipp8sc);
        break;
    }
    case ipp16u:
    {
        *sizeof_datatype = sizeof(Ipp16u);
        break;
    }
    case ipp16uc:
    {
        *sizeof_datatype = sizeof(Ipp16uc);
        break;
    }
    case ipp16s:
    {
        *sizeof_datatype = sizeof(Ipp16s);
        break;
    }
    case ipp16sc:
    {
        *sizeof_datatype = sizeof(Ipp16sc);
        break;
    }
    case ipp32u:
    {
        *sizeof_datatype = sizeof(Ipp32u);
        break;
    }
    case ipp32s:
    {
        *sizeof_datatype = sizeof(Ipp32s);
        break;
    }
    case ipp32sc:
    {
        *sizeof_datatype = sizeof(Ipp32sc);
        break;
    }
    case ipp32f:
    {
        *sizeof_datatype = sizeof(Ipp32f);
        break;
    }
    case ipp32fc:
    {
        *sizeof_datatype = sizeof(Ipp32fc);
        break;
    }
    case ipp64u:
    {
        *sizeof_datatype = sizeof(Ipp64u);
        break;
    }
    case ipp64s:
    {
        *sizeof_datatype = sizeof(Ipp64s);
        break;
    }
    case ipp64sc:
    {
        *sizeof_datatype = sizeof(Ipp64sc);
        break;
    }
    case ipp64f:
    {
        *sizeof_datatype = sizeof(Ipp64f);
        break;
    }
    case ipp64fc:
    {
        *sizeof_datatype = sizeof(Ipp64fc);
        break;
    }
    default:
    {
        status = ippStsDataTypeErr;
    }
    }
    return status;
}
