# Scikit-IPP

* filters (Gaussian)

Corrently it works for dtypes: uint8, uint16, int16, float32

### TODO
1) handle the ipp status check
2) delete numChannels from funcs signature
3) currently `gaussian` doesn't use multichannel, cval, truncate values (will fix)
4) integrate numpy conversions
5) test on scikit-image gausssian_test
6) multichannels

### issues:
1) Ipp has more than 6 types of borders, is it needed to use all of them?
