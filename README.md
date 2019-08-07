# Scikit-IPP

* filters (Gaussian)

### TODO
1) fix bug for RGB images [case: output is None]
2) handle the ipp status check
3) delete numChannels from funcs signature
4) currently `gaussian` doesn't use multichannel, cval, truncate values (will fix)
5) integrate numpy conversions
6) test on scikit-image gausssian_test
7) multichannels
8) code style

### issues:
1) Ipp has more than 6 types of borders, is it needed to use all of them?
2) conversions in-place or not?
