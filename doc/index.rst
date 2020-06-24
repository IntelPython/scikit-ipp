.. scikit-ipp documentation master file, created by
   sphinx-quickstart on Mon Apr 13 05:56:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-IPP (skipp)
==================

`scikit-ipp` is optimization of open-source image processing library `scikit-image <https://scikit-image.org/>`_ by using Intel® Integrated Performance Primitives (Intel® IPP) library.

`scikit-ipp` is a standalone package, provided scikit-image-like API to some of Intel® IPP functions.

- `Documentation <https://github.intel.com/pages/SAT/scikit-ipp/>`_
- `Source Code <https://github.intel.com/SAT/scikit-ipp>`_
- `About Intel® IPP <https://software.intel.com/en-us/intel-ipp>`_

`scikit-ipp` is easily built from source with the majority of the necessary prerequisites available on conda.  The instructions below detail how to gather the prerequisites, setting one's build environment, and finally building and installing the completed package.  `scikit-ipp` can be built for all three major platforms (Windows, Linux, macOS).

The build-process (using setup.py) happens in 2 stages:

1. Running cython on C and Cython sources
2. Compiling and linking

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
