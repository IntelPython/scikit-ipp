************
Installation
************

Building scikit-ipp using conda-build
=====================================
The easiest way to build `scikit-ipp` is using the conda-build with the provided recipe.

Prerequisites
-------------
* Python version >= 3.6
* conda-build version >= 3
* C compiler


Building scikit-ipp
-------------------

.. code:: console

    $ cd <checkout-dir>
    $ conda build -c intel conda-recipe

This will build the conda package and tell you where to find it (```.../scikit-ipp*.tar.bz2```).


Installing the built scikit-ipp conda package
---------------------------------------------

.. code:: console

    $ conda install <path-to-conda-package-as-built-above>

To actually use your `scikit-ipp`, dependent packages need to be installed. To ensure, do

Linux or Windows:

.. code:: console

    $ conda install -c intel numpy ipp

Building documentation for scikit-ipp
=====================================
Prerequisites for creating documentation
----------------------------------------
* sphinx >= 3.0
* sphinx_rtd_theme >= 0.4
* sphinx-gallery >= 0.3.1
* matplotlib > = 3.0.1

Building documentation
----------------------
1. Install scikit-ipp into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```
