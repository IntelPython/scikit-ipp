from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

if __name__ == '__main__':
    module1 = Extension("Scikit_IPP_filters",
                        ["GaussianFilter_wrapper.pyx"],
                        language="c",
                        include_dirs=[get_include(), "$IPPROOT/include"],
                        libraries=["ippcv",
                                   "ippcore",
                                   "ippvm",
                                   "ipps",
                                   "ippi"],
                        library_dirs=["$IPPROOT/intel64_win"]
                        )

    setup(name='Scikit_IPP_filters',
          version='1.0',
          ext_modules=cythonize([module1])
          )
