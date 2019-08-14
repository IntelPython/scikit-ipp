from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

if __name__ == '__main__':
    gaussian_module = Extension("Scikit_IPP_filters", 
                        ["_edges.pyx"], 
                        language="c",
                        include_dirs=[get_include(),"$IPPROOT/include"],
                        libraries = ["ippcv","ippcore", "ippvm", "ipps", "ippi"],
                        library_dirs = ["$IPPROOT/lib/intel64_lin"]
                        )
    setup (name = 'Scikit_IPP_filters',
           version = '1.0',
           ext_modules = cythonize([gaussian_module])
           )