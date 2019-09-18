from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include
import os
import os.path

ipp_root = os.environ.get('IPPROOT', None)
if ipp_root is None or not os.path.isdir(ipp_root):
    raise ValueError("Set IPPROOT environment variable to point to Intel(R) Performance Primitive installation directory.")

if __name__ == '__main__':
    gaussian_module = Extension("Scikit_IPP_filters", 
                        ["_edges.pyx"], 
                        language="c",
                        include_dirs=[get_include(), os.path.join([ipp_root, "include"])],
                        libraries = ["ippcv","ippcore", "ippvm", "ipps", "ippi"],
                        library_dirs = [os.path.join([ipp_root, 'lib'])]
                        )
    setup (name = 'Scikit_IPP_filters',
           version = '1.0',
           ext_modules = cythonize([gaussian_module])
           )
