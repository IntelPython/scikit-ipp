from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

if __name__ == '__main__':
	module1 = Extension("Scikit_IPP_filters", 
						["GaussianFilter_wrapper.pyx"], 
						language="c",
						include_dirs=[get_include(),"C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019\\windows\\ipp\\include"],
						libraries = ["ippcv","ippcore", "ippvm", "ipps", "ippi"],
						library_dirs = ["C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019\\windows\\ipp\\lib\\intel64_win"]
						)

	setup (name = 'Scikit_IPP_filters',
		   version = '1.0',
		   ext_modules = cythonize([module1])
		   )