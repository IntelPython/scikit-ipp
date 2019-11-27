from __future__ import division, print_function, absolute_import
import io
import re
import os

from os.path import join, split, dirname, abspath
from numpy import get_include as get_numpy_include
from distutils.sysconfig import get_python_inc as get_python_include


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('skipp', parent_package, top_path)

    ipp_root = os.environ['IPPROOT']

    ipp_include_dir = [join(ipp_root, 'include')]
    ipp_library_dirs = [join(ipp_root, 'lib')]
    ipp_libraries = ["ippcv", "ippcore", "ippvm", "ipps", "ippi"]

    filters_dir = 'filters'
    filters_dir_w = join(filters_dir, 'src')

    try:
        from Cython.Build import cythonize
        sources = [join(filters_dir, '_filters.pyx')]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(filters_dir, '_filters.c')]
        if not exists(sources[0]):
            raise ValueError(str(e) + '. ' +
                             'Cython is required to build the initial .c file.')

    include_dirs = [get_numpy_include(), get_python_include()]
    include_dirs.extend(ipp_include_dir)

    config.add_extension(
        name='filters',
        sources=sources,
        language="c",
        libraries=ipp_libraries,
        include_dirs=include_dirs,
        library_dirs=ipp_library_dirs
    )

    if have_cython:
        config.ext_modules = cythonize(config.ext_modules,
                                       include_path=[filters_dir, filters_dir_w])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
