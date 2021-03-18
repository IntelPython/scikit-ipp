# ******************************************************************************
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************

import os
import platform

from os.path import join, split, dirname, abspath
from numpy import get_include as get_numpy_include
from distutils.sysconfig import get_python_inc as get_python_include


IS_WIN = platform.system() == 'Windows'
IS_LIN = platform.system() == 'Linux'
IS_MAC = platform.system() == 'Darwin'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    def _get_sources_and_includes(dirs_list):
        """
        Returns list of source files and list of includes
        Parameters
        ----------
        dirs_list : list:
            List of directories.
        Returns
        ----------
        sources : list
            List of source files.
        includes :list
            List of include dirs.
        Note:
        ----------
            All given `dirs_list` item (directory) should have
            `include` and `src` sub dirs
        """
        sources = []
        includes = []
        for directory in dirs_list:
            source_dir = join(os.path.dirname(os.path.realpath(__file__)),
                              directory, 'src')
            for source in os.listdir(source_dir):
                sources.append(join(source_dir, source))
            includes.append(join(directory, 'include'))
        return sources, includes

    config = Configuration('skipp', parent_package, top_path)

    lib_root = os.environ['LIBROOT']

    use_omp = True if 'USE_OPENMP' in os.environ else False

    include_dir = [join(lib_root, 'include')]
    library_dirs = [join(lib_root, 'lib')]
    libraries = ["ippcv", "ippcore", "ippvm", "ipps", "ippi"]

    _ipp_utils_dir = ['_ipp_utils']
    _ipp_wr_dir = ['_ipp_wr']

    if IS_LIN:
        extra_compile_args=['-fopenmp']
        extra_link_args=['-fopenmp']
    elif IS_WIN:
        extra_compile_args=['-openmp']
        extra_link_args=['-openmp']
    elif IS_MAC:
        extra_compile_args=['-fopenmp']
        extra_link_args=['-fopenmp=libiomp5']
    if IS_MAC:
       libraries.append("iomp5")
        
    extension_names = []  # extension names and their dir names are the same
    extension_cy_src = {}

    extension_names.append('filters')
    extension_names.append('morphology')
    extension_names.append('transform')
    extension_sources, extension_includes = _get_sources_and_includes(extension_names +
                                                                      _ipp_utils_dir +
                                                                      _ipp_wr_dir)
    try:
        from Cython.Build import cythonize
        for extension_name in extension_names:
            extension_dir = extension_name
            extension_cy_src[extension_name] = join(extension_dir, f'_{extension_name}.pyx')
        have_cython = True
    except ImportError as e:
        have_cython = False
        for extension_name in extension_names:
            extension_dir = extension_name
            extension_cy_src[extension_name] = join(extension_dir, f'_{extension_name}.c')
        for source in extension_cy_src.values():
            if not exists(source):
                raise ValueError(str(e) + '. ' +
                                 'Cython is required to build the initial .c file.')

    include_dirs = [get_numpy_include(), get_python_include()]
    include_dirs.extend(include_dir)
    include_dirs.extend(extension_includes)

    for extension_name in extension_names:
        config.add_extension(
            name=extension_name,
            sources=extension_sources +
                    [extension_cy_src[extension_name]],
            language="c",
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs)
    if have_cython:
        config.ext_modules = cythonize(config.ext_modules)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
