import os

from os.path import join, split, dirname, abspath
from numpy import get_include as get_numpy_include
from distutils.sysconfig import get_python_inc as get_python_include


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

    ipp_root = os.environ['IPPROOT']

    ipp_include_dir = [join(ipp_root, 'include')]
    ipp_library_dirs = [join(ipp_root, 'lib')]
    ipp_libraries = ["ippcv", "ippcore", "ippvm", "ipps", "ippi"]

    _ipp_utils_dir = ['_ipp_utils']

    extension_names = []  # extension names and their dir names are the same
    extension_cy_src = {}

    extension_names.append('filters')
    # extension_names.append('morphology')
    # extension_names.append('transform')
    extension_sources, extension_includes = _get_sources_and_includes(extension_names +
                                                                      _ipp_utils_dir)
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
    include_dirs.extend(ipp_include_dir)
    include_dirs.extend(extension_includes)

    for extension_name in extension_names:
        config.add_extension(
            name=extension_name,
            sources=extension_sources +
                    [extension_cy_src[extension_name]],
            language="c",
            libraries=ipp_libraries,
            include_dirs=include_dirs,
            library_dirs=ipp_library_dirs)
    if have_cython:
        config.ext_modules = cythonize(config.ext_modules)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
