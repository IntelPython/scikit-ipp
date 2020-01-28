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

    extension_names = []
    extension_sources = {}
    extension_includes = {}
    include_path = []

    extension_names.append('filters')
    # extension_names.append('morphology')
    # extension_names.append('transform')

    for extension_name in extension_names:
        extension_dir = extension_name
        extension_src_dir = join(extension_dir, 'src')
        extension_includes[extension_name] = [extension_dir, extension_src_dir]
        include_path.extend(extension_includes[extension_name])
    try:
        from Cython.Build import cythonize
        for extension_name in extension_names:
            extension_source = join(extension_includes[extension_name][0],
                                    f'_{extension_name}.pyx')
            extension_sources[extension_name] = [extension_source]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = []
        for extension_name in extension_names:
            extension_source = join(extension_includes[extension_name][0],
                                    f'_{extension_name}.c')
            sources.append(extension_source)
            extension_sources[extension_name] = [extension_source]
        for source in sources:
            if not exists(source):
                raise ValueError(str(e) + '. ' +
                                 'Cython is required to build the initial .c file.')

    include_dirs = [get_numpy_include(), get_python_include()]
    include_dirs.extend(ipp_include_dir)

    for extension_name in extension_names:
        config.add_extension(
            name=extension_name,
            sources=extension_sources[extension_name],
            language="c",
            libraries=ipp_libraries,
            include_dirs=include_dirs,
            library_dirs=ipp_library_dirs)
    if have_cython:
        config.ext_modules = cythonize(config.ext_modules,
                                       include_path=include_path)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
