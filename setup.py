#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import io
import os
import re
import sys

with io.open('skipp/_version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


VERSION = version

CLASSIFIERS = ""


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('skipp', parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False)

    config.add_subpackage('skipp')

    config.version = VERSION

    return config


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    from setuptools import setup, find_packages
    from numpy.distutils.core import setup

    metadata = dict(
        name='skipp',
        install_requires=['numpy'],
        packages=find_packages(),
        configuration=configuration
        )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return None


if __name__ == '__main__':
    setup_package()
