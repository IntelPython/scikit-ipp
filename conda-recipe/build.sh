#!/bin/bash
set -ex

if [ `uname` != Darwin ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
	# currently intel-openmp does not work for osx
    export USE_OPENMP=1
else
    WHEELS_BUILD_ARGS=""
    export LDFLAGS="-headerpad_max_install_names $LDFLAGS"
fi

export LIBROOT=$PREFIX
$PYTHON setup.py build install --old-and-unmanageable bdist_wheel ${WHEELS_BUILD_ARGS}
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    cp dist/scikit-ipp*.whl ${WHEELS_OUTPUT_FOLDER}
fi
