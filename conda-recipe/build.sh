#!/bin/bash
set -ex

if [ `uname` != Darwin ]; then
    if [ "$CONDA_PY" == "36" ]; then
        WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
    else
        WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
    fi
	# currently intel-openmp does not work for osx
    export USE_OPENMP=1
else
    WHEELS_BUILD_ARGS=""
    export LDFLAGS="-headerpad_max_install_names $LDFLAGS"
fi

export LIBROOT=$PREFIX
$PYTHON setup.py build install --old-and-unmanageable

# Build wheel package
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/scikit_ipp*.whl ${WHEELS_OUTPUT_FOLDER}
fi
