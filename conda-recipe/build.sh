if [ `uname` != Darwin ]; then
	# currently intel-openmp does not work for osx
    export USE_OPENMP=1
fi

LIBROOT=$PREFIX $PYTHON setup.py build install --old-and-unmanageable
