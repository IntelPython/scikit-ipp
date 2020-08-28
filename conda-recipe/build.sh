if [ `uname` != Darwin ]; then
	# currently intel-openmp does not work for osx
    export USE_OPENMP=1
else
    export LDFLAGS="-headerpad_max_install_names $LDFLAGS"
fi

LIBROOT=$PREFIX $PYTHON setup.py build install --old-and-unmanageable
