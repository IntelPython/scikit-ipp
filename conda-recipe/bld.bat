@rem Remember to activate compiler, if needed

set LIBROOT=%PREFIX%
set USE_OPENMP=1
%PYTHON% setup.py build --force install --old-and-unmanageable
if errorlevel 1 exit 1
