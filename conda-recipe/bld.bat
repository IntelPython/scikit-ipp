@rem Remember to activate compiler, if needed

set LIBROOT=%PREFIX%
set USE_OPENMP=1
%PYTHON% setup.py build --force install --old-and-unmanageable bdist_wheel
if errorlevel 1 exit 1
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" copy dist\scikit-ipp*.whl %WHEELS_OUTPUT_FOLDER%
if errorlevel 1 exit 1
