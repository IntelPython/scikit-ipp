@rem Remember to activate compiler, if needed

set LIBROOT=%PREFIX%
set USE_OPENMP=1
%PYTHON% setup.py build --force install --old-and-unmanageable
if errorlevel 1 exit 1

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\scikit_ipp*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
