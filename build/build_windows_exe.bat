@echo off
setlocal

REM Build on Windows in a clean virtual environment.
REM Put this BAT file in the same folder as:
REM   enzyme_kinetics_gui_unified_v6.py
REM   enzyme_kinetics_gui_unified_v6.spec

echo Creating virtual environment...
py -m venv .venv
call .venv\Scripts\activate.bat

echo Upgrading pip...
py -m pip install --upgrade pip

echo Installing dependencies...
py -m pip install pyinstaller pandas numpy matplotlib scipy openpyxl

echo Building executable...
py -m PyInstaller --clean --noconfirm enzyme_kinetics_gui_unified_v6.spec

echo.
echo Build complete.
echo Your Windows app is in:
echo   dist\EnzymeKineticsAnalyzer\
echo.
echo Recommended distribution:
echo   Zip the whole EnzymeKineticsAnalyzer folder and give that to students.
echo.
pause
