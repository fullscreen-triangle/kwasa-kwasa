@echo off
echo ============================================================
echo Testing Turbulance Compiler
echo ============================================================
echo.

echo Test 1: Points
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_point.trb
echo.
echo.

echo Test 2: Resolutions
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_resolution.trb
echo.
echo.

echo Test 3: BMDs
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_bmd.trb
echo.
echo.

echo ============================================================
echo All tests complete!
echo ============================================================

