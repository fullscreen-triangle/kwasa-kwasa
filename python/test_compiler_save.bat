@echo off
echo ============================================================
echo Testing Turbulance Compiler (WITH OUTPUT SAVING)
echo ============================================================
echo.

echo Test 1: Points
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_point.trb --save-output
echo.
echo.

echo Test 2: Resolutions
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_resolution.trb --save-output
echo.
echo.

echo Test 3: BMDs
echo ----------------------------------------
python turbulance.py ../examples/turbulance/simple_bmd.trb --save-output
echo.
echo.

echo ============================================================
echo All tests complete! Check validation_outputs/ directory.
echo ============================================================

