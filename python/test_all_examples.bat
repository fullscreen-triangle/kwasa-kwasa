@echo off
echo ============================================================
echo Testing ALL Turbulance Examples (WITH OUTPUT SAVING)
echo ============================================================
echo.

echo ============================================================
echo BASIC EXAMPLES (Simple Demonstrations)
echo ============================================================
echo.

echo [1/6] Points (Basic)
python turbulance.py ../examples/turbulance/simple_point.trb --save-output
echo.
pause

echo [2/6] Resolutions (Basic)
python turbulance.py ../examples/turbulance/simple_resolution.trb --save-output
echo.
pause

echo [3/6] BMDs (Basic)
python turbulance.py ../examples/turbulance/simple_bmd.trb --save-output
echo.
pause

echo ============================================================
echo ADVANCED EXAMPLES (Detailed Demonstrations)
echo ============================================================
echo.

echo [4/6] Points (Advanced - Full Uncertainty Propagation)
python turbulance.py ../examples/turbulance/01_point_demo_executable.trb --save-output
echo.
pause

echo [5/6] Resolutions (Advanced - Full Bayesian Integration)
python turbulance.py ../examples/turbulance/02_resolution_demo_executable.trb --save-output
echo.
pause

echo [6/6] BMDs (Advanced - Full Frame Selection)
python turbulance.py ../examples/turbulance/03_bmd_demo_executable.trb --save-output
echo.
pause

echo ============================================================
echo All tests complete!
echo ============================================================
echo.
echo Check validation_outputs/ directory for saved outputs.
echo.
echo Files created:
dir validation_outputs\*_latest.txt /B
echo.
pause

