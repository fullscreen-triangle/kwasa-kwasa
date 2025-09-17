@echo off
echo 🔧 Installing Kwasa-Kwasa VSCode Extensions...

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    exit /b 1
)

:: Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed. Please install npm first.
    exit /b 1
)

:: Check if vsce is installed, if not install it
vsce --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing vsce Visual Studio Code Extension manager...
    npm install -g vsce
)

:: Check if code command is available
code --version >nul 2>&1
if errorlevel 1 (
    echo ❌ VSCode 'code' command not found. Please install VSCode and ensure 'code' is in PATH.
    exit /b 1
)

echo 🛠️  Building and installing extensions...

:: Extensions to install
set extensions=turbulance-language-server knowledge-resolution-engine pattern-evidence-workbench metacognitive-orchestration-debugger boundary-detection-studio

for %%e in (%extensions%) do (
    if exist "%%e" (
        echo 📁 Processing %%e...

        cd "%%e"

        echo    📦 Installing dependencies...
        call npm install

        echo    🔨 Compiling TypeScript...
        call npm run compile

        echo    📦 Packaging extension...
        call vsce package --out ../%%e.vsix

        echo    ✅ Installing extension...
        call code --install-extension ../%%e.vsix

        cd ..

        echo    ✨ %%e installed successfully!
        echo.
    ) else (
        echo    ⚠️  Directory %%e not found, skipping...
    )
)

echo 🎉 All extensions installed successfully!
echo.
echo 🔄 Please restart VSCode to activate all extensions.
echo.
echo 📖 Next steps:
echo    1. Create a new .turb file
echo    2. Write some Turbulance code
echo    3. Use Ctrl+Shift+P and search for 'Turbulance' or 'Knowledge Engine' commands
echo    4. Check the Activity Bar for new icons Knowledge Engine, Pattern Workbench, etc.

pause
