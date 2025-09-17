#!/bin/bash

# Install script for Kwasa-Kwasa VSCode Extensions
set -e

echo "🔧 Installing Kwasa-Kwasa VSCode Extensions..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if vsce is installed, if not install it
if ! command -v vsce &> /dev/null; then
    echo "📦 Installing vsce (Visual Studio Code Extension manager)..."
    npm install -g vsce
fi

# Check if code command is available
if ! command -v code &> /dev/null; then
    echo "❌ VSCode 'code' command not found. Please install VSCode and ensure 'code' is in PATH."
    exit 1
fi

# Array of extension directories
extensions=(
    "turbulance-language-server"
    "knowledge-resolution-engine"
    "pattern-evidence-workbench"
    "metacognitive-orchestration-debugger"
    "boundary-detection-studio"
)

echo "🛠️  Building and installing extensions..."

# Install dependencies and build each extension
for ext in "${extensions[@]}"; do
    if [ -d "$ext" ]; then
        echo "📁 Processing $ext..."

        cd "$ext"

        # Install dependencies
        echo "   📦 Installing dependencies..."
        npm install

        # Compile TypeScript
        echo "   🔨 Compiling TypeScript..."
        npm run compile

        # Package extension
        echo "   📦 Packaging extension..."
        vsce package --out ../"${ext}".vsix

        # Install extension
        echo "   ✅ Installing extension..."
        code --install-extension ../"${ext}".vsix

        cd ..

        echo "   ✨ $ext installed successfully!"
        echo
    else
        echo "   ⚠️  Directory $ext not found, skipping..."
    fi
done

echo "🎉 All extensions installed successfully!"
echo
echo "🔄 Please restart VSCode to activate all extensions."
echo
echo "📖 Next steps:"
echo "   1. Create a new .turb file"
echo "   2. Write some Turbulance code"
echo "   3. Use Ctrl+Shift+P and search for 'Turbulance' or 'Knowledge Engine' commands"
echo "   4. Check the Activity Bar for new icons (Knowledge Engine, Pattern Workbench, etc.)"
