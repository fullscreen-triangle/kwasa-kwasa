#!/bin/bash

# Rust Analyzer Fix Script for Cursor/VS Code
# This script helps diagnose and fix common rust-analyzer issues

set -e

echo "🔧 Rust Analyzer Diagnostic and Fix Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Rust installation
print_status "Checking Rust installation..."
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    print_success "Rust is installed: $RUST_VERSION"
else
    print_error "Rust is not installed. Please install Rust first."
    exit 1
fi

# Check Cargo
if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version)
    print_success "Cargo is available: $CARGO_VERSION"
else
    print_error "Cargo is not available."
    exit 1
fi

# Check rust-analyzer
print_status "Checking rust-analyzer installation..."
if command -v rust-analyzer &> /dev/null; then
    RA_VERSION=$(rust-analyzer --version)
    print_success "rust-analyzer is installed: $RA_VERSION"
else
    print_warning "rust-analyzer not found in PATH. Installing..."
    rustup component add rust-analyzer
    if command -v rust-analyzer &> /dev/null; then
        print_success "rust-analyzer installed successfully"
    else
        print_error "Failed to install rust-analyzer"
        exit 1
    fi
fi

# Check if we're in a Rust project
print_status "Checking project structure..."
if [ -f "Cargo.toml" ]; then
    print_success "Found Cargo.toml - this is a Rust project"
else
    print_error "No Cargo.toml found. Are you in a Rust project directory?"
    exit 1
fi

# Check if project builds
print_status "Testing project build..."
if cargo check --quiet; then
    print_success "Project builds successfully"
else
    print_warning "Project has build errors. rust-analyzer may not work properly until these are fixed."
fi

# Clear rust-analyzer cache
print_status "Clearing rust-analyzer cache..."
if [ -d "$HOME/.cache/rust-analyzer" ]; then
    rm -rf "$HOME/.cache/rust-analyzer"
    print_success "Cleared rust-analyzer cache"
else
    print_status "No rust-analyzer cache found"
fi

# Check VS Code/Cursor settings
print_status "Checking VS Code/Cursor configuration..."
if [ -f ".vscode/settings.json" ]; then
    print_success "Found .vscode/settings.json"

    # Check if rust-analyzer is configured
    if grep -q "rust-analyzer" ".vscode/settings.json"; then
        print_success "rust-analyzer is configured in settings"
    else
        print_warning "rust-analyzer not found in VS Code settings"
    fi
else
    print_warning "No .vscode/settings.json found"
fi

# Check workspace configuration
if [ -f "kwasa-kwasa.code-workspace" ]; then
    print_success "Found workspace configuration file"
else
    print_status "No workspace file found (this is optional)"
fi

# Suggest fixes
echo ""
echo "🔍 Troubleshooting Steps for Cursor:"
echo "====================================="
echo "1. Restart Cursor completely"
echo "2. Open Command Palette (Cmd+Shift+P) and run:"
echo "   - 'Rust Analyzer: Restart Server'"
echo "   - 'Developer: Reload Window'"
echo "3. Check if rust-analyzer extension is installed and enabled"
echo "4. Try opening the workspace file: kwasa-kwasa.code-workspace"
echo ""

# Extension recommendations
echo "📦 Recommended Extensions:"
echo "========================="
echo "Primary: rust-lang.rust-analyzer"
echo "Fallback: rust-lang.rust (if rust-analyzer doesn't work)"
echo "Debugging: vadimcn.vscode-lldb"
echo "Crates: serayuzgur.crates"
echo "TOML: tamasfe.even-better-toml"
echo ""

# Cursor-specific advice
echo "🎯 Cursor-Specific Tips:"
echo "======================="
echo "1. Update Cursor to the latest version"
echo "2. Try disabling Cursor's AI features temporarily"
echo "3. Use 'Developer: Reset Extension Host' if needed"
echo "4. Check Extension Host logs for errors"
echo ""

# Performance tips
echo "⚡ Performance Optimization:"
echo "=========================="
echo "- Use 'framework-core' feature instead of 'full'"
echo "- Disable unused rust-analyzer features"
echo "- Exclude large directories from file watching"
echo ""

print_success "Diagnostic complete! Check the troubleshooting guide for more help."
