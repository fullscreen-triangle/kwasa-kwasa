#!/bin/bash

echo "🚨 COMPREHENSIVE TURBULANCE SYNTAX FIX 🚨"
echo "Fixing all syntax errors across the entire codebase..."

# Fix all .turb files: var -> item
echo "1. Fixing .turb files: var -> item"
find . -name "*.turb" -exec sed -i '' 's/\bvar \([a-zA-Z_][a-zA-Z0-9_]*\) =/item \1 =/g' {} \;

# Fix all .rs files: incorrect Turbulance syntax in comments/strings
echo "2. Fixing .rs files: if -> given in Turbulance examples"
find . -name "*.rs" -exec sed -i '' 's/if sentence contains/given sentence contains/g' {} \;
find . -name "*.rs" -exec sed -i '' 's/if resolution is within/given resolution is within/g' {} \;

# Fix all .md files: var -> item in code blocks
echo "3. Fixing .md files: var -> item in Turbulance code blocks"
find . -name "*.md" -exec sed -i '' '/```turbulance/,/```/ s/\bvar \([a-zA-Z_][a-zA-Z0-9_]*\) =/item \1 =/g' {} \;

echo "✅ Syntax fix complete!"
echo "Files processed:"
find . -name "*.turb" -o -name "*.rs" -o -name "*.md" | wc -l 