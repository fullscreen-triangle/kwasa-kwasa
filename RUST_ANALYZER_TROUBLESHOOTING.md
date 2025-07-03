# Rust Analyzer Troubleshooting Guide for Cursor/VS Code

## Common Issues and Solutions

### 1. Rust Analyzer Not Working at All

**Check Extension Installation:**
- Open Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux)
- Run: `Extensions: Show Installed Extensions`
- Verify `rust-lang.rust-analyzer` is installed and enabled

**Alternative Extensions to Try:**
If rust-analyzer doesn't work, try these alternatives:
- `rust-lang.rust` (official Rust extension)
- `kalitaalexey.vscode-rust` (community extension)
- `panicbit.cargo` (Cargo integration)

### 2. Rust Analyzer Server Issues

**Restart the Language Server:**
```
Cmd/Ctrl + Shift + P → "Rust Analyzer: Restart Server"
```

**Check Server Status:**
```
Cmd/Ctrl + Shift + P → "Rust Analyzer: Status"
```

**Manual Server Installation:**
```bash
# Install/update rust-analyzer manually
rustup component add rust-analyzer
# Or install via cargo
cargo install rust-analyzer
```

### 3. Project Not Recognized

**Reload Window:**
```
Cmd/Ctrl + Shift + P → "Developer: Reload Window"
```

**Check Cargo.toml:**
- Ensure `Cargo.toml` is in workspace root
- Verify workspace configuration is correct

**Clear Cache:**
```bash
# Clear rust-analyzer cache
rm -rf ~/.cache/rust-analyzer/
# Or on Windows
rmdir /s %USERPROFILE%\.cache\rust-analyzer\
```

### 4. Performance Issues

**Reduce Memory Usage:**
Add to `.vscode/settings.json`:
```json
{
    "rust-analyzer.cargo.loadOutDirsFromCheck": false,
    "rust-analyzer.procMacro.enable": false,
    "rust-analyzer.cargo.buildScripts.enable": false
}
```

**Exclude Large Directories:**
```json
{
    "rust-analyzer.files.excludeDirs": [
        "target",
        "node_modules",
        ".git"
    ]
}
```

### 5. Cursor-Specific Issues

**Check Cursor Compatibility:**
- Cursor is based on VS Code but may have compatibility issues
- Try these steps:

1. **Update Cursor:**
   - Ensure you're using the latest version of Cursor
   - Check for updates in Cursor settings

2. **Reset Extension Host:**
   ```
   Cmd/Ctrl + Shift + P → "Developer: Reset Extension Host"
   ```

3. **Disable Cursor AI temporarily:**
   - Sometimes AI features can interfere with language servers
   - Try disabling Cursor's AI features temporarily

### 6. Workspace Configuration

**Multi-Root Workspace Setup:**
If using workspace with multiple Rust projects:
```json
// workspace.code-workspace
{
    "folders": [
        {
            "path": "."
        },
        {
            "path": "./turbulance"
        }
    ],
    "settings": {
        "rust-analyzer.linkedProjects": [
            "./Cargo.toml",
            "./turbulance/Cargo.toml"
        ]
    }
}
```

### 7. Environment Issues

**Check Rust Installation:**
```bash
# Verify Rust is installed
rustc --version
cargo --version

# Check toolchain
rustup show

# Update if needed
rustup update
```

**Path Issues:**
Ensure Rust is in your PATH:
```bash
# Add to your shell profile (.zshrc, .bashrc, etc.)
export PATH="$HOME/.cargo/bin:$PATH"
```

### 8. Alternative Debugging Setup

**If LLDB doesn't work, try CodeLLDB:**
```json
// In .vscode/extensions.json
{
    "recommendations": [
        "vadimcn.vscode-lldb",
        "ms-vscode.cpptools"
    ]
}
```

**Native Debug Configuration:**
```json
// In .vscode/launch.json
{
    "name": "Debug with Native",
    "type": "cppdbg",
    "request": "launch",
    "program": "${workspaceFolder}/target/debug/kwasa-kwasa",
    "args": [],
    "stopAtEntry": false,
    "cwd": "${workspaceFolder}",
    "environment": [],
    "externalConsole": false,
    "MIMode": "lldb"
}
```

### 9. Complete Reset Procedure

If nothing works, try this complete reset:

1. **Close Cursor completely**

2. **Remove all Rust extensions:**
   ```
   Cmd/Ctrl + Shift + P → "Extensions: Show Installed Extensions"
   # Uninstall all Rust-related extensions
   ```

3. **Clear extension cache:**
   ```bash
   # macOS
   rm -rf ~/.vscode/extensions/rust-lang.*
   rm -rf ~/Library/Application\ Support/Cursor/User/workspaceStorage/

   # Windows
   rmdir /s %USERPROFILE%\.vscode\extensions\rust-lang.*
   rmdir /s %APPDATA%\Cursor\User\workspaceStorage\

   # Linux
   rm -rf ~/.vscode/extensions/rust-lang.*
   rm -rf ~/.config/Cursor/User/workspaceStorage/
   ```

4. **Restart Cursor and reinstall extensions**

### 10. Recommended Extension Combination

**Primary Setup:**
- `rust-lang.rust-analyzer` (main language server)
- `vadimcn.vscode-lldb` (debugging)
- `serayuzgur.crates` (dependency management)
- `tamasfe.even-better-toml` (Cargo.toml editing)

**Fallback Setup (if rust-analyzer fails):**
- `rust-lang.rust` (official Rust extension)
- `ms-vscode.cpptools` (debugging alternative)
- `serayuzgur.crates` (dependency management)

### 11. Cursor-Specific Settings

Add these to your Cursor settings:
```json
{
    "rust-analyzer.server.path": "rust-analyzer",
    "rust-analyzer.updates.channel": "stable",
    "rust-analyzer.cargo.allFeatures": true,
    "rust-analyzer.diagnostics.enable": true,
    "rust-analyzer.completion.addCallParentheses": false,
    "rust-analyzer.completion.addCallArgumentSnippets": false
}
```

### 12. Getting Help

**Check Logs:**
```
Cmd/Ctrl + Shift + P → "Developer: Show Logs..." → "Extension Host"
```

**Rust Analyzer Logs:**
```
Cmd/Ctrl + Shift + P → "Rust Analyzer: Show RA Logs"
```

**Report Issues:**
- [Rust Analyzer GitHub](https://github.com/rust-lang/rust-analyzer/issues)
- [Cursor Discord/Community](https://cursor.sh/community)

## Quick Fixes Checklist

- [ ] Extension installed and enabled
- [ ] Rust toolchain installed (`rustup show`)
- [ ] Cargo.toml exists in workspace root
- [ ] Restart language server
- [ ] Reload window
- [ ] Clear cache
- [ ] Check logs for errors
- [ ] Try alternative extensions
- [ ] Update Cursor to latest version

## Performance Optimization

For large projects like Kwasa-Kwasa:
```json
{
    "rust-analyzer.cargo.loadOutDirsFromCheck": true,
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.cargo.buildScripts.enable": true,
    "rust-analyzer.checkOnSave.command": "check",
    "rust-analyzer.checkOnSave.allTargets": false,
    "rust-analyzer.cargo.features": ["framework-core"]
}
```
