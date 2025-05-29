# Installation Guide

## Prerequisites

Before installing Kwasa-Kwasa, ensure you have the following prerequisites:

- Rust (latest stable version)
- Cargo (comes with Rust)
- Git

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kwasa-kwasa.git
   cd kwasa-kwasa
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run the tests to ensure everything is working:
   ```bash
   cargo test
   ```

## Using as a Dependency

To use Kwasa-Kwasa in your own Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
kwasa-kwasa = { git = "https://github.com/yourusername/kwasa-kwasa" }
```

## Development Setup

For development, you might want to install additional tools:

1. Install recommended VS Code extensions (if using VS Code)
2. Set up pre-commit hooks:
   ```bash
   git config core.hooksPath .github/hooks
   ```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have the latest stable Rust version:
   ```bash
   rustup update stable
   ```

2. Clear your Cargo cache if needed:
   ```bash
   cargo clean
   ```

3. Check the [Issue Tracker](https://github.com/yourusername/kwasa-kwasa/issues) for known problems

For additional help, please [open an issue](https://github.com/yourusername/kwasa-kwasa/issues/new).

## Running Kwasa-Kwasa

### Using the CLI

The Kwasa-Kwasa CLI provides several commands for working with Turbulance:

1. **Run a Turbulance script**
   ```bash
   # Using debug build
   cargo run -- run path/to/script.turb
   
   # Or using release build
   ./target/release/kwasa-kwasa run path/to/script.turb
   ```

2. **Validate a script without executing**
   ```bash
   cargo run -- validate path/to/script.turb
   ```

3. **Process a document with embedded Turbulance**
   ```bash
   cargo run -- process path/to/document.md
   
   # Interactive mode
   cargo run -- process path/to/document.md --interactive
   ```

4. **Start the interactive REPL**
   ```bash
   cargo run -- repl
   ```

### Installing Globally

To make the `kwasa-kwasa` command available system-wide:

```bash
cargo install --path .
```

After this, you can run commands directly:

```bash
kwasa-kwasa repl
kwasa-kwasa run path/to/script.turb
```

## Directory Structure

The Kwasa-Kwasa codebase is organized as follows:
