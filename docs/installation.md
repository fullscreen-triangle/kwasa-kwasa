# Kwasa-Kwasa Installation Guide

This document provides comprehensive instructions for installing, building, and running the Kwasa-Kwasa metacognitive text processing framework.

## Prerequisites

Before installing Kwasa-Kwasa, ensure you have the following dependencies:

1. **Rust (1.65.0+)** - The core programming language
   ```bash
   # Check if Rust is installed
   rustc --version
   
   # If not installed, use rustup (recommended)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   # Or visit https://www.rust-lang.org/tools/install for other options
   ```

2. **Cargo** - Rust's package manager (comes with Rust installation)
   ```bash
   # Verify Cargo installation
   cargo --version
   ```

3. **SQLite3** - Required for the knowledge database component
   ```bash
   # On Ubuntu/Debian
   sudo apt install sqlite3 libsqlite3-dev
   
   # On macOS with Homebrew
   brew install sqlite
   
   # On Windows
   # Download from https://www.sqlite.org/download.html
   ```

4. **Git** - For cloning the repository
   ```bash
   # Install Git if needed
   # Ubuntu/Debian
   sudo apt install git
   
   # macOS
   brew install git
   
   # Windows
   # Download from https://git-scm.com/download/win
   ```

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/kwasa-kwasa.git
   cd kwasa-kwasa
   ```

2. **Configure environment variables**
   
   Create a `.env` file in the project root with the following content:
   ```
   KWASA_LOG_LEVEL=debug
   KWASA_DB_PATH=./data/knowledge.db
   KWASA_CACHE_DIR=./data/cache
   ```

3. **Build the project**
   ```bash
   # Build in debug mode
   cargo build
   
   # Or build with optimizations for better performance
   cargo build --release
   ```

4. **Run tests to verify installation**
   ```bash
   cargo test
   ```

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
