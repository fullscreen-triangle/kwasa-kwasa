# Kwasa-Kwasa Development Makefile
.PHONY: help build test clean format lint docs install dev release check all

# Default target
all: check test build

help: ## Show this help message
	@echo "Kwasa-Kwasa Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development commands
dev: ## Start development environment
	@echo "🚀 Starting Kwasa-Kwasa development environment..."
	cargo run --features="full"

install: ## Install dependencies and tools
	@echo "📦 Installing dependencies and development tools..."
	rustup component add rustfmt clippy rust-src rust-analyzer
	rustup target add wasm32-unknown-unknown
	cargo install cargo-watch cargo-audit cargo-outdated cargo-tree
	@echo "✅ Installation complete!"

# Build commands
build: ## Build the project in debug mode
	@echo "🔨 Building Kwasa-Kwasa..."
	cargo build

build-release: ## Build the project in release mode
	@echo "🔨 Building Kwasa-Kwasa (release)..."
	cargo build --release

build-wasm: ## Build WebAssembly target
	@echo "🌐 Building WebAssembly target..."
	cargo build --target wasm32-unknown-unknown

# Testing commands
test: ## Run all tests
	@echo "🧪 Running tests..."
	cargo test

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	cargo test -- --nocapture

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	cargo test --test '*'

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	cargo test --lib

test-docs: ## Test documentation examples
	@echo "📚 Testing documentation examples..."
	cargo test --doc

# Code quality commands
format: ## Format code using rustfmt
	@echo "🎨 Formatting code..."
	cargo fmt

format-check: ## Check if code is properly formatted
	@echo "🎨 Checking code formatting..."
	cargo fmt -- --check

lint: ## Run clippy linter
	@echo "📋 Running clippy linter..."
	cargo clippy --all-targets --all-features -- -D warnings

lint-fix: ## Run clippy with automatic fixes
	@echo "📋 Running clippy with fixes..."
	cargo clippy --fix --all-targets --all-features

check: ## Run cargo check
	@echo "✅ Running cargo check..."
	cargo check --all-targets --all-features

check-all: format-check lint check ## Run all checks

# Documentation commands
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	cargo doc --no-deps --all-features

docs-open: ## Generate and open documentation
	@echo "📚 Generating and opening documentation..."
	cargo doc --no-deps --all-features --open

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation on http://localhost:8000..."
	@cd target/doc && python3 -m http.server 8000

# Benchmark commands
bench: ## Run benchmarks
	@echo "⚡ Running benchmarks..."
	cargo bench

bench-baseline: ## Run benchmarks and save as baseline
	@echo "⚡ Running benchmarks (baseline)..."
	cargo bench -- --save-baseline main

# Security and maintenance
audit: ## Run security audit
	@echo "🔒 Running security audit..."
	cargo audit

outdated: ## Check for outdated dependencies
	@echo "📦 Checking for outdated dependencies..."
	cargo outdated

update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	cargo update

tree: ## Show dependency tree
	@echo "🌳 Showing dependency tree..."
	cargo tree

# Clean commands
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	cargo clean

clean-all: clean ## Clean everything including target directory
	@echo "🧹 Deep cleaning..."
	rm -rf target/
	rm -rf Cargo.lock

# Release commands
release: check-all test build-release ## Prepare for release
	@echo "🚀 Release build complete!"

release-dry-run: ## Dry run of cargo publish
	@echo "🚀 Running release dry run..."
	cargo publish --dry-run

publish: ## Publish to crates.io
	@echo "🚀 Publishing to crates.io..."
	cargo publish

# Development workflow
watch: ## Watch for changes and run tests
	@echo "👀 Watching for changes..."
	cargo watch -x "test"

watch-run: ## Watch for changes and run the program
	@echo "👀 Watching for changes and running..."
	cargo watch -x "run --features=full"

# Docker commands (if using Docker)
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t kwasa-kwasa .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm kwasa-kwasa

# Git workflow helpers
pre-commit: check-all test ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed!"

pre-push: pre-commit bench ## Run pre-push checks
	@echo "✅ Pre-push checks passed!"

# Performance profiling
profile: ## Run with profiling
	@echo "📊 Running with profiling..."
	cargo build --release
	perf record --call-graph=dwarf target/release/kwasa-kwasa
	perf report

# WebAssembly specific
wasm-pack: ## Build with wasm-pack
	@echo "📦 Building with wasm-pack..."
	wasm-pack build --target web --out-dir pkg

# Scientific computing specific
validate-examples: ## Validate all example files
	@echo "🔬 Validating scientific examples..."
	@for file in examples/*.turb; do \
		echo "Validating $$file..."; \
		cargo run -- validate "$$file" || exit 1; \
	done
	@echo "✅ All examples validated!"

# Help with common development tasks
setup-dev: install ## Complete development setup
	@echo "🛠️  Setting up development environment..."
	@make install
	@make check
	@echo "✅ Development environment ready!"

quick-test: ## Quick test suite for development
	@echo "⚡ Running quick tests..."
	cargo test --lib --bins

full-ci: check-all test bench audit ## Full CI pipeline
	@echo "🏗️  Running full CI pipeline..."
	@echo "✅ CI pipeline complete!" 