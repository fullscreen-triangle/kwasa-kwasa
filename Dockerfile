# Multi-stage Dockerfile for Kwasa-Kwasa
# Stage 1: Build environment
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY turbulance/Cargo.toml ./turbulance/

# Create dummy source files to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN mkdir turbulance/src && echo "" > turbulance/src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release && rm -rf src turbulance/src

# Copy actual source code
COPY . .

# Build the application
RUN cargo build --release --features="framework-core,autobahn-reasoning"

# Stage 2: Runtime environment
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false kwasa

# Create application directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/kwasa-kwasa /usr/local/bin/kwasa-kwasa

# Copy configuration files
COPY --from=builder /app/docs/ /app/docs/
COPY --from=builder /app/examples/ /app/examples/

# Set ownership
RUN chown -R kwasa:kwasa /app

# Switch to non-root user
USER kwasa

# Expose port (if web interface is added)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD kwasa-kwasa --version || exit 1

# Default command
CMD ["kwasa-kwasa", "--help"]

# Labels for metadata
LABEL maintainer="Kundai <kundai@example.com>"
LABEL description="Kwasa-Kwasa: Metacognitive text processing framework"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/fullscreen-triangle/kwasa-kwasa" 