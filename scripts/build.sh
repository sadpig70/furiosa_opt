#!/bin/bash
# Build script for furiosa-optimizer
# Usage: ./scripts/build.sh [--release] [--python] [--test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
RELEASE=""
PYTHON=""
TEST=""

for arg in "$@"; do
    case $arg in
        --release)
            RELEASE="--release"
            ;;
        --python)
            PYTHON="1"
            ;;
        --test)
            TEST="1"
            ;;
    esac
done

echo "=========================================="
echo "Building furiosa-optimizer"
echo "=========================================="

# Check dependencies
echo "Checking dependencies..."
command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found"; exit 1; }
command -v protoc >/dev/null 2>&1 || { echo "Warning: protoc not found, prost-build may fail"; }

# Run tests if requested
if [ -n "$TEST" ]; then
    echo ""
    echo "Running tests..."
    cargo test --all-features
    
    echo ""
    echo "Running clippy..."
    cargo clippy --all-targets --all-features -- -D warnings
fi

# Build Rust library
echo ""
echo "Building Rust library..."
cargo build $RELEASE --all-features

# Build Python wheel if requested
if [ -n "$PYTHON" ]; then
    echo ""
    echo "Building Python wheel..."
    command -v maturin >/dev/null 2>&1 || pip install maturin
    
    if [ -n "$RELEASE" ]; then
        maturin build --release --features python
    else
        maturin develop --features python
    fi
    
    echo ""
    echo "Python wheel location:"
    ls -la target/wheels/*.whl 2>/dev/null || echo "(development install, no wheel)"
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
