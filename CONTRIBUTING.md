# Contributing to Furiosa Optimizer

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Rust 1.70+
- Python 3.8+
- protoc (Protocol Buffers compiler)

### Ubuntu/Debian

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc
sudo apt-get install -y protobuf-compiler

# Install Python dependencies
pip install maturin pytest onnx
```

### macOS

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc
brew install protobuf

# Install Python dependencies
pip install maturin pytest onnx
```

## Building

```bash
# Build Rust library
cargo build --release

# Run tests
cargo test

# Build Python wheel
maturin build --release --features python

# Development install (editable)
maturin develop --features python
```

## Code Style

- Rust: Follow `rustfmt` formatting
- Run `cargo clippy` before committing
- Add tests for new functionality

```bash
# Format code
cargo fmt

# Check lints
cargo clippy --all-targets --all-features -- -D warnings
```

## Adding a New Transformer

1. Create file in `src/transformers/`
2. Implement `OnnxTransformer` trait
3. Add to `mod.rs` exports
4. Add tests
5. Update `OptimizationPipeline` if applicable

Example structure:

```rust
use crate::graph::GraphContext;
use crate::transformers::{OnnxTransformer, TransformResult};
use crate::error::OnnxResult;

pub struct MyTransformer;

impl MyTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl OnnxTransformer for MyTransformer {
    fn name(&self) -> &'static str {
        "MyTransformer"
    }

    fn transform(&self, ctx: &mut GraphContext) -> OnnxResult<TransformResult> {
        let mut result = TransformResult::default();
        // Implementation
        Ok(result)
    }
}
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run tests: `cargo test`
5. Run clippy: `cargo clippy`
6. Commit with descriptive message
7. Push and create Pull Request

## Reporting Issues

- Use GitHub Issues
- Include minimal reproduction case
- Include ONNX model version and opset
- Include error messages and stack traces

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
