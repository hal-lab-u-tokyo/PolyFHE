# HiFive: Fully Homomorphic Encryption Compiler for GPUs

## Build and Run

### Build MLIR (Only Once)
```bash
./scripts/build-mlir.sh
```

### Build HiFive and Run
```bash
# First time
./scripts/build.sh

# Subsequent times
cmake --build build --target hifive-opt

# Run
./build/tools/hifive-opt -help

# Build and run tests
cmake --build build --target check-hifive
```