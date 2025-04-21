# PolyFHE: Fully Homomorphic Encryption Compiler for GPUs

## Build
```
git clone --recursive git@github.com:ainozaki/PolyFHE.git
cd PolyFHE
cmake -S . -B build
cmake --build build -j $(nproc)
./script/build-example.sh opt
```

To install python interface,
```
uv pip  install -e .
uv run ./example/ckks_HMult/example.py
```

## Dependencies
- Boost C++ Libraries

## License
PolyFHE is licensed under the GPLv3 license. For the full license text, please refer to the [LICENSE](LICENSE) file.
This project includes modified code from the following third-party projects:
- [Phantom](https://github.com/encryptorion-lab/phantom-fhe), licensed under GPLv3
- [NNFusion](https://github.com/microsoft/nnfusion/), licensed under the MIT License

In accordance with the terms of these licenses, all derivative code is distributed under GPLv3.

## Memo
```
nvcc -o ./build/bench build/generated.cu polyfhe/kernel/device_context.cu polyfhe/kernel/polynomial.cu -g -std=c++17 -O2 -I./  --relocatable-device-code true
```