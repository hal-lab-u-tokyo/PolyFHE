# Fully Homomorphic Encryption Compiler for GPUs


## Build and Run
```
git clone --recursive git@github.com:ainozaki/HiFive.git
cd HiFive
cmake -B build -S .
cmake --build build -j $(nproc)
./build/test/test_ckks
```

## TODO
- [x] Prepare Params
- [x] En/Decrypt
- [x] Prepare DeviceVector
- [ ] Impl HAdd
- [ ] Impl NTT
- [ ] Impl HMul
- [ ] Understand Key-Decomposed ModUp
- [ ] Impl ModUp
- [ ] Prepare DecomposedKey
- [ ] Impl KeySwitch
- [ ] Impl ModDown


## Tips
- Memory check
```
compute-sanitizer --tool memcheck ./build/test/test_ckks --gtest_filter="*GPU*"
```