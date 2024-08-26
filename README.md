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
- [ ] Prepare Params
- [ ] En/Decrypt
- [ ] Prepare DeviceVector
- [ ] Impl HAdd
- [ ] Impl HMul
- [ ] Understand Key-Decomposed ModUp
- [ ] Impl ModUp
- [ ] Impl KeySwitch
- [ ] Impl ModDown