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
- [x] Impl HAdd
- [ ] Design how to fuse kernels
- [ ] Prepare Precomputed Values
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

## License
This software containes the modified code from [Phantom](https://github.com/encryptorion-lab/phantom-fhe), which is released under GPLv3 License.