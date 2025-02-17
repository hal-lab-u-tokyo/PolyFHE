# PolyFHE: Fully Homomorphic Encryption Compiler for GPUs

## Build and Run

PolyFHE requirements: boost

```
Hifive Options:
  -o [ --opt ]          Optimize graph
  -h [ --help ]         Print help message
  -i [ --input ] arg    Input dot file
```

To test and benchmark HAdd, run the following commands:
```
make run TARGET=./data/graph_poly_hadd.dot
```

## Benchmark
|           | iwashi(A4000) | rump(A100) | Phantom(A100) | 100x(V100)  |
|-----------|---------------|------------|---------|-------|
| N         | 65536         | 65536      | 65536   | 65536 |
| L         | 20            | 20         | 29      | 32    |
| HAdd[us]  | 178           | 89         | 70.5    | 162   |
| HMult[us] |               |            |         |       |

## Memo
```
nvcc -o ./build/bench build/generated.cu polyfhe/kernel/device_context.cu polyfhe/kernel/polynomial.cu -g -std=c++17 -O2 -I./  --relocatable-device-code true
```