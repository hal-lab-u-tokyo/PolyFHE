# HiFive: Fully Homomorphic Encryption Compiler for GPUs

## Build and Run

HiFive requirements: boost

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
| L         | 20            |            | 29      | 32    |
| HAdd[us]  | 352.5         | 90.5       | 70.5    | 162   |
| HMult[us] |               |            |         |       |