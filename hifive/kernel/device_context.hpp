#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <memory>

#include "hifive/core/logger.hpp"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError_t err, const char* filename, int line);
inline void __checkCudaErrors(cudaError_t err, const char* filename, int line) {
    assert(filename);
    if (cudaSuccess != err) {
        const char* ename = cudaGetErrorName(err);
        LOG_ERROR(
            "CUDA API Error %04d: \"%s\" from file <%s>, "
            "line %i.\n",
            err, ((ename != NULL) ? ename : "Unknown"), filename, line);
        exit(err);
    }
}

struct NTTParams {
    uint64_t N;
    int logN;
    int n1; // N = n1 * n2
    int n2;
    int batch;
    uint64_t* q;
    uint64_t* root;
    uint64_t* root_inv;
    uint64_t* N_inv;
    uint64_t** roots_pow;
    uint64_t** roots_pow_shoup;
    uint64_t** roots_pow_inv;
    uint64_t** roots_pow_inv_shoup;
};

class Params {
public:
    Params(int logN, int L);
    Params() = default;
    ~Params() = default;

    // Encryption parameters
    int logN; ///< Logarithm of Ring Dimension
    int N;
    int n1;
    int n2;

    int L;    ///< Maximum Level that we want to support
    int limb; // Current limb Level
    int K;    ///< The number of special modulus (usually L + 1)

    double sigma = 3.2;

    uint64_t* qVec;
    uint64_t* pVec;

    NTTParams* ntt_params;
};

class FHEContext {
public:
    FHEContext();
    FHEContext(const int logN, const int L);
    ~FHEContext() = default;

    Params* get_d_params() { return d_params; }
    NTTParams* get_d_ntt_params() { return d_ntt_params; }
    std::shared_ptr<Params> get_h_params() { return h_params; }

private:
    void Init(const int logN, const int L);
    void CopyParamsToDevice();

    std::shared_ptr<Params> h_params;
    Params* d_params;
    NTTParams* d_ntt_params;
};

uint64_t NTTSampleSize(const uint64_t logN);