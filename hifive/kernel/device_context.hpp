#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <memory>

#include "hifive/core/logger.hpp"
#include "hifive/kernel/FullRNS-HEAAN/src/Context.h"

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

using HEAANContext = Context;
class Params {
public:
    Params(std::shared_ptr<HEAANContext> context);
    Params() = default;
    ~Params() = default;

    // Encryption parameters
    long logN; ///< Logarithm of Ring Dimension
    long N;
    int n1;
    int n2;

    long L;   ///< Maximum Level that we want to support
    int limb; // Current limb Level
    long K;   ///< The number of special modulus (usually L + 1)

    double sigma;

    uint64_t* qVec;
    uint64_t* qrVec;
    long* qTwok;
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
    std::shared_ptr<HEAANContext> get_heaan_context() { return heaan_context; }

private:
    void Init(const int logN, const int L);
    void CopyParamsToDevice();

    std::shared_ptr<HEAANContext> heaan_context;
    std::shared_ptr<Params> h_params;
    Params* d_params;
    NTTParams* d_ntt_params;
};

uint64_t NTTSampleSize(const uint64_t logN);