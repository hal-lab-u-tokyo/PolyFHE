#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <memory>

#include "phantom-fhe/include/rns.cuh"
#include "polyfhe/core/logger.hpp"

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
        // exit(err);
    }
}

struct NTTParams {
    uint64_t N;
    int logN;
    int n1; // N = n1 * n2
    int n2;
    int batch;
    uint64_t* q;
    uint64_t* p;
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
    Params(int logN, int L, int dnum) {
        this->logN = logN;
        this->N = 1 << logN;
        this->n2 = 1 << (logN / 2);
        this->n1 = this->N / this->n2;
        this->L = L;
        this->dnum = dnum;
        this->limb = L;
        this->alpha = (L + 1) / dnum;
        this->K = this->alpha;
        this->KL = this->K + this->L;
    }
    Params() = default;
    ~Params() = default;

    // Encryption parameters
    int logN;
    int N; // Number of coefficients in a polynomial
    int n1;
    int n2;

    int L;    // Maximum limb
    int limb; // Current limb
    int K;    // The number of special modulus
    int KL;
    int alpha; // Number of limbs in a digit, ceil((L + 1) / dnum)
    int dnum;  // Number of digits
    const size_t pad = 4;
    const int per_thread_ntt_size = 8;

    double sigma = 3.2;

    uint64_t* qVec;
    uint64_t* pVec;
    uint64_t* modulus_const_ratio;

    uint64_t* itwiddle;
    uint64_t* itwiddle_shoup;
    uint64_t* n_inv;
    uint64_t* n_inv_shoup;
    uint64_t* inv_degree_modulo;
    uint64_t* inv_degree_modulo_shoup;

    NTTParams* ntt_params;
    std::vector<phantom::DRNSTool*> rns_tools;
    const DNTTTable* ntt_tables;
    const int per_block_pad = 4;
};

class FHEContext {
public:
    FHEContext(const int logN, const int L, const int dnum);
    ~FHEContext() = default;

    Params* get_d_params() { return d_params; }
    NTTParams* get_d_ntt_params() { return d_ntt_params; }
    std::shared_ptr<Params> get_h_params() { return h_params; }

private:
    void CopyParamsToDevice();

    std::shared_ptr<Params> h_params;
    Params* d_params;
    NTTParams* d_ntt_params;
};