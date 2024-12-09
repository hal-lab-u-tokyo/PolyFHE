#include <cstring>

#include "device_context.hpp"
#include "hifive/core/param.hpp"
#include "hifive/kernel/FullRNS-HEAAN/src/Context.h"
#include "hifive/kernel/FullRNS-HEAAN/src/EvaluatorUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Numb.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Scheme.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SchemeAlgo.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SecretKey.h"
#include "hifive/kernel/FullRNS-HEAAN/src/StringUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/TimeUtils.h"
#include "hifive/kernel/device_context.hpp"

uint64_t NTTSampleSize(const uint64_t logN) {
    if (logN == 12) {
        return 1 << 6;
    } else if (logN == 13) {
        return 1 << 7;
    } else if (logN == 14) {
        return 1 << 7;
    } else if (logN == 15) {
        return 1 << 8;
    } else if (logN == 16) {
        return 1 << 8;
    } else if (logN == 17) {
        return 1 << 9;
    } else {
        LOG_ERROR("Invalid logN: %ld\n", logN);
        return 0;
    }
}

uint64_t compute_shoup(const uint64_t operand, const uint64_t modulus) {
    if (operand >= modulus) {
        throw "Operand must be less than modulus";
    }
    // Using __uint128_t to avoid overflow during multiplication
    __uint128_t temp = operand;
    temp <<= 64; // multiplying by 2^64
    return temp / modulus;
}

Params::Params(std::shared_ptr<HEAANContext> context) {
    logN = context->logN;
    L = context->L;
    K = context->K;
    N = context->N;
    sigma = context->sigma;

    qVec = context->qVec;
    qrVec = context->qrVec;
    qTwok = context->qTwok;
    pVec = context->pVec;

    ntt_params = new NTTParams();
    ntt_params->N = N;
    ntt_params->n1 = NTTSampleSize(logN);
    ntt_params->n2 = N / ntt_params->n1;
    ntt_params->logN = logN;
    ntt_params->batch = L;

    ntt_params->q = qVec;
    ntt_params->root = context->qRoots;
    ntt_params->root_inv = context->qRootsInv;
    ntt_params->N_inv = context->NInvModp;

    ntt_params->roots_pow = new uint64_t *[L];
    ntt_params->roots_pow_inv = new uint64_t *[L];
    ntt_params->roots_pow_shoup = new uint64_t *[L];
    ntt_params->roots_pow_inv_shoup = new uint64_t *[L];
    memcpy(ntt_params->roots_pow, context->qRootPows, L * sizeof(uint64_t *));
    memcpy(ntt_params->roots_pow_inv, context->qRootPowsInv,
           L * sizeof(uint64_t *));
    for (int i = 0; i < L; i++) {
        ntt_params->roots_pow_shoup[i] = new uint64_t[N];
        ntt_params->roots_pow_inv_shoup[i] = new uint64_t[N];
        for (int j = 0; j < N; j++) {
            ntt_params->roots_pow_shoup[i][j] =
                compute_shoup(context->qRootPows[i][j], qVec[i]);
            ntt_params->roots_pow_inv_shoup[i][j] =
                compute_shoup(context->qRootPowsInv[i][j], qVec[i]);
        }
    }
}

void FHEContext::CopyParamsToDevice() {
    Params params_tmp;
    memcpy(&params_tmp, h_params.get(), sizeof(Params));

    uint64_t *d_qVec;
    uint64_t *d_qrVec;
    long *d_qTwok;
    uint64_t *d_pVec;
    checkCudaErrors(cudaMalloc(&d_qVec, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_qrVec, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_qTwok, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_pVec, h_params->K * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_qVec, h_params->qVec,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_qrVec, h_params->qrVec,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_qTwok, h_params->qTwok,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pVec, h_params->pVec,
                               params_tmp.K * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    params_tmp.qVec = d_qVec;
    params_tmp.qrVec = d_qrVec;
    params_tmp.qTwok = d_qTwok;
    params_tmp.pVec = d_pVec;

    NTTParams ntt_params_tmp;
    memcpy(&ntt_params_tmp, h_params->ntt_params, sizeof(NTTParams));
    ntt_params_tmp.q = d_qVec;
    uint64_t *d_root;
    uint64_t *d_root_inv;
    uint64_t *d_N_inv;
    checkCudaErrors(cudaMalloc(&d_root, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_root_inv, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_N_inv, h_params->L * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_root, h_params->ntt_params->root,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_root_inv, h_params->ntt_params->root_inv,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_N_inv, h_params->ntt_params->N_inv,
                               params_tmp.L * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    uint64_t **d_roots_pow = new uint64_t *[h_params->L];
    uint64_t **d_roots_pow_inv = new uint64_t *[h_params->L];
    uint64_t **d_roots_pow_shoup = new uint64_t *[h_params->L];
    uint64_t **d_roots_pow_inv_shoup = new uint64_t *[h_params->L];
    for (int i = 0; i < h_params->L; i++) {
        checkCudaErrors(
            cudaMalloc(&d_roots_pow[i], h_params->N * sizeof(uint64_t)));
        checkCudaErrors(
            cudaMalloc(&d_roots_pow_inv[i], h_params->N * sizeof(uint64_t)));
        checkCudaErrors(
            cudaMalloc(&d_roots_pow_shoup[i], h_params->N * sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc(&d_roots_pow_inv_shoup[i],
                                   h_params->N * sizeof(uint64_t)));
        checkCudaErrors(
            cudaMemcpy(d_roots_pow[i], h_params->ntt_params->roots_pow[i],
                       h_params->N * sizeof(uint64_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            d_roots_pow_inv[i], h_params->ntt_params->roots_pow_inv[i],
            h_params->N * sizeof(uint64_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            d_roots_pow_shoup[i], h_params->ntt_params->roots_pow_shoup[i],
            h_params->N * sizeof(uint64_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_roots_pow_inv_shoup[i],
                                   h_params->ntt_params->roots_pow_inv_shoup[i],
                                   h_params->N * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow,
                               h_params->L * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_inv,
                               h_params->L * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_shoup,
                               h_params->L * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_inv_shoup,
                               h_params->L * sizeof(uint64_t *)));
    checkCudaErrors(cudaMemcpy(ntt_params_tmp.roots_pow, d_roots_pow,
                               h_params->L * sizeof(uint64_t *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ntt_params_tmp.roots_pow_inv, d_roots_pow_inv,
                               h_params->L * sizeof(uint64_t *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(ntt_params_tmp.roots_pow_shoup, d_roots_pow_shoup,
                   h_params->L * sizeof(uint64_t *), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(ntt_params_tmp.roots_pow_inv_shoup, d_roots_pow_inv_shoup,
                   h_params->L * sizeof(uint64_t *), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_ntt_params, sizeof(NTTParams)));
    checkCudaErrors(cudaMemcpy(d_ntt_params, &ntt_params_tmp, sizeof(NTTParams),
                               cudaMemcpyHostToDevice));
    params_tmp.ntt_params = d_ntt_params;

    checkCudaErrors(cudaMalloc((void **) &d_params, sizeof(Params)));
    checkCudaErrors(cudaMemcpy(d_params, &params_tmp, sizeof(Params),
                               cudaMemcpyHostToDevice));

    d_ntt_params = params_tmp.ntt_params;
}

void FHEContext::Init(const int logN, const int L) {
    LOG_INFO("Initializing Params\n");
    const uint64_t logp = 55;
    HEAANContext heaan(logN, logp, L, L + 1);
    SecretKey secretKey(heaan);
    Scheme scheme(secretKey, heaan);
    Key key = scheme.keyMap.at(MULTIPLICATION);

    heaan_context = std::make_shared<HEAANContext>(heaan);
    h_params = std::make_shared<Params>(heaan_context);
    CopyParamsToDevice();
}

FHEContext::FHEContext() { Init(hifive::logN, hifive::L); }

FHEContext::FHEContext(const int logN, const int L) { Init(logN, L); }