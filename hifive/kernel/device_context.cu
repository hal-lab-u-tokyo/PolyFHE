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

void FHEContext::Init(const int logN, const int L) {
    LOG_INFO("Initializing DeviceContext\n");
    const uint64_t logp = 55;
    HEAANContext heaan_context(logN, logp, L, L + 1);
    SecretKey secretKey(heaan_context);
    Scheme scheme(secretKey, heaan_context);
    Key key = scheme.keyMap.at(MULTIPLICATION);

    d_context_in_cpu = std::make_shared<DeviceContext>(heaan_context);
    h_context = std::make_shared<HEAANContext>(heaan_context);

    // Copy context_d to device
    checkCudaErrors(
        cudaMalloc((void **) &d_context_in_gpu, sizeof(DeviceContext)));
    checkCudaErrors(cudaMemcpy(d_context_in_gpu, d_context_in_cpu.get(),
                               sizeof(DeviceContext), cudaMemcpyHostToDevice));
}

FHEContext::FHEContext() { Init(hifive::logN, hifive::L); }

FHEContext::FHEContext(const int logN, const int L) { Init(logN, L); }

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

uint64_t Inverse(const uint64_t op, const uint64_t prime) {
    uint64_t tmp = op > prime ? (op % prime) : op;
    return powMod(tmp, prime - 2, prime);
}

DeviceContext::DeviceContext(HEAANContext &context) {
    logN = context.logN;
    logNh = context.logNh;
    L = context.L;
    K = context.K;
    N = context.N;
    N1 = NTTSampleSize(logN);
    N2 = N / N1;
    M = context.M;
    Nh = context.Nh;
    logp = context.logp;
    p = context.p;
    h = context.h;
    sigma = context.sigma;

    // qVec, qrVec, qTwok, qkVec, qdVec
    // pVec, prVec, pTwok, pkVec, pdVec
    const long size_L = L * sizeof(uint64_t);
    const long size_K = K * sizeof(uint64_t);
    checkCudaErrors(cudaMalloc((void **) &qVec, size_L));
    checkCudaErrors(cudaMalloc((void **) &qrVec, size_L));
    checkCudaErrors(cudaMalloc((void **) &qTwok, size_L));
    checkCudaErrors(cudaMalloc((void **) &qkVec, size_L));
    checkCudaErrors(cudaMalloc((void **) &qdVec, size_L));
    checkCudaErrors(cudaMalloc((void **) &pVec, size_K));
    checkCudaErrors(cudaMalloc((void **) &prVec, size_K));
    checkCudaErrors(cudaMalloc((void **) &pTwok, size_K));
    checkCudaErrors(cudaMalloc((void **) &pkVec, size_K));
    checkCudaErrors(cudaMalloc((void **) &pdVec, size_K));
    checkCudaErrors(
        cudaMemcpy(qVec, context.qVec, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(qrVec, context.qrVec, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(qTwok, context.qTwok, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(qkVec, context.qkVec, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(qdVec, context.qdVec, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pVec, context.pVec, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(prVec, context.prVec, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pTwok, context.pTwok, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pkVec, context.pkVec, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pdVec, context.pdVec, size_K, cudaMemcpyHostToDevice));

    // qRoots, qRootsInv, qRootPows, qRootPowsInv
    // pRoots, pRootsInv, pRootPows, pRootPowsInv
    const uint64_t size_N = N * sizeof(uint64_t);
    checkCudaErrors(cudaMalloc((void **) &qRoots, size_L));
    checkCudaErrors(cudaMalloc((void **) &qRootsInv, size_L));
    checkCudaErrors(cudaMalloc((void **) &pRoots, size_K));
    checkCudaErrors(cudaMalloc((void **) &pRootsInv, size_K));
    checkCudaErrors(
        cudaMemcpy(qRoots, context.qRoots, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(qRootsInv, context.qRootsInv, size_L,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pRoots, context.pRoots, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pRootsInv, context.pRootsInv, size_L,
                               cudaMemcpyHostToDevice));

    uint64_t **tmp = new uint64_t *[L];
    uint64_t **tmp2 = new uint64_t *[L];
    uint64_t *tmpN = new uint64_t[N];
    uint64_t *tmpN2 = new uint64_t[N];

    // qRootPows
    for (int i = 0; i < L; i++) {
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(cudaMemcpy(tmp[i], context.qRootPows[i], size_N,
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &qRootPows, size_L));
    checkCudaErrors(cudaMemcpy(qRootPows, tmp, size_L, cudaMemcpyHostToDevice));

    // qRootPowsShoup
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            Shoup(tmpN[j], context.qRootPows[i][j], context.qVec[i]);
        }
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(
            cudaMemcpy(tmp[i], tmpN, size_N, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &qRootPowsShoup, size_L));
    checkCudaErrors(
        cudaMemcpy(qRootPowsShoup, tmp, size_L, cudaMemcpyHostToDevice));

    // qRootPowsInv
    for (int i = 0; i < L; i++) {
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(cudaMemcpy(tmp[i], context.qRootPowsInv[i], size_N,
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &qRootPowsInv, size_L));
    checkCudaErrors(
        cudaMemcpy(qRootPowsInv, tmp, size_L, cudaMemcpyHostToDevice));

    // qRootPowsInvShoup
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            Shoup(tmpN2[j], context.qRootPowsInv[i][j], context.qVec[i]);
        }
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(
            cudaMemcpy(tmp[i], tmpN, size_N, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &qRootPowsInvShoup, size_L));
    checkCudaErrors(
        cudaMemcpy(qRootPowsInvShoup, tmp, size_L, cudaMemcpyHostToDevice));

    tmp = new uint64_t *[K];
    // pRootPows
    for (int i = 0; i < K; i++) {
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(cudaMemcpy(tmp[i], context.pRootPows[i], size_N,
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &pRootPows, size_K));
    checkCudaErrors(cudaMemcpy(pRootPows, tmp, size_K, cudaMemcpyHostToDevice));

    // pRootPowsInv
    for (int i = 0; i < K; i++) {
        checkCudaErrors(cudaMalloc((void **) &tmp[i], size_N));
        checkCudaErrors(cudaMemcpy(tmp[i], context.pRootPowsInv[i], size_N,
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc((void **) &pRootPowsInv, size_K));
    checkCudaErrors(
        cudaMemcpy(pRootPowsInv, tmp, size_K, cudaMemcpyHostToDevice));
    LOG_INFO("DeviceContext initialized\n");
}