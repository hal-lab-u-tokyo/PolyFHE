#include "device_context.hpp"
#include "hifive/kernel/FullRNS-HEAAN/src/Context.h"
#include "hifive/kernel/FullRNS-HEAAN/src/EvaluatorUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Numb.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Scheme.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SchemeAlgo.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SecretKey.h"
#include "hifive/kernel/FullRNS-HEAAN/src/StringUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/TimeUtils.h"
#include "hifive/kernel/device_context.hpp"

FHEContext::FHEContext() {
    LOG_INFO("Initializing DeviceContext\n");
    const int logN = 15;
    // const int N = (1 << logN);
    const int L = 44;
    const uint64_t logp = 55;
    // const uint64_t logSlots = 3;
    // const uint64_t slots = (1 << logSlots);
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

DeviceContext::DeviceContext(HEAANContext &context) {
    logN = context.logN;
    logNh = context.logNh;
    L = context.L;
    K = context.K;
    N = context.N;
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
    const uint64_t size_NL = N * L * sizeof(uint64_t);
    const uint64_t size_NK = N * K * sizeof(uint64_t);
    checkCudaErrors(cudaMalloc((void **) &qRoots, size_L));
    checkCudaErrors(cudaMalloc((void **) &qRootsInv, size_L));
    checkCudaErrors(cudaMalloc((void **) &qRootPows, size_NL));
    checkCudaErrors(cudaMalloc((void **) &qRootPowsInv, size_NL));
    checkCudaErrors(cudaMalloc((void **) &pRoots, size_K));
    checkCudaErrors(cudaMalloc((void **) &pRootsInv, size_K));
    checkCudaErrors(cudaMalloc((void **) &pRootPows, size_NK));
    checkCudaErrors(cudaMalloc((void **) &pRootPowsInv, size_NK));
    checkCudaErrors(
        cudaMemcpy(qRoots, context.qRoots, size_L, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(qRootsInv, context.qRootsInv, size_L,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(pRoots, context.pRoots, size_K, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pRootsInv, context.pRootsInv, size_L,
                               cudaMemcpyHostToDevice));
    for (int i = 0; i < L; i++) {
        checkCudaErrors(cudaMemcpy(qRootPows + i * N, context.qRootPows[i],
                                   N * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(qRootPowsInv + i * N, context.qRootPowsInv[i],
                       N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < K; i++) {
        checkCudaErrors(cudaMemcpy(pRootPows + i * N, context.pRootPows[i],
                                   N * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(pRootPowsInv + i * N, context.pRootPowsInv[i],
                       N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
}