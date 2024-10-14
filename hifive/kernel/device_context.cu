#include "hifive/kernel/FullRNS-HEAAN/src/Context.h"
#include "hifive/kernel/FullRNS-HEAAN/src/EvaluatorUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Numb.h"
#include "hifive/kernel/FullRNS-HEAAN/src/Scheme.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SchemeAlgo.h"
#include "hifive/kernel/FullRNS-HEAAN/src/SecretKey.h"
#include "hifive/kernel/FullRNS-HEAAN/src/StringUtils.h"
#include "hifive/kernel/FullRNS-HEAAN/src/TimeUtils.h"
#include "hifive/kernel/device_context.hpp"

DeviceContext::DeviceContext() {
    LOG_INFO("Initializing DeviceContext\n");
    const int logN = 15;
    // const int N = (1 << logN);
    const int L = 44;
    const uint64_t logp = 55;
    // const uint64_t logSlots = 3;
    // const uint64_t slots = (1 << logSlots);
    Context context(logN, logp, L, L + 1);
    SecretKey secretKey(context);
    Scheme scheme(secretKey, context);
    Key key = scheme.keyMap.at(MULTIPLICATION);

    set_params(context);
}

void DeviceContext::set_params(Context &context) {
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
}