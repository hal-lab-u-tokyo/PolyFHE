#include <cstring>
#include <vector>

#include "device_context.hpp"
#include "hifive/kernel/device_context.hpp"
#include "hifive/utils.hpp"

// from https://github.com/snucrypto/HEAAN, 131d275
void mulMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
    unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
    mul %= static_cast<unsigned __int128>(m);
    r = static_cast<uint64_t>(mul);
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t powMod(uint64_t x, uint64_t y, uint64_t modulus) {
    uint64_t res = 1;
    while (y > 0) {
        if (y & 1) {
            mulMod(res, res, x, modulus);
        }
        y = y >> 1;
        mulMod(x, x, x, modulus);
    }
    return res;
}

// from https://github.com/snucrypto/HEAAN, 131d275
void findPrimeFactors(std::vector<uint64_t> &s, uint64_t number) {
    while (number % 2 == 0) {
        s.push_back(2);
        number /= 2;
    }
    for (uint64_t i = 3; i < sqrt(number); i++) {
        while (number % i == 0) {
            s.push_back(i);
            number /= i;
        }
    }
    if (number > 2) {
        s.push_back(number);
    }
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t findPrimitiveRoot(uint64_t modulus) {
    std::vector<uint64_t> s;
    uint64_t phi = modulus - 1;
    findPrimeFactors(s, phi);
    for (uint64_t r = 2; r <= phi; r++) {
        bool flag = false;
        for (auto it = s.begin(); it != s.end(); it++) {
            if (powMod(r, phi / (*it), modulus) == 1) {
                flag = true;
                break;
            }
        }
        if (flag == false) {
            return r;
        }
    }
    throw "Cannot find the primitive root of unity";
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

Params::Params(const int logN, const int L, const int dnum)
    : logN(logN), L(L), dnum(dnum) {
    limb = L;
    N = 1 << logN;
    alpha = std::ceil((L + 1) / dnum);
    K = alpha;
    n1 = hifive::NTTSampleSize(logN);
    n2 = N / n1;
    sigma = 3.2;
    qVec = new uint64_t[L + K];
    pVec = qVec + L;
    for (int i = 0; i < L; i++) {
        qVec[i] = 998244353; // 51-bit
    }
    for (int i = 0; i < K; i++) {
        pVec[i] = 998244353; // 51-bit
    }

    ntt_params = new NTTParams();
    ntt_params->N = N;
    ntt_params->n1 = n1;
    ntt_params->n2 = n2;
    ntt_params->logN = logN;
    ntt_params->batch = L;
    ntt_params->q = qVec;
    ntt_params->p = pVec;
    ntt_params->root = new uint64_t[L + K];
    ntt_params->root_inv = new uint64_t[L + K];
    ntt_params->N_inv = new uint64_t[L + K];
    for (int i = 0; i < L + K; i++) {
        ntt_params->root[i] = findPrimitiveRoot(qVec[i]);
        ntt_params->root_inv[i] =
            powMod(ntt_params->root[i], qVec[i] - 2, qVec[i]);
        ntt_params->N_inv[i] = powMod(N, qVec[i] - 2, qVec[i]);
    }
    ntt_params->roots_pow = new uint64_t *[L + K];
    ntt_params->roots_pow_shoup = new uint64_t *[L + K];
    ntt_params->roots_pow_inv = new uint64_t *[L + K];
    ntt_params->roots_pow_inv_shoup = new uint64_t *[L + K];
    for (int i = 0; i < (L + K); i++) {
        ntt_params->roots_pow[i] = new uint64_t[N];
        ntt_params->roots_pow_inv[i] = new uint64_t[N];
        ntt_params->roots_pow_shoup[i] = new uint64_t[N];
        ntt_params->roots_pow_inv_shoup[i] = new uint64_t[N];
        ntt_params->roots_pow[i][0] = 1;
        ntt_params->roots_pow_inv[i][0] = 1;
        ntt_params->roots_pow_shoup[i][0] = compute_shoup(1, ntt_params->q[i]);
        ntt_params->roots_pow_inv_shoup[i][0] =
            compute_shoup(1, ntt_params->q[i]);
        for (int j = 1; j < N; j++) {
            mulMod(ntt_params->roots_pow[i][j], ntt_params->roots_pow[i][j - 1],
                   ntt_params->root[i], ntt_params->q[i]);
            mulMod(ntt_params->roots_pow_inv[i][j],
                   ntt_params->roots_pow_inv[i][j - 1], ntt_params->root_inv[i],
                   ntt_params->q[i]);
            ntt_params->roots_pow_shoup[i][j] =
                compute_shoup(ntt_params->roots_pow[i][j], ntt_params->q[i]);
            ntt_params->roots_pow_inv_shoup[i][j] = compute_shoup(
                ntt_params->roots_pow_inv[i][j], ntt_params->q[i]);
        }
    }
}

void FHEContext::CopyParamsToDevice() {
    Params params_tmp;
    memcpy(&params_tmp, h_params.get(), sizeof(Params));
    const int L = h_params->L;
    const int K = h_params->K;

    uint64_t *d_qVec;
    checkCudaErrors(cudaMalloc(&d_qVec, (L + K) * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_qVec, h_params->qVec,
                               (L + K) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    params_tmp.qVec = d_qVec;
    params_tmp.pVec = d_qVec + L;

    NTTParams ntt_params_tmp;
    memcpy(&ntt_params_tmp, h_params->ntt_params, sizeof(NTTParams));
    ntt_params_tmp.q = d_qVec;
    ntt_params_tmp.p = d_qVec + L;
    uint64_t *d_root;
    uint64_t *d_root_inv;
    uint64_t *d_N_inv;
    checkCudaErrors(cudaMalloc(&d_root, (L + K) * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_root_inv, (L + K) * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc(&d_N_inv, (L + K) * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_root, h_params->ntt_params->root,
                               (L + K) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_root_inv, h_params->ntt_params->root_inv,
                               (L + K) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_N_inv, h_params->ntt_params->N_inv,
                               (L + K) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));
    uint64_t **d_roots_pow = new uint64_t *[L + K];
    uint64_t **d_roots_pow_inv = new uint64_t *[L + K];
    uint64_t **d_roots_pow_shoup = new uint64_t *[L + K];
    uint64_t **d_roots_pow_inv_shoup = new uint64_t *[L + K];
    for (int i = 0; i < L + K; i++) {
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
    checkCudaErrors(
        cudaMalloc(&ntt_params_tmp.roots_pow, (L + K) * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_inv,
                               (L + K) * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_shoup,
                               (L + K) * sizeof(uint64_t *)));
    checkCudaErrors(cudaMalloc(&ntt_params_tmp.roots_pow_inv_shoup,
                               (L + K) * sizeof(uint64_t *)));
    checkCudaErrors(cudaMemcpy(ntt_params_tmp.roots_pow, d_roots_pow,
                               (L + K) * sizeof(uint64_t *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ntt_params_tmp.roots_pow_inv, d_roots_pow_inv,
                               (L + K) * sizeof(uint64_t *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ntt_params_tmp.roots_pow_shoup,
                               d_roots_pow_shoup, (L + K) * sizeof(uint64_t *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(ntt_params_tmp.roots_pow_inv_shoup, d_roots_pow_inv_shoup,
                   (L + K) * sizeof(uint64_t *), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_ntt_params, sizeof(NTTParams)));
    checkCudaErrors(cudaMemcpy(d_ntt_params, &ntt_params_tmp, sizeof(NTTParams),
                               cudaMemcpyHostToDevice));
    params_tmp.ntt_params = d_ntt_params;

    checkCudaErrors(cudaMalloc((void **) &d_params, sizeof(Params)));
    checkCudaErrors(cudaMemcpy(d_params, &params_tmp, sizeof(Params),
                               cudaMemcpyHostToDevice));

    d_ntt_params = params_tmp.ntt_params;
}

FHEContext::FHEContext(const int logN, const int L, const int dnum) {
    LOG_INFO("Initializing Params\n");
    h_params = std::make_shared<Params>(logN, L, dnum);
    CopyParamsToDevice();
}