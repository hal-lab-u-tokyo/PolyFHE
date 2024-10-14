#pragma once

#include <complex>
#include <cstdint>

class DeviceContext {
public:
    DeviceContext();

    // Encryption parameters
    long logN;  ///< Logarithm of Ring Dimension
    long logNh; ///< Logarithm of Ring Dimension - 1
    long L;     ///< Maximum Level that we want to support
    long K;     ///< The number of special modulus (usually L + 1)

    long N;
    long M;
    long Nh;

    long logp;
    long p;

    long h;
    double sigma;

    uint64_t* qVec;
    uint64_t* pVec;

    uint64_t* qrVec; // Barrett reduction
    uint64_t* prVec; // Barrett recution

    long* qTwok; // Barrett reduction
    long* pTwok; // Barrett reduction

    uint64_t* qkVec; // Montgomery reduction
    uint64_t* pkVec; // Montgomery reduction

    uint64_t* qdVec;
    uint64_t* pdVec;

    uint64_t* qInvVec;
    uint64_t* pInvVec;

    uint64_t* qRoots;
    uint64_t* pRoots;

    uint64_t* qRootsInv;
    uint64_t* pRootsInv;

    uint64_t** qRootPows;
    uint64_t** pRootPows;

    uint64_t** qRootPowsInv;
    uint64_t** pRootPowsInv;

    uint64_t* NInvModq;
    uint64_t* NInvModp;

    uint64_t** qRootScalePows;
    uint64_t** pRootScalePows;

    uint64_t** qRootScalePowsOverq;
    uint64_t** pRootScalePowsOverp;

    uint64_t** qRootScalePowsInv;
    uint64_t** pRootScalePowsInv;

    uint64_t* NScaleInvModq; // [i]
    uint64_t* NScaleInvModp; // [k]

    uint64_t** qHatModq; // [l][i] (phat_i)_l mod p_i
    uint64_t* pHatModp;  // [k] qhat_k mod q_k

    uint64_t** qHatInvModq; // [l][i] (qhat_i)_l^-1 mod q_i
    uint64_t* pHatInvModp;  // [k] phat_k^-1 mod p_k

    uint64_t*** qHatModp; // [l] [i] [k]  (phat_i)_l mod q_k

    uint64_t** pHatModq; // [k][i] qhat_k mod p_i

    uint64_t* PModq;    // [i] qprod mod p_i
    uint64_t* PInvModq; // [i] qprod mod p_i

    uint64_t** QModp;    // [i] qprod mod p_i
    uint64_t** QInvModp; // [i] qprod mod p_i

    uint64_t** qInvModq; // [i][j] p_i^-1 mod p_j

    long* rotGroup; ///< precomputed rotation group indexes

    uint64_t* p2coeff;
    uint64_t* pccoeff;
    uint64_t* p2hcoeff;
};