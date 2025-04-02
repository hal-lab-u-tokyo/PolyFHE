#pragma once

#include <cstdint>

#include "polyfhe/kernel/device_context.hpp"

// GPU
extern "C" {

__device__ void NTTPhase1Internal(uint64_t* buffer, NTTParams* params,
                                  const size_t batch_idx,
                                  const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n1;

    for (int m = 1; m < params->n1; m *= 2) {
        t = t / 2;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx = t * j * params->n2;
        uint64_t S = params->roots_pow[batch_idx][rootidx];
        __syncthreads();
        uint64_t U = buffer[k + j];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        __syncthreads();
    }
}

__device__ void NTTPhase1BlockedInternal(uint64_t* buffer, NTTParams* params,
                                         const int batch_idx,
                                         const size_t thread_idx, int loopidx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n1;
    int sync_idx = batch_idx % 15 + 1;
    int n_sync_threads = params->n1 / 2;
    for (int m = 1; m < params->n1; m *= 2) {
        t = t / 2;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx = t * j * params->n2;
        uint64_t S = params->roots_pow[batch_idx][rootidx];
        asm volatile("bar.sync %0, %1;" ::"r"(sync_idx), "r"(n_sync_threads));
        uint64_t U = buffer[k + j];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        asm volatile("bar.sync %0, %1;" ::"r"(sync_idx), "r"(n_sync_threads));
    }
}

__device__ void NTTPhase2Internal(uint64_t* buffer, NTTParams* params,
                                  const size_t batch_idx,
                                  const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n2;
    const size_t block_idx = blockIdx.x % params->n1;
    for (int m = 1; m < params->n2; m *= 2) {
        t = t / 2;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow[batch_idx][rootidx];
        __syncthreads();
        uint64_t U = buffer[k + j];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        __syncthreads();
    }
}

__device__ void NTTPhase2BlockedInternal(uint64_t* buffer, NTTParams* params,
                                         const size_t batch_idx,
                                         const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t = params->n2;
    int barrier_threads = params->n2 / 2;
    int barrier_group = batch_idx % 15 + 1;
    const size_t block_idx = blockIdx.x % params->n1;
    for (int m = 1; m < params->n2; m *= 2) {
        t = t / 2;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow[batch_idx][rootidx];
        asm volatile("bar.sync %0, %1;" ::"r"((int32_t) barrier_group),
                     "r"((int32_t) barrier_threads));
        uint64_t U = buffer[k + j];
        uint64_t V = (buffer[k + j + m] * S) % q;
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        tmp = U + q - V;
        buffer[k + j + m] = tmp >= q ? tmp - q : tmp;
        asm volatile("bar.sync %0, %1;" ::"r"((int32_t) barrier_group),
                     "r"((int32_t) barrier_threads));
    }
}

__device__ void iNTTPhase2Internal(uint64_t* buffer, NTTParams* params,
                                   const size_t batch_idx,
                                   const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    const size_t block_idx = blockIdx.x % params->n1;
    for (int m = params->n2 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n2 / step;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        __syncthreads();
    }
}

__device__ void iNTTPhase2BlockedInternal(uint64_t* buffer, NTTParams* params,
                                          const size_t batch_idx,
                                          const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    int barrier_threads = params->n2 / 2;
    int barrier_group = batch_idx % 15 + 1;
    const size_t block_idx = blockIdx.x % params->n1;
    for (int m = params->n2 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n2 / step;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx =
            t * j * params->n1 + block_idx * params->n2 / (2 * m);

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        asm volatile("bar.sync %0, %1;" ::"r"((int32_t) barrier_group),
                     "r"((int32_t) barrier_threads));
    }
}

__device__ void iNTTPhase1Internal(uint64_t* buffer, NTTParams* params,
                                   const size_t batch_idx,
                                   const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    for (int m = params->n1 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n1 / step;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx = t * j * params->n2;

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        __syncthreads();
    }
    const uint64_t Ninv = params->N_inv[batch_idx];
    buffer[thread_idx] = (buffer[thread_idx] * Ninv) % q;
    buffer[thread_idx + params->n1 / 2] =
        (buffer[thread_idx + params->n1 / 2] * Ninv) % q;
}

__device__ void iNTTPhase1BlockedInternal(uint64_t* buffer, NTTParams* params,
                                          const size_t batch_idx,
                                          const size_t thread_idx) {
    uint64_t q = params->q[batch_idx];
    uint64_t t, step;
    int barrier_threads = params->n1 / 2;
    int barrier_group = batch_idx % 15 + 1;
    for (int m = params->n1 / 2; m >= 1; m /= 2) {
        step = m * 2;
        t = params->n1 / step;
        int j = thread_idx & (m - 1);
        int k = 2 * m * (thread_idx / m);
        const int rootidx = t * j * params->n2;

        uint64_t S = params->roots_pow_inv[batch_idx][rootidx];
        uint64_t U = buffer[k + j];
        uint64_t V = buffer[k + j + m];
        uint64_t tmp = U + V;
        buffer[k + j] = tmp >= q ? tmp - q : tmp;
        buffer[k + j + m] = (((U - V + q) % q) * S) % q;
        asm volatile("bar.sync %0, %1;" ::"r"((int32_t) barrier_group),
                     "r"((int32_t) barrier_threads));
    }
    const uint64_t Ninv = params->N_inv[batch_idx];
    buffer[thread_idx] = (buffer[thread_idx] * Ninv) % q;
    buffer[thread_idx + params->n1 / 2] =
        (buffer[thread_idx + params->n1 / 2] * Ninv) % q;
}

} // extern "C"