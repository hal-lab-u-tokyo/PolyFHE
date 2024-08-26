#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>

namespace hifive {

void __checkCudaErrors(cudaError_t err, const char *filename, int line);
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

class GPUContext {
public:
    GPUContext();
    ~GPUContext() = default;
    void GetGPUInfo(bool verbose);

    int max_threads_per_block_;
};

class gpu_ptr {
public:
    gpu_ptr() = default;
    explicit gpu_ptr(uint64_t *d_ptr, uint64_t size)
        : d_ptr_(d_ptr), n_(size) {}
    ~gpu_ptr() = default;

    // copy operator
    gpu_ptr &operator=(const gpu_ptr &other) {
        n_ = other.n_;
        d_ptr_ = other.d_ptr_;
        return *this;
    }

    // copy constructor
    gpu_ptr(const gpu_ptr &obj) {
        this->n_ = obj.n_;
        this->d_ptr_ = obj.d_ptr_;
    }

    void copy_to_cpu(uint64_t *dst, uint64_t size) const {
        assert(size <= size_);
        checkCudaErrors(cudaMemcpy(dst, d_ptr_, size * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost));
    }

    uint64_t *get() const { return d_ptr_; }
    uint64_t size() const { return n_; }

private:
    uint64_t *d_ptr_ = nullptr;
    uint64_t n_ = 0;
};

gpu_ptr make_and_copy_gpu_ptr(uint64_t *src, uint64_t size);
gpu_ptr make_gpu_ptr(uint64_t size);
} // namespace hifive