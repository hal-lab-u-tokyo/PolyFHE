#pragma once

#include <cstdint>

namespace hifive {

// ntt block threads, max = 2^16 * coeff_mod_size / (8*thread) as we do 8
// pre-thread ntt
constexpr dim3 gridDimNTT(4096);
constexpr dim3 blockDimNTT(128);
// radix-8 nwt, DO NOT change this variable !!!
constexpr size_t per_thread_sample_size = 8;
// per_block_pad can be 1, 2, 4, etc., per_block_pad * phase1_sample_size / 8 <=
// blockDim.x per_block_pad = 4 seems to be most optimized for n=4096, max pad =
// 8
constexpr size_t per_block_pad = 4;

class DModulus {
private:
    uint64_t value_ = 0;
    uint64_t const_ratio_[2] = {0,
                                0}; // 0 corresponding low, 1 corresponding high

public:
    DModulus() = default;

    DModulus(const uint64_t value, const uint64_t ratio0, const uint64_t ratio1)
        : value_(value), const_ratio_{ratio0, ratio1} {}

    void set(const uint64_t value, const uint64_t const_ratio0,
             const uint64_t const_ratio1) {
        cudaMemcpyAsync(&value_, &value, sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(const_ratio_[0]), &const_ratio0, sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(const_ratio_[1]), &const_ratio1, sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
    }

    // Returns a const pointer to the value of the current Modulus.
    __device__ __host__ inline const uint64_t *data() const noexcept {
        return &value_;
    }

    __device__ __host__ inline uint64_t value() const { return value_; }

    __device__ __host__ inline auto &const_ratio() const {
        return const_ratio_;
    }
};

class DNTTTable {
public:
    DNTTTable() = default;
    ~DNTTTable() = default;
    uint64_t n() const { return n_; }

private:
    uint64_t n_ = 0;
    uint64_t size_ = 0;
};

void nwt_2d_radix8_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables,
                                   uint64_t coeff_modulus_size,
                                   uint64_t start_modulus_idx);

} // namespace hifive