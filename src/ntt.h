#pragma once

#include <cstdint>
#include <vector>

#include "gpu_utils.h"
#include "seal/seal.h"

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
    uint64_t size() const { return size_; }
    DModulus *modulus() const { return modulus_; }
    uint64_t *twiddle() const { return twiddle_.get(); }
    uint64_t *twiddle_shoup() const { return twiddle_shoup_.get(); }
    uint64_t *itwiddle() const { return itwiddle_.get(); }
    uint64_t *itwiddle_shoup() const { return itwiddle_shoup_.get(); }
    uint64_t *n_inv_mod_q() const { return n_inv_mod_q_.get(); }
    uint64_t *n_inv_mod_q_shoup() const { return n_inv_mod_q_shoup_.get(); }

private:
    uint64_t n_ = 0;
    uint64_t size_ = 0;
    // TODO: wrap DMoulus in a gpu_ptr
    DModulus *modulus_;         // modulus for this NWT
    gpu_ptr twiddle_;           // forward NTT table
    gpu_ptr twiddle_shoup_;     // forward NTT table
    gpu_ptr itwiddle_;          // inverse NTT table
    gpu_ptr itwiddle_shoup_;    // inverse NTT table
    gpu_ptr n_inv_mod_q_;       // n^(-1) modulo q
    gpu_ptr n_inv_mod_q_shoup_; // n^(-1) modulo q, shoup version
};

class NTTTable {
public:
    explicit NTTTable(int coeff_count_power, const seal::Modulus &modulus);

    uint64_t get_root() const { return root_; }

    auto &get_from_root_powers() const { return root_powers_; }

    auto &get_from_root_powers_shoup() const { return root_powers_shoup_; }

    auto &get_from_inv_root_powers() const { return inv_root_powers_; }

    auto &get_from_inv_root_powers_shoup() const {
        return inv_root_powers_shoup_;
    }

    const uint64_t &inv_degree_modulo() const { return inv_degree_modulo_; }

    const uint64_t &inv_degree_modulo_shoup() const {
        return inv_degree_modulo_shoup_;
    }

    const seal::Modulus &modulus() const { return modulus_; }

    int coeff_count_power() const { return coeff_count_power_; }

    size_t coeff_count() const { return coeff_count_; }

private:
    std::uint64_t root_ = 0;
    std::uint64_t inv_root_ = 0;
    int coeff_count_power_ = 0;
    std::size_t coeff_count_ = 0;
    seal::Modulus modulus_;

    // Inverse of coeff_count_ modulo modulus_.
    uint64_t inv_degree_modulo_;
    uint64_t inv_degree_modulo_shoup_;

    // Holds 1~(n-1)-th powers of root_ in bit-reversed order, the 0-th power is
    // left unset.
    std::vector<uint64_t> root_powers_;
    std::vector<uint64_t> root_powers_shoup_;

    // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power
    // is left unset.
    std::vector<uint64_t> inv_root_powers_;
    std::vector<uint64_t> inv_root_powers_shoup_;
};

void nwt_2d_radix8_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables,
                                   size_t coeff_modulus_size,
                                   size_t start_modulus_idx);
void nwt_2d_radix8_backward_inplace(uint64_t *inout,
                                    const DNTTTable &ntt_tables,
                                    size_t coeff_modulus_size,
                                    size_t start_modulus_idx);

} // namespace hifive