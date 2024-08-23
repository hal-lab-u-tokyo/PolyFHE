#include <gtest/gtest.h>

#include <vector>

#include "example.h"
#include "phantom.h"
#include "polynomial.h"

void test_ntt(int log_dim) {
    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();
    size_t dim = 1 << log_dim;
    size_t batch_size = 1;

    // generate modulus in host
    const auto h_modulus = phantom::arith::CoeffModulus::Create(
        dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; i++) {
        modulus.get()[i].set(h_modulus[i].value(),
                             h_modulus[i].const_ratio()[0],
                             h_modulus[i].const_ratio()[1]);
    }

    auto twiddles =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto twiddles_shoup =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto itwiddles =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto itwiddles_shoup =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto d_n_inv_mod_q =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);
    auto d_n_inv_mod_q_shoup =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);

    for (size_t i = 0; i < batch_size; i++) {
        // generate twiddles in host
        auto h_ntt_table = phantom::arith::NTT(log_dim, h_modulus[i]);
        // copy twiddles to device
        cudaMemcpyAsync(twiddles.get() + i * dim,
                        h_ntt_table.get_from_root_powers().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(twiddles_shoup.get() + i * dim,
                        h_ntt_table.get_from_root_powers_shoup().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(itwiddles.get() + i * dim,
                        h_ntt_table.get_from_inv_root_powers().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(itwiddles_shoup.get() + i * dim,
                        h_ntt_table.get_from_inv_root_powers_shoup().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_n_inv_mod_q.get() + i,
                        &h_ntt_table.inv_degree_modulo(), sizeof(uint64_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_n_inv_mod_q_shoup.get() + i,
                        &h_ntt_table.inv_degree_modulo_shoup(),
                        sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    }

    // create input
    auto h_idata = std::make_unique<uint64_t[]>(batch_size * dim);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < dim; j++) {
            h_idata.get()[i * dim + j] = j;
        }
    }

    auto d_data =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemcpyAsync(d_data.get(), h_idata.get(),
                    batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice,
                    s);

    // Verify
    auto h_odata = std::make_unique<uint64_t[]>(batch_size * dim);
    cudaMemcpyAsync(h_odata.get(), d_data.get(),
                    batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                    s);
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < batch_size * dim; i++) {
        std::cout << i << " " << h_idata.get()[i] << " != " << h_idata.get()[i]
                  << std::endl;
        if (h_idata.get()[i] != h_odata.get()[i]) {
            std::cout << i << " " << h_idata.get()[i]
                      << " != " << h_idata.get()[i] << std::endl;
            throw std::logic_error("Error");
        }
    }
}

TEST(NTTTest, NTT) { test_ntt(15); }
