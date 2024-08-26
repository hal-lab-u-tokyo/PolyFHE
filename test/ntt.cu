#include <memory>

#include "test.h"

void test_ntt(size_t log_dim, size_t batch_size) {
    size_t dim = 1 << log_dim;
    const auto &s = cudaStreamPerThread;

    // generate modulus
    const auto h_modulus = phantom::arith::CoeffModulus::Create(
        dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; i++) {
        modulus.get()[i].set(h_modulus[i].value(),
                             h_modulus[i].const_ratio()[0],
                             h_modulus[i].const_ratio()[1]);
    }

    // generate NTT tables using Phantom
    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size, s);
    for (int i = 0; i < batch_size; i++) {
        auto h_ntt_table = phantom::arith::NTT(log_dim, h_modulus[i]);
        d_ntt_tables.set(&modulus.get()[i],
                         h_ntt_table.get_from_root_powers().data(),
                         h_ntt_table.get_from_root_powers_shoup().data(),
                         h_ntt_table.get_from_inv_root_powers().data(),
                         h_ntt_table.get_from_inv_root_powers_shoup().data(),
                         h_ntt_table.inv_degree_modulo(),
                         h_ntt_table.inv_degree_modulo_shoup(), i, s);
    }

    // create input
    auto h_data = std::make_unique<uint64_t[]>(batch_size * dim);
    for (size_t i = 0; i < batch_size * dim; i++) {
        h_data[i] = 2;
    }

    auto d_data =
        phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemcpyAsync(d_data.get(), h_data.get(),
                    batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice,
                    s);

    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    nwt_2d_radix8_backward_inplace(d_data.get(), d_ntt_tables, batch_size, 0,
                                   s);

    cudaMemcpyAsync(h_data.get(), d_data.get(),
                    batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                    s);
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < batch_size * dim; i++) {
        if (h_data.get()[i] != 2) {
            std::cout << i << " " << h_data.get()[i] << std::endl;
            throw std::logic_error("Error");
        }
    }
}

TEST(NTT, NTTSingle) {
    test_ntt(12, 1);
    test_ntt(13, 1);
    test_ntt(14, 1);
    test_ntt(15, 1);
    test_ntt(16, 1);
    test_ntt(17, 1);
}

TEST(NTT, NTTBatch) {
    test_ntt(12, 10);
    test_ntt(13, 10);
    test_ntt(14, 10);
    test_ntt(15, 10);
    test_ntt(16, 10);
    test_ntt(17, 10);
}