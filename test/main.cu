#include "hifive/core/logger.hpp"
#include "hifive/kernel/device_context.hpp"

void test_poly_add(DeviceContext *dc, const int N, const int L,
                   const int block_x, const int block_y);

int main() {
    LOG_INFO("Start testing...\n");
    DeviceContext dc;

    test_poly_add(&dc, 1 << 14, 20, 1 << 7, 20);
    test_poly_add(&dc, 1 << 15, 20, 1 << 7, 20);
    test_poly_add(&dc, 1 << 16, 20, 1 << 8, 20);
    test_poly_add(&dc, 1 << 17, 20, 1 << 8, 20);
}