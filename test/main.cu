#include "hifive/core/logger.hpp"
#include "hifive/kernel/device_context.hpp"

int test_poly_add(DeviceContext *dc);

int main() {
    LOG_INFO("Start testing...\n");
    DeviceContext dc;

    if (test_poly_add(&dc) == 0) {
        LOG_INFO("test_poly_add...Passed.\n");
    } else {
        LOG_INFO("test_poly_add...Failed.\n");
    }
}