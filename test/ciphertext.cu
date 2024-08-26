#include "test.h"

TEST(Ciphertext, New) {
    const uint64_t n = 40;
    const uint64_t l = 20;
    hifive::Ciphertext ct(n, l);
    check_can_access<<<1, 1>>>(ct.ax(), 0);
    check_can_access<<<1, 1>>>(ct.ax(), n * l - 1);
    check_can_access<<<1, 1>>>(ct.bx(), 0);
    check_can_access<<<1, 1>>>(ct.bx(), n * l - 1);
}
