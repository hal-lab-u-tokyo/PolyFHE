#include "test/test.hpp"

int main() {
    LOG_INFO("Start testing...\n");

    /*
        test_poly_add(context, 1 << 14, 20, 1 << 7, 20);
        test_poly_add(context, 1 << 15, 20, 1 << 7, 20);
        test_poly_add(context, 1 << 16, 20, 1 << 8, 20);
        test_poly_add(context, 1 << 17, 20, 1 << 8, 20);

        test_poly_mult(context, 1 << 14, 20, 1 << 7, 20);
        test_poly_mult(context, 1 << 15, 20, 1 << 7, 20);
        test_poly_mult(context, 1 << 16, 20, 1 << 8, 20);
        test_poly_mult(context, 1 << 17, 20, 1 << 8, 20);
        */

    /*
    {
        FHEContext context(17, 20);
        test_ntt(context, 17, 20);
    }
    */
    {
        FHEContext context(12, 2);
        test_ntt(context, 12, 2);
    }
}