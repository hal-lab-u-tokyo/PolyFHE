#include "polyfhe/utils.hpp"

#include "polyfhe/core/logger.hpp"

namespace polyfhe {
uint64_t NTTSampleSize(const uint64_t logN) {
    if (logN == 12) {
        return 1 << 6;
    } else if (logN == 13) {
        return 1 << 7;
    } else if (logN == 14) {
        return 1 << 7;
    } else if (logN == 15) {
        return 1 << 8;
    } else if (logN == 16) {
        return 1 << 8;
    } else if (logN == 17) {
        return 1 << 9;
    } else {
        LOG_ERROR("Invalid logN: %ld\n", logN);
        return 0;
    }
}

} // namespace polyfhe