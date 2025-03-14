#include "polyfhe/utils.hpp"

#include "polyfhe/core/logger.hpp"

namespace polyfhe {
uint64_t NTTSampleSize(const uint64_t logN) { return 1 << (logN / 2); }

} // namespace polyfhe