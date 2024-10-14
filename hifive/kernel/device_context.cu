#include "hifive/core/logger.hpp"
#include "hifive/kernel/FullRNS-HEAAN/src/Context.h"
#include "hifive/kernel/device_context.hpp"

DeviceContext::DeviceContext() {
    LOG_INFO("Initializing DeviceContext\n");
    const int logN = 15;
    const int N = (1 << logN);
    const int L = 44;
    const uint64_t logp = 55;
    const uint64_t logSlots = 3;
    const uint64_t slots = (1 << logSlots);
    Context context(logN, logp, L, L + 1);
}