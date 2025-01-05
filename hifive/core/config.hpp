#pragma once

#include <string>

namespace hifive {
class Config {
public:
    Config(std::string filename);
    int logN = 16;
    int N = 1 << logN;
    int L = 4;
    int SharedMemKB = 120;
};

} // namespace hifive