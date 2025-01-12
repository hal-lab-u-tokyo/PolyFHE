#pragma once

#include <string>

namespace hifive {
class Config {
public:
    Config(std::string filename);
    int logN;
    int N;
    int L;
    int dnum;
    int SharedMemKB;
};

} // namespace hifive