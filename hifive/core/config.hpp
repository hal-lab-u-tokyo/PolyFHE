#pragma once

#include <string>

namespace hifive {
class Config {
public:
    Config(std::string filename);
    int logN;
    int N;
    int L;
    int k;
    int alpha;
    int dnum;
    int SharedMemKB;
};

} // namespace hifive