#pragma once

#include <string>

namespace polyfhe {
class Config {
public:
    Config(std::string filename);
    int logN;
    int N;
    int n1;
    int n2;
    int L;
    int k;
    int alpha;
    int dnum;
    int SharedMemKB;
};

} // namespace polyfhe