#include "hifive/core/config.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace hifive {

Config::Config(std::string filename) {
    // Read config file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string key;
        int value;

        if (std::getline(ss, key, ',') && ss >> value) {
            std::cout << "Key: " << key << ", Value: " << value << std::endl;
            if (key == "logN") {
                logN = value;
                N = 1 << logN;
            } else if (key == "N") {
                N = value;
            } else if (key == "L") {
                L = value;
            } else if (key == "dnum") {
                dnum = value;
            } else if (key == "SharedMemKB") {
                SharedMemKB = value;
            }
        }
    }
    file.close();

    alpha = std::ceil((L + 1) / dnum);
    k = alpha;
}

} // namespace hifive