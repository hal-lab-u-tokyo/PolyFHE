#include "phantom.h"

#include <iostream>

int main(){
    std::cout << "Phantom example" << std::endl;
    phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
    return 0;
}