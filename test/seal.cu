#include <gtest/gtest.h>

// SEAL
#include "seal/seal.h"

TEST(SEAL, Param){
    // SEAL
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
}