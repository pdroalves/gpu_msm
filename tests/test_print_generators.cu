#include "fp.h"
#include "curve.h"
#include <cstdio>
#include <gtest/gtest.h>

TEST(PrintGenerators, GetMontgomeryValues) {
    init_device_modulus(nullptr, 0);
    init_device_generators(nullptr, 0);
    
    G1Point g1 = g1_generator();
    G2Point g2 = g2_generator();
    
    fp_to_montgomery(g1.x, g1.x);
    fp_to_montgomery(g1.y, g1.y);
    fp_to_montgomery(g2.x.c0, g2.x.c0);
    fp_to_montgomery(g2.x.c1, g2.x.c1);
    fp_to_montgomery(g2.y.c0, g2.y.c0);
    fp_to_montgomery(g2.y.c1, g2.y.c1);
    
    printf("\n=== G1 Generator (Montgomery form) ===\n");
    printf("__constant__ const G1Point DEVICE_G1_GENERATOR_MONT = {\n");
    printf("    {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g1.x.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}, // x\n");
    printf("    {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g1.y.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}, // y\n");
    printf("    false // infinity\n");
    printf("};\n\n");
    
    printf("=== G2 Generator (Montgomery form) ===\n");
    printf("__constant__ const G2Point DEVICE_G2_GENERATOR_MONT = {\n");
    printf("    { // x\n");
    printf("        {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g2.x.c0.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}, // c0\n");
    printf("        {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g2.x.c1.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}  // c1\n");
    printf("    },\n");
    printf("    { // y\n");
    printf("        {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g2.y.c0.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}, // c0\n");
    printf("        {");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL", g2.y.c1.limb[i]);
        if (i < 6) printf(", ");
    }
    printf("}  // c1\n");
    printf("    },\n");
    printf("    false // infinity\n");
    printf("};\n");
}
