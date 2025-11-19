
// Temporary test to print generator values
TEST(CurveTest, PrintGeneratorValues) {
    // Access via the public function (standard form on host)
    G1Point g1 = g1_generator();
    G2Point g2 = g2_generator();
    
    // Convert to Montgomery form for diagnostics
    fp_to_montgomery(g1.x, g1.x);
    fp_to_montgomery(g1.y, g1.y);
    fp_to_montgomery(g2.x.c0, g2.x.c0);
    fp_to_montgomery(g2.x.c1, g2.x.c1);
    fp_to_montgomery(g2.y.c0, g2.y.c0);
    fp_to_montgomery(g2.y.c1, g2.y.c1);
    
    printf("\n=== G1 Generator (Montgomery form) ===\n");
    printf("X: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g1.x.limb[i], i < 6 ? ", " : "\n");
    }
    printf("Y: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g1.y.limb[i], i < 6 ? ", " : "\n");
    }
    
    printf("\n=== G2 Generator (Montgomery form) ===\n");
    printf("X.c0: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g2.x.c0.limb[i], i < 6 ? ", " : "\n");
    }
    printf("X.c1: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g2.x.c1.limb[i], i < 6 ? ", " : "\n");
    }
    printf("Y.c0: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g2.y.c0.limb[i], i < 6 ? ", " : "\n");
    }
    printf("Y.c1: ");
    for (int i = 0; i < 7; i++) {
        printf("0x%016llxULL%s", g2.y.c1.limb[i], i < 6 ? ", " : "\n");
    }
    printf("\n");
}
