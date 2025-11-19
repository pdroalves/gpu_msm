#include "curve.h"
#include "device.h"
#include <stdio.h>

int main() {
    // G1 generator is provided in standard form on the host
    G1Point G = g1_generator();
    G1Point G_mont = G;
    fp_to_montgomery(G_mont.x, G_mont.x);
    fp_to_montgomery(G_mont.y, G_mont.y);
    
    printf("Generator (Montgomery form):\n");
    printf("  x[0] = 0x%016llx\n", G_mont.x.limb[0]);
    printf("  y[0] = 0x%016llx\n", G_mont.y.limb[0]);
    
    // Convert to normal form
    G1Point G_normal = G;
    G_normal.infinity = G.infinity;
    
    printf("\nGenerator (normal form):\n");
    printf("  x[0] = 0x%016llx\n", G_normal.x.limb[0]);
    printf("  y[0] = 0x%016llx\n", G_normal.y.limb[0]);
    
    // Compute 15G = (1+2+3+4+5)G using scalar multiplication
    uint64_t scalar = 15;
    G1Point result_mont;
    point_scalar_mul(result_mont, G_mont, &scalar, 1);
    
    printf("\n15G (Montgomery form):\n");
    printf("  x[0] = 0x%016llx\n", result_mont.x.limb[0]);
    printf("  y[0] = 0x%016llx\n", result_mont.y.limb[0]);
    
    // Convert to normal form
    G1Point result_normal;
    fp_from_montgomery(result_normal.x, result_mont.x);
    fp_from_montgomery(result_normal.y, result_mont.y);
    result_normal.infinity = result_mont.infinity;
    
    printf("\n15G (normal form):\n");
    printf("  x[0] = 0x%016llx\n", result_normal.x.limb[0]);
    printf("  x[1] = 0x%016llx\n", result_normal.x.limb[1]);
    printf("  x[2] = 0x%016llx\n", result_normal.x.limb[2]);
    printf("  x[3] = 0x%016llx\n", result_normal.x.limb[3]);
    printf("  x[4] = 0x%016llx\n", result_normal.x.limb[4]);
    printf("  x[5] = 0x%016llx\n", result_normal.x.limb[5]);
    printf("  x[6] = 0x%016llx\n", result_normal.x.limb[6]);
    printf("  y[0] = 0x%016llx\n", result_normal.y.limb[0]);
    
    return 0;
}

