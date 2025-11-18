#pragma once

#include "fp2.h"

// CUDA kernel declarations and host wrappers for Fp2

// Host wrapper: Add two arrays element-wise
// c[i] = a[i] + b[i]
void fp2_add_array_host(Fp2* c, const Fp2* a, const Fp2* b, int n);

// Host wrapper: Multiply two arrays element-wise
// c[i] = a[i] * b[i]
void fp2_mul_array_host(Fp2* c, const Fp2* a, const Fp2* b, int n);

