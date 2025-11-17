#pragma once

#include "fp.h"

// CUDA kernel declarations and host wrappers

// Host wrapper: Add two arrays element-wise
// c[i] = a[i] + b[i] mod p
void fp_add_array_host(Fp* c, const Fp* a, const Fp* b, int n);

// Host wrapper: Multiply two arrays element-wise
// c[i] = a[i] * b[i] mod p
void fp_mul_array_host(Fp* c, const Fp* a, const Fp* b, int n);

