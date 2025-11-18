#include "fp2.h"

// Comparison: lexicographic (c0 first, then c1)
__host__ __device__ int fp2_cmp(const Fp2& a, const Fp2& b) {
    int cmp0 = fp_cmp(a.c0, b.c0);
    if (cmp0 != 0) return cmp0;
    return fp_cmp(a.c1, b.c1);
}

// Check if a == 0
__host__ __device__ bool fp2_is_zero(const Fp2& a) {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}

// Check if a == 1
__host__ __device__ bool fp2_is_one(const Fp2& a) {
    return fp_is_one(a.c0) && fp_is_zero(a.c1);
}

// Set to zero
__host__ __device__ void fp2_zero(Fp2& a) {
    fp_zero(a.c0);
    fp_zero(a.c1);
}

// Set to one (1 + 0*i)
__host__ __device__ void fp2_one(Fp2& a) {
    fp_one(a.c0);
    fp_zero(a.c1);
}

// Copy: dst = src
__host__ __device__ void fp2_copy(Fp2& dst, const Fp2& src) {
    fp_copy(dst.c0, src.c0);
    fp_copy(dst.c1, src.c1);
}

// Addition: c = a + b
// (a0 + a1*i) + (b0 + b1*i) = (a0 + b0) + (a1 + b1)*i
__host__ __device__ void fp2_add(Fp2& c, const Fp2& a, const Fp2& b) {
    fp_add(c.c0, a.c0, b.c0);
    fp_add(c.c1, a.c1, b.c1);
}

// Subtraction: c = a - b
// (a0 + a1*i) - (b0 + b1*i) = (a0 - b0) + (a1 - b1)*i
__host__ __device__ void fp2_sub(Fp2& c, const Fp2& a, const Fp2& b) {
    fp_sub(c.c0, a.c0, b.c0);
    fp_sub(c.c1, a.c1, b.c1);
}

// Multiplication: c = a * b
// (a0 + a1*i) * (b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
// Uses Karatsuba-like optimization: (a0 + a1*i)*(b0 + b1*i) = a0*b0 - a1*b1 + (a0*b1 + a1*b0)*i
// We can optimize by computing: t0 = a0*b0, t1 = a1*b1, t2 = (a0 + a1)*(b0 + b1)
// Then: c0 = t0 - t1, c1 = t2 - t0 - t1
// NOTE: This assumes inputs are in normal form and converts to/from Montgomery
__host__ __device__ void fp2_mul(Fp2& c, const Fp2& a, const Fp2& b) {
    Fp t0, t1, t2, t3;
    
    // t0 = a0 * b0
    fp_mul(t0, a.c0, b.c0);
    
    // t1 = a1 * b1
    fp_mul(t1, a.c1, b.c1);
    
    // t2 = a0 + a1
    fp_add(t2, a.c0, a.c1);
    
    // t3 = b0 + b1
    fp_add(t3, b.c0, b.c1);
    
    // t2 = (a0 + a1) * (b0 + b1)
    fp_mul(t2, t2, t3);
    
    // c0 = a0*b0 - a1*b1 = t0 - t1
    fp_sub(c.c0, t0, t1);
    
    // c1 = a0*b1 + a1*b0 = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 = t2 - t0 - t1
    fp_sub(c.c1, t2, t0);
    fp_sub(c.c1, c.c1, t1);
}

// Montgomery multiplication: c = a * b (all in Montgomery form)
// (a0 + a1*i) * (b0 + b1*i) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i
// Uses Karatsuba-like optimization
// NOTE: All inputs and outputs are in Montgomery form
__host__ __device__ void fp2_mont_mul(Fp2& c, const Fp2& a, const Fp2& b) {
    Fp t0, t1, t2, t3;
    
    // t0 = a0 * b0 (Montgomery multiply)
    fp_mont_mul(t0, a.c0, b.c0);
    
    // t1 = a1 * b1 (Montgomery multiply)
    fp_mont_mul(t1, a.c1, b.c1);
    
    // t2 = a0 + a1
    fp_add(t2, a.c0, a.c1);
    
    // t3 = b0 + b1
    fp_add(t3, b.c0, b.c1);
    
    // t2 = (a0 + a1) * (b0 + b1) (Montgomery multiply)
    fp_mont_mul(t2, t2, t3);
    
    // c0 = a0*b0 - a1*b1 = t0 - t1
    fp_sub(c.c0, t0, t1);
    
    // c1 = a0*b1 + a1*b0 = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 = t2 - t0 - t1
    fp_sub(c.c1, t2, t0);
    fp_sub(c.c1, c.c1, t1);
}

// Squaring: c = a^2
// (a0 + a1*i)^2 = (a0^2 - a1^2) + 2*a0*a1*i
// Optimized: t0 = a0^2, t1 = a1^2, t2 = (a0 + a1)^2
// Then: c0 = t0 - t1, c1 = t2 - t0 - t1
__host__ __device__ void fp2_square(Fp2& c, const Fp2& a) {
    Fp t0, t1, t2;
    
    // t0 = a0^2
    fp_mul(t0, a.c0, a.c0);
    
    // t1 = a1^2
    fp_mul(t1, a.c1, a.c1);
    
    // t2 = a0 + a1
    fp_add(t2, a.c0, a.c1);
    
    // t2 = (a0 + a1)^2
    fp_mul(t2, t2, t2);
    
    // c0 = a0^2 - a1^2 = t0 - t1
    fp_sub(c.c0, t0, t1);
    
    // c1 = 2*a0*a1 = (a0 + a1)^2 - a0^2 - a1^2 = t2 - t0 - t1
    fp_sub(c.c1, t2, t0);
    fp_sub(c.c1, c.c1, t1);
}

// Negation: c = -a
// -(a0 + a1*i) = -a0 - a1*i
__host__ __device__ void fp2_neg(Fp2& c, const Fp2& a) {
    fp_neg(c.c0, a.c0);
    fp_neg(c.c1, a.c1);
}

// Conjugation: c = a.conjugate() = a0 - a1*i
__host__ __device__ void fp2_conjugate(Fp2& c, const Fp2& a) {
    fp_copy(c.c0, a.c0);
    fp_neg(c.c1, a.c1);
}

// Helper function: Modular inversion using Fermat's little theorem
// Computes a^(-1) = a^(p-2) mod p
__host__ __device__ void fp_inv_fermat(Fp& result, const Fp& a) {
    // Check for zero
    if (fp_is_zero(a)) {
        fp_zero(result);
        return;
    }
    
    // Get p-2
    Fp p_minus_2;
    Fp one, two;
    fp_one(one);
    fp_one(two);
    two.limb[0] = 2;
    const Fp& p = fp_modulus();
    fp_sub(p_minus_2, p, two);  // p - 2
    
    // Binary exponentiation: result = a^(p-2) mod p
    fp_one(result);
    Fp base;
    fp_copy(base, a);
    
    // Process bits from MSB to LSB
    bool started = false;
    for (int limb = FP_LIMBS - 1; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started || ((p_minus_2.limb[limb] >> bit) & 1)) {
                started = true;
                // Square result
                Fp temp;
                fp_mul(temp, result, result);
                fp_copy(result, temp);
                
                // Multiply by base if bit is set
                if ((p_minus_2.limb[limb] >> bit) & 1) {
                    fp_mul(temp, result, base);
                    fp_copy(result, temp);
                }
            }
        }
    }
}

// Inversion: c = a^(-1)
// Uses the formula: (a0 + a1*i)^(-1) = (a0 - a1*i) / (a0^2 + a1^2)
// We compute: norm = a0^2 + a1^2, then c = conjugate(a) / norm
__host__ __device__ void fp2_inv(Fp2& c, const Fp2& a) {
    // Check for zero (should not happen in practice, but handle gracefully)
    if (fp2_is_zero(a)) {
        fp2_zero(c);
        return;
    }
    
    Fp t0, t1, norm, norm_inv;
    
    // t0 = a0^2
    fp_mul(t0, a.c0, a.c0);
    
    // t1 = a1^2
    fp_mul(t1, a.c1, a.c1);
    
    // norm = a0^2 + a1^2
    fp_add(norm, t0, t1);
    
    // Compute norm^(-1) mod p using Fermat's little theorem
    fp_inv_fermat(norm_inv, norm);
    
    // Now compute c = conjugate(a) * norm_inv
    fp_mul(c.c0, a.c0, norm_inv);
    
    // c1 = -a1 * norm_inv
    fp_neg(c.c1, a.c1);
    fp_mul(c.c1, c.c1, norm_inv);
}

// Montgomery inversion: c = a^(-1) (all in Montgomery form)
// NOTE: All inputs and outputs are in Montgomery form
__host__ __device__ void fp2_mont_inv(Fp2& c, const Fp2& a) {
    // Check for zero
    if (fp2_is_zero(a)) {
        fp2_zero(c);
        return;
    }
    
    Fp t0, t1, norm, norm_inv;
    
    // t0 = a0^2 (Montgomery multiply)
    fp_mont_mul(t0, a.c0, a.c0);
    
    // t1 = a1^2 (Montgomery multiply)
    fp_mont_mul(t1, a.c1, a.c1);
    
    // norm = a0^2 + a1^2
    fp_add(norm, t0, t1);
    
    // norm_inv = 1 / norm (use Montgomery inversion)
    fp_mont_inv(norm_inv, norm);
    
    // c0 = a0 * norm_inv (Montgomery multiply)
    fp_mont_mul(c.c0, a.c0, norm_inv);
    
    // c1 = -a1 * norm_inv (Montgomery multiply)
    fp_neg(c.c1, a.c1);
    fp_mont_mul(c.c1, c.c1, norm_inv);
}

// Division: c = a / b = a * b^(-1)
__host__ __device__ void fp2_div(Fp2& c, const Fp2& a, const Fp2& b) {
    Fp2 b_inv;
    fp2_inv(b_inv, b);
    fp2_mul(c, a, b_inv);
}

// Conditional assignment: if condition, dst = src, else dst unchanged
__host__ __device__ void fp2_cmov(Fp2& dst, const Fp2& src, uint64_t condition) {
    fp_cmov(dst.c0, src.c0, condition);
    fp_cmov(dst.c1, src.c1, condition);
}

// Frobenius map: c = a^p
// For Fp2, the Frobenius map is: (a0 + a1*i)^p = a0 - a1*i = conjugate
// This is because i^p = i^(p mod 4) = i^(-1) = -i (since p ≡ 3 mod 4 for BLS12 curves)
__host__ __device__ void fp2_frobenius(Fp2& c, const Fp2& a) {
    fp2_conjugate(c, a);
}

// Multiply by i: c = a * i
// (a0 + a1*i) * i = -a1 + a0*i
__host__ __device__ void fp2_mul_by_i(Fp2& c, const Fp2& a) {
    Fp temp;
    fp_copy(temp, a.c0);
    fp_neg(c.c0, a.c1);
    fp_copy(c.c1, temp);
}

// Multiply by non-residue: For BLS12, non-residue is typically i
// This is the same as mul_by_i
__host__ __device__ void fp2_mul_by_non_residue(Fp2& c, const Fp2& a) {
    fp2_mul_by_i(c, a);
}

