#include "fp.h"
#include "device.h"
#include <cstring>
#include <cstdio>

// Prime modulus p for BLS12-446 (Fq field)
// From tfhe-rs/tfhe-zk-pok/src/curve_446/mod.rs
// Modulus: 172824703542857155980071276579495962243492693522789898437834836356385656662277472896902502740297183690175962001546428467344062165330603
// This is a 448-bit prime (Fp448)
// Format: little-endian, limb[0] is least significant
// Converted from decimal to hex and split into 7 limbs of 64 bits each

// For CUDA device code, we use __constant__ memory
// Constants are hardcoded at compile time (like sppark) to avoid cudaMemcpyToSymbol
__constant__ const Fp DEVICE_MODULUS = {
    {0x311c0026aab0aaabULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
     0x0f75a64bbac71602ULL, 0x0095a4b78a02fe32ULL, 0x200fc34965aad640ULL,
     0x3cdee0fb28c5e535ULL}
};

__constant__ const Fp DEVICE_R2 = {
    {0x2AFF01DDDC752B45ULL, 0x92C772A7421CCF5BULL, 0x140EEF29C347DAD6ULL,
     0xF5A1400C22EA595EULL, 0x99D91C9FEC145218ULL, 0x3BB6537F90143D4BULL,
     0x3627854C9BE7974FULL}
};

__constant__ const Fp DEVICE_R_INV = {
    {0xCE2560B51652D82FULL, 0xA0166C2F90C0838EULL, 0x6C2028836577CA52ULL,
     0x28BE97CD54A76C2CULL, 0x0C01F5F4B5806D69ULL, 0x498338A6A4F43367ULL,
     0x32E6A14BC7F5FA16ULL}
};

__constant__ const uint64_t DEVICE_P_PRIME = 0xcd63fd900035fffdULL;

// Host-side helper functions removed - values are now hardcoded directly
// in the accessor functions (fp_modulus, fp_r2, fp_r_inv, fp_p_prime)


// Helper to get modulus reference
// On device: returns DEVICE_MODULUS from constant memory
// On host: returns a hardcoded copy (same as device)
__host__ __device__ const Fp& fp_modulus() {
#ifdef __CUDA_ARCH__
    return DEVICE_MODULUS;
#else
    static const Fp host_mod = {
        {0x311c0026aab0aaabULL, 0x56ee4528c573b5ccULL, 0x824e6dc3e23acdeeULL,
         0x0f75a64bbac71602ULL, 0x0095a4b78a02fe32ULL, 0x200fc34965aad640ULL,
         0x3cdee0fb28c5e535ULL}
    };
    return host_mod;
#endif
}

// Helper to get R2 reference
__host__ __device__ const Fp& fp_r2() {
#ifdef __CUDA_ARCH__
    return DEVICE_R2;
#else
    static const Fp host_r2 = {
        {0x2AFF01DDDC752B45ULL, 0x92C772A7421CCF5BULL, 0x140EEF29C347DAD6ULL,
         0xF5A1400C22EA595EULL, 0x99D91C9FEC145218ULL, 0x3BB6537F90143D4BULL,
         0x3627854C9BE7974FULL}
    };
    return host_r2;
#endif
}

// Helper to get R_INV reference
__host__ __device__ const Fp& fp_r_inv() {
#ifdef __CUDA_ARCH__
    return DEVICE_R_INV;
#else
    static const Fp host_r_inv = {
        {0xCE2560B51652D82FULL, 0xA0166C2F90C0838EULL, 0x6C2028836577CA52ULL,
         0x28BE97CD54A76C2CULL, 0x0C01F5F4B5806D69ULL, 0x498338A6A4F43367ULL,
         0x32E6A14BC7F5FA16ULL}
    };
    return host_r_inv;
#endif
}

// Helper to get p' (Montgomery constant)
__host__ __device__ uint64_t fp_p_prime() {
#ifdef __CUDA_ARCH__
    return DEVICE_P_PRIME;
#else
    return 0xcd63fd900035fffdULL;
#endif
}

__host__ __device__ int fp_cmp(const Fp& a, const Fp& b) {
    for (int i = FP_LIMBS - 1; i >= 0; i--) {
        if (a.limb[i] > b.limb[i]) return 1;
        if (a.limb[i] < b.limb[i]) return -1;
    }
    return 0;
}

__host__ __device__ bool fp_is_zero(const Fp& a) {
    for (int i = 0; i < FP_LIMBS; i++) {
        if (a.limb[i] != 0) return false;
    }
    return true;
}

__host__ __device__ bool fp_is_one(const Fp& a) {
    if (a.limb[0] != 1) return false;
    for (int i = 1; i < FP_LIMBS; i++) {
        if (a.limb[i] != 0) return false;
    }
    return true;
}

__host__ __device__ void fp_zero(Fp& a) {
    for (int i = 0; i < FP_LIMBS; i++) {
        a.limb[i] = 0;
    }
}

__host__ __device__ void fp_one(Fp& a) {
    // Return 1 in Montgomery form: 1 * R mod p = R mod p
    // But for compatibility, we return normal 1
    // For Montgomery form, use fp_to_montgomery after setting to 1
    a.limb[0] = 1;
    for (int i = 1; i < FP_LIMBS; i++) {
        a.limb[i] = 0;
    }
}

// Set to one in Montgomery form: R mod p
__host__ __device__ void fp_one_montgomery(Fp& a) {
    const Fp& r2 = fp_r2();
    // R mod p = R^2 * R_INV mod p
    // Actually, R mod p is just the Montgomery form of 1
    // We can compute it by converting 1 to Montgomery
    Fp one;
    fp_one(one);
    fp_to_montgomery(a, one);
}

__host__ __device__ void fp_copy(Fp& dst, const Fp& src) {
    for (int i = 0; i < FP_LIMBS; i++) {
        dst.limb[i] = src.limb[i];
    }
}

// Addition with carry propagation
__host__ __device__ uint64_t fp_add_raw(Fp& c, const Fp& a, const Fp& b) {
    uint64_t carry = 0;
    
    for (int i = 0; i < FP_LIMBS; i++) {
        // Add with carry: c = a + b + carry
        uint64_t sum = a.limb[i] + carry;
        carry = (sum < a.limb[i]) ? 1 : 0;  // Check for overflow
        sum += b.limb[i];
        carry += (sum < b.limb[i]) ? 1 : 0;  // Check for overflow
        c.limb[i] = sum;
    }
    
    return carry;
}

// Subtraction with borrow propagation
__host__ __device__ uint64_t fp_sub_raw(Fp& c, const Fp& a, const Fp& b) {
    uint64_t borrow = 0;
    
    for (int i = 0; i < FP_LIMBS; i++) {
        // Subtract with borrow: c = a - b - borrow
        uint64_t diff = a.limb[i] - borrow;
        borrow = (diff > a.limb[i]) ? 1 : 0;  // Check for underflow
        uint64_t old_diff = diff;
        diff -= b.limb[i];
        borrow += (diff > old_diff) ? 1 : 0;  // Check for underflow
        c.limb[i] = diff;
    }
    
    return borrow;
}

// Addition with modular reduction
__host__ __device__ void fp_add(Fp& c, const Fp& a, const Fp& b) {
    Fp sum;
    uint64_t carry = fp_add_raw(sum, a, b);
    
    // If there's a carry or sum >= MODULUS, we need to reduce
    const Fp& p = fp_modulus();
    if (carry || fp_cmp(sum, p) >= 0) {
        Fp reduced;
        fp_sub_raw(reduced, sum, p);
        fp_copy(c, reduced);
    } else {
        fp_copy(c, sum);
    }
}

// Subtraction with modular reduction
__host__ __device__ void fp_sub(Fp& c, const Fp& a, const Fp& b) {
    Fp diff;
    uint64_t borrow = fp_sub_raw(diff, a, b);
    
    // If there was a borrow, we need to add MODULUS
    const Fp& p = fp_modulus();
    if (borrow) {
        fp_add_raw(c, diff, p);
    } else {
        fp_copy(c, diff);
    }
}

// Helper function for 64x64 -> 128 bit multiply
// Returns (hi, lo) as a struct would, but we use output parameters
__host__ __device__ inline void mul64x64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#ifdef __CUDA_ARCH__
    // Use CUDA intrinsics for device code
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    // Host code: use __uint128_t if available, otherwise manual implementation
#ifdef __SIZEOF_INT128__
    __uint128_t product = (__uint128_t)a * (__uint128_t)b;
    lo = (uint64_t)product;
    hi = (uint64_t)(product >> 64);
#else
    // Fallback for systems without __uint128_t
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;
    
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    
    uint64_t mid1 = p1 + (p0 >> 32);
    uint64_t carry1 = (mid1 < p1) ? 1 : 0;
    
    uint64_t mid2 = mid1 + p2;
    uint64_t carry2 = (mid2 < mid1) ? 1 : 0;
    
    lo = (p0 & 0xFFFFFFFFULL) | (mid2 << 32);
    hi = p3 + (mid2 >> 32) + (carry1 << 32) + (carry2 << 32);
#endif
#endif
}

// Multiplication using schoolbook method
// Result is stored in c[0..2*FP_LIMBS-1]
__host__ __device__ void fp_mul_raw(uint64_t* c, const Fp& a, const Fp& b) {
    // Initialize result to zero
    for (int i = 0; i < 2 * FP_LIMBS; i++) {
        c[i] = 0;
    }
    
    // Schoolbook multiplication: c[i+j] += a[i] * b[j]
    for (int i = 0; i < FP_LIMBS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < FP_LIMBS; j++) {
            // Multiply a[i] * b[j] to get 128-bit result
            uint64_t lo, hi;
            mul64x64(a.limb[i], b.limb[j], hi, lo);
            
            // Add lo to c[i+j]
            uint64_t sum_lo = c[i + j] + lo;
            uint64_t carry_lo = (sum_lo < c[i + j]) ? 1 : 0;
            c[i + j] = sum_lo;
            
            // Add hi + carry + carry_lo to c[i + j + 1]
            uint64_t old_val = c[i + j + 1];
            uint64_t sum1 = old_val + hi;
            uint64_t carry1 = (sum1 < old_val) ? 1 : 0;
            
            uint64_t sum2 = sum1 + carry;
            uint64_t carry2 = (sum2 < sum1) ? 1 : 0;
            
            uint64_t sum3 = sum2 + carry_lo;
            uint64_t carry3 = (sum3 < sum2) ? 1 : 0;
            
            c[i + j + 1] = sum3;
            carry = carry1 + carry2 + carry3;
        }
        
        // Propagate remaining carry through higher limbs
        int idx = i + FP_LIMBS;
        while (carry && idx < 2 * FP_LIMBS) {
            uint64_t sum = c[idx] + carry;
            carry = (sum < c[idx]) ? 1 : 0;
            c[idx] = sum;
            idx++;
        }
    }
}

// Modular reduction (legacy - kept for compatibility)
// Input a is 2*FP_LIMBS limbs (result of multiplication)
// Output c is FP_LIMBS limbs, reduced mod p
__host__ __device__ void fp_reduce(Fp& c, const uint64_t* a) {
    // Use Montgomery reduction
    fp_mont_reduce(c, a);
}

// Montgomery reduction: c = (a * R_INV) mod p
// Input a is 2*FP_LIMBS limbs (result of multiplication)
// Output c is FP_LIMBS limbs in Montgomery form
// Algorithm: Standard Montgomery reduction for R = 2^448
__host__ __device__ void fp_mont_reduce(Fp& c, const uint64_t* a) {
    const Fp& p = fp_modulus();
    uint64_t p_prime = fp_p_prime();
    
    // Working array: copy input
    uint64_t t[2 * FP_LIMBS + 1];
    for (int i = 0; i < 2 * FP_LIMBS; i++) {
        t[i] = a[i];
    }
    t[2 * FP_LIMBS] = 0;
    
    // Montgomery reduction: for each limb, compute u = t[i] * p' mod 2^64
    // then add u * p to t, which zeros out t[i]
    for (int i = 0; i < FP_LIMBS; i++) {
        uint64_t u = t[i] * p_prime;  // u = t[i] * p' mod 2^64
        
        // Add u * p to t, starting at position i
        uint64_t carry = 0;
        for (int j = 0; j < FP_LIMBS; j++) {
            uint64_t hi, lo;
            mul64x64(u, p.limb[j], hi, lo);
            
            // Three-way addition: t[i+j] + lo + carry
            // Do it in two steps to handle carries properly
            uint64_t temp = t[i + j] + lo;
            uint64_t carry1 = (temp < t[i + j]) ? 1 : 0;
            
            uint64_t sum = temp + carry;
            uint64_t carry2 = (sum < temp) ? 1 : 0;
            
            t[i + j] = sum;
            
            // Next carry is hi + carry1 + carry2
            carry = hi + carry1 + carry2;
        }
        
        // Propagate remaining carry
        int idx = i + FP_LIMBS;
        while (carry != 0 && idx <= 2 * FP_LIMBS) {
            uint64_t sum = t[idx] + carry;
            carry = (sum < t[idx]) ? 1 : 0;
            t[idx] = sum;
            idx++;
        }
    }
    
    // Result is in t[FP_LIMBS..2*FP_LIMBS-1] (high half)
    // But we also need to check if there's a carry into t[2*FP_LIMBS]
    // Copy to output
    for (int i = 0; i < FP_LIMBS; i++) {
        c.limb[i] = t[i + FP_LIMBS];
    }
    
    // Final reduction: if c >= p, subtract p
    // Also handle any carry that might have gone into t[2*FP_LIMBS]
    if (t[2 * FP_LIMBS] != 0 || fp_cmp(c, p) >= 0) {
        Fp reduced;
        fp_sub_raw(reduced, c, p);
        fp_copy(c, reduced);
    }
}

// Montgomery multiplication: c = (a * b * R_INV) mod p
// Both a and b are in Montgomery form, result is in Montgomery form
__host__ __device__ void fp_mont_mul(Fp& c, const Fp& a, const Fp& b) {
    uint64_t product[2 * FP_LIMBS];
    fp_mul_raw(product, a, b);
    fp_mont_reduce(c, product);
}

// Convert to Montgomery form: c = (a * R) mod p
// Input a is in normal form, output c is in Montgomery form
__host__ __device__ void fp_to_montgomery(Fp& c, const Fp& a) {
    // c = a * R mod p = a * R^2 * R_INV mod p
    // First compute a * R^2, then reduce
    uint64_t product[2 * FP_LIMBS];
    const Fp& r2 = fp_r2();
    fp_mul_raw(product, a, r2);
    fp_mont_reduce(c, product);
}

// Convert from Montgomery form: c = (a * R_INV) mod p
// Input a is in Montgomery form, output c is in normal form
__host__ __device__ void fp_from_montgomery(Fp& c, const Fp& a) {
    // To convert from Montgomery form, we use Montgomery reduction directly!
    // If a represents (value * R) mod p, then Montgomery reducing [a, 0, 0, ..., 0]
    // gives us (a * R_INV) mod p = value mod p
    
    // Create double-width array with a in low part, zeros in high part
    uint64_t extended[2 * FP_LIMBS];
    for (int i = 0; i < FP_LIMBS; i++) {
        extended[i] = a.limb[i];
    }
    for (int i = FP_LIMBS; i < 2 * FP_LIMBS; i++) {
        extended[i] = 0;
    }
    
    // Montgomery reduction: computes (extended * R_INV) mod p = (a * R_INV) mod p = value mod p
    fp_mont_reduce(c, extended);
}

// Batch conversion to Montgomery form
__host__ __device__ void fp_to_montgomery_batch(Fp* dst, const Fp* src, int n) {
    for (int i = 0; i < n; i++) {
        fp_to_montgomery(dst[i], src[i]);
    }
}

// Batch conversion from Montgomery form
__host__ __device__ void fp_from_montgomery_batch(Fp* dst, const Fp* src, int n) {
    for (int i = 0; i < n; i++) {
        fp_from_montgomery(dst[i], src[i]);
    }
}

// Multiplication with modular reduction
// NOTE: This assumes inputs are in normal form and converts to/from Montgomery
// For better performance, keep values in Montgomery form and use fp_mont_mul
__host__ __device__ void fp_mul(Fp& c, const Fp& a, const Fp& b) {
    // Convert to Montgomery, multiply, convert back
    Fp a_mont, b_mont;
    fp_to_montgomery(a_mont, a);
    fp_to_montgomery(b_mont, b);
    
    Fp c_mont;
    fp_mont_mul(c_mont, a_mont, b_mont);
    fp_from_montgomery(c, c_mont);
}

// Negation: c = -a mod p = p - a
__host__ __device__ void fp_neg(Fp& c, const Fp& a) {
    if (fp_is_zero(a)) {
        fp_zero(c);
    } else {
        const Fp& p = fp_modulus();
        fp_sub(c, p, a);
    }
}

// Exponentiation by squaring (helper for inversion and pow)
// Computes base^exp mod p where exp is a big integer
// Uses Montgomery form internally for efficiency
__host__ __device__ static void fp_pow_internal(Fp& result, const Fp& base, const uint64_t* exp, int exp_limbs) {
    // Convert base to Montgomery form
    Fp base_mont;
    fp_to_montgomery(base_mont, base);
    
    // Result starts as 1 in Montgomery form
    Fp one;
    fp_one(one);
    fp_to_montgomery(result, one);
    
    // Find the most significant bit
    int msb_idx = exp_limbs - 1;
    while (msb_idx >= 0 && exp[msb_idx] == 0) {
        msb_idx--;
    }
    
    if (msb_idx < 0) {
        // Exponent is zero, result is 1
        fp_one(one);
        fp_to_montgomery(result, one);
        // result is already in Montgomery form, convert back to normal
        Fp result_normal;
        fp_from_montgomery(result_normal, result);
        fp_copy(result, result_normal);
        return;
    }
    
    // Find the most significant bit in the highest non-zero limb
    uint64_t msb_val = exp[msb_idx];
    int bit_pos = 63;
    while (bit_pos >= 0 && ((msb_val >> bit_pos) & 1) == 0) {
        bit_pos--;
    }
    
    // Square-and-multiply algorithm
    for (int limb_idx = msb_idx; limb_idx >= 0; limb_idx--) {
        int start_bit = (limb_idx == msb_idx) ? bit_pos : 63;
        
        for (int bit = start_bit; bit >= 0; bit--) {
            // Square result
            Fp temp;
            fp_mont_mul(temp, result, result);
            fp_copy(result, temp);
            
            // Multiply by base if current bit is set
            if ((exp[limb_idx] >> bit) & 1) {
                fp_mont_mul(temp, result, base_mont);
                fp_copy(result, temp);
            }
        }
    }
    
    // Convert result back from Montgomery form
    Fp result_normal;
    fp_from_montgomery(result_normal, result);
    fp_copy(result, result_normal);
}

// Exponentiation with 64-bit exponent
__host__ __device__ void fp_pow_u64(Fp& c, const Fp& a, uint64_t e) {
    uint64_t exp_array[1] = {e};
    fp_pow_internal(c, a, exp_array, 1);
}

// Exponentiation with big integer exponent
__host__ __device__ void fp_pow(Fp& c, const Fp& a, const uint64_t* e, int e_limbs) {
    // Limit to FP_LIMBS to avoid issues
    int actual_limbs = (e_limbs > FP_LIMBS) ? FP_LIMBS : e_limbs;
    fp_pow_internal(c, a, e, actual_limbs);
}

// Inversion: c = a^(-1) mod p
// Uses Fermat's little theorem: a^(p-2) = a^(-1) mod p
// NOTE: Assumes input is in normal form and converts to/from Montgomery
__host__ __device__ void fp_inv(Fp& c, const Fp& a) {
    if (fp_is_zero(a)) {
        // Division by zero: return 0
        fp_zero(c);
        return;
    }
    
    // Compute a^(p-2) mod p
    const Fp& p = fp_modulus();
    
    // p - 2 in little-endian format
    Fp p_minus_2;
    Fp two;
    fp_one(two);
    two.limb[0] = 2;
    fp_sub(p_minus_2, p, two);
    
    // Compute a^(p-2) mod p
    fp_pow_internal(c, a, p_minus_2.limb, FP_LIMBS);
}

// Montgomery inversion: c = a^(-1) mod p (all in Montgomery form)
// NOTE: Input and output are in Montgomery form (no conversions)
__host__ __device__ void fp_mont_inv(Fp& c, const Fp& a) {
    if (fp_is_zero(a)) {
        fp_zero(c);
        return;
    }
    
    // Convert from Montgomery to normal form for fp_pow_internal
    Fp a_normal;
    fp_from_montgomery(a_normal, a);
    
    // Compute a_normal^(p-2) mod p (fp_pow_internal handles conversions)
    const Fp& p = fp_modulus();
    Fp p_minus_2;
    Fp two;
    fp_one(two);
    two.limb[0] = 2;
    fp_sub(p_minus_2, p, two);
    
    Fp result_normal;
    fp_pow_internal(result_normal, a_normal, p_minus_2.limb, FP_LIMBS);
    
    // Convert result back to Montgomery form
    fp_to_montgomery(c, result_normal);
}

// Division: c = a / b mod p = a * b^(-1) mod p
__host__ __device__ void fp_div(Fp& c, const Fp& a, const Fp& b) {
    if (fp_is_zero(b)) {
        // Division by zero: return 0
        fp_zero(c);
        return;
    }
    
    Fp b_inv;
    fp_inv(b_inv, b);
    fp_mul(c, a, b_inv);
}

// Helper: Divide Fp by 2 (right shift by 1)
__host__ __device__ static void fp_div_by_2(Fp& result, const Fp& a) {
    uint64_t carry = 0;
    for (int i = FP_LIMBS - 1; i >= 0; i--) {
        uint64_t new_val = (a.limb[i] >> 1) | (carry << 63);
        carry = a.limb[i] & 1;
        result.limb[i] = new_val;
    }
}

// Helper: Divide Fp by 4 (right shift by 2)
__host__ __device__ static void fp_div_by_4(Fp& result, const Fp& a) {
    // Divide by 4 by calling fp_div_by_2 twice
    // This is simpler and more reliable than trying to handle remainders directly
    Fp temp;
    fp_div_by_2(temp, a);    // a / 2
    fp_div_by_2(result, temp);  // (a / 2) / 2 = a / 4
}

// Check if a is a quadratic residue using Euler's criterion
// Returns true if a^((p-1)/2) = 1 mod p
__host__ __device__ bool fp_is_quadratic_residue(const Fp& a) {
    if (fp_is_zero(a)) {
        return true;  // 0 is a quadratic residue (0^2 = 0)
    }
    
    const Fp& p = fp_modulus();
    
    // Compute (p-1)/2
    Fp p_minus_1;
    Fp one;
    fp_one(one);
    fp_sub(p_minus_1, p, one);
    
    // Divide by 2 using helper
    Fp exp_direct;
    fp_div_by_2(exp_direct, p_minus_1);
    
    // Compute a^((p-1)/2) mod p
    Fp result;
    fp_pow_internal(result, a, exp_direct.limb, FP_LIMBS);
    
    // If result == 1, a is a quadratic residue
    return fp_is_one(result);
}

// Square root computation
// For primes p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4) mod p (if a is a quadratic residue)
// BLS12-446 has p ≡ 3 (mod 4), so we use the fast method
__host__ __device__ bool fp_sqrt(Fp& c, const Fp& a) {
    if (fp_is_zero(a)) {
        fp_zero(c);
        return true;  // sqrt(0) = 0
    }
    
    // Check if a is a quadratic residue
    if (!fp_is_quadratic_residue(a)) {
        fp_zero(c);
        return false;  // No square root exists
    }
    
    // For p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4) mod p
    // Since p = 4k + 3, we have p+1 = 4(k+1), so (p+1)/4 = k+1
    // We compute this as: (p-3)/4 + 1 = k + 1
    const Fp& p = fp_modulus();
    Fp three, p_minus_3, exp;
    fp_zero(three);
    three.limb[0] = 3;
    // Compute p-3. Since p > 3, (p-3) mod p = p-3
    fp_sub(p_minus_3, p, three);  // p - 3 = 4k
    // Divide by 4: (p-3)/4 = k
    fp_div_by_4(exp, p_minus_3);  // (p-3)/4 = k
    // Add 1 to get (p+1)/4 = k+1
    Fp one;
    fp_one(one);
    fp_add(exp, exp, one);  // exp = k+1 = (p+1)/4
    
    // Compute a^((p+1)/4) mod p
    // Note: fp_pow_internal already converts result back from Montgomery form
    fp_pow_internal(c, a, exp.limb, FP_LIMBS);
    
    // Verify: c^2 should equal a (mod p)
    Fp c_squared;
    fp_mul(c_squared, c, c);
    
    // Check if c^2 == a
    if (fp_cmp(c_squared, a) == 0) {
        return true;  // Correct square root found
    }
    
    // If not, try the other square root: p - c
    // In finite fields, if c is a square root, so is p - c
    Fp alt_c;
    fp_sub(alt_c, p, c);
    fp_mul(c_squared, alt_c, alt_c);
    if (fp_cmp(c_squared, a) == 0) {
        fp_copy(c, alt_c);
        return true;
    }
    
    // If verification fails, the computed value might still be correct
    // but there could be a reduction issue. Try reducing c modulo p if needed.
    // However, since fp_pow_internal should already return a reduced value,
    // this shouldn't be necessary. But let's ensure c is properly reduced.
    if (fp_cmp(c, p) >= 0) {
        Fp reduced_c;
        fp_sub(reduced_c, c, p);
        fp_copy(c, reduced_c);
        fp_mul(c_squared, c, c);
        if (fp_cmp(c_squared, a) == 0) {
            return true;
        }
    }
    
    // If all checks fail, there's likely a bug in the computation
    // Return false to indicate failure
    return false;
}

// Conditional move: if condition != 0, dst = src
__host__ __device__ void fp_cmov(Fp& dst, const Fp& src, uint64_t condition) {
    // condition should be 0 or all-ones for constant-time
    uint64_t mask = -(condition & 1);  // 0 or 0xFFFFFFFFFFFFFFFF
    
    for (int i = 0; i < FP_LIMBS; i++) {
        dst.limb[i] = (dst.limb[i] & ~mask) | (src.limb[i] & mask);
    }
}

