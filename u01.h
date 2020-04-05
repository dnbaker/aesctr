#ifndef FAST_U01_H__
#define FAST_U01_H__
#include "wy.h"
#include <x86intrin.h>

#define U01_SIMD_ENABLED 0

namespace wy {

#define FILL_BUF_WY \
    for(unsigned i = 0; i < unroll_count; ++i) {\
        buf[i] = hf.apply(&seed);\
    }

static inline double val(uint64_t x) {
    const constexpr double mul = 1. / (1ull << 52);
    return mul * (x >> 11);
}
static inline float val(uint32_t x) {
    const constexpr float mul = 1. / ((1ull << 20) - 1);
    return mul * (x >> 11);
}
#if U01_SIMD_ENABLED
#if __AVX512F__
#define RNGSIZE 8
union simsimd {
    __m512  f_;
    __m512d d_;
    __m512i i_;
    float far[sizeof(__m512) / sizeof(float)];
    double dar[sizeof(__m512) / sizeof(double)];
    int iar[sizeof(__m512) / sizeof(int)];
    static constexpr size_t FN = sizeof(far) / sizeof(float);
    static constexpr size_t DN = FN / 2;
};
#define U01_SIMD_ENABLED 1
#elif __AVX2__
union simsimd {
    __m256  f_;
    __m256d d_;
    __m256i i_;
    float far[sizeof(__m256) / sizeof(float)];
    double dar[sizeof(__m256) / sizeof(double)];
    int iar[sizeof(__m256) / sizeof(int)];
    static constexpr size_t FN = sizeof(far) / sizeof(float);
    static constexpr size_t DN = FN / 2;
};
#define RNGSIZE 4
#define U01_SIMD_ENABLED 1
#else
#define RNGSIZE 1
#define U01_SIMD_ENABLED 0
#endif
#else
#define RNGSIZE 1
#endif

#if U01_SIMD_ENABLED
#define DEF_UNROLL RNGSIZE
#else
#define DEF_UNROLL 2
#endif

template<typename FT, size_t unroll_count=DEF_UNROLL, typename=typename std::enable_if<std::is_floating_point<FT>::value>::type, typename HashFunc=WyHashFunc>
void fill_fastu01(FT *ptr, size_t nelem, uint64_t seed=0, HashFunc hf=HashFunc()) {
    const FT div = sizeof(FT) == 8 ? FT(1. / ((1ull << 53) - 1)): FT(1. / ((1ull << 21) - 1));
    static constexpr bool is_32bit = sizeof(FT) == 4;
    auto v = wyhash64_stateless(&seed);
    CONST_IF(sizeof(FT) == 4) {
        if((uint64_t)ptr % sizeof(double)) {
            *ptr++ = val(uint32_t(v));
            --nelem;
        }
    }
    uint64_t buf[unroll_count];
    FT *const end = ptr + nelem;
#if U01_SIMD_ENABLED
    static_assert(unroll_count % RNGSIZE == 0, "For simd operations, unroll count must be a multiple of vector width");
    simsimd mul;
#if __AVX512F__
    CONST_IF(is_32bit) mul.f_ = _mm512_set1_ps(div);
    else               mul.d_ = _mm512_set1_pd(div);
#elif __AVX2__
    CONST_IF(is_32bit) mul.f_ = _mm256_set1_ps(div);
    else               mul.d_ = _mm256_set1_pd(div);
#endif
    if(is_32bit) for(const auto v: mul.far) std::fprintf(stderr, "%g\n", v);
    else         for(const auto v: mul.dar) std::fprintf(stderr, "%g\n", v);
    if(is_32bit || RNGSIZE == 8) {
        for(;ptr <= end - unroll_count * sizeof(uint64_t) / sizeof(FT);) {
            for(unsigned bufind = 0; bufind < unroll_count / RNGSIZE; ++bufind) {
                FILL_BUF_WY
#if __AVX512F__
                CONST_IF(is_32bit) {
                    _mm512_storeu_ps((__m512 *)ptr, _mm512_mul_ps(_mm512_cvtepu32_ps(_mm512_srli_epi32(_mm512_loadu_si512((const __m512i *)buf + bufind), 11)), mul.f_));
                } else {
#  if __AVX512DQ__
                    _mm512_storeu_pd((__m512d *)ptr, _mm512_mul_pd(_mm512_cvtepu64_pd(_mm512_srli_epi64(_mm512_loadu_si512((const __m512i *)buf + bufind), 11)), mul.d_));
#  else
                    simsimd tmp;
                    tmp.i_ = _mm512_srli_epi64(_mm512_loadu_si512((const __m512i *)buf + bufind), 11);
                    for(unsigned i = 0; i < tmp.DN; ++i) {
                        tmp.d_[i] = static_cast<double>(((uint64_t *)&tmp)[i]);
                    }
                    tmp.d_ = _mm512_mul_pd(tmp.d_, mul.d_);
                    _mm512_storeu_pd((double *)ptr, tmp.d_);
#  endif
                }
#elif __AVX2__
                CONST_IF(is_32bit) {
                    simsimd tmp;
                    // Load, shift right by 11, convert to float, multiply by ( 1. / ((1 << 21) - 1)), and then storeu
                    _mm256_storeu_ps((float *)ptr,
                                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_loadu_si256((const __m256i *)buf + bufind), 11)), mul.f_) // * 1. / ((1 << 21) - 1)
                    );
                } else {
                    simsimd tmp;
                    tmp.i_ = _mm256_srli_epi64(_mm256_loadu_si256((const __m256i *)buf + bufind), 11);
                    for(unsigned i = 0; i < tmp.DN; ++i) {
                        tmp.d_[i] = static_cast<double>((((uint64_t *)&tmp)[i]));
                    }
                    tmp.d_ = _mm256_mul_pd(tmp.d_, mul.d_);
                    _mm256_storeu_pd((double *)ptr, tmp.d_);
                }
#endif
                ptr += RNGSIZE * sizeof(uint64_t) / sizeof(FT);
            }
        }
    }
#endif
    while(ptr + unroll_count * sizeof(uint64_t) / sizeof(FT) <= end) {
        FILL_BUF_WY
        for(unsigned i = 0; i < unroll_count; ++i) {
            CONST_IF(sizeof(FT) == 4) {
                ptr[2 * i] = div * (reinterpret_cast<uint32_t *>(buf)[2 * i] >> 11);
                ptr[2 * i + 1] = div * ((reinterpret_cast<uint32_t *>(buf)[2 * i + 1] >> 11));
            } else {
                ptr[i] = div * (buf[i] >> 11);
            }
        }
        ptr += unroll_count * sizeof(uint64_t) / sizeof(FT);
    }
    FILL_BUF_WY
    unsigned bi = 0;
    while(ptr < end) {
        CONST_IF(sizeof(FT) == 4) {
            *ptr++ = div * (reinterpret_cast<uint32_t *>(buf)[bi++] >> 11);
        } else {
            *ptr++ = div * (buf[bi++] >> 11);
        }
    }
}

} // namespace wy

#undef FILL_BUF_WY

#endif /* FAST_U01_H__ */
