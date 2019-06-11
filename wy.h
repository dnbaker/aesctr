// adapted to the lemire/testingRNG project by D. Lemire, from https://github.com/wangyi-fudan/wyhash/blob/master/wyhash.h
// This uses mum hashing.
// Further adapted by D. Baker from https://github.com/lemire/testingRNG/blob/42a3a76feef1126d632f7a56181dacb77ba1ccc7/source/wyhash.h

// XXH3 ported from https://github.com/Cyan4973/xxHash/blob/4229399fc96a034fac522525946f5452d5bf0a65/xxh3.h
#ifndef WYHASH_RNG_H__
#define WYHASH_RNG_H__
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cinttypes>
#include <cstdio>

#ifndef CONST_IF
#  if __cpp_if_constexpr
#    define CONST_IF(x) if constexpr(x)
#  else
#    define CONST_IF(x) if(x)
#  endif
#endif
#ifndef NO_UNIQUE_ADDRESS
#  if __has_cpp_attribute(no_unique_address)
#    define NO_UNIQUE_ADDRESS [[no_unique_address]]
#  else
#    define NO_UNIQUE_ADDRESS
#    if VERBOSE_AF
#      pragma messsage("no unique address not supported")
#    endif
#  endif
#endif

namespace wy {
using std::uint64_t;
using std::size_t;

// call wyhash64_seed before calling wyhash64
static inline constexpr uint64_t wyhash64_stateless(uint64_t *seed) {
  *seed += UINT64_C(0x60bee2bee120fc15);
  __uint128_t tmp = (__uint128_t)*seed * UINT64_C(0xa3b195354a39b70d);
  uint64_t m1 = (tmp >> 64) ^ tmp;
  tmp = (__uint128_t)m1 * UINT64_C(0x1b03738712fad5c9);
  uint64_t m2 = (tmp >> 64) ^ tmp;
  return m2;
}

struct WyHashFunc {
    static constexpr uint64_t apply(uint64_t *x) {
        return wyhash64_stateless(x);
    }
};

struct XXH3Func {
    static constexpr uint64_t PRIME64_3 =  1609587929392839161ULL;  // 0b0001011001010110011001111011000110011110001101110111100111111001
    static constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
    static constexpr uint64_t apply(uint64_t *x) {
        *x += PRIME64_1;
        *x ^= *x >> 29;
        *x *= PRIME64_3;
        *x ^= *x >> 32;
        return *x;
    }
};

template<typename T=std::uint64_t, size_t unroll_count=0, typename HashFunc=WyHashFunc>
class WyHash {
public:
    static constexpr size_t UNROLL_COUNT = unroll_count ? unroll_count: size_t(1);
    using result_type = T;
private:
    uint64_t state_;
    uint64_t unrolled_stuff_[UNROLL_COUNT];
    unsigned offset_;
    unsigned &off() {return offset_;}
public:
    WyHash(uint64_t seed=0): state_(seed ? seed: uint64_t(1337)) {
        std::memset(unrolled_stuff_, 0, sizeof(unrolled_stuff_));
        CONST_IF(unroll_count) off() = sizeof(unrolled_stuff_);
    }
    void seed(uint64_t newseed) {
        state_ = newseed;
        CONST_IF(unroll_count) {
            off() = sizeof(unrolled_stuff_);
        }
    }
    const uint8_t *as_bytes() const {return reinterpret_cast<const uint8_t *>(unrolled_stuff_);}
    uint64_t next_value() {
        return HashFunc().apply(&state_);
    }
    static auto constexpr min() {return std::numeric_limits<T>::min();}
    static auto constexpr max() {return std::numeric_limits<T>::max();}
    T operator()() {
        if(unroll_count) {
            if(off() >= sizeof(unrolled_stuff_)) {
                for(size_t i = 0; i < UNROLL_COUNT; unrolled_stuff_[i++] = next_value());
                off() = 0;
            }
            T ret;
            std::memcpy(&ret, as_bytes() + off(), sizeof(T));
            off() += sizeof(T);
            return ret;
        } else {
            CONST_IF(sizeof(T) <= sizeof(uint64_t)) {
                return static_cast<T>(this->next_value());
            } else {
                T ret;
                size_t offset = 0;
                for(size_t i = 0; i < (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t); ++i) {
                    uint64_t v = next_value();
                    std::memcpy(&ret, &v, std::min(sizeof(v), sizeof(T) - offset * sizeof(uint64_t)));
                    ++offset;
                }
                return ret;
            }
        }
    }
};

} // namespace wy

#endif /* #ifndef WYHASH_RNG_H__ */
