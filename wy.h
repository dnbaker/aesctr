// adapted to the lemire/testingRNG project by D. Lemire, from https://github.com/wangyi-fudan/wyhash/blob/master/wyhash.h
// This uses mum hashing.
// Further adapted by D. Baker from https://github.com/lemire/testingRNG/blob/42a3a76feef1126d632f7a56181dacb77ba1ccc7/source/wyhash.h
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>

#ifndef CONST_IF
#  if __cpp_if_constexpr
#    define CONST_IF(x) if constexpr(x)
#  else
#    define CONST_IF(x) if(x)
#  endif
#endif

namespace wy {
using std::uint64_t;
using std::size_t;

// call wyhash64_seed before calling wyhash64
static inline uint64_t wyhash64_stateless(uint64_t *seed) {
  *seed += UINT64_C(0x60bee2bee120fc15);
  __uint128_t tmp;
  tmp = (__uint128_t)*seed * UINT64_C(0xa3b195354a39b70d);
  uint64_t m1 = (tmp >> 64) ^ tmp;
  tmp = (__uint128_t)m1 * UINT64_C(0x1b03738712fad5c9);
  uint64_t m2 = (tmp >> 64) ^ tmp;
  return m2;
}

template<typename T=std::uint64_t, size_t unroll_count=0>
class WyHash {
    uint64_t state_;
    uint64_t unrolled_stuff_[unroll_count];
    unsigned offset_[!!unroll_count];
    unsigned &off() {return offset_[0];}
public:
    WyHash(uint64_t seed=0): state_(seed) {
        std::memset(unrolled_stuff_, 0, sizeof(unrolled_stuff_));
        if(unroll_count) off() = 0;
    }
    const uint8_t *as_bytes() const {return reinterpret_cast<const uint8_t *>(unrolled_stuff_);}
    uint64_t next_value() {
        return wyhash64_stateless(&state_);
    }
    static auto constexpr min() {return std::numeric_limits<T>::min();}
    static auto constexpr max() {return std::numeric_limits<T>::max();}
    T operator()() {
        CONST_IF(unroll_count) {
            if(off() == sizeof(unrolled_stuff_) / sizeof(T)) {
                for(size_t i = 0; i < unroll_count; unrolled_stuff_[i++] = next_value());
                off() = 0;
            }
            T ret;
            std::memcpy(&ret, as_bytes() + off(), sizeof(T));
            off() += sizeof(T);
            return ret;
        } else {
            CONST_IF(sizeof(T) <= sizeof(uint64_t))
                return static_cast<T>(this->next_value());
            else {
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

}
