#pragma once
#include <arm_acle.h>
#include <cstdint>
#include <array>

extern "C" {
    struct UUIDWrapper {
        uint64_t data[2] = {0x0040000000000000, 0};
        
        UUIDWrapper();
        
        static const UUIDWrapper zero;
        
        inline UUIDWrapper(uint64_t lower, uint64_t upper) {
            data[0] = lower;
            data[1] = upper;
        }

        inline explicit UUIDWrapper(uint64_t* ptr) {
            std::memcpy(data, ptr, 16);
        }

        inline explicit UUIDWrapper(const uint8_t bytes[16]) {
            std::memcpy(data, bytes, 16);
        }
        
        inline void unwrapTo(uint8_t dst[16]) const {
            memcpy(dst, data, 16);
        }
    
        inline bool operator==(const UUIDWrapper& other) const noexcept {
            return this->data[0] == other.data[0] && data[1] == other.data[1];
        }

        inline bool operator==(const UUIDWrapper* other) const noexcept {
            return this->data[0] == other->data[0] && data[1] == other->data[1];
        }

        inline bool operator<(const UUIDWrapper& other) const noexcept {
            return (this->data[0] < other.data[0]) || (this->data[0] == other.data[0] && data[1] < other.data[1]);
        }
        
        inline UUIDWrapper operator^(const UUIDWrapper& other) const noexcept {
            return { data[0] ^ other.data[0], data[1] ^ other.data[1] };
        }

        inline UUIDWrapper operator+(const UUIDWrapper& other) const noexcept {
            auto res = *this;
            return res += other; 
        }

        inline UUIDWrapper& operator^=(const UUIDWrapper& other) noexcept {
            data[0] ^= other.data[0]; // Adhere to UUID v4 standards
            data[1] ^= other.data[1];
            return *this;
        }

        inline UUIDWrapper& operator+=(const UUIDWrapper& other) noexcept {
            uint64_t low = data[0] + other.data[0];
            uint64_t overflow = low < data[0];
            data[0] = low; // Adhere to UUID v4 standards
            data[1] += other.data[1] + overflow;
            return *this;
        }
    };
}

inline const UUIDWrapper UUIDWrapper::zero = {0x0040000000000000, 0};
    
namespace std {
    template<>
    struct hash<UUIDWrapper> {
        std::size_t operator()(const UUIDWrapper& uuid) const noexcept {
            size_t h = std::hash<uint64_t>{}(uuid.data[0]);
            h ^= std::hash<uint64_t>{}(uuid.data[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return static_cast<size_t>(h);
        }
    };
}
