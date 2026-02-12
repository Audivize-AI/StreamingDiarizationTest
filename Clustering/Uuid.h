#pragma once
#include <cstdint>
#include <array>

#ifdef __cplusplus
extern "C" {
#endif
    
    struct UUID {
    public:
        UUID();
    
        UUID(uint64_t a, uint64_t b) {
            data[0] = a;
            data[1] = b;
        }
        
        explicit UUID(const uint8_t bytes[16]) {
            std::memcpy(data, bytes, 16);
        }
    
        bool operator==(const UUID& other) const {
            return this->data[0] == other.data[0] && data[1] == other.data[1];
        }
    
        bool operator==(const UUID* other) const {
            return this->data[0] == other->data[0] && data[1] == other->data[1];
        }
    private:
        uint64_t data[2] = {0, 0};
        friend struct std::hash<UUID>;
    };
    
#ifdef __cplusplus
}
#endif
    
namespace std {
    template<>
    struct hash<UUID> {
        std::size_t operator()(const UUID& uuid) const noexcept {
            auto h1 = std::hash<uint64_t>{}(uuid.data[0]);
            auto h2 = std::hash<uint64_t>{}(uuid.data[1]);
            return h1 ^ (h2 << 1);
        }
    };
}


