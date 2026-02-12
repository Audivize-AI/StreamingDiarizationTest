//
// Created by Benjamin Lee on 1/26/26.
//
#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>
#include "SharedArray.hpp"
#include "Uuid.h"

class Embedding {
public:
    static constexpr std::size_t dims = 192;
    using Vector = SharedArray<float, dims>;
    
    Embedding() = default;
    Embedding(const Embedding& other) = default;
    Embedding(Embedding&& other) noexcept;
    
    /**
     * @brief Construct an embedding that copies the vector from a buffer
     * @param buffer Const pointer to buffer containing the vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(const float* buffer, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Construct an embedding that copies the vector from a buffer
     * @param id Embedding ID
     * @param buffer Const pointer to buffer containing the vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(UUID id, const float* buffer, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Construct an embedding with a pre-initialized buffer
     * @param buffer Pointer to buffer containing the vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(float* buffer, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Construct an embedding with a pre-initialized buffer
     * @param id Embedding ID
     * @param buffer Pointer to buffer containing the vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(UUID id, float* buffer, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Construct an embedding with a SharedArray<192> object
     * @param vector SharedArray<192> object that holds the embedding vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(Vector vector, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Construct an embedding with a SharedArray<192> object
     * @param id Embedding ID
     * @param vector SharedArray<192> object that holds the embedding vector
     * @param speakerId Slot in the sortformer output (-1 for no speakerId)
     * @param normalize Whether to normalize the vector
     * @param weight The total duration of audio that this vector was built on
     * @param spread The spread of the cluster this embedding represents. If it's alone, then leave as 0
     */
    explicit Embedding(UUID id, Vector vector, long speakerId = -1, bool normalize = true, float weight=1, float spread=0);

    /**
     * @brief Get the squared L2 distance to another embedding
     * @param other Another embedding
     * @return |a - b|^2
     */
    [[nodiscard]] float squaredDistanceTo(const Embedding& other) const;

    /**
     * @brief Get the cosine distance to another embedding
     * @param other Another embedding
     * @return max(0, 1 - (a • b) / (|a||b|))
     */
    [[nodiscard]] float cosineDistanceTo(const Embedding& other) const;

    /**
     * @brief Get the cosine dissimilarity to another embedding, assuming both are unit vectors
     * @param other Another unit embedding
     * @return max(0, 1 - a • b)
     */
    [[nodiscard]] float unitCosineDistanceTo(const Embedding& other) const;

    /**
     * @brief Get the dot product with another embedding
     * @param other Another embedding
     * @return Dot product with `other` if both are active
     */
    [[nodiscard]] float dot(const Embedding& other) const;
    
    // Rescale the embedding _vector to length 1
    inline Embedding& normalize() { return rescaleToLength(1.f); }
    
    // Rescale the embedding _vector to length 1
    [[nodiscard]] inline Embedding normalized() const { return rescaledToLength(1.f); }
    
    /**
     * @brief Rescale the embedding _vector to a new length
     * @param newLength New embedding _vector length
     * @returns: This embedding
     */
    Embedding& rescaleToLength(float newLength);

    /**
     * @brief Rescale the embedding _vector to a new length
     * @param newLength New embedding _vector length
     * @returns: The rescaled embedding
     */
    [[nodiscard]] Embedding rescaledToLength(float newLength) const;
    
    // Flag as deleted in the spread matrix
    inline void releaseVector() { _vector.release(); }
    
    // Set row in the spread matrix
    inline void setMatrixIndex(long index) { _matrixIndex = index; }
    
    // Make this Embedding own its _vector
    void takeOwnership();
    
    // Set the weight of this vector
    inline void setWeight(float weight) { _weight = weight; }

    // Set the weight of this vector
    inline void setSpread(float spread) {
        if (!std::isfinite(spread)) {
            _spread = 2.f;
            return;
        }
        _spread = std::clamp(spread, 0.f, 2.f);
    }
    
    // Set the speaker ID of this embedding vector
    inline void setSpeakerId(long speakerId) { _speakerId = speakerId; }
    
    // Get its row in the spread matrix
    [[nodiscard]] inline long matrixIndex() const { return _matrixIndex; }
    
    // Get weight
    [[nodiscard]] inline float weight() const { return _weight; }

    // Get speaker ID
    [[nodiscard]] inline float speakerId() const { return _speakerId; }
    
    // Get spread
    [[nodiscard]] inline float spread() const { return _spread; }
    
    // Get vector buffer
    [[nodiscard]] inline Vector vector() const { return _vector; }
    
    // Get the embedding UUID
    [[nodiscard]] inline UUID id() const { return _id; }
    
    // Whether this embedding is inactive/deleted from the matrix
    [[nodiscard]] inline bool expired() const { return !static_cast<bool>(_vector); }

    [[nodiscard]] inline bool hasVector() const { return static_cast<bool>(_vector); }
    [[nodiscard]] inline float* get() const { return _vector.get(); }
    
    // Whether this embedding is active in the matrix
    [[nodiscard]] inline explicit operator bool() const { return hasVector(); }
    
    // Whether this embedding owns its get
    [[nodiscard]] inline bool ownsVector() const { return _vector.isManaged(); }
    
    // Embedding _vector norm
    [[nodiscard]] float norm() const;
    
    [[nodiscard]] float normSquared() const;
    
    bool operator==(const Embedding& other) const;
    
    Embedding operator+(const Embedding& other) const;
    
    Embedding operator-(const Embedding& other) const;

    Embedding operator*(float scalar) const;
    
    Embedding operator/(float scalar) const;

    Embedding& operator=(const Embedding& other) = default;
    Embedding& operator=(Embedding&& other) noexcept;
    Embedding& operator+=(const Embedding& other);
    Embedding& operator-=(const Embedding& other);
    Embedding& operator*=(float scalar);
    Embedding& operator/=(float scalar);
    
    // Compute C = w_a * E_a + w_b * E_b in-place
    static void weightedSum(float weightA, const Embedding& a, float weightB, const Embedding& b, Embedding& result);
    
    // Compute C = w_a * E_a + w_b * E_b
    static Embedding weightedSum(float weightA, const Embedding& a, float weightB, const Embedding& b);
    
    // Compute C = (A + B) / |A + B| * length in-place
    static void rescaledSum(const Embedding& a, const Embedding& b, float length, Embedding& result);
    
    // Compute C = (A + B) / |A + B| * length
    static Embedding rescaledSum(const Embedding& a, const Embedding& b, float length);

protected:
    UUID _id { UUID(0, 0) };
    Vector _vector {};
    long _speakerId = -1;
    long _matrixIndex = -1;
    float _weight{1};
    float _spread{0};
    
    friend struct std::hash<Embedding>;
};

namespace std {
    template<>
    struct hash<Embedding> {
        inline std::size_t operator()(const Embedding &e) const noexcept {
            return std::hash<UUID>{}(e._id);
        }
    };
}
