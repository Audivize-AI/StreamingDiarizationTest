//
// Created by Benjamin Lee on 1/26/26.
//
#pragma once

#include <cstdint>
#include <vector>
#include "Uuid.hpp"

extern "C" {
    void swiftSpeakerEmbeddingCreate(void** swiftPtrOut, uint64_t* idIn, uint64_t* idOut, float** vectorOut);
    void swiftSpeakerEmbeddingLoad(const void* swiftPtr, uint64_t* idOut, float** vectorOut, float* weightOut);
    void swiftSpeakerEmbeddingRetain(const void* swiftPtr);
    void swiftSpeakerEmbeddingRelease(const void* swiftPtr);
}

class SpeakerEmbeddingWrapper {
public:
    static constexpr std::size_t dims = 192;
    
    SpeakerEmbeddingWrapper() = default;
    SpeakerEmbeddingWrapper(const SpeakerEmbeddingWrapper& other);
    SpeakerEmbeddingWrapper(SpeakerEmbeddingWrapper&& other) noexcept;

    /**
     * @brief Construct a new SpeakerEmbedding 
     * @param weight Speaker embedding weight. Used as a metric of embedding quality.
     */
    explicit SpeakerEmbeddingWrapper(float weight, std::vector<UUIDWrapper>&& segmentIds);

    /**
     * @brief Construct a new SpeakerEmbedding 
     * @param weight Speaker embedding weight. Used as a metric of embedding quality.
     */
    explicit SpeakerEmbeddingWrapper(float weight, std::vector<UUIDWrapper> const& segmentIds = {});

    /**
     * @brief Construct a new SpeakerEmbedding with a set weight and ID
     * @param id Speaker embedding UUIDWrapper.
     * @param weight Speaker embedding weight. Used as a metric of embedding quality.
     */
    explicit SpeakerEmbeddingWrapper(UUIDWrapper id, float weight, std::vector<UUIDWrapper>&& segmentIds);

    /**
     * @brief Construct a new SpeakerEmbedding with a set weight and ID
     * @param id Speaker embedding UUIDWrapper.
     * @param weight Speaker embedding weight. Used as a metric of embedding quality.
     */
    explicit SpeakerEmbeddingWrapper(UUIDWrapper id, float weight = 1, std::vector<UUIDWrapper> const& segmentIds = {});
    
    /**
     * @brief Construct from a Swift SpeakerEmbedding 
     * @param swiftPtr Pointer to Swift Speaker Embedding object
     */
    explicit SpeakerEmbeddingWrapper(const void* swiftPtr);
    
    virtual ~SpeakerEmbeddingWrapper();
    
    /** 
     * @brief Get the squared L2 distance to another embedding
     * @param other Another embedding 
     * @return |a - b|^2
     */
    [[nodiscard]] float squaredDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the weighted squared L2 distance to another embedding
     * @param other Another embedding 
     * @return |a - b|^2
     */
    [[nodiscard]] float wardDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the cosine distance to another embedding
     * @param other Another embedding 
     * @return max(0, 1 - (a • b) / (|a||b|))
     */
    [[nodiscard]] float cosineDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the cosine dissimilarity to another embedding, assuming both are unit vectors
     * @param other Another unit embedding 
     * @return max(0, 1 - a • b)
     */
    [[nodiscard]] float unitCosineDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the dot product with another embedding
     * @param other Another embedding 
     * @return Dot product with `other` if both are active
     */
    [[nodiscard]] float dot(const SpeakerEmbeddingWrapper& other) const;
    
    // Normalize this embedding vector in place
    inline SpeakerEmbeddingWrapper& normalizedInPlace() { return rescaledInPlaceToLength(1.f); }

    // Return a normalized copy of this embedding vector
    [[nodiscard]] inline SpeakerEmbeddingWrapper normalized() const { return rescaledToLength(1.f); }
    
    /** 
     * @brief Rescale the embedding _vector to a new length
     * @param newLength New embedding vector length
     * @returns: This embedding
     */
    SpeakerEmbeddingWrapper& rescaledInPlaceToLength(float newLength);

    /** 
     * @brief Rescale a copy of this embedding vector to a new length
     * @param newLength New embedding vector length
     * @returns: Rescaled embedding
     */
    [[nodiscard]] SpeakerEmbeddingWrapper rescaledToLength(float newLength) const;
    
    // Release the vector it held
    inline void releaseVector() {
        if (!this->swiftPtr) return;
        swiftSpeakerEmbeddingRelease(this->swiftPtr); 
        this->swiftPtr = nullptr;
        this->_vector = nullptr; 
        this->_id = {0, 0}; // NULL
    }

    // Get weight
    [[nodiscard]] inline float weight() const { return _weight; }
    
    // Get weight reference
    [[nodiscard]] inline float& weight() { return _weight; }
    
    // Get the embedding ID
    [[nodiscard]] inline UUIDWrapper id() const { return _id; }

    // Get reference to embedding ID
    [[nodiscard]] inline UUIDWrapper& id() { return _id; }

    // Get IDs of the corresponding segments
    [[nodiscard]] inline std::vector<UUIDWrapper> const& segmentIds() const { return this->_segmentIds; }
    
    // Get reference to IDs of the corresponding segments
    [[nodiscard]] inline std::vector<UUIDWrapper>& segmentIds() { return _segmentIds; }
    
    // Check whether this embedding is nil
    [[nodiscard]] inline bool expired() const { return !_vector; }

    // Check if this has a vector
    [[nodiscard]] inline bool hasVector() const { return _vector; }
    
    // Get raw pointer to the embedding vector
    [[nodiscard]] inline float* vector() const { return _vector; }
    
    // Whether this embedding is active in the matrix
    [[nodiscard]] inline explicit operator bool() const { return _vector; }
    
    // Embedding vector norm
    [[nodiscard]] float norm() const;
    
    // Embedding vector norm squared
    [[nodiscard]] float normSquared() const;
    
    bool operator==(const SpeakerEmbeddingWrapper& other) const;
    
    SpeakerEmbeddingWrapper operator*(float scalar) const;
    SpeakerEmbeddingWrapper operator/(float scalar) const;
    SpeakerEmbeddingWrapper& operator=(const SpeakerEmbeddingWrapper& other);
    SpeakerEmbeddingWrapper& operator=(SpeakerEmbeddingWrapper&& other) noexcept;
    SpeakerEmbeddingWrapper& operator*=(float scalar);
    SpeakerEmbeddingWrapper& operator/=(float scalar);

protected:
    UUIDWrapper _id{0, 0};
    const void* swiftPtr{nullptr};
    float* _vector{nullptr};
    std::vector<UUIDWrapper> _segmentIds{};
    float _weight{1.f};
    
    friend struct std::hash<SpeakerEmbeddingWrapper>;
};

namespace std {
    template<> 
    struct hash<SpeakerEmbeddingWrapper> {
        inline std::size_t operator()(const SpeakerEmbeddingWrapper &e) const noexcept {
            return std::hash<UUIDWrapper>{}(e._id);
        }
    };
}
