//
// Created by Benjamin Lee on 1/26/26.
//

#include "SpeakerEmbeddingWrapper.hpp"
//#include "FluidAudio/Sortformer/SpeakerEmbeddingCppBridge.h"

#include <utility>
#include "Accelerate/Accelerate.h"

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(float weight, std::vector<UUIDWrapper>&& segmentIds): 
        _weight(weight), _segmentIds(segmentIds)
{
    swiftSpeakerEmbeddingCreate(const_cast<void **>(&swiftPtr), nullptr, _id.data, &_vector);
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(UUIDWrapper id, float weight, std::vector<UUIDWrapper>&& segmentIds): 
        _id(id), _weight(weight), _segmentIds(segmentIds)
{
    swiftSpeakerEmbeddingCreate(const_cast<void **>(&swiftPtr), id.data, nullptr, &_vector);
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(float weight, std::vector<UUIDWrapper> const& segmentIds): 
        _weight(weight), _segmentIds(segmentIds)
{
    swiftSpeakerEmbeddingCreate(const_cast<void **>(&swiftPtr), nullptr, _id.data, &_vector);
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(UUIDWrapper id, float weight, std::vector<UUIDWrapper> const& segmentIds): 
        _id(id), _weight(weight), _segmentIds(segmentIds)
{
    swiftSpeakerEmbeddingCreate(const_cast<void **>(&swiftPtr), id.data, nullptr, &_vector);
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(const void* swiftPtr): swiftPtr(swiftPtr) {
    if (swiftPtr) {
        swiftSpeakerEmbeddingLoad(swiftPtr, _id.data, &_vector, &_weight);
    }
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(const SpeakerEmbeddingWrapper& other):
        _id(other._id),
        _vector(other._vector),
        _weight(other._weight),
        swiftPtr(other.swiftPtr) 
{
    swiftSpeakerEmbeddingRetain(swiftPtr);
}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(SpeakerEmbeddingWrapper&& other) noexcept: 
        _id(other._id), 
        _vector(other._vector), 
        _weight(other._weight),
        swiftPtr(other.swiftPtr)
{
    other.swiftPtr = nullptr;
    other._vector = nullptr;
    other._id = {0, 0};
}

SpeakerEmbeddingWrapper::~SpeakerEmbeddingWrapper() {
    if (!this->swiftPtr) return;
    swiftSpeakerEmbeddingRelease(this->swiftPtr);
}
        
float SpeakerEmbeddingWrapper::squaredDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    if (this->expired() || other.expired())
        return std::numeric_limits<float>::infinity();
    
    float dist;
    vDSP_distancesq(
            this->_vector, 1,
            other._vector, 1,
            &dist,
            SpeakerEmbeddingWrapper::dims
    );
    return dist; 
}

float SpeakerEmbeddingWrapper::wardDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    if (this->expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dist;
    vDSP_distancesq(
            this->_vector, 1,
            other._vector, 1,
            &dist,
            SpeakerEmbeddingWrapper::dims
    );
    return (this->_weight * other._weight) / (this->_weight + other._weight) * dist;
}

float SpeakerEmbeddingWrapper::cosineDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    if (this->expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );
    
    float normSqA, normSqB;
    vDSP_svesq(this->_vector, 1, &normSqA, SpeakerEmbeddingWrapper::dims);
    vDSP_svesq(other._vector, 1, &normSqB, SpeakerEmbeddingWrapper::dims);
    return std::max(0.f, 1.f - dot / std::sqrt(normSqA * normSqB));
}

float SpeakerEmbeddingWrapper::unitCosineDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    if (this->expired() || other.expired())
        return std::numeric_limits<float>::infinity();
    
    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );

    return std::max(0.f, 1.f - dot);
}

float SpeakerEmbeddingWrapper::dot(const SpeakerEmbeddingWrapper& other) const {
    if (this->expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );
    return dot;
}

float SpeakerEmbeddingWrapper::norm() const {
    float normSq;
    vDSP_svesq(this->_vector, 1, &normSq, SpeakerEmbeddingWrapper::dims);
    return std::sqrt(normSq);
}

float SpeakerEmbeddingWrapper::normSquared() const {
    float normSq;
    vDSP_svesq(this->_vector, 1, &normSq, SpeakerEmbeddingWrapper::dims);
    return normSq;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::rescaledInPlaceToLength(float newLength) {
    auto length = norm();
    if (length < 1e-6) return *this;
    return *this *= newLength / length;
}

SpeakerEmbeddingWrapper SpeakerEmbeddingWrapper::rescaledToLength(float newLength) const {
    auto length = norm();
    if (length < 1e-6) return *this;
    return *this * (newLength / length);
}

bool SpeakerEmbeddingWrapper::operator==(const SpeakerEmbeddingWrapper& other) const {
    return this->_id == other._id;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator=(SpeakerEmbeddingWrapper&& other) noexcept {
    if (this->_id == other._id)
        return *this;
    if (this->swiftPtr) swiftSpeakerEmbeddingRelease(this->swiftPtr);
    this->swiftPtr = other.swiftPtr;
    this->_id = other._id;
    this->_vector = other._vector;
    this->_weight = other._weight;
    this->_segmentIds = std::move(other._segmentIds);
    other.swiftPtr = nullptr;
    other._vector = nullptr;
    other._id = {0, 0};
    return *this;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator=(const SpeakerEmbeddingWrapper& other) {
    if (this == &other) return *this;
    if (this->_id == other._id) return *this;
    if (this->swiftPtr) swiftSpeakerEmbeddingRelease(this->swiftPtr);
    this->swiftPtr = other.swiftPtr;
    this->_id = other._id;
    this->_vector = other._vector;
    this->_weight = other._weight;
    this->_segmentIds = other._segmentIds;
    if (this->swiftPtr) swiftSpeakerEmbeddingRetain(this->swiftPtr);
    return *this;
}

SpeakerEmbeddingWrapper SpeakerEmbeddingWrapper::operator*(float scalar) const {
    auto result = SpeakerEmbeddingWrapper(this->_weight, this->_segmentIds);
    vDSP_vsmul(
            _vector, 1,
            &scalar,
            result._vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return result;
}

SpeakerEmbeddingWrapper SpeakerEmbeddingWrapper::operator/(float scalar) const {
    auto result = SpeakerEmbeddingWrapper(this->_weight, this->_segmentIds);
    vDSP_vsdiv(
            _vector, 1,
            &scalar,
            result._vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return result;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator*=(float scalar) {
    vDSP_vsmul(
            _vector, 1,
            &scalar,
            _vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return *this;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator/=(float scalar) {
    vDSP_vsdiv(
            _vector, 1,
            &scalar,
            _vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return *this;
}
