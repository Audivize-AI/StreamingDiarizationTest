//
// Created by Benjamin Lee on 1/26/26.
//

#include "Embedding.hpp"

#include <utility>
#include "Accelerate/Accelerate.h"

Embedding::Embedding(const float* buffer, long speakerId, bool normalize, float weight, float spread):
        Embedding(Vector::copyFrom(buffer), speakerId, normalize, weight, spread) {}

Embedding::Embedding(UUID id, const float* buffer, long speakerId, bool normalize, float weight, float spread):
        Embedding(id, Vector::copyFrom(buffer), speakerId, normalize, weight, spread) {}

Embedding::Embedding(float* buffer, long speakerId, bool normalize, float weight, float spread):
        Embedding(Vector::view(buffer), speakerId, normalize, weight, spread) {}

Embedding::Embedding(UUID id, float* buffer, long speakerId, bool normalize, float weight, float spread):
        Embedding(id, Vector::view(buffer), speakerId, normalize, weight, spread) {}

Embedding::Embedding(Vector vector, long speakerId, bool normalize, float weight, float spread):
        _id(), _vector(std::move(vector)), _speakerId(speakerId), _weight(weight), _spread(spread) {
    if (normalize) this->normalize();
}

Embedding::Embedding(UUID id, Vector vector, long speakerId, bool normalize, float weight, float spread):
        _id(id), _vector(std::move(vector)), _speakerId(speakerId), _weight(weight), _spread(spread) {
    if (normalize) this->normalize();
}

Embedding::Embedding(Embedding&& other) noexcept:
        _id(std::move(other._id)),
        _vector(std::move(other._vector)),
        _speakerId(other._speakerId),
        _matrixIndex(other._matrixIndex),
        _weight(other._weight),
        _spread(other._spread)
{}
        
float Embedding::squaredDistanceTo(const Embedding& other) const {
    if (expired() || other.expired())
        return std::numeric_limits<float>::infinity();
    
    float dist;
    vDSP_distancesq(
            this->_vector.get(), 1,
            other._vector.get(), 1,
            &dist,
            Embedding::dims
    );
    return dist;
}

float Embedding::cosineDistanceTo(const Embedding& other) const {
    if (expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dot;
    vDSP_dotpr(
            this->_vector.get(), 1,
            other._vector.get(), 1,
            &dot,
            Embedding::dims
    );
    
    float normSqA, normSqB;
    vDSP_svesq(this->_vector.get(), 1, &normSqA, Embedding::dims);
    vDSP_svesq(other._vector.get(), 1, &normSqB, Embedding::dims);
    return std::max(0.f, 1.f - dot / std::sqrt(normSqA * normSqB));
}

float Embedding::unitCosineDistanceTo(const Embedding& other) const {
    if (expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dot;
    vDSP_dotpr(
            this->_vector.get(), 1,
            other._vector.get(), 1,
            &dot,
            Embedding::dims
    );

    return std::max(0.f, 1.f - dot);
}

float Embedding::dot(const Embedding& other) const {
    if (expired() || other.expired())
        return std::numeric_limits<float>::infinity();

    float dot;
    vDSP_dotpr(
            this->_vector.get(), 1,
            other._vector.get(), 1,
            &dot,
            Embedding::dims
    );
    return dot;
}

float Embedding::norm() const {
    float normSq;
    vDSP_svesq(this->_vector.get(), 1, &normSq, Embedding::dims);
    return std::sqrt(normSq);
}

float Embedding::normSquared() const {
    float normSq;
    vDSP_svesq(this->_vector.get(), 1, &normSq, Embedding::dims);
    return normSq;
}

Embedding& Embedding::rescaleToLength(float newLength) {
    auto length = norm();
    if (length < 1e-6) return *this;
    return *this *= newLength / length;
}

Embedding Embedding::rescaledToLength(float newLength) const {
    auto length = norm();
    if (length < 1e-6) return *this;
    return *this * (newLength / length);
}

void Embedding::takeOwnership() {
    if (ownsVector()) return;
    this->_vector = Vector::copyFrom(_vector);
}

bool Embedding::operator==(const Embedding& other) const {
    return this->_id == other._id;
}

Embedding& Embedding::operator=(Embedding&& other) noexcept {
    this->_id = other._id;
    this->_vector = std::move(other._vector);
    this->_speakerId = other._speakerId;
    this->_matrixIndex = other._matrixIndex;
    this->_weight = other._weight;
    this->_spread = other._spread;
    return *this;
}

Embedding Embedding::operator+(const Embedding& other) const {
    auto result = Vector::makeNew();
    vDSP_vadd(
            this->_vector.get(), 1,
            other._vector.get(), 1,
            result.get(), 1,
            Embedding::dims
    );
    return Embedding(result, this->_speakerId, false);
}

Embedding Embedding::operator-(const Embedding& other) const {
    auto result = Vector::makeNew();
    vDSP_vsub(
            other._vector.get(), 1,
            this->_vector.get(), 1,
            result.get(), 1,
            Embedding::dims
    );
    return Embedding(result, this->_speakerId, false);
}

Embedding Embedding::operator*(float scalar) const {
    auto result = Vector::makeNew();
    vDSP_vsmul(
            this->_vector.get(), 1,
            &scalar,
            result.get(), 1,
            Embedding::dims
    );
    return Embedding(result, this->_speakerId, false);
}

Embedding Embedding::operator/(float scalar) const {
    auto result = Vector::makeNew();
    vDSP_vsdiv(
            _vector.get(), 1,
            &scalar,
            result.get(), 1,
            Embedding::dims
    );
    return Embedding(result, this->_speakerId, false);
}

Embedding& Embedding::operator+=(const Embedding& other) {
    auto ptr = this->_vector.get();
    vDSP_vadd(
            ptr, 1,
            other._vector.get(), 1,
            ptr, 1,
            Embedding::dims
    );
    return *this;
}

Embedding& Embedding::operator-=(const Embedding& other) {
    auto ptr = this->_vector.get();
    vDSP_vsub(
            other._vector.get(), 1,
            ptr, 1,
            ptr, 1,
            Embedding::dims
    );
    return *this;
}

Embedding& Embedding::operator*=(float scalar) {
    auto ptr = _vector.get();
    vDSP_vsmul(
            ptr, 1,
            &scalar,
            ptr, 1,
            Embedding::dims
    );
    return *this;
}

Embedding& Embedding::operator/=(float scalar) {
    auto ptr = this->_vector.get();
    vDSP_vsdiv(
            ptr, 1,
            &scalar,
            ptr, 1,
            Embedding::dims
    );
    return *this;
}

void Embedding::weightedSum(float weightA, const Embedding& a, float weightB, const Embedding& b, Embedding& result) {
    vDSP_vsmsma(
            a._vector.get(), 1,
            &weightA,
            b._vector.get(), 1,
            &weightB,
            result._vector.get(), 1,
            Embedding::dims
    );
}

Embedding Embedding::weightedSum(float weightA, const Embedding& a, float weightB, const Embedding& b) {
    auto result = Vector::makeNew();
    vDSP_vsmsma(
            a._vector.get(), 1,
            &weightA,
            b._vector.get(), 1,
            &weightB,
            result.get(), 1,
            Embedding::dims
    );
    return Embedding(result, a._speakerId, false);
}

// Compute C = (A + B) / |A + B| * length in-place
void Embedding::rescaledSum(const Embedding& a, const Embedding& b, float length, Embedding& result) {
    vDSP_vadd(
            a._vector.get(), 1,
            b._vector.get(), 1,
            result._vector.get(), 1,
            Embedding::dims
    );
    result.rescaleToLength(length);
}

// Compute C = (A + B) / |A + B| * length
Embedding Embedding::rescaledSum(const Embedding& a, const Embedding& b, float length) {
    return (a+b).rescaleToLength(length);
}
