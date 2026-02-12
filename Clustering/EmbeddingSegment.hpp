#pragma once
#include "Embedding.hpp"
#include <unordered_set>
#include <memory>

struct EmbeddingSegment {
    std::vector<Embedding> embeddings;
    long speakerId;

    explicit EmbeddingSegment(long speakerId, long count): speakerId(speakerId), embeddings() {
        embeddings.reserve(count);
    }
    
    explicit EmbeddingSegment(std::vector<Embedding>&& embeddings): 
            speakerId(-1), embeddings(std::move(embeddings)) {}
    
    explicit EmbeddingSegment(const std::vector<Embedding>& embeddings): 
            speakerId(-1), embeddings(embeddings) {}
            
    EmbeddingSegment(long speakerId, std::vector<Embedding>&& embeddings):
            speakerId(speakerId), embeddings(std::move(embeddings)) {
        for (auto &e: this->embeddings) e.setSpeakerId(speakerId);
    }
    
    EmbeddingSegment(long speakerId, const std::vector<Embedding>& embeddings): 
            speakerId(speakerId), embeddings(embeddings) {
        for (auto &e: this->embeddings) e.setSpeakerId(speakerId);
    }
    
    inline void add(Embedding& e) {
        e.setSpeakerId(this->speakerId);
        embeddings.emplace_back(e);
    }

    inline void add(Embedding&& e) {
        e.setSpeakerId(this->speakerId);
        embeddings.emplace_back(e);
    }
    
    inline void add(UUID id, const float* buffer, bool normalize, float weight) {
        embeddings.emplace_back(id, buffer, speakerId, normalize, weight);
    }
};
