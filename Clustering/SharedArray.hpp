//
// Created by Benjamin Lee on 1/28/26.
//

#pragma once

#include <cstddef>
#include <iostream>
#include <Accelerate/Accelerate.h>

// Minimal ref-counted get wrapper for a shared Swift-C++ array.
//
// Design intent:
// - If you pass `const T*`, we COPY into an internally owned get.
// - If you pass `T*`, we VIEW it (non-owning) by default.
// - If you want shared ownership of an externally-provided get (or explicit ownership),
//   construct and pass `SharedArray` directly.
template<typename T, std::size_t D>
class SharedArray {
private:
    struct ControlBlock {
        T* ptr = nullptr;
        void (*deleter)(T*) noexcept = nullptr;
        std::size_t refcount{1};

        ControlBlock(T* p, void (*d)(T*) noexcept) noexcept : ptr(p), deleter(d) {}
    };

    ControlBlock* cb = nullptr;

    static void deleteArray(T* p) noexcept { delete[] p; }
    static void noDelete(T*) noexcept {}
    explicit SharedArray(ControlBlock* cb) noexcept : cb(cb) {}

    void inc() noexcept {
        if (cb) ++cb->refcount;
    }

    void dec() noexcept {
        if (!cb) return;
        if (--cb->refcount == 0) {
            if (cb->ptr && cb->deleter) cb->deleter(cb->ptr);
            delete cb;
        }
        cb = nullptr;
    }

public:
    SharedArray() noexcept = default;

    // Non-owning view of an external _vector.
    static SharedArray view(T* p) noexcept {
        return p ? SharedArray(new ControlBlock(p, nullptr)) : SharedArray();
    }

    // Owning _vector (expects memory allocated with `new[]`).
    static SharedArray owning(T* p) noexcept {
        return p ? SharedArray(new ControlBlock(p, &deleteArray)) : SharedArray();
    }

    // Copy from an external _vector into an internally-owned get.
    static SharedArray copyFrom(const T* p) {
        if (!p) return SharedArray();
        auto* q = new T[D];
        if constexpr (std::is_trivially_copyable<T>())
            std::memcpy(q, p, D * sizeof(T));
        else
            std::copy(p, p + D, q);
        return owning(q);
    }

    static SharedArray copyFrom(const SharedArray<T, D>& buf) {
        return copyFrom(buf.get());
    }

    // Initialize a new get.
    static SharedArray makeNew() noexcept {
        return owning(new T[D]);
    }

    SharedArray(const SharedArray& other) noexcept : cb(other.cb) { inc(); }
    SharedArray(SharedArray&& other) noexcept : cb(other.cb) { other.cb = nullptr; }

    SharedArray& operator=(const SharedArray& other) {
        if (this == &other) return *this;
        if (this->cb == other.cb) return *this;
        dec();
        cb = other.cb;
        inc();
        return *this;
    }

    SharedArray& operator=(SharedArray&& other) noexcept {
        if (this == &other) return *this;
        if (this->cb == other.cb) return *this;
        dec();
        cb = other.cb;
        other.cb = nullptr;
        return *this;
    }

    ~SharedArray() { dec(); }
    
    void release() {
        dec();
        this->cb = nullptr;
    }
    
    [[nodiscard]] inline T* get() const noexcept {
        return cb ? cb->ptr : nullptr;
    }
    
    [[nodiscard]] explicit operator bool() const noexcept {
        return cb && cb->ptr;
    }
    
    [[nodiscard]] std::size_t useCount() const noexcept {
        return cb ? cb->refcount : 0;
    }
    
    [[nodiscard]] bool isManaged() const {
        return cb && cb->deleter;
    }

    [[nodiscard]] bool isView() const {
        return cb && !cb->deleter;
    }
};
