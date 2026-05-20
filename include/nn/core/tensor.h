#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cstdint>

// 32-byte alignment for AVX SIMD operations
#define TENSOR_ALIGNMENT 32

template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    AlignedAllocator() noexcept {}
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, TENSOR_ALIGNMENT, n * sizeof(T));
        if (ret != 0) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, std::size_t) {
        free(ptr);
    }
    
    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

class Tensor {
public:
    // Flat contiguous storage with SIMD-friendly alignment
    std::vector<double, AlignedAllocator<double>> data;
    size_t rows = 0, cols = 0;

    Tensor() {}
    Tensor(size_t r, size_t c);
    Tensor(const std::vector<std::vector<double>>& d);
    // Copy constructor (deep copy)
    Tensor(const Tensor& other);
    // Copy assignment (deep copy)
    Tensor& operator=(const Tensor& other);
    // Construct from flat data pointer (row-wise copy)
    Tensor(size_t r, size_t c, const double* flat_data);
    static Tensor zeros(size_t r, size_t c);
    static Tensor random(size_t r, size_t c, double scale);
    Tensor operator+(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;
    Tensor& operator-=(const Tensor& other);
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor transpose() const;
    Tensor concatenate(const Tensor& other, bool along_cols = true) const;
    Tensor hadamard(const Tensor& other) const;
    template<typename F>
    Tensor apply(F func) const {
        Tensor res(rows, cols);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j, ++idx)
                res.data[idx] = func(data[idx]);
        return res;
    }
    double sum() const;
    double max() const;
    void fill(double val);

    // Backward-compatible subscript — returns pointer into flat buffer
    double* operator[](size_t i) { return data.data() + i * cols; }
    const double* operator[](size_t i) const { return data.data() + i * cols; }

    // 2D subscript accessor
    double& operator()(size_t r, size_t c) { return data[r * cols + c]; }
    const double& operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    // Row extraction: returns row i as (1, cols) tensor
    Tensor get_row(size_t i) const;
    // Explicit deep clone — use instead of copy constructor when semantics need to be unambiguous
    Tensor clone() const { Tensor t(rows, cols); t.data = data; return t; }
};

#endif