#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <functional>

class Tensor {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

    Tensor() : rows(0), cols(0) {}
    Tensor(size_t r, size_t c);
    Tensor(const std::vector<std::vector<double>>& d);
    static Tensor zeros(size_t r, size_t c);
    static Tensor random(size_t r, size_t c, double scale);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor transpose() const;
    Tensor hadamard(const Tensor& other) const;
    template<typename F>
    Tensor apply(F func) const {
        Tensor res(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                res.data[i][j] = func(data[i][j]);
        return res;
    }
    double sum() const;
    double max() const;
    void fill(double val);
    std::vector<double>& operator[](size_t i);
    const std::vector<double>& operator[](size_t i) const;
    // Row extraction: returns row i as (1, cols) tensor
    Tensor get_row(size_t i) const;
};

#endif