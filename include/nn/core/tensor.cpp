#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(size_t r, size_t c) : rows(r), cols(c) {
    data.resize(r * c);
}

Tensor::Tensor(const Tensor& other) : data(other.data), rows(other.rows), cols(other.cols) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

Tensor::Tensor(const std::vector<std::vector<double>>& d)
    : rows(d.size()), cols(d.empty() ? 0 : d[0].size()) {
    data.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i * cols + j] = d[i][j];
}

Tensor::Tensor(size_t r, size_t c, const double* flat_data)
    : rows(r), cols(c) {
    data.resize(r * c);
    for (size_t idx = 0; idx < r * c; ++idx)
        data[idx] = flat_data[idx];
}

Tensor Tensor::zeros(size_t r, size_t c) {
    Tensor t(r, c);
    t.fill(0.0);
    return t;
}

Tensor Tensor::random(size_t r, size_t c, double scale) {
    Tensor t(r, c);
    size_t total = r * c;
    for (size_t idx = 0; idx < total; ++idx)
        t.data[idx] = ((double)rand() / RAND_MAX) * scale;
    return t;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    Tensor res(rows, cols);
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        res.data[idx] = data[idx] + other.data[idx];
    return res;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    Tensor res(rows, cols);
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        res.data[idx] = data[idx] - other.data[idx];
    return res;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        data[idx] += other.data[idx];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        data[idx] -= other.data[idx];
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (cols != other.rows) throw std::invalid_argument("Tensor multiplication dimension mismatch");
    Tensor res(rows, other.cols);
    // i-k-j: optimal cache locality for row-major storage.
    // Inner loop over j accesses A[i][k] (sequential), B[k][j] (sequential within row),
    // and C[i][j] (sequential write). Avoids stride jumps through B that i-j-k suffers.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < cols; ++k) {
            double a_ik = (*this)[i][k];
            for (size_t j = 0; j < other.cols; ++j) {
                res[i][j] += a_ik * other[k][j];
            }
        }
    }
    return res;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor res(rows, cols);
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        res.data[idx] = data[idx] * scalar;
    return res;
}

Tensor Tensor::transpose() const {
    Tensor res(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res[j][i] = (*this)[i][j];
    return res;
}

Tensor Tensor::hadamard(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Hadamard dimension mismatch");
    Tensor res(rows, cols);
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        res.data[idx] = data[idx] * other.data[idx];
    return res;
}

double Tensor::sum() const {
    double s = 0.0;
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        s += data[idx];
    return s;
}

double Tensor::max() const {
    double m = data[0];
    size_t total = rows * cols;
    for (size_t idx = 1; idx < total; ++idx)
        if (data[idx] > m) m = data[idx];
    return m;
}

void Tensor::fill(double val) {
    size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx)
        data[idx] = val;
}

Tensor Tensor::get_row(size_t i) const {
    Tensor row(1, cols);
    for (size_t j = 0; j < cols; ++j) row[0][j] = (*this)[i][j];
    return row;
}