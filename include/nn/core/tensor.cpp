#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(size_t r, size_t c) : rows(r), cols(c) {
    data.assign(r, std::vector<double>(c, 0.0));
}

Tensor::Tensor(const std::vector<std::vector<double>>& d) : data(d), rows(d.size()), cols(d.empty() ? 0 : d[0].size()) {}

Tensor Tensor::zeros(size_t r, size_t c) {
    return Tensor(r, c);
}

Tensor Tensor::random(size_t r, size_t c, double scale) {
    Tensor t(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t.data[i][j] = ((double)rand() / RAND_MAX) * scale;
    return t;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    Tensor res(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res.data[i][j] = data[i][j] + other.data[i][j];
    return res;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    Tensor res(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res.data[i][j] = data[i][j] - other.data[i][j];
    return res;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i][j] += other.data[i][j];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Tensor dimensions mismatch");
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i][j] -= other.data[i][j];
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (cols != other.rows) throw std::invalid_argument("Tensor multiplication dimension mismatch");
    Tensor res(rows, other.cols);
    res.fill(0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t k = 0; k < cols; ++k)
            for (size_t j = 0; j < other.cols; ++j)
                res.data[i][j] += data[i][k] * other.data[k][j];
    return res;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor res(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res.data[i][j] = data[i][j] * scalar;
    return res;
}

Tensor Tensor::transpose() const {
    Tensor res(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res.data[j][i] = data[i][j];
    return res;
}

Tensor Tensor::hadamard(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Hadamard dimension mismatch");
    Tensor res(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res.data[i][j] = data[i][j] * other.data[i][j];
    return res;
}

double Tensor::sum() const {
    double s = 0.0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            s += data[i][j];
    return s;
}

double Tensor::max() const {
    double m = data[0][0];
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (data[i][j] > m) m = data[i][j];
    return m;
}

void Tensor::fill(double val) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i][j] = val;
}

std::vector<double>& Tensor::operator[](size_t i) {
    return data[i];
}

const std::vector<double>& Tensor::operator[](size_t i) const {
    return data[i];
}

Tensor Tensor::get_row(size_t i) const {
    Tensor row(1, cols);
    for (size_t j = 0; j < cols; ++j) row[0][j] = (*this)[i][j];
    return row;
}
