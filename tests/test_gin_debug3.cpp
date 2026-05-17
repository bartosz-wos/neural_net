// Minimal debug
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gin.h"

using namespace std;

double compute_loss(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t(i, j);
    return s;
}

int main() {
    size_t N = 3, in_f = 2, out_f = 4;
    
    Tensor adj(N, N);
    adj(0, 1) = 1; adj(0, 2) = 1;
    adj(1, 0) = 1; adj(1, 2) = 1;
    adj(2, 0) = 1; adj(2, 1) = 1;

    Tensor input(N, in_f);
    input(0, 0) = 0.5; input(0, 1) = -0.3;
    input(1, 0) = 0.8; input(1, 1) = 0.2;
    input(2, 0) = -0.1; input(2, 1) = 0.4;

    GINLayer layer(in_f, out_f, 16, 2);
    
    cout << "First forward:" << endl;
    Tensor out1 = layer.forward_with_adj(input, adj);
    cout << "Loss1 = " << compute_loss(out1) << endl;
    cout << "Output1:" << endl;
    for (size_t i = 0; i < out1.rows; ++i) {
        for (size_t j = 0; j < out1.cols; ++j)
            cout << " " << out1(i,j);
        cout << endl;
    }
    
    cout << "\nSecond forward (same input):" << endl;
    Tensor out2 = layer.forward_with_adj(input, adj);
    cout << "Loss2 = " << compute_loss(out2) << endl;
    cout << "Output2:" << endl;
    for (size_t i = 0; i < out2.rows; ++i) {
        for (size_t j = 0; j < out2.cols; ++j)
            cout << " " << out2(i,j);
        cout << endl;
    }
    
    return 0;
}