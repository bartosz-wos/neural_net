// Debug: trace gradient flow direction
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gin.h"

using namespace std;

int main() {
    cout << "=== Adjacency Direction Debug ===" << endl;

    size_t N = 3;
    size_t in_f = 2;

    // Symmetric adjacency (undirected)
    Tensor adj(N, N);
    adj(0, 1) = 1; adj(0, 2) = 1;  // 0 connects to 1,2
    adj(1, 0) = 1; adj(1, 2) = 1;  // 1 connects to 0,2
    adj(2, 0) = 1; adj(2, 1) = 1;  // 2 connects to 0,1

    Tensor input(N, in_f);
    input(0, 0) = 0.5;  input(0, 1) = -0.3;
    input(1, 0) = 0.8;  input(1, 1) = 0.2;
    input(2, 0) = -0.1; input(2, 1) = 0.4;

    // Forward aggregation: adj[i][j] means "j contributes to i"
    // agg[i] = sum_{j where adj[i][j]=1} input[j]
    cout << "\n--- Forward: agg[i] = sum_{j->i} input[j] ---" << endl;
    for (size_t i = 0; i < N; ++i) {
        double agg0 = 0, agg1 = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 1e-9) {
                cout << "  adj[" << i << "][" << j << "]=1: input[" << j << "]=["
                     << input(j,0) << "," << input(j,1) << "] -> agg[" << i << "]" << endl;
                agg0 += input(j, 0);
                agg1 += input(j, 1);
            }
        }
        cout << "agg[" << i << "] = [" << agg0 << "," << agg1 << "]" << endl;
    }

    cout << "\n--- Backward: grad flows from i to j where adj[j][i]=1 ---" << endl;
    cout << "(i.e., grad_agg[i] should go to all j that contributed to i)" << endl;
    for (size_t i = 0; i < N; ++i) {
        cout << "grad_agg[" << i << "] flows to:" << endl;
        for (size_t j = 0; j < N; ++j) {
            // Current code uses adj_(i,j): if adj_(i,j) > 0, add grad_agg[i] to grad_input[j]
            // But this is wrong direction! Should be adj_(j,i) > 0
            bool current_code = (adj(i, j) > 1e-9);  // adj_(i,j) - WRONG
            bool correct = (adj(j, i) > 1e-9);       // adj_(j,i) - correct
            cout << "  j=" << j << ": current uses adj[" << i << "][" << j << "]="
                 << (current_code ? "true" : "false")
                 << ", correct uses adj[" << j << "][" << i << "]="
                 << (correct ? "true" : "false") << endl;
        }
    }

    return 0;
}