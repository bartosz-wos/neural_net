// Manual gradient computation trace
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/gin.h"

using namespace std;

int main() {
    size_t N = 3, in_f = 2, out_f = 4;

    Tensor adj(N, N);
    adj(0, 1) = 1; adj(0, 2) = 1;
    adj(1, 0) = 1; adj(1, 2) = 1;
    adj(2, 0) = 1; adj(2, 1) = 1;

    Tensor input(N, in_f);
    input(0, 0) = 0.5;  input(0, 1) = -0.3;
    input(1, 0) = 0.8;  input(1, 1) = 0.2;
    input(2, 0) = -0.1; input(2, 1) = 0.4;

    GIN0Layer layer(in_f, out_f);

    // Get W
    const Tensor& W = *layer.parameters()[0];

    cout << "=== Manual gradient trace ===" << endl;

    // Forward pass
    cout << "\n--- Forward pass ---" << endl;
    cout << "adj matrix (row i means 'i aggregates from j'):" << endl;
    for (size_t i = 0; i < N; ++i) {
        cout << "Row " << i << ": ";
        for (size_t j = 0; j < N; ++j) cout << adj(i,j) << " ";
        cout << endl;
    }

    // Compute input_plus_agg manually
    cout << "\n--- input_plus_agg = input + agg ---" << endl;
    for (size_t i = 0; i < N; ++i) {
        double agg0 = 0, agg1 = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i,j) > 1e-9) {
                agg0 += input(j,0);
                agg1 += input(j,1);
            }
        }
        cout << "Node " << i << ": input=[" << input(i,0) << "," << input(i,1)
             << "] agg=[" << agg0 << "," << agg1 << "]"
             << " combined=[" << input(i,0)+agg0 << "," << input(i,1)+agg1 << "]" << endl;
    }

    // Compute output: combined @ W^T
    cout << "\n--- Output (combined @ W^T) ---" << endl;
    double output_sum = 0;
    for (size_t i = 0; i < N; ++i) {
        cout << "Node " << i << " output features:" << endl;
        for (size_t k = 0; k < out_f; ++k) {
            double val = 0;
            for (size_t j = 0; j < in_f; ++j) {
                val += (input(i,j) + (j==0 ? (adj(i,0)>0?input(0,0):0)+(adj(i,1)>0?input(1,0):0)+(adj(i,2)>0?input(2,0):0)
                                        : (adj(i,0)>0?input(0,1):0)+(adj(i,1)>0?input(1,1):0)+(adj(i,2)>0?input(2,1):0)) ) * W(k,j);
            }
            // Recalculate combined
            double combined0 = input(i,0);
            double combined1 = input(i,1);
            for (size_t j = 0; j < N; ++j) {
                if (adj(i,j) > 1e-9) {
                    combined0 += input(j,0);
                    combined1 += input(j,1);
                }
            }
            val = 0;
            for (size_t j = 0; j < in_f; ++j) {
                val += combined0 * W(0,j) + combined1 * W(1,j);
            }
            (void)val;
        }
    }

    // Just print actual output
    Tensor out = layer.forward_with_adj(input, adj);
    output_sum = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < out_f; ++k) {
            output_sum += out(i,k);
        }
    }
    cout << "Loss (sum of output) = " << output_sum << endl;

    // Now compute numerical gradient for input[0][0]
    double eps = 1e-5;
    double orig = input(0,0);

    auto compute_loss = [](const Tensor& t) { double s=0; for(size_t i=0;i<t.rows;i++) for(size_t j=0;j<t.cols;j++) s+=t(i,j); return s; };

    // loss_plus
    Tensor input_plus = input.clone();
    input_plus(0,0) = orig + eps;
    Tensor out_plus = layer.forward_with_adj(input_plus, adj);
    double loss_plus = compute_loss(out_plus);

    // loss_minus
    Tensor input_minus = input.clone();
    input_minus(0,0) = orig - eps;
    Tensor out_minus = layer.forward_with_adj(input_minus, adj);
    double loss_minus = compute_loss(out_minus);

    double num_grad = (loss_plus - loss_minus) / (2*eps);
    cout << "\n--- Numerical gradient for input[0][0] ---" << endl;
    cout << "loss_plus = " << loss_plus << endl;
    cout << "loss_minus = " << loss_minus << endl;
    cout << "num_grad = " << num_grad << endl;

    // Now compute analytical gradient step by step
    cout << "\n--- Analytical gradient computation ---" << endl;

    // grad_output = all ones
    Tensor grad_output(N, out_f);
    grad_output.fill(1.0);

    // Step 1: grad_agg = grad_output @ W  (N x in_f)
    cout << "\nStep 1: grad_agg = grad_output @ W" << endl;
    cout << "grad_agg (N=" << N << ", in_f=" << in_f << "):" << endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            double sum = 0;
            for (size_t k = 0; k < out_f; ++k) {
                sum += grad_output(i,k) * W(k,j);
            }
            cout << " grad_agg[" << i << "][" << j << "] = " << sum << endl;
        }
    }

    // Step 2: grad_W = grad_agg^T @ input_plus_agg
    cout << "\nStep 2: grad_W (out x in) = grad_agg^T @ input_plus_agg" << endl;

    // Recompute input_plus_agg
    double input_plus_agg[3][2];
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_f; ++f) {
            double agg = 0;
            for (size_t j = 0; j < N; ++j) {
                if (adj(i,j) > 1e-9) agg += input(j,f);
            }
            input_plus_agg[i][f] = input(i,f) + agg;
        }
    }

    cout << "input_plus_agg:" << endl;
    for (size_t i = 0; i < N; ++i) {
        cout << " [" << input_plus_agg[i][0] << ", " << input_plus_agg[i][1] << "]" << endl;
    }

    // grad_W[i][j] = sum_k grad_agg[k][i] * input_plus_agg[k][j]
    cout << "\nGrad_W:" << endl;
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            double sum = 0;
            for (size_t k = 0; k < N; ++k) {
                // grad_agg[k][i] - use formula above
                double g_ki = 0;
                for (size_t m = 0; m < out_f; ++m) {
                    g_ki += grad_output(k,m) * W(m,i);
                }
                sum += g_ki * input_plus_agg[k][j];
            }
            cout << " grad_W[" << i << "][" << j << "] = " << sum << endl;
        }
    }

    // Step 3: grad_input = grad_agg (self-loop) + aggregation contribution
    cout << "\nStep 3: grad_input = grad_agg (self-loop) + agg_contrib" << endl;

    // Self-loop
    double grad_input[3][2];
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_f; ++f) {
            grad_input[i][f] = 0;
        }
    }

    // For self-loop, grad_agg[i][f] is correct
    // grad_agg[i][f] = sum_k grad_output[i][k] * W(f,k)
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_f; ++f) {
            double g_agg = 0;
            for (size_t k = 0; k < out_f; ++k) {
                g_agg += grad_output(i,k) * W(k,f);
            }
            grad_input[i][f] += g_agg;
        }
    }

    cout << "After self-loop contribution:" << endl;
    for (size_t i = 0; i < N; ++i) {
        cout << " grad_input[" << i << "] = [" << grad_input[i][0] << ", " << grad_input[i][1] << "]" << endl;
    }

    // Aggregation contribution: for each edge j->i (adj[i][j]=1 means j contributes to i),
    // grad_agg[i] flows back to node j
    cout << "\nAggregation contribution (using adj_(j,i) - correct direction):" << endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj(j, i) > 1e-9) {  // j contributes to i in forward
                cout << "  Edge " << j << " -> " << i << " (adj[" << j << "][" << i << "]=" << adj(j,i) << ")" << endl;
                double g_agg_i0 = 0, g_agg_i1 = 0;
                for (size_t k = 0; k < out_f; ++k) {
                    g_agg_i0 += grad_output(i,k) * W(k,0);
                    g_agg_i1 += grad_output(i,k) * W(k,1);
                }
                cout << "    grad_agg[" << i << "] = [" << g_agg_i0 << ", " << g_agg_i1 << "]" << endl;
                cout << "    adds to grad_input[" << j << "]" << endl;
                grad_input[j][0] += g_agg_i0;
                grad_input[j][1] += g_agg_i1;
            }
        }
    }

    cout << "\nFinal grad_input:" << endl;
    for (size_t i = 0; i < N; ++i) {
        cout << " [" << grad_input[i][0] << ", " << grad_input[i][1] << "]" << endl;
    }

    return 0;
}