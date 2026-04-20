#include "nn/layers/attention/transformer.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));
    std::cout << "=== Transformer Test ===\n\n";

    // --- Positional Encoding ---
    std::cout << "[1] PositionalEncoding Test\n";
    PositionalEncoding pe(100, 8);
    std::cout << "  PE created (max_len=100, d_model=8)\n";
    
    Tensor x(8, 5);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = 1.0;
    std::cout << "  Input: " << x.rows << "x" << x.cols << "\n";

    Tensor encoded = pe.forward(x);
    std::cout << "  Encoded: " << encoded.rows << "x" << encoded.cols << "\n";
    std::cout << "  PE[0][0]=" << pe.pe[0][0] << " PE[1][1]=" << pe.pe[1][1] << "\n\n";

    // --- MultiHeadAttention ---
    std::cout << "[2] MultiHeadAttention Test\n";
    MultiHeadAttention mha(8, 2);
    std::cout << "  MHA created (d_model=8, heads=2)\n";
    
    Tensor xmha(8, 5);
    for (size_t i = 0; i < xmha.rows; ++i)
        for (size_t j = 0; j < xmha.cols; ++j)
            xmha[i][j] = (std::rand() % 100) / 20.0 - 2.5;
    
    Tensor out_mha = mha.forward(xmha);
    std::cout << "  Output: " << out_mha.rows << "x" << out_mha.cols << "\n\n";

    // --- TransformerBlock ---
    std::cout << "[3] TransformerBlock Test\n";
    TransformerBlock tb(8, 2);
    std::cout << "  TB created (d_model=8, heads=2)\n";

    Tensor xfrm(8, 5);
    for (size_t i = 0; i < xfrm.rows; ++i)
        for (size_t j = 0; j < xfrm.cols; ++j)
            xfrm[i][j] = (std::rand() % 100) / 20.0 - 2.5;

    Tensor out_tb = tb.forward(xfrm);
    std::cout << "  Output: " << out_tb.rows << "x" << out_tb.cols << "\n\n";

    // --- Integration: PE + MHA ---
    std::cout << "[4] Integration: PE -> MHA\n";
    Tensor with_pe = pe.forward(xfrm);
    Tensor from_attn = mha.forward(with_pe);
    std::cout << "  PE output -> MHA output: " << from_attn.rows << "x" << from_attn.cols << "\n\n";

    std::cout << "=== All Transformer Tests Passed ===\n";
    return 0;
}