#include <iostream>
#include <cassert>
#include <cmath>
#include <nn/layers/pooling/pool_layer.h>
#include <nn/core/tensor.h>

int main() {
    std::cout << "=== AvgPool2D Test ===\n";

    // Test 1: kH=2, kW=2, H_in=4, W_in=4, stride=2 -> H_out=2, W_out=2
    {
        AvgPool2D pool(2, 2, 4, 4, 2, 2);

        // Input: batch=1, channels=1, H=4, W=4 -> cols = 1*4*4 = 16
        // Layout: [c*H*W + h*W + w]
        // [[1, 2, 3, 4],
        //  [5, 6, 7, 8],
        //  [9, 10, 11, 12],
        //  [13, 14, 15, 16]]
        Tensor input(1, 16);
        input[0][0] = 1;  input[0][1] = 2;  input[0][2] = 3;  input[0][3] = 4;
        input[0][4] = 5;  input[0][5] = 6;  input[0][6] = 7;  input[0][7] = 8;
        input[0][8] = 9;  input[0][9] = 10; input[0][10] = 11; input[0][11] = 12;
        input[0][12] = 13; input[0][13] = 14; input[0][14] = 15; input[0][15] = 16;

        Tensor output = pool.forward(input);

        // Output shape: (1, 1, H_out, W_out) = (1, 1*2*2) = (1, 4)
        std::cout << "Output shape: (" << output.rows << ", " << output.cols << ")\n";
        assert(output.rows == 1 && output.cols == 4);

        // Top-left window (0,0): [[1,2],[5,6]] -> avg = 14/4 = 3.5
        // Top-right window (0,1): [[3,4],[7,8]] -> avg = 22/4 = 5.5
        // Bottom-left window (1,0): [[9,10],[13,14]] -> avg = 46/4 = 11.5
        // Bottom-right window (1,1): [[11,12],[15,16]] -> avg = 54/4 = 13.5
        double expected[4] = {3.5, 5.5, 11.5, 13.5};
        std::cout << "Output values: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << output[0][i] << " ";
            assert(std::abs(output[0][i] - expected[i]) < 1e-9);
        }
        std::cout << "\n";
        std::cout << "Test 1 PASSED: forward values correct\n";
    }

    // Test 2: backward pass
    {
        AvgPool2D pool(2, 2, 4, 4, 2, 2);
        Tensor input(1, 16);
        for (int i = 0; i < 16; ++i) input[0][i] = static_cast<double>(i + 1);

        Tensor output = pool.forward(input);

        // Gradient output all 1s
        Tensor grad_out(1, 4);
        grad_out.fill(1.0);

        Tensor grad_input = pool.backward(grad_out, 0.01);

        std::cout << "Grad input shape: (" << grad_input.rows << ", " << grad_input.cols << ")\n";
        assert(grad_input.rows == 1 && grad_input.cols == 16);

        // Each input position belongs to exactly 1 window (stride=2, no overlap)
        // So grad_input[n] = grad_out[window_of_n] / 4 = 1/4 for all
        // Check non-zero and finite
        bool all_finite = true;
        bool all_positive = true;
        for (int i = 0; i < 16; ++i) {
            double v = grad_input[0][i];
            if (!std::isfinite(v)) all_finite = false;
            if (v <= 0) all_positive = false;
        }
        assert(all_finite);
        assert(all_positive);
        std::cout << "Grad input sum: " << grad_input[0][0] << " (expected 0.25)\n";
        assert(std::abs(grad_input[0][0] - 0.25) < 1e-9);
        std::cout << "Test 2 PASSED: backward gradients non-zero and finite\n";
    }

    // Test 3: stride=1 (overlapping windows)
    {
        AvgPool2D pool(2, 2, 4, 4, 1, 1);
        // H_out = (4-2)/1+1 = 3, W_out = 3
        Tensor input(1, 16);
        for (int i = 0; i < 16; ++i) input[0][i] = static_cast<double>(i + 1);

        Tensor output = pool.forward(input);
        std::cout << "Output shape (stride=1): (" << output.rows << ", " << output.cols << ")\n";
        assert(output.rows == 1 && output.cols == 9); // 1*3*3=9

        // Window (0,0): [[1,2],[5,6]] -> avg = 14/4 = 3.5
        assert(std::abs(output[0][0] - 3.5) < 1e-9);
        std::cout << "Test 3 PASSED: stride=1 works correctly\n";
    }

    std::cout << "\n=== ALL AvgPool2D TESTS PASSED ===\n";
    return 0;
}