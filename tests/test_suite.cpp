// test_suite.cpp — main test runner aggregating all tests
#include <iostream>
#include <iomanip>
#include <cstdlib>

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "       Neural Net Test Suite               " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;

    // =================================================================
    // Run S4 layer tests
    // =================================================================
    std::cout << ">>> Running S4 Layer Tests..." << std::endl;
    int ret_s4 = std::system("./build/test_s4");
    int exit_s4 = WEXITSTATUS(ret_s4);

    // =================================================================
    // Run Gradient Correctness tests
    // =================================================================
    std::cout << std::endl << ">>> Running Gradient Correctness Tests..." << std::endl;
    int ret_gc = std::system("./build/test_gradient_check");
    int exit_gc = WEXITSTATUS(ret_gc);

    // =================================================================
    // Run RMSNorm gradient tests
    // =================================================================
    std::cout << std::endl << ">>> Running RMSNorm Gradient Tests..." << std::endl;
    int ret_rms = std::system("./build/test_rmsnorm");
    int exit_rms = WEXITSTATUS(ret_rms);

    // =================================================================
    // Run WGAN-GP gradient tests
    // =================================================================
    std::cout << std::endl << ">>> Running WGAN-GP Gradient Tests..." << std::endl;
    int ret_wgan = std::system("./build/test_wgan_gp");
    int exit_wgan = WEXITSTATUS(ret_wgan);

    // =================================================================
    // Run FlashAttention gradient tests
    // =================================================================
    std::cout << std::endl << ">>> Running FlashAttention Gradient Tests..." << std::endl;
    int ret_flash = std::system("./build/test_flash_attention");
    int exit_flash = WEXITSTATUS(ret_flash);

    // =================================================================
    // Run ViT gradient tests
    // =================================================================
    std::cout << std::endl << ">>> Running ViT Gradient Tests..." << std::endl;
    int ret_vit = std::system("./build/test_vit");
    int exit_vit = WEXITSTATUS(ret_vit);

    // =================================================================
    // Summary
    // =================================================================
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  S4 Tests:           " << (exit_s4 == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Gradient Tests:     " << (exit_gc == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  RMSNorm Tests:      " << (exit_rms == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  WGAN-GP Tests:      " << (exit_wgan == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  FlashAttn Tests:    " << (exit_flash == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  ViT Tests:          " << (exit_vit == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================" << std::endl;

    // Any failure propagates
    return (exit_s4 != 0 || exit_gc != 0 || exit_rms != 0 ||
            exit_wgan != 0 || exit_flash != 0 || exit_vit != 0) ? 1 : 0;
}