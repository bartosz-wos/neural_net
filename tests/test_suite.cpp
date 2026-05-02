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
    // Summary
    // =================================================================
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  S4 Tests:       " << (exit_s4 == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Gradient Tests: " << (exit_gc == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================" << std::endl;

    // Any failure propagates
    return (exit_s4 != 0 || exit_gc != 0) ? 1 : 0;
}