// probe_inert_m7.cpp — differential probe for the M7 mutation.
//
// M7 removes the final clamp `if (best_T < t_min) best_T = t_min;`. The claim
// under test is that this mutation is INERT, i.e. removing the line changes no
// observable output on any of these fixtures. The harness runs this probe
// twice — once against the mutated source and once against the pristine
// baseline — and requires the output to be BIT-IDENTICAL. If it is not, the
// "inert" claim is wrong and the harness fails.
//
// The fixtures are chosen to cover every path that assigns `best_T`:
//   * interior optimum inside the bracket  -> `best_T = exp(mid)`
//   * optimum AT the lower boundary       -> `best_T = t_min` (the init)
//   * optimum AT the upper boundary       -> `best_T = t_max` (the endpoint eval)
//   * bracket excluding the true optimum  -> must return the nearest endpoint
#include "nn/utils/temperature_scaling.h"
#include <iostream>
#include <iomanip>
#include <vector>

static Tensor mk(std::vector<std::vector<double>> r) { return Tensor(r); }
static Tensor lab(std::vector<double> y) {
    Tensor t(y.size(), 1);
    for (size_t i = 0; i < y.size(); ++i) t(i, 0) = y[i];
    return t;
}

int main() {
    std::cout << std::setprecision(15);

    // FIX_C2: interior optimum at T* = 11.6951
    auto c2 = mk({{3.0, 0.5, 0.0}, {0.5, 3.0, 0.0}, {9.0, -8.0, 0.5}, {4.0, 0.0, 3.0}});
    auto c2y = lab({0, 1, 2, 2});
    std::cout << "c2 default      = " << fit_temperature(c2, c2y) << "\n";
    std::cout << "c2 [0.5,2]      = " << fit_temperature(c2, c2y, 0.5, 2.0) << "\n";
    std::cout << "c2 [5,50]       = " << fit_temperature(c2, c2y, 5.0, 50.0) << "\n";
    std::cout << "c2 [0.1,10]     = " << fit_temperature(c2, c2y, 0.1, 10.0) << "\n";
    std::cout << "c2 [1e-3,1e3]   = " << fit_temperature(c2, c2y, 1e-3, 1e3) << "\n";

    // Separable: optimum is the lower boundary (T -> 0 limit).
    auto sep = mk({{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}});
    auto sepy = lab({0, 1, 2});
    std::cout << "separable       = " << fit_temperature(sep, sepy) << "\n";
    std::cout << "sep nll         = " << temperature_nll(sep, sepy, fit_temperature(sep, sepy)) << "\n";
    std::cout << "sep [1e-4,1]    = " << fit_temperature(sep, sepy, 1e-4, 1.0) << "\n";

    // Underconfident: optimum below 1 but interior.
    auto e = mk({{0.05, -0.05, 0.0}, {-0.04, 0.06, 0.0},
                 {0.10, 0.0, -0.10}, {-0.05, 0.05, 0.0}});
    auto ey = lab({0, 1, 0, 2});
    std::cout << "underconfident  = " << fit_temperature(e, ey) << "\n";
    std::cout << "uc [1e-2,1]     = " << fit_temperature(e, ey, 1e-2, 1.0) << "\n";
    std::cout << "uc [1,1e2]      = " << fit_temperature(e, ey, 1.0, 1e2) << "\n";

    // Closed-form fixture: interior optimum at m/ln4 = 2.164.
    std::vector<std::vector<double>> rows;
    std::vector<double> y;
    for (size_t i = 0; i < 30; ++i) {
        std::vector<double> r(3, 0.0);
        r[0] = 3.0;
        rows.push_back(r);
        y.push_back(i % 3 < 2 ? 0.0 : 1.0);
    }
    auto z = mk(rows);
    auto zy = lab(y);
    std::cout << "closed form     = " << fit_temperature(z, zy) << "\n";

    // calibrate_logits also routes through fit_temperature.
    std::cout << "calibrated      = " << calibrate_logits(c2, c2y).sum() << "\n";
    std::cout << "probs row0      = " << calibrate_logits(c2, c2y)[0][0] << " "
              << calibrate_logits(c2, c2y)[0][1] << " " << calibrate_logits(c2, c2y)[0][2] << "\n";
    return 0;
}