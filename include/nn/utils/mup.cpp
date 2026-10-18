#include "mup.h"

#include <cmath>
#include <stdexcept>

namespace {

void require_nonzero(size_t v, const char* what) {
    // (D7) A zero dimension would make every downstream value inf or NaN.
    if (v == 0) {
        throw std::invalid_argument(std::string("mup: ") + what + " must be nonzero");
    }
}

// The ratio the whole relative API is built on. Exact 1.0 at w == w0 (D4),
// measured over 200000 randomised pairs before the test was written.
double width_ratio(const MupSpec& s) {
    require_nonzero(s.width, "width");
    require_nonzero(s.base_width, "base_width");
    return static_cast<double>(s.width) / static_cast<double>(s.base_width);
}

double check_finite(double v, const char* what) {
    if (!std::isfinite(v)) {
        throw std::invalid_argument(std::string("mup: ") + what +
                                    " produced a non-finite value");
    }
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Absolute form -- Table 3's cells, given the layer's own dims.
// ---------------------------------------------------------------------------

double mup_init_var_input(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}

double mup_init_var_hidden(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}

// (D1) The one squared cell in Table 3. Eq. 3's W3 ~ N(0, 1/n^2) confirms
// it independently.
double mup_init_var_output(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    const double n = static_cast<double>(fan_in);
    return 1.0 / (n * n);
}

// (D3) SGD scales on the input row. `fan_in` is not used: an input weight's
// fan_out is what matters here, and it is passed in. Kept in the signature
// so the call site reads symmetrically with the other two-row functions and
// so a future caller can see that both dims were considered.
double mup_sgd_lr_input(size_t fan_out, size_t /*fan_in*/) {
    require_nonzero(fan_out, "fan_out");
    return static_cast<double>(fan_out);
}

double mup_sgd_lr_hidden(size_t /*fan_in*/) {
    return 1.0;
}

double mup_sgd_lr_output(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}

// (D3) Adam does NOT scale on the input row. This is the cell Table 3's SP
// column puts the other way round, and the reason an SP-tuned Adam LR drifts
// with width.
double mup_adam_lr_input(size_t /*fan_out*/, size_t /*fan_in*/) {
    return 1.0;
}

double mup_adam_lr_hidden(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}

double mup_adam_lr_output(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}

// (D6) Biases take the input row. Appendix J: "just think of it as an input
// weight with input 1" -- so fan_out is the bias's own length.
double mup_bias_init_var(size_t fan_out) {
    return mup_init_var_input(fan_out);
}

double mup_sgd_lr_bias(size_t fan_out) {
    return mup_sgd_lr_input(fan_out, 1);
}

double mup_adam_lr_bias(size_t /*fan_out*/) {
    return mup_adam_lr_input(1, 1);
}

// ---------------------------------------------------------------------------
// Relative form.
// ---------------------------------------------------------------------------

double mup_init_var_rel(const MupSpec& spec, MupRole role) {
    const double r = width_ratio(spec);
    switch (role) {
        // The absolute cell is 1/fan_in, which SHRINKS as width grows, so
        // the multiplier from base_width to width is (1/w)/(1/w0) = 1/r.
        case MupRole::Input:
        case MupRole::Hidden:
            return check_finite(1.0 / r, "init_var_rel");
        // (D1) 1/fan_in^2 => the same ratio squared.
        case MupRole::Output:
            return check_finite(1.0 / (r * r), "init_var_rel");
    }
    throw std::invalid_argument("mup: unknown MupRole");
}

double mup_sgd_lr_rel(const MupSpec& spec, MupRole role) {
    const double r = width_ratio(spec);
    switch (role) {
        // (D3) scales UP with width.
        case MupRole::Input:
            return check_finite(r, "sgd_lr_rel");
        // Hidden LRs do not move with width at all.
        case MupRole::Hidden:
            return 1.0;
        // 1/fan_in => 1/r.
        case MupRole::Output:
            return check_finite(1.0 / r, "sgd_lr_rel");
    }
    throw std::invalid_argument("mup: unknown MupRole");
}

double mup_adam_lr_rel(const MupSpec& spec, MupRole role) {
    const double r = width_ratio(spec);
    switch (role) {
        // (D3) the invariant row.
        case MupRole::Input:
            return 1.0;
        case MupRole::Hidden:
        case MupRole::Output:
            return check_finite(1.0 / r, "adam_lr_rel");
    }
    throw std::invalid_argument("mup: unknown MupRole");
}

double mup_bias_init_var_rel(const MupSpec& spec) {
    return mup_init_var_rel(spec, MupRole::Input);
}

double mup_sgd_lr_bias_rel(const MupSpec& spec) {
    return mup_sgd_lr_rel(spec, MupRole::Input);
}

double mup_adam_lr_bias_rel(const MupSpec& spec) {
    return mup_adam_lr_rel(spec, MupRole::Input);
}

// ---------------------------------------------------------------------------
// Attention.
// ---------------------------------------------------------------------------

// (D5) sqrt(d0/d), not 1/d and not d0/d. Derivation in the header: the
// paper's `sqrt(d0)/d * q^T k` is the COMPLETE logit from the raw inner
// product, so the factor applied to a logit the caller already scaled by
// 1/sqrt(d) must supply the other half, giving sqrt(d0/d).
double mup_attn_scale(size_t head_dim, size_t base_head_dim, double alpha_attn) {
    require_nonzero(head_dim, "head_dim");
    require_nonzero(base_head_dim, "base_head_dim");
    const double d = static_cast<double>(head_dim);
    const double d0 = static_cast<double>(base_head_dim);
    return check_finite(alpha_attn * std::sqrt(d0 / d), "attn_scale");
}

double mup_attn_scale_rel(size_t head_dim, size_t base_head_dim, double alpha_attn) {
    return mup_attn_scale(head_dim, base_head_dim, alpha_attn);
}

double mup_attn_scale_rel_spec(const MupSpec& spec, double alpha_attn) {
    return mup_attn_scale(spec.head_dim, spec.base_head_dim, alpha_attn);
}

// ---------------------------------------------------------------------------
// Bundles.
// ---------------------------------------------------------------------------

namespace {

Tensor build_tensor(const std::vector<std::string>& roles, const MupSpec& spec,
                    bool adam) {
    Tensor out(1, roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        const std::string& r = roles[i];
        if (r == "input") {
            out(0, i) = adam ? mup_adam_lr_rel(spec, MupRole::Input)
                             : mup_sgd_lr_rel(spec, MupRole::Input);
        } else if (r == "hidden") {
            out(0, i) = adam ? mup_adam_lr_rel(spec, MupRole::Hidden)
                             : mup_sgd_lr_rel(spec, MupRole::Hidden);
        } else if (r == "output") {
            out(0, i) = adam ? mup_adam_lr_rel(spec, MupRole::Output)
                             : mup_sgd_lr_rel(spec, MupRole::Output);
        } else if (r == "bias") {
            out(0, i) = adam ? mup_adam_lr_bias_rel(spec)
                             : mup_sgd_lr_bias_rel(spec);
        } else {
            throw std::invalid_argument(
                "mup: unknown role '" + r +
                "' (expected input, hidden, output or bias)");
        }
    }
    return out;
}

}  // namespace

Tensor mup_adam_lr_tensor(const std::vector<std::string>& roles, const MupSpec& spec) {
    return build_tensor(roles, spec, /*adam=*/true);
}

Tensor mup_sgd_lr_tensor(const std::vector<std::string>& roles, const MupSpec& spec) {
    return build_tensor(roles, spec, /*adam=*/false);
}

// ---------------------------------------------------------------------------

double mup_update_ratio(double update_norm, double activation_norm) {
    // (D7) refuse rather than return NaN.
    if (activation_norm == 0.0) {
        throw std::invalid_argument("mup: activation_norm must be nonzero");
    }
    return update_norm / activation_norm;
}