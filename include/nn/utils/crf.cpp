#include "crf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {
namespace utils {
namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// logsumexp over a slice. Returns -inf when every term is -inf, which is
// the correct value for "no path reaches here" (S8: a masked transition).
// ---------------------------------------------------------------------------
double logsumexp(const std::vector<double>& v) {
    double mx = kNegInf;
    for (double x : v) {
        if (x > mx) mx = x;
    }
    if (mx == kNegInf) return kNegInf;
    double sum = 0.0;
    for (double x : v) sum += std::exp(x - mx);
    return mx + std::log(sum);
}

// ---------------------------------------------------------------------------
// Shape validation. The messages here are DISTINCT from one another and from
// the solvers' own guards: a substring that appears in two different messages
// discriminates nothing, so the tests match on the unique text.
// ---------------------------------------------------------------------------
void validate_shapes(const Tensor& emissions, const Tensor& transitions) {
    if (emissions.rows == 0 || emissions.cols == 0) {
        throw std::invalid_argument("crf: emissions must be non-empty");
    }
    if (transitions.rows != emissions.cols || transitions.cols != emissions.cols) {
        throw std::invalid_argument(
            "crf: transitions must be (K, K) matching the emissions' label count");
    }
    for (size_t t = 0; t < emissions.rows; ++t) {
        for (size_t k = 0; k < emissions.cols; ++k) {
            if (!std::isfinite(emissions(t, k))) {
                throw std::invalid_argument("crf: emissions must be finite");
            }
        }
    }
    for (size_t i = 0; i < transitions.rows; ++i) {
        for (size_t j = 0; j < transitions.cols; ++j) {
            const double v = transitions(i, j);
            // -inf is LEGAL (S8): it masks a label pair. NaN is not.
            if (std::isnan(v) || v == std::numeric_limits<double>::infinity()) {
                throw std::invalid_argument("crf: transitions must be finite or -inf");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The forward-backward messages, in the log domain (S7).
//
//   alpha[t][j] = log sum over paths ending at (t, j) of exp(score)
//     alpha[0][j] = emissions(0, j)                       <- S2: no transition
//     alpha[t][j] = emissions(t,j) + logsumexp_i(alpha[t-1][i] + A[i][j])   <- S1
//
//   beta[t][i] = log sum over paths from (t, i) to the end of exp(score)
//     beta[T-1][i] = 0                                    <- S3
//     beta[t][i]  = logsumexp_j(A[i][j] + e[t+1][j] + beta[t+1][j])
//
// The pair is returned as flat vectors of size T*K.
// ---------------------------------------------------------------------------
struct Messages {
    std::vector<double> alpha;   // T*K
    std::vector<double> beta;    // T*K
    double log_z = 0.0;
};

Messages compute_messages(const Tensor& emissions, const Tensor& transitions) {
    const size_t T = emissions.rows;
    const size_t K = emissions.cols;

    Messages m;
    m.alpha.assign(T * K, kNegInf);
    m.beta.assign(T * K, kNegInf);

    std::vector<double> terms(K);

    // --- forward: alpha -----------------------------------------------------
    // alpha_0 is EXACTLY the emissions (S2). There is no transition before
    // position 0, and adding one here is a classic off-by-one.
    for (size_t j = 0; j < K; ++j) {
        m.alpha[j] = emissions(0, j);
    }
    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < K; ++j) {
            for (size_t i = 0; i < K; ++i) {
                // transitions(i, j): FROM label i TO label j (S1).
                terms[i] = m.alpha[(t - 1) * K + i] + transitions(i, j);
            }
            m.alpha[t * K + j] = emissions(t, j) + logsumexp(terms);
        }
    }

    // --- the partition function ---------------------------------------------
    for (size_t j = 0; j < K; ++j) terms[j] = m.alpha[(T - 1) * K + j];
    m.log_z = logsumexp(terms);

    // --- backward: beta -----------------------------------------------------
    // beta_{T-1} is ZERO, not -inf (S3).
    for (size_t i = 0; i < K; ++i) {
        m.beta[(T - 1) * K + i] = 0.0;
    }
    // t runs from T-2 down to 0 INCLUSIVE: beta[t] is built from beta[t+1].
    for (size_t t = T - 1; t-- > 0;) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                terms[j] = transitions(i, j) + emissions(t + 1, j) + m.beta[(t + 1) * K + j];
            }
            m.beta[t * K + i] = logsumexp(terms);
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// The node marginals and the edge marginals (S5).
// ---------------------------------------------------------------------------
Tensor node_marginals_from(const Messages& m, size_t T, size_t K) {
    Tensor out(T, K);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < K; ++k) {
            const double v = m.alpha[t * K + k] + m.beta[t * K + k] - m.log_z;
            out(t, k) = (v == kNegInf) ? 0.0 : std::exp(v);
        }
    }
    return out;
}

Tensor edge_marginals_from(const Messages& m, const Tensor& emissions,
                            const Tensor& transitions, size_t T, size_t K) {
    Tensor out(T > 0 ? T - 1 : 0, K * K);
    for (size_t t = 0; t + 1 < T; ++t) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const double v = m.alpha[t * K + i] + transitions(i, j) +
                                 emissions(t + 1, j) + m.beta[(t + 1) * K + j] - m.log_z;
                out(t, i * K + j) = (v == kNegInf) ? 0.0 : std::exp(v);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Read a label index out of a (T, 1) tensor, with a range check.
// ---------------------------------------------------------------------------
std::vector<long long> read_labels(const Tensor& labels, size_t T, size_t K,
                                   const char* what) {
    if (labels.rows != T) {
        throw std::invalid_argument(std::string("crf: ") + what +
                                    " must have one row per position");
    }
    if (labels.cols != 1) {
        throw std::invalid_argument(std::string("crf: ") + what +
                                    " must be a single column of label indices");
    }
    std::vector<long long> out(T);
    for (size_t t = 0; t < T; ++t) {
        const double v = labels(t, 0);
        if (!std::isfinite(v) || v < 0.0 ||
            v >= static_cast<double>(K) || v != std::floor(v)) {
            throw std::invalid_argument(std::string("crf: ") + what +
                                        " must be label indices inside [0, K)");
        }
        out[t] = static_cast<long long>(v);
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
double crf_log_partition(const Tensor& emissions, const Tensor& transitions) {
    validate_shapes(emissions, transitions);
    return compute_messages(emissions, transitions).log_z;
}

// ---------------------------------------------------------------------------
Tensor crf_node_marginals(const Tensor& emissions, const Tensor& transitions) {
    validate_shapes(emissions, transitions);
    const Messages m = compute_messages(emissions, transitions);
    return node_marginals_from(m, emissions.rows, emissions.cols);
}

// ---------------------------------------------------------------------------
Tensor crf_edge_marginals(const Tensor& emissions, const Tensor& transitions) {
    validate_shapes(emissions, transitions);
    const Messages m = compute_messages(emissions, transitions);
    return edge_marginals_from(m, emissions, transitions, emissions.rows, emissions.cols);
}

// ---------------------------------------------------------------------------
double crf_viterbi(const Tensor& emissions, const Tensor& transitions,
                   Tensor* path_out) {
    validate_shapes(emissions, transitions);
    const size_t T = emissions.rows;
    const size_t K = emissions.cols;

    // delta[t][j] = the best score of a path ending at (t, j).
    std::vector<double> delta(T * K, kNegInf);
    std::vector<size_t> back(T * K, 0);

    // delta_0 is EXACTLY the emissions (S2), matching the alpha base case.
    for (size_t j = 0; j < K; ++j) delta[j] = emissions(0, j);

    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < K; ++j) {
            double best = kNegInf;
            size_t best_i = 0;
            for (size_t i = 0; i < K; ++i) {
                // transitions(i, j): FROM label i TO label j (S1).
                const double cand = delta[(t - 1) * K + i] + transitions(i, j);
                // Ties are broken by the SMALLEST label index, which makes the
                // decode deterministic. `cand > best` (not `>=`) does that.
                if (cand > best) {
                    best = cand;
                    best_i = i;
                }
            }
            delta[t * K + j] = emissions(t, j) + best;
            back[t * K + j] = best_i;
        }
    }

    double best = kNegInf;
    size_t best_j = 0;
    for (size_t j = 0; j < K; ++j) {
        if (delta[(T - 1) * K + j] > best) {
            best = delta[(T - 1) * K + j];
            best_j = j;
        }
    }

    if (path_out != nullptr) {
        Tensor path(T, 1);
        std::vector<size_t> idx(T);
        idx[T - 1] = best_j;
        for (size_t t = T - 1; t-- > 0;) {
            idx[t] = back[(t + 1) * K + idx[t + 1]];
        }
        for (size_t t = 0; t < T; ++t) {
            path(t, 0) = static_cast<double>(idx[t]);
        }
        *path_out = path;
    }
    return best;
}

// ---------------------------------------------------------------------------
void crf_marginal_log_likelihood_grad(const Tensor& emissions,
                                      const Tensor& transitions,
                                      const Tensor& g,
                                      Tensor* grad_emissions,
                                      Tensor* grad_transitions) {
    validate_shapes(emissions, transitions);
    const size_t T = emissions.rows;
    const size_t K = emissions.cols;

    if (g.rows != T || g.cols != K) {
        throw std::invalid_argument(
            "crf: the marginal weights g must have the emissions' shape");
    }
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < K; ++k) {
            if (!std::isfinite(g(t, k))) {
                throw std::invalid_argument("crf: the marginal weights g must be finite");
            }
        }
    }

    const Messages m = compute_messages(emissions, transitions);
    const Tensor node = node_marginals_from(m, T, K);
    const Tensor edge = edge_marginals_from(m, emissions, transitions, T, K);

    // --- the ALPHA side -----------------------------------------------------
    // alpha_{t+1}(j) = x_{t+1}(j) + logsumexp_i(alpha_t(i) + A[i,j]), so the
    // adjoint flows alpha_{t+1} -> alpha_t, i.e. DOWNWARD in t.
    //
    // `a[t][j]` is dObjective/d alpha[t][j], where Objective_alpha is
    // sum_{t,k} g[t,k] * alpha[t,k].
    std::vector<double> a(T * K, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < K; ++k) a[t * K + k] = g(t, k);
    }
    std::vector<double> terms(K);
    // The loop runs t = T-1 down to t = 1 INCLUSIVE, and it uses alpha[t-1].
    // A `t-- > 0` post-decrement would start at t = T-2 and run OFF THE END at
    // t = 0, where `t - 1` underflows a size_t to SIZE_MAX -- which is exactly
    // what the first build did: an out-of-bounds read that the -O0 build
    // turned into an abort and the -O2 build turned into heap corruption.
    // (This repo's own lesson: a debug build hides out-of-bounds reads that
    // -O2 turns into heap corruption.)
    for (size_t t = T - 1; t >= 1; --t) {
        // r[j] = logsumexp_i(alpha_{t-1}(i) + A[i,j]) = alpha[t][j] - x[t][j]
        for (size_t j = 0; j < K; ++j) {
            terms[j] = m.alpha[t * K + j] - emissions(t, j);
        }
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const double v = m.alpha[(t - 1) * K + i] + transitions(i, j) - terms[j];
                const double w = (v == kNegInf) ? 0.0 : std::exp(v);
                a[(t - 1) * K + i] += a[t * K + j] * w;
            }
        }
    }

    // --- the BETA side ------------------------------------------------------
    // beta_t(i) = logsumexp_j(A[i,j] + x_{t+1}(j) + beta_{t+1}(j)), so the
    // adjoint flows beta_t -> beta_{t+1}, i.e. UPWARD in t. The two
    // directions are OPPOSITE, and getting either backwards silently
    // produces a plausible gradient.
    std::vector<double> b(T * K, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < K; ++k) b[t * K + k] = g(t, k);
    }
    std::vector<double> grad_e_beta(T * K, 0.0);
    for (size_t t = 0; t + 1 < T; ++t) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const double v = transitions(i, j) + emissions(t + 1, j) +
                                 m.beta[(t + 1) * K + j] - m.beta[t * K + i];
                const double w = (v == kNegInf) ? 0.0 : std::exp(v);
                b[(t + 1) * K + j] += b[t * K + i] * w;
                grad_e_beta[(t + 1) * K + j] += b[t * K + i] * w;
            }
        }
    }

    // --- the -logZ part -----------------------------------------------------
    // Every log node[t,k] = alpha[t,k] + beta[t,k] - logZ carries a -logZ, so
    // the whole objective contributes -G * (the marginals), with G = sum g.
    double G = 0.0;
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < K; ++k) G += g(t, k);
    }

    if (grad_emissions != nullptr) {
        Tensor out(T, K);
        for (size_t t = 0; t < T; ++t) {
            for (size_t k = 0; k < K; ++k) {
                // d/dx of the alpha part: alpha[t][k] contains x[t][k]
                // directly, and the adjoint `a` already carries the effect of
                // the recursion through the higher positions.
                const double alpha_part = a[t * K + k];
                // d/dx of the beta part: x[t][k] appears in beta[t-1][*], so
                // the beta contribution to position t was accumulated there.
                // grad_e_beta[0] is ZERO (position 0 has no beta predecessor)
                // and grad_e_beta[T-1] is NOT -- an off-by-one here silently
                // drops a whole position's contribution.
                const double beta_part = grad_e_beta[t * K + k];
                out(t, k) = alpha_part + beta_part - G * node(t, k);
            }
        }
        *grad_emissions = out;
    }

    if (grad_transitions != nullptr) {
        Tensor out(K, K);
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                double acc = 0.0;
                // the alpha side: A[i,j] appears in alpha_{t+1}(j) for every t
                for (size_t t = 0; t + 1 < T; ++t) {
                    const double r = m.alpha[(t + 1) * K + j] - emissions(t + 1, j);
                    const double v = m.alpha[t * K + i] + transitions(i, j) - r;
                    const double w = (v == kNegInf) ? 0.0 : std::exp(v);
                    acc += a[(t + 1) * K + j] * w;
                }
                // the beta side: A[i,j] appears in beta_t(i) for every t
                for (size_t t = 0; t + 1 < T; ++t) {
                    const double v = transitions(i, j) + emissions(t + 1, j) +
                                     m.beta[(t + 1) * K + j] - m.beta[t * K + i];
                    const double w = (v == kNegInf) ? 0.0 : std::exp(v);
                    acc += b[t * K + i] * w;
                }
                // and the -logZ part: dlogZ/dA[i,j] = sum_t edge[t,i,j]
                double edge_sum = 0.0;
                for (size_t t = 0; t + 1 < T; ++t) {
                    edge_sum += edge(t, i * K + j);
                }
                out(i, j) = acc - G * edge_sum;
            }
        }
        *grad_transitions = out;
    }
}

// ---------------------------------------------------------------------------
void crf_joint_nll_grad(const Tensor& emissions, const Tensor& transitions,
                        const Tensor& labels,
                        Tensor* grad_emissions,
                        Tensor* grad_transitions) {
    validate_shapes(emissions, transitions);
    const size_t T = emissions.rows;
    const size_t K = emissions.cols;

    const std::vector<long long> y = read_labels(labels, T, K, "labels");

    const Messages m = compute_messages(emissions, transitions);
    const Tensor node = node_marginals_from(m, T, K);
    const Tensor edge = edge_marginals_from(m, emissions, transitions, T, K);

    if (grad_emissions != nullptr) {
        Tensor out(T, K);
        for (size_t t = 0; t < T; ++t) {
            for (size_t k = 0; k < K; ++k) {
                out(t, k) = node(t, k);
            }
            out(t, static_cast<size_t>(y[t])) -= 1.0;
        }
        *grad_emissions = out;
    }

    if (grad_transitions != nullptr) {
        Tensor out(K, K);
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t + 1 < T; ++t) {
                    acc += edge(t, i * K + j);
                }
                out(i, j) = acc;
            }
        }
        for (size_t t = 0; t + 1 < T; ++t) {
            out(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1])) -= 1.0;
        }
        *grad_transitions = out;
    }
}

// ---------------------------------------------------------------------------
double crf_sequence_score(const Tensor& emissions, const Tensor& transitions,
                          const Tensor& labels) {
    validate_shapes(emissions, transitions);
    const size_t T = emissions.rows;
    const size_t K = emissions.cols;
    const std::vector<long long> y = read_labels(labels, T, K, "labels");

    double score = 0.0;
    for (size_t t = 0; t < T; ++t) {
        score += emissions(t, static_cast<size_t>(y[t]));
        if (t + 1 < T) {
            score += transitions(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1]));
        }
    }
    return score;
}

}  // namespace utils
}  // namespace nn
