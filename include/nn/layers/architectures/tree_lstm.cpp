#include "tree_lstm.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ============================================================================
// TreeLSTM implementation
// ============================================================================

TreeLSTM::TreeLSTM(size_t in_dim, size_t hidden_size)
    : input_size_(in_dim), hidden_size_(hidden_size), last_num_nodes_(0)
{
    // Combined weight matrices, gate layout [i, f, o, u]
    W = Tensor(4 * hidden_size, in_dim);
    U = Tensor(4 * hidden_size, hidden_size);
    b = Tensor(1, 4 * hidden_size);
    grad_W = Tensor(4 * hidden_size, in_dim);
    grad_U = Tensor(4 * hidden_size, hidden_size);
    grad_b = Tensor(1, 4 * hidden_size);
    grad_W.fill(0.0);
    grad_U.fill(0.0);
    grad_b.fill(0.0);

    // Xavier/Glorot init
    double scale_w = std::sqrt(2.0 / (in_dim + hidden_size));
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale_w);
    for (size_t i = 0; i < W.rows; ++i)
        for (size_t j = 0; j < W.cols; ++j)
            W(i, j) = dis(gen);
    for (size_t i = 0; i < U.rows; ++i)
        for (size_t j = 0; j < U.cols; ++j)
            U(i, j) = dis(gen);

    // Biases: zero except forget-gate bias = 1.0 (rows [H, 2H) of b).
    b.fill(0.0);
    for (size_t i = hidden_size; i < 2 * hidden_size; ++i) {
        b(0, i) = 1.0;
    }
}

Tensor TreeLSTM::forward(const Tensor& node_features,
                         const std::vector<std::vector<int>>& children) {
    size_t N = node_features.rows;
    if (node_features.cols != input_size_) {
        throw std::invalid_argument("TreeLSTM: input feature dim mismatch");
    }

    // Cache input (deep clone so the user's tensor isn't corrupted by
    // numerical-perturbation gradient checks).
    last_input_ = Tensor(N, input_size_);
    for (size_t i = 0; i < N * input_size_; ++i) {
        last_input_.data[i] = node_features.data[i];
    }
    last_children_ = children;
    last_num_nodes_ = static_cast<int>(N);

    // Post-order: children before parents.
    last_post_order_ = build_post_order(children, static_cast<int>(N));

    // Allocate per-node caches.
    last_h_      = Tensor(N, hidden_size_);
    last_c_      = Tensor(N, hidden_size_);
    last_h_sum_  = Tensor(N, hidden_size_);
    last_i_pre_  = Tensor(N, hidden_size_);
    last_f_pre_  = Tensor(N, hidden_size_);  // will be filled per-edge below
    last_o_pre_  = Tensor(N, hidden_size_);
    last_u_pre_  = Tensor(N, hidden_size_);
    last_i_gate_ = Tensor(N, hidden_size_);
    last_o_gate_ = Tensor(N, hidden_size_);
    last_u_tanh_ = Tensor(N, hidden_size_);

    // Edge-level forget-gate cache. Total edges M = Σ |C(j)|.
    size_t M = 0;
    for (size_t j = 0; j < N; ++j) M += children[j].size();
    last_edge_parent_.resize(M);
    last_edge_child_.resize(M);
    last_edge_offset_.resize(N + 1, 0);
    {
        size_t e = 0;
        for (size_t j = 0; j < N; ++j) {
            last_edge_offset_[j] = e;
            for (int k : children[j]) {
                last_edge_parent_[e] = static_cast<int>(j);
                last_edge_child_[e]  = k;
                ++e;
            }
        }
        last_edge_offset_[N] = M;
    }
    last_f_gates_ = Tensor(M, hidden_size_);

    // Zero h, c, h_sum caches.
    for (size_t i = 0; i < N; ++i) {
        for (size_t h = 0; h < hidden_size_; ++h) {
            last_h_(i, h) = 0.0;
            last_c_(i, h) = 0.0;
            last_h_sum_(i, h) = 0.0;
        }
    }

    // Allocate backward scratch buffers.
    grad_input_ = Tensor(N, input_size_);
    grad_h_sum_ = Tensor(N, hidden_size_);
    grad_c_     = Tensor(N, hidden_size_);
    dh_sum_     = Tensor(N, hidden_size_);

    // Forward in post-order: compute h_j, c_j for each j after its children.
    for (int j : last_post_order_) {
        // h_sum_j = Σ_{k ∈ C(j)} h_k.  For leaves this is 0 (no children).
        for (int k : children[j]) {
            for (size_t h = 0; h < hidden_size_; ++h) {
                last_h_sum_(j, h) += last_h_(k, h);
            }
        }

        // Combined W @ x_j + b  (size 4*hidden, result stored into per-gate
        // pre-activations). Layout [i, f, o, u].
        // We'll do this gate-by-gate to keep the index math explicit.
        for (size_t h = 0; h < hidden_size_; ++h) {
            double sum_i = b(0, h);
            double sum_f = b(0, hidden_size_ + h);
            double sum_o = b(0, 2 * hidden_size_ + h);
            double sum_u = b(0, 3 * hidden_size_ + h);
            for (size_t d = 0; d < input_size_; ++d) {
                double xjd = last_input_(j, d);
                sum_i += W(h, d) * xjd;        // i gate uses W rows [0, H)
                sum_f += W(hidden_size_ + h, d) * xjd;  // f gate: [H, 2H)
                sum_o += W(2 * hidden_size_ + h, d) * xjd;  // o gate: [2H, 3H)
                sum_u += W(3 * hidden_size_ + h, d) * xjd;  // u gate: [3H, 4H)
            }
            // h_sum_j contribution
            for (size_t hh = 0; hh < hidden_size_; ++hh) {
                double hsh = last_h_sum_(j, hh);
                sum_i += U(h, hh) * hsh;
                sum_f += U(hidden_size_ + h, hh) * hsh;
                sum_o += U(2 * hidden_size_ + h, hh) * hsh;
                sum_u += U(3 * hidden_size_ + h, hh) * hsh;
            }
            last_i_pre_(j, h) = sum_i;
            last_o_pre_(j, h) = sum_o;
            last_u_pre_(j, h) = sum_u;
        }

        // i_j = σ(sum_i), o_j = σ(sum_o), u_j = tanh(sum_u)
        for (size_t h = 0; h < hidden_size_; ++h) {
            last_i_gate_(j, h) = sigmoid(last_i_pre_(j, h));
            last_o_gate_(j, h) = sigmoid(last_o_pre_(j, h));
            last_u_tanh_(j, h) = std::tanh(last_u_pre_(j, h));
        }

        // c_j = i_j ⊙ u_j + Σ_{k ∈ C(j)} f_jk ⊙ c_k
        // f_jk = σ(W_f · x_j + U_f · h_k + b_f)  — computed per child k
        // using each child h_k (not h_sum).
        for (size_t ei = last_edge_offset_[j]; ei < last_edge_offset_[j + 1]; ++ei) {
            int k = last_edge_child_[ei];
            for (size_t h = 0; h < hidden_size_; ++h) {
                double sum_f = b(0, hidden_size_ + h);
                for (size_t d = 0; d < input_size_; ++d) {
                    sum_f += W(hidden_size_ + h, d) * last_input_(j, d);
                }
                // U_f rows [H, 2H) dot h_k (the specific child, not h_sum)
                for (size_t hh = 0; hh < hidden_size_; ++hh) {
                    sum_f += U(hidden_size_ + h, hh) * last_h_(k, hh);
                }
                double f_jk = sigmoid(sum_f);
                last_f_gates_(ei, h) = f_jk;
                last_c_(j, h) += f_jk * last_c_(k, h);
            }
        }
        // Add the i ⊙ u term.
        for (size_t h = 0; h < hidden_size_; ++h) {
            last_c_(j, h) += last_i_gate_(j, h) * last_u_tanh_(j, h);
        }

        // h_j = o_j ⊙ tanh(c_j)
        for (size_t h = 0; h < hidden_size_; ++h) {
            last_h_(j, h) = last_o_gate_(j, h) * std::tanh(last_c_(j, h));
        }
    }

    return last_h_.clone();
}

Tensor TreeLSTM::forward(const Tensor& input) {
    // Single-Tensor forward: requires the last_children_ to be set. This is
    // the drop-in Layer-interface variant (matches LSTM, etc.); for the
    // primary API use forward(features, children).
    if (last_children_.empty()) {
        // No prior tree was set: treat the input as a single node.
        std::vector<std::vector<int>> children(1);
        return forward(input, children);
    }
    return forward(input, last_children_);
}

Tensor TreeLSTM::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t N = static_cast<size_t>(last_num_nodes_);
    if (N == 0 || last_post_order_.empty()) {
        return Tensor(0, 0);
    }
    if (grad_output.rows != N || grad_output.cols != hidden_size_) {
        throw std::invalid_argument("TreeLSTM: grad_output dim mismatch");
    }

    // Reset gradient accumulators.
    grad_W.fill(0.0);
    grad_U.fill(0.0);
    grad_b.fill(0.0);
    grad_input_.fill(0.0);
    grad_h_sum_.fill(0.0);
    grad_c_.fill(0.0);
    dh_sum_.fill(0.0);

    // We traverse in REVERSE post-order: parents before children. This
    // ensures that by the time we compute d h_k for a child k, we have
    // already accumulated its contribution from the parent.

    // First, seed grad_h_sum_ (= dL/dh_j accumulator) from grad_output.
    for (size_t i = 0; i < N; ++i)
        for (size_t h = 0; h < hidden_size_; ++h)
            grad_h_sum_(i, h) = grad_output(i, h);

    // Now process nodes in REVERSE post-order (parents before children).
    for (auto it = last_post_order_.rbegin(); it != last_post_order_.rend(); ++it) {
        int j = *it;

        // ----- d h_j is in grad_h_sum_(j, .) -----
        // h_j = o_j ⊙ tanh(c_j)  =>  d tanh(c_j) = d_h_j ⊙ o_j
        //                          and d o_j = d_h_j ⊙ tanh(c_j)
        // We'll compute d_tanh_c = grad_h_sum_(j,h) * o_gate; then split
        // into d c (via the c->h chain) and into the gate gradients.
        for (size_t h = 0; h < hidden_size_; ++h) {
            double o_val = last_o_gate_(j, h);
            double tc    = std::tanh(last_c_(j, h));
            double dh    = grad_h_sum_(j, h);
            // d/d tanh(c_j) from h_j
            double d_tanh_c = dh * o_val;
            // d c_j (from h chain; add to the c_j chain which already
            // includes the contribution from f_jk paths feeding into it).
            double d_c = d_tanh_c * (1.0 - tc * tc);
            grad_c_(j, h) += d_c;

            // d o_j = d_h_j * tanh(c_j); the o gate output is sigmoid(o_pre).
            double d_o = dh * tc;
            // d o_pre = d_o * o_gate * (1 - o_gate)
            double d_o_pre = d_o * o_val * (1.0 - o_val);
            // Accumulate into grad_W, grad_U, grad_b for the o-gate row.
            // grad_b row [2H, 3H) += d_o_pre  (summed over N, so it's already
            // a per-gate-row total — but since we're per-node, we add
            // d_o_pre directly; the total bias gradient sums across all
            // nodes).
            grad_b(0, 2 * hidden_size_ + h) += d_o_pre;
            // grad_W rows [2H, 3H) column d += d_o_pre * x_j[d]
            for (size_t d = 0; d < input_size_; ++d) {
                grad_W(2 * hidden_size_ + h, d) += d_o_pre * last_input_(j, d);
            }
            // grad_U rows [2H, 3H) column hh += d_o_pre * h_sum_j[hh]
            for (size_t hh = 0; hh < hidden_size_; ++hh) {
                grad_U(2 * hidden_size_ + h, hh) += d_o_pre * last_h_sum_(j, hh);
            }
        }

        // ----- c_j chain -----
        // d c_j has two contributors: (a) the h_j->c_j chain (added above),
        // (b) for each child k, the c_k path through f_jk (already in
        // grad_c_(j, .) from a later step? No — we compute parents before
        // children, so c_k hasn't been processed yet. We have to PROPOGATE
        // d c_j BACK to d c_k via the forget gates.)
        // The f-gate path: c_j = i_j ⊙ u_j + Σ_k f_jk ⊙ c_k
        //   d c_k (from this) += d_c_j ⊙ f_jk
        //   d f_jk = d_c_j ⊙ c_k
        // We'll handle this child-by-child inside the forget-gate loop.

        // ----- i_j and u_j chains (the i*u term in c_j) -----
        // d c_j also feeds into d i_j = d_c_j ⊙ u_j and d u_j = d_c_j ⊙ i_j.
        // d i_j = d_c_j ⊙ u_tanh  =>  d i_pre = d_i * i * (1-i)
        // d u_j = d_c_j ⊙ i_gate  =>  d u_pre = d_u * (1 - tanh²(u_pre))
        for (size_t h = 0; h < hidden_size_; ++h) {
            double dc = grad_c_(j, h);
            double i_val = last_i_gate_(j, h);
            double u_val = last_u_tanh_(j, h);

            // i gate chain
            double d_i  = dc * u_val;
            double d_ip = d_i * i_val * (1.0 - i_val);
            grad_b(0, h) += d_ip;
            for (size_t d = 0; d < input_size_; ++d) {
                grad_W(h, d) += d_ip * last_input_(j, d);
            }
            for (size_t hh = 0; hh < hidden_size_; ++hh) {
                grad_U(h, hh) += d_ip * last_h_sum_(j, hh);
            }

            // u gate chain
            double d_u  = dc * i_val;
            double d_up = d_u * (1.0 - u_val * u_val);
            grad_b(0, 3 * hidden_size_ + h) += d_up;
            for (size_t d = 0; d < input_size_; ++d) {
                grad_W(3 * hidden_size_ + h, d) += d_up * last_input_(j, d);
            }
            for (size_t hh = 0; hh < hidden_size_; ++hh) {
                grad_U(3 * hidden_size_ + h, hh) += d_up * last_h_sum_(j, hh);
            }

            // x_j gradient (W rows [0, H) ∪ [H, 2H) ∪ [2H, 3H) ∪ [3H, 4H)
            // all contribute). Accumulate here.
        }

        // Accumulate grad_input_(j, .) from all 4 gates' W contributions.
        // grad_input_(j, d) = Σ_h d_gate_pre[h] * W[gate_row, d]
        for (size_t d = 0; d < input_size_; ++d) {
            double acc = 0.0;
            for (size_t h = 0; h < hidden_size_; ++h) {
                // i gate
                double i_val = last_i_gate_(j, h);
                double u_val = last_u_tanh_(j, h);
                double d_ip = (grad_c_(j, h) * u_val) * i_val * (1.0 - i_val);
                acc += d_ip * W(h, d);
                // o gate
                double o_val = last_o_gate_(j, h);
                double tc    = std::tanh(last_c_(j, h));
                double d_op = (grad_h_sum_(j, h) * tc) * o_val * (1.0 - o_val);
                acc += d_op * W(2 * hidden_size_ + h, d);
                // u gate
                double d_up = (grad_c_(j, h) * i_val) * (1.0 - u_val * u_val);
                acc += d_up * W(3 * hidden_size_ + h, d);
            }
            // f-gate (per child) — separate loop below.
            grad_input_(j, d) = acc;
        }

        // h_sum_j gradient: each of the 4 gates (except f, which uses h_k
        // specifically) gets d_gate_pre * U[gate_row, hh] contribution.
        // We'll accumulate into a per-node buffer, then distribute to each
        // child k of j as d_h_k += d_h_sum_j[hh] * U_f_row[hh].
        // First: d_h_sum_j[hh] from the i, o, u gates.
        // d_h_sum_j from forget gates: zero (forget uses h_k not h_sum).

        for (size_t hh = 0; hh < hidden_size_; ++hh) {
            double acc = 0.0;
            for (size_t h = 0; h < hidden_size_; ++h) {
                // i gate
                double i_val = last_i_gate_(j, h);
                double u_val = last_u_tanh_(j, h);
                double d_ip = (grad_c_(j, h) * u_val) * i_val * (1.0 - i_val);
                acc += d_ip * U(h, hh);
                // o gate
                double o_val = last_o_gate_(j, h);
                double tc    = std::tanh(last_c_(j, h));
                double d_op = (grad_h_sum_(j, h) * tc) * o_val * (1.0 - o_val);
                acc += d_op * U(2 * hidden_size_ + h, hh);
                // u gate
                double d_up = (grad_c_(j, h) * i_val) * (1.0 - u_val * u_val);
                acc += d_up * U(3 * hidden_size_ + h, hh);
            }
            dh_sum_(j, hh) = acc;  // dL/dh_sum_j (per-parent), separate
                                   // from dL/dh_j which lives in grad_h_sum_
        }

        // For each child k of j, distribute dL/dh_sum_j back into dL/dh_k.
        // d h_k contribution from the i, o, u gates is via the
        // h_sum -> child split. Each child k of j gets d_h_sum_j[hh]
        // added to its d_h_k[hh] (since h_sum_j[hh] = Σ_{k'} h_{k'}[hh],
        // and the partial w.r.t. h_k[hh] is 1.0).
        for (size_t ei = last_edge_offset_[j]; ei < last_edge_offset_[j + 1]; ++ei) {
            int k = last_edge_child_[ei];
            for (size_t hh = 0; hh < hidden_size_; ++hh) {
                grad_h_sum_(k, hh) += dh_sum_(j, hh);
            }
        }

        // Forget-gate backward: for each edge (j, k) with forget gate f_jk,
        // c_j = i_j ⊙ u_j + Σ_{k'} f_jk' ⊙ c_{k'}
        //   d c_k (from c_j) += d_c_j ⊙ f_jk     (propagate to child)
        //   d f_jk   = d_c_j ⊙ c_k                (compute f-gate grad)
        //   d f_pre_jk = d_f_jk * f_jk * (1 - f_jk)
        //   grad_W row [H, 2H) col d += d_f_pre_jk * x_j[d]
        //   grad_U row [H, 2H) col hh += d_f_pre_jk * h_k[hh]   (NOT h_sum)
        //   grad_b row [H, 2H) += d_f_pre_jk
        //   grad_input_(j, d) += d_f_pre_jk * W[hidden_size_+h, d]
        //   grad_h_k[hh] += d_f_pre_jk * U[hidden_size_+h, hh]    (per-child h_k)

        for (size_t ei = last_edge_offset_[j]; ei < last_edge_offset_[j + 1]; ++ei) {
            int k = last_edge_child_[ei];
            for (size_t h = 0; h < hidden_size_; ++h) {
                double f_val = last_f_gates_(ei, h);
                double dc    = grad_c_(j, h);

                // d c_k += d_c_j ⊙ f_jk
                grad_c_(k, h) += dc * f_val;

                // d f_jk
                double d_f = dc * last_c_(k, h);
                double d_fp = d_f * f_val * (1.0 - f_val);

                grad_b(0, hidden_size_ + h) += d_fp;
                for (size_t d = 0; d < input_size_; ++d) {
                    grad_W(hidden_size_ + h, d) += d_fp * last_input_(j, d);
                }
                for (size_t hh = 0; hh < hidden_size_; ++hh) {
                    grad_U(hidden_size_ + h, hh) += d_fp * last_h_(k, hh);
                }
                // x_j gradient
                for (size_t d = 0; d < input_size_; ++d) {
                    grad_input_(j, d) += d_fp * W(hidden_size_ + h, d);
                }
                // h_k gradient (per-child h_k, not h_sum)
                for (size_t hh = 0; hh < hidden_size_; ++hh) {
                    grad_h_sum_(k, hh) += d_fp * U(hidden_size_ + h, hh);
                }
            }
        }
    }

    return grad_input_.clone();
}

void TreeLSTM::update_weights(double learning_rate) {
    // NoOp — the Model / caller is expected to consume the gradients via
    // parameters()/gradients() and apply an optimizer. To keep the layer
    // self-contained we still apply a basic SGD step here.
    for (size_t i = 0; i < W.rows; ++i)
        for (size_t j = 0; j < W.cols; ++j)
            W(i, j) -= learning_rate * grad_W(i, j);
    for (size_t i = 0; i < U.rows; ++i)
        for (size_t j = 0; j < U.cols; ++j)
            U(i, j) -= learning_rate * grad_U(i, j);
    for (size_t i = 0; i < b.rows; ++i)
        for (size_t j = 0; j < b.cols; ++j)
            b(i, j) -= learning_rate * grad_b(i, j);
}

void TreeLSTM::zero_grad() {
    grad_W.fill(0.0);
    grad_U.fill(0.0);
    grad_b.fill(0.0);
    if (grad_input_.rows > 0) grad_input_.fill(0.0);
    if (grad_h_sum_.rows > 0) grad_h_sum_.fill(0.0);
    if (grad_c_.rows > 0)     grad_c_.fill(0.0);
    if (dh_sum_.rows > 0)     dh_sum_.fill(0.0);
}

std::vector<Tensor*> TreeLSTM::parameters() {
    return { &W, &U, &b };
}

std::vector<Tensor*> TreeLSTM::gradients() {
    return { &grad_W, &grad_U, &grad_b };
}

Tensor TreeLSTM::get_weights() const { return W; }
Tensor TreeLSTM::get_gradients() const { return grad_W; }

std::vector<int> TreeLSTM::build_post_order(
    const std::vector<std::vector<int>>& children, int num_nodes)
{
    // We expect `children[j]` to list the children of node j. The root
    // is the unique node that does not appear as anyone's child. (If
    // children[0] is empty and 0 doesn't appear in any list, root = 0.)
    std::vector<char> is_child(num_nodes, 0);
    for (int j = 0; j < num_nodes; ++j) {
        for (int k : children[j]) {
            if (k >= 0 && k < num_nodes) is_child[k] = 1;
        }
    }
    int root = 0;
    for (int j = 0; j < num_nodes; ++j) {
        if (!is_child[j]) { root = j; break; }
    }

    // Iterative DFS post-order from root.
    std::vector<int> post;
    post.reserve(num_nodes);
    // We use an explicit stack with (node, state) where state 0 = first
    // visit, state 1 = post-visit.
    std::vector<std::pair<int, int>> stack;
    stack.push_back({root, 0});
    while (!stack.empty()) {
        auto [v, state] = stack.back();
        stack.pop_back();
        if (state == 1) {
            post.push_back(v);
        } else {
            stack.push_back({v, 1});
            // Push children in reverse so we visit them in the given order.
            const auto& c = children[v];
            for (auto it = c.rbegin(); it != c.rend(); ++it) {
                stack.push_back({*it, 0});
            }
        }
    }
    return post;
}

// ============================================================================
// TreeLSTMModel
// ============================================================================

TreeLSTMModel::TreeLSTMModel(size_t in_dim, size_t hidden_size, size_t out_dim)
    : cell(in_dim, hidden_size),
      classifier(hidden_size, out_dim),
      input_size_(in_dim),
      hidden_size_(hidden_size),
      output_size_(out_dim),
      last_num_nodes_(0)
{}

Tensor TreeLSTMModel::forward_with_tree(const Tensor& node_features,
                                        const std::vector<std::vector<int>>& children) {
    last_children_ = children;
    last_num_nodes_ = static_cast<int>(node_features.rows);
    Tensor h_all = cell.forward(node_features, children);
    // For the classifier: take h at the root (the only node not appearing
    // as anyone's child). Build a (1, hidden) tensor.
    std::vector<char> is_child(node_features.rows, 0);
    for (size_t j = 0; j < node_features.rows; ++j) {
        for (int k : children[j]) {
            if (k >= 0 && k < (int)node_features.rows) {
                is_child[k] = 1;
            }
        }
    }
    int root = 0;
    for (size_t j = 0; j < node_features.rows; ++j) {
        if (!is_child[j]) { root = j; break; }
    }
    Tensor root_h(1, hidden_size_);
    for (size_t h = 0; h < hidden_size_; ++h) {
        root_h(0, h) = h_all(root, h);
    }
    Tensor logits = classifier.forward(root_h);
    return logits;
}

Tensor TreeLSTMModel::forward(const Tensor& input) {
    if (last_children_.empty()) {
        std::vector<std::vector<int>> children(1);
        return forward_with_tree(input, children);
    }
    return forward_with_tree(input, last_children_);
}

Tensor TreeLSTMModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output is (1, out_dim) — gradient of loss w.r.t. classifier output.
    // Step 1: backward through the classifier to get d_root_h.
    Tensor grad_root_h = classifier.backward(grad_output, 0.0);

    // Step 2: scatter d_root_h into a (N, hidden) tensor and call
    // cell.backward().
    size_t N = static_cast<size_t>(last_num_nodes_);
    Tensor grad_h(N, hidden_size_);
    grad_h.fill(0.0);

    // Find the root (same logic as in forward).
    std::vector<char> is_child(N, 0);
    for (size_t j = 0; j < N; ++j) {
        for (int k : last_children_[j]) {
            if (k >= 0 && k < (int)N) is_child[k] = 1;
        }
    }
    int root = 0;
    for (size_t j = 0; j < N; ++j) {
        if (!is_child[j]) { root = j; break; }
    }
    for (size_t h = 0; h < hidden_size_; ++h) {
        grad_h(root, h) = grad_root_h(0, h);
    }
    return cell.backward(grad_h, 0.0);
}

void TreeLSTMModel::update_weights(double learning_rate) {
    cell.update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}

void TreeLSTMModel::zero_grad() {
    cell.zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> TreeLSTMModel::parameters() {
    auto p = cell.parameters();
    auto q = classifier.parameters();
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

std::vector<Tensor*> TreeLSTMModel::gradients() {
    auto g = cell.gradients();
    auto h = classifier.gradients();
    g.insert(g.end(), h.begin(), h.end());
    return g;
}

Tensor TreeLSTMModel::get_weights() const { return cell.get_weights(); }
Tensor TreeLSTMModel::get_gradients() const { return cell.get_gradients(); }
