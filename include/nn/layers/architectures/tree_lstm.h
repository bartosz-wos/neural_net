#ifndef TREE_LSTM_H
#define TREE_LSTM_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Tree-LSTM — Tai, Socher, Manning 2015
//   "Improved Semantic Representations From Tree-Structured Long Short-Term
//    Memory Networks" (ACL).
//
// A generalization of the LSTM to tree-structured inputs. Instead of a single
// previous hidden state h_{t-1}, each node's hidden state is a function of
// ALL of its children's hidden states. The two variants from the paper:
//
//   1. Child-Sum Tree-LSTM (the one implemented here) — sums the children's
//      h and c vectors, with separate forget gates per child. Used for
//      dependency trees (where the number of children per node is variable
//      and the children have no positional ordering).
//
//   2. N-ary Tree-LSTM — each child slot has its own parameters, used for
//      constituency parses where the branching factor is fixed. We do NOT
//      implement N-ary here; the child-sum version is more general.
//
// ----------------------------------------------------------------------------
// Math (Child-Sum Tree-LSTM, Tai et al. 2015 §2.2):
//
//   For each node j with children C(j) (indices of direct children):
//
//     h̃_j          = Σ_{k ∈ C(j)} h_k                    (sum of child hidden states)
//
//     i_j          = σ(W_i · x_j + U_i · h̃_j + b_i)    (input gate)
//     f_jk         = σ(W_f · x_j + U_f · h_k + b_f)    (forget gate, ONE per child k)
//     o_j          = σ(W_o · x_j + U_o · h̃_j + b_o)    (output gate)
//     u_j          = tanh(W_u · x_j + U_u · h̃_j + b_u)  (candidate)
//
//     c_j          = i_j ⊙ u_j + Σ_{k ∈ C(j)} f_jk ⊙ c_k    (cell state)
//     h_j          = o_j ⊙ tanh(c_j)                           (hidden state)
//
//   Note: per-child forget gates is the key innovation vs the chain LSTM
//   (which has a single forget gate shared across all "children" of t, i.e.
//   just h_{t-1}). This lets the network learn which sibling to keep
//   information from.
//
// ----------------------------------------------------------------------------
// Initialization convention (matching the LSTM in this codebase):
//   * Weights W_*, U_*: xavier-uniform scaling (Dense default).
//   * Biases b_*: zero, EXCEPT forget-gate bias = 1.0 (PyTorch convention;
//     this "opens" the forget gate at init so gradients flow).
//   * We use a single combined W (4*hidden, in_dim) and combined U
//     (4*hidden, hidden) for efficiency, with gate layout [i, f, o, u]
//     (matching the LSTM convention in this codebase).
//
// ----------------------------------------------------------------------------
// Shape convention:
//   - input:    `node_features` (N, in_dim)     — x_j for j=0..N-1
//   - tree:     `children`     std::vector<std::vector<int>>
//               of size N. children[j] is the list of child node indices of
//               node j. The root is node 0 (a single node not appearing as
//               a child of anyone) — or detect automatically.
//   - output:   (N, hidden_size)  — h_j for all nodes (not just root).
//               Use output row 0 (or the implicit root) for downstream
//               classification.
//
// ----------------------------------------------------------------------------
// BPTT (Back-Propagation Through Tree-Structure):
//   - Forward: post-order traversal (children before parents).
//   - Backward: reverse post-order traversal (parents before children).
//   - For each node j, accumulate d h_j and d c_j. d c_j feeds back through
//     the per-child forget gates into d c_k for each child k.
//   - The h̃-sum means d W, d U, d b, d x must accumulate across the entire
//     tree (each gate uses the SAME h̃). Naive impl would be O(N*children).
//     We do this efficiently by computing per-node gate values once and
//     reusing them.
// ============================================================================

class TreeLSTM : public Layer {
public:
    // in_dim:     input feature dimension (e.g. word-embedding dim)
    // hidden_size: hidden state / cell state dim
    TreeLSTM(size_t in_dim, size_t hidden_size);

    // Forward pass on a tree.
    //   node_features: (N, in_dim)  — input features at every node
    //   children:      length-N vector where children[j] is the list of
    //                  node indices that are direct children of node j.
    //                  Root is conventionally node 0.
    //   returns:       (N, hidden_size) — h_j for every node.
    Tensor forward(const Tensor& node_features,
                   const std::vector<std::vector<int>>& children);

    // The Layer interface expects forward(input) with a single Tensor. Since
    // Tree-LSTM is fundamentally a function of (features, tree), we ALSO
    // provide a single-Tensor forward that uses the LAST cached tree.
    // This makes the layer drop-in compatible with the Layer base class
    // (e.g. for use inside a Model) but the (features, children) overload
    // is the primary API.
    Tensor forward(const Tensor& input) override;

    // Backward pass. The (features, children) used in the most recent
    // forward must still be in scope; we cache a reference via the cache
    // tensors below. `grad_output`: (N, hidden_size) — d L / d h_j.
    // returns: (N, in_dim) — d L / d x_j.
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "TreeLSTM"; }

    // Accessors for tests
    size_t input_size()  const { return input_size_; }
    size_t hidden_size() const { return hidden_size_; }

    // Debug: returns the i-gate pre-activation of node j (test-only).
    const Tensor& debug_i_pre()  const { return last_i_pre_; }
    const Tensor& debug_o_pre()  const { return last_o_pre_; }
    const Tensor& debug_u_pre()  const { return last_u_pre_; }
    const Tensor& debug_i_gate() const { return last_i_gate_; }
    const Tensor& debug_o_gate() const { return last_o_gate_; }
    const Tensor& debug_u_tanh() const { return last_u_tanh_; }
    const Tensor& debug_h()      const { return last_h_; }
    const Tensor& debug_c()      const { return last_c_; }
    const Tensor& debug_h_sum()  const { return last_h_sum_; }
    const Tensor& debug_f_gates() const { return last_f_gates_; }

    // Public access to the last input gradient (returned by backward()).
    const Tensor& grad_input() const { return grad_input_; }

    // Public parameters (for gradient tests / debugging)
    Tensor W;          // (4*hidden, in_dim)    combined [i, f, o, u] @ x_j
    Tensor U;          // (4*hidden, hidden)   combined [i, f, o, u] @ h_sum
    Tensor b;          // (1, 4*hidden)        combined bias for [i, f, o, u]
    Tensor grad_W;     // (4*hidden, in_dim)
    Tensor grad_U;     // (4*hidden, hidden)
    Tensor grad_b;     // (1, 4*hidden)

private:
    size_t input_size_;
    size_t hidden_size_;

    // Cached tree from the most recent forward call. We deep-clone the
    // children list (it's tiny) and reference the feature Tensor via a
    // pointer. Re-clone is fine — the user passes the same list each call.
    std::vector<std::vector<int>> last_children_;
    int last_num_nodes_;
    // Per-node post-order indices (children before parents).
    std::vector<int> last_post_order_;

    // Cached forward-pass tensors (size N, hidden_size each unless noted).
    Tensor last_input_;         // (N, in_dim) — features passed to last forward
    Tensor last_h_;             // (N, hidden_size)
    Tensor last_c_;             // (N, hidden_size)
    Tensor last_h_sum_;        // (N, hidden_size) — Σ_{k∈C(j)} h_k, with root=0
    // Per-node gate pre-activations (size N, hidden_size each):
    Tensor last_i_pre_;
    Tensor last_f_pre_;         // NOTE: per-child; we cache only the one for the
                               // "first child" path — see notes in .cpp
    Tensor last_o_pre_;
    Tensor last_u_pre_;
    // Per-node post-activation gates (size N, hidden_size):
    Tensor last_i_gate_;
    Tensor last_o_gate_;
    Tensor last_u_tanh_;
    // For forget gates we keep a per-(parent, child-position) cache. The
    // total number of (parent, child) edges in the tree is M = sum |C(j)|.
    // We store forget-gate values as a flat (M, hidden_size) tensor indexed
    // by an edge offset.
    std::vector<int> last_edge_parent_;   // size M: parent of each edge
    std::vector<int> last_edge_child_;    // size M: child of each edge
    std::vector<size_t> last_edge_offset_; // size N: starting offset for each node
                                           // (cumulative sum of |C(j)|) — for
                                           // per-node indexing into f_gates
    Tensor last_f_gates_;                 // (M, hidden_size)

    // Backward scratch buffers
    Tensor grad_input_;          // (N, in_dim)
    Tensor grad_h_sum_;          // (N, hidden_size) — dL/dh_sum_j accumulator
                                 //   (only used as a per-node dL/dh_sum, NOT
                                 //   the same as dL/dh_k which is also stored
                                 //   in grad_h_sum_ at the START of backward
                                 //   but re-loaded for each node from a
                                 //   separate buffer — see below).
    Tensor grad_c_;              // (N, hidden_size) — per-node accumulated dL/dc
    Tensor dh_sum_;              // (N, hidden_size) — dL/dh_sum_j (per-parent)

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }

    // Build post-order: children[j] is the list of node indices that are
    // direct children of j. Root = 0 (or autodetected as the unique node
    // never appearing as a child).
    static std::vector<int> build_post_order(
        const std::vector<std::vector<int>>& children, int num_nodes);
};

// ============================================================================
// TreeLSTMModel — TreeLSTM + linear classifier on the root node.
//
// A simple "tree classifier" — runs the TreeLSTM over a tree, then takes the
// hidden state of the root node (or h_sum if no clear root) and projects it
// to `out_dim` logits. Useful for sentence-level classification tasks where
// the input is a dependency tree (e.g. SST-5 with Stanford parser).
// ============================================================================

class TreeLSTMModel : public Layer {
public:
    TreeLSTMModel(size_t in_dim, size_t hidden_size, size_t out_dim);

    // Forward: returns (N, out_dim) where row 0 (the root) holds the
    // classification logits and the other rows mirror the TreeLSTM output
    // through a zero projection (so shapes line up with the TreeLSTM).
    // Use `forward_with_tree` for the full API.
    Tensor forward(const Tensor& input) override;
    Tensor forward_with_tree(const Tensor& node_features,
                             const std::vector<std::vector<int>>& children);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "TreeLSTMModel"; }

    size_t input_size()  const { return input_size_; }
    size_t hidden_size() const { return hidden_size_; }
    size_t output_size() const { return output_size_; }

    // Public sub-layers
    TreeLSTM cell;
    Dense classifier;   // (hidden -> out_dim)

private:
    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;

    // Cached tree
    std::vector<std::vector<int>> last_children_;
    int last_num_nodes_;
};

#endif // TREE_LSTM_H
