#include "vq_vae.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ===========================================================================
// Helper: validate that `z` is (M, D) and `w` is (D, K).
// Throws rather than reading out of range -- `Tensor::operator()` indexes
// `data[r * cols + c]` with NO bounds check, so a shape mismatch is silent
// corruption, not an exception. (The repo has already been bitten by exactly
// this in the VAT fixture, where a one-past-the-end write landed on the
// adjacent heap block.)
// ===========================================================================
static void validate(const Tensor& z, const Tensor& w) {
    if (z.cols != w.rows)
        throw std::invalid_argument(
            "vq: latent width (" + std::to_string(z.cols) +
            ") must equal the codebook embedding dim (" +
            std::to_string(w.rows) + ")");
    if (w.cols == 0)
        throw std::invalid_argument("vq: codebook must have at least one code");
    if (z.rows == 0)
        throw std::invalid_argument("vq: latent batch must be non-empty");
}

// ===========================================================================
// Free functions -- the testable core.
// ===========================================================================

Tensor vq_squared_distances(const Tensor& z, const Tensor& w) {
    validate(z, w);
    const size_t M = z.rows, D = z.cols, K = w.cols;
    Tensor out(M, K);
    // Same expansion the reference uses: |x|^2 - 2 x.w + |w|^2. The |w|^2 term
    // is per-COLUMN, which is why the codebook is stored (D, K) with column k
    // holding e_k (D3).
    std::vector<double> col_sq(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        double s = 0.0;
        for (size_t d = 0; d < D; ++d) s += w(d, k) * w(d, k);
        col_sq[k] = s;
    }
    for (size_t i = 0; i < M; ++i) {
        double row_sq = 0.0;
        for (size_t d = 0; d < D; ++d) row_sq += z(i, d) * z(i, d);
        for (size_t k = 0; k < K; ++k) {
            double dot = 0.0;
            for (size_t d = 0; d < D; ++d) dot += z(i, d) * w(d, k);
            out(i, k) = row_sq - 2.0 * dot + col_sq[k];
        }
    }
    return out;
}

std::vector<size_t> vq_nearest_indices(const Tensor& z, const Tensor& w) {
    Tensor d = vq_squared_distances(z, w);
    std::vector<size_t> out(z.rows, 0);
    for (size_t i = 0; i < z.rows; ++i) {
        // STRICT `<`, so the FIRST minimum wins. The reference is
        // `argmax(-distances, 1)` and argmin/argmax both return the first
        // extremum (D4). Using `<=` here would make the tie-break LAST-wins,
        // which is observably different on an exact tie and is caught by the
        // D4 fixture.
        size_t best = 0;
        for (size_t k = 1; k < w.cols; ++k)
            if (d(i, k) < d(i, best)) best = k;
        out[i] = best;
    }
    return out;
}

Tensor vq_quantize(const Tensor& z, const Tensor& w) {
    std::vector<size_t> idx = vq_nearest_indices(z, w);
    Tensor out(z.rows, z.cols);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d)
            out(i, d) = w(d, idx[i]);   // copied, never recomputed
    return out;
}

double vq_codebook_loss(const Tensor& z, const Tensor& z_q) {
    // Mean over ALL M*D elements (D2) -- `reduce_mean` over the whole tensor
    // in the reference, not a per-row mean.
    double s = 0.0;
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d) {
            const double diff = z_q(i, d) - z(i, d);
            s += diff * diff;
        }
    return s / static_cast<double>(z.rows * z.cols);
}

double vq_commitment_loss(const Tensor& z, const Tensor& z_q, double beta) {
    // Same scalar as the dictionary term -- the difference is entirely in
    // WHICH OPERAND IS STOPPED, which is invisible in this value (see the
    // header). beta is applied HERE, not to the dictionary term (D1).
    double s = 0.0;
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d) {
            const double diff = z(i, d) - z_q(i, d);
            s += diff * diff;
        }
    return beta * s / static_cast<double>(z.rows * z.cols);
}

Tensor vq_codebook_gradient(const Tensor& z, const Tensor& w) {
    // d/dw [ ||sg[z_q] - z_e||^2 ] = (2/N) * sum_{i: idx_i = k} (e_k - z_e[i])
    // No beta here (D1). Rows accumulate onto the single column they were
    // assigned, so a code hit by two rows gets the SUM of both contributions,
    // and a code hit by NO row gets EXACTLY zero (the Tensor starts at 0 and
    // nothing ever touches it -- not "small", not stale).
    //
    // SIGN: (2/N)*(z_q - z_e) = -(2/N)*(z_e - z_q). The codebook moves TOWARD
    // the encoder output under gradient DESCENT, so the gradient must point
    // away from z_e. Gradient descent then subtracts it and the code travels
    // toward z_e -- which is what test 9 asserts by checking the codebook
    // value INCREASES toward z_e.
    //
    // Takes the codebook rather than a separate z_q because K (the codebook
    // width) is needed for the gradient's shape and is NOT derivable from
    // z_e's width: those are two different quantities, and passing the
    // codebook keeps one buffer, one meaning.
    validate(z, w);
    const size_t M = z.rows, D = z.cols, K = w.cols;
    Tensor g(D, K);
    if (M == 0) return g;
    std::vector<size_t> idx = vq_nearest_indices(z, w);
    const double inv = 2.0 / static_cast<double>(M * D);
    for (size_t i = 0; i < M; ++i) {
        const size_t k = idx[i];
        for (size_t d = 0; d < D; ++d)
            g(d, k) += inv * (w(d, k) - z(i, d));
    }
    return g;
}

Tensor vq_codebook_gradient_with_beta(const Tensor& z, const Tensor& w, double beta) {
    // WRONG ON PURPOSE -- exists so the D1 discriminator in the tests has
    // something concrete to disagree with. See the header.
    return vq_codebook_gradient(z, w) * beta;
}

Tensor vq_commitment_gradient(const Tensor& z, const Tensor& z_q, double beta) {
    // d/dz_e [ beta*||z_e - sg[e]||^2 ] = (2*beta/N) * (z_e - z_q)
    // Carries a factor of EXACTLY beta and nothing else.
    const size_t M = z.rows, D = z.cols;
    const double inv = 2.0 * beta / static_cast<double>(M * D);
    Tensor g(M, D);
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            g(i, d) = inv * (z(i, d) - z_q(i, d));
    return g;
}

Tensor vq_straight_through(const Tensor& z, const Tensor& z_q) {
    // The VALUE of the straight-through estimator is z_q, bit-identically.
    // The reference writes `inputs + stop_gradient(quantized - inputs)`, which
    // cancels in exact arithmetic to `quantized` but is NOT bit-identical in
    // floating point -- so return z_q directly rather than round-tripping.
    // `z` is unused by design; it is kept in the signature because it names
    // the operation and because a caller reading the call site should see
    // which operand is being replaced.
    (void)z;
    return z_q.clone();
}

double vq_perplexity(const std::vector<size_t>& indices, size_t num_codes) {
    if (indices.empty() || num_codes == 0) return 0.0;
    std::vector<double> counts(num_codes, 0.0);
    for (size_t k : indices) {
        if (k >= num_codes)
            throw std::invalid_argument("vq_perplexity: index out of range");
        counts[k] += 1.0;
    }
    const double inv_n = 1.0 / static_cast<double>(indices.size());
    double acc = 0.0;
    for (size_t k = 0; k < num_codes; ++k) {
        const double p = counts[k] * inv_n;
        if (p > 0.0)
            acc += p * std::log(p + 1e-10);   // the +1e-10 is load-bearing (D9)
    }
    return std::exp(-acc);
}

Tensor vq_ema_step(const Tensor& z, const Tensor& w,
                   const Tensor& cluster_size, const Tensor& ema_w,
                   double decay, double epsilon) {
    validate(z, w);
    if (cluster_size.rows * cluster_size.cols != w.cols)
        throw std::invalid_argument("vq_ema_step: cluster_size must have one entry per code");
    if (ema_w.rows != w.rows || ema_w.cols != w.cols)
        throw std::invalid_argument("vq_ema_step: ema_w must match the codebook shape");
    if (!(decay >= 0.0 && decay <= 1.0))
        throw std::invalid_argument("vq_ema_step: decay must lie in [0, 1]");

    const size_t M = z.rows, D = w.rows, K = w.cols;
    std::vector<size_t> idx = vq_nearest_indices(z, w);

    // new_n[k] = decay*n[k] + (1-decay)*count[k]
    std::vector<double> new_n(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        double count = 0.0;
        for (size_t i = 0; i < M; ++i) if (idx[i] == k) count += 1.0;
        const double old = (k < cluster_size.rows || k < cluster_size.cols)
                             ? cluster_size[static_cast<size_t>(0)][k] : 0.0;
        new_n[k] = decay * old + (1.0 - decay) * count;
    }

    // new_dw[:,k] = decay*dw[:,k] + (1-decay)*sum_{i: idx_i=k} z[i]
    Tensor new_dw(D, K);
    for (size_t d = 0; d < D; ++d) {
        for (size_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < M; ++i) if (idx[i] == k) acc += z(i, d);
            new_dw(d, k) = decay * ema_w(d, k) + (1.0 - decay) * acc;
        }
    }

    // Laplace smoothing (D7): n_k <- (n_k + eps)/(N + K*eps) * N.
    double total = 0.0;
    for (size_t k = 0; k < K; ++k) total += new_n[k];

    Tensor out(D, K);
    for (size_t d = 0; d < D; ++d) {
        for (size_t k = 0; k < K; ++k) {
            const double denom = total + static_cast<double>(K) * epsilon;
            double smoothed = (new_n[k] + epsilon) / denom * total;
            // DEFECT FOUND BY THE ORACLE, not by inspection: with decay == 1
            // and a zero-valued statistics state, `total` is 0, so the
            // Laplace smoothing computes 0/0 and the WHOLE codebook comes back
            // NaN. The oracle reproduces it (`decay=1` -> all NaN), so it is a
            // property of the formula as written -- but shipping a function
            // that returns NaN for a legal, in-range argument is a trap, and
            // `decay = 1` is a legitimate configuration meaning "keep the
            // codebook, add no data".
            //
            // Guarded rather than "corrected": when nothing has been observed
            // yet (total == 0) there is no evidence to update FROM, so the only
            // defensible answer is the codebook we were handed.
            if (!(total > 0.0)) return w;
            if (!(smoothed > 0.0)) smoothed = epsilon;   // never divide by 0
            // A code assigned by NO row has new_dw == 0 and smoothed == eps,
            // so it collapses to EXACTLY zero (D8). It does not keep its
            // initial value -- that is a real property of the reference.
            out(d, k) = new_dw(d, k) / smoothed;
        }
    }
    return out;
}

// ===========================================================================
// VectorQuantizer
// ===========================================================================

VectorQuantizer::VectorQuantizer(size_t embedding_dim, size_t num_embeddings,
                                 double beta)
    : embedding_dim_(embedding_dim), num_embeddings_(num_embeddings), beta_(beta),
      codebook_(embedding_dim, num_embeddings),
      grad_codebook_(embedding_dim, num_embeddings) {
    if (embedding_dim == 0 || num_embeddings == 0)
        throw std::invalid_argument("VectorQuantizer: dims must be non-zero");
    // Small uniform init. Deliberately NOT zero: an all-zero codebook makes
    // every distance identical, so the assignment is decided purely by the
    // tie-break and no distance-based test can see anything.
    codebook_.apply([](double) { return 0.0; });
    for (size_t d = 0; d < embedding_dim; ++d)
        for (size_t k = 0; k < num_embeddings; ++k)
            codebook_(d, k) = 0.01 * (static_cast<double>((d * 7 + k * 13) % 11) - 5.0);
    grad_codebook_.fill(0.0);
}

void VectorQuantizer::set_codebook(const Tensor& w) {
    if (w.rows != embedding_dim_ || w.cols != num_embeddings_)
        throw std::invalid_argument("VectorQuantizer::set_codebook: expected (" +
                                    std::to_string(embedding_dim_) + "," +
                                    std::to_string(num_embeddings_) + ")");
    codebook_ = w.clone();
}

Tensor VectorQuantizer::forward(const Tensor& input) {
    last_z_e_ = input.clone();
    last_z_q_ = vq_quantize(input, codebook_);
    indices_ = vq_nearest_indices(input, codebook_);

    codebook_loss_ = vq_codebook_loss(input, last_z_q_);
    commitment_loss_ = vq_commitment_loss(input, last_z_q_, beta_);
    perplexity_ = vq_perplexity(indices_, num_embeddings_);

    used_codes_.clear();
    dead_codes_.clear();
    std::vector<bool> used(num_embeddings_, false);
    for (size_t k : indices_) if (k < num_embeddings_) used[k] = true;
    for (size_t k = 0; k < num_embeddings_; ++k)
        (used[k] ? used_codes_ : dead_codes_).push_back(k);

    // The forward VALUE is z_q, bit-identically -- this is the whole point of
    // a quantizer (D5). Returning `input` here would be an autoencoder.
    return last_z_q_.clone();
}

Tensor VectorQuantizer::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    // The codebook gradient is accumulated on EVERY backward (the dictionary
    // term), independent of what flows through the straight-through.
    grad_codebook_.fill(0.0);
    Tensor g = vq_codebook_gradient(last_z_e_, codebook_);
    for (size_t d = 0; d < embedding_dim_; ++d)
        for (size_t k = 0; k < num_embeddings_; ++k)
            grad_codebook_(d, k) = g(d, k);

    // D5: the straight-through passes grad_output UNALTERED to z_e.
    //
    // SHAPE + BROADCAST CONTRACT (a real bug, found by a standalone probe and
    // NOT by reading). Two things are wrong with the obvious implementation
    // `Tensor result = grad_output.clone();`:
    //
    //  (a) SHAPE. It returns whatever shape the CALLER passed. A caller
    //      seeding with a (1,1) scalar -- the obvious way to ask "is this the
    //      identity?" -- silently gets a (1,1) gradient back, truncating the
    //      commitment term to a single cell.
    //  (b) BROADCAST. My first fix allocated (M, D) and copied element-wise
    //      under a bounds guard, which zero-filled everything outside the
    //      seed's extent. The probe caught it immediately: gz(0,1) came back
    //      0.00208 (commitment only) where it must be 1.00208 (seed +
    //      commitment). A (1,1) tensor is a SCALAR SEED and must be added to
    //      every cell, not only to (0,0).
    //
    // So: a (1,1) grad_output broadcasts as a scalar over the whole result; a
    // matching (M, D) tensor is used element-wise. Anything else is a caller
    // error and is rejected rather than silently zero-filled.
    const bool is_scalar_seed = (grad_output.rows == 1 && grad_output.cols == 1);
    const bool is_full_shape =
        (grad_output.rows == last_z_e_.rows && grad_output.cols == last_z_e_.cols);
    if (!is_scalar_seed && !is_full_shape)
        throw std::invalid_argument(
            "VectorQuantizer::backward: grad_output must be the input shape (" +
            std::to_string(last_z_e_.rows) + "," + std::to_string(last_z_e_.cols) +
            ") or a (1,1) scalar seed");

    Tensor result(last_z_e_.rows, last_z_e_.cols);
    for (size_t i = 0; i < result.rows; ++i)
        for (size_t d = 0; d < result.cols; ++d)
            result(i, d) = is_scalar_seed ? grad_output(0, 0)
                                          : grad_output(i, d);

    // ...and the commitment term is the one EXTRA gradient path that does not
    // come through the STE. Adding it here (rather than leaving it to the
    // caller) is what makes the layer's contract self-contained.
    Tensor gc = vq_commitment_gradient(last_z_e_, last_z_q_, beta_);
    for (size_t i = 0; i < result.rows; ++i)
        for (size_t d = 0; d < result.cols; ++d)
            result(i, d) += gc(i, d);
    return result;
}

void VectorQuantizer::update_weights(double learning_rate) {
    for (size_t d = 0; d < embedding_dim_; ++d)
        for (size_t k = 0; k < num_embeddings_; ++k)
            codebook_(d, k) -= learning_rate * grad_codebook_(d, k);
}

void VectorQuantizer::zero_grad() {
    grad_codebook_.fill(0.0);
}

// ===========================================================================
// VectorQuantizerEMA
// ===========================================================================

VectorQuantizerEMA::VectorQuantizerEMA(size_t embedding_dim, size_t num_embeddings,
                                       double beta, double decay, double epsilon)
    : embedding_dim_(embedding_dim), num_embeddings_(num_embeddings),
      beta_(beta), decay_(decay), epsilon_(epsilon),
      codebook_(embedding_dim, num_embeddings),
      cluster_size_(1, num_embeddings),
      ema_w_(embedding_dim, num_embeddings) {
    if (embedding_dim == 0 || num_embeddings == 0)
        throw std::invalid_argument("VectorQuantizerEMA: dims must be non-zero");
    for (size_t d = 0; d < embedding_dim; ++d)
        for (size_t k = 0; k < num_embeddings; ++k)
            codebook_(d, k) = 0.01 * (static_cast<double>((d * 7 + k * 13) % 11) - 5.0);
    cluster_size_.fill(0.0);
    ema_w_.fill(0.0);
}

void VectorQuantizerEMA::set_codebook(const Tensor& w) {
    if (w.rows != embedding_dim_ || w.cols != num_embeddings_)
        throw std::invalid_argument("VectorQuantizerEMA::set_codebook: expected (" +
                                    std::to_string(embedding_dim_) + "," +
                                    std::to_string(num_embeddings_) + ")");
    codebook_ = w.clone();
}

void VectorQuantizerEMA::set_ema_state(const Tensor& cs, const Tensor& ew) {
    if (cs.rows * cs.cols != num_embeddings_)
        throw std::invalid_argument("VectorQuantizerEMA::set_ema_state: cluster_size must have one entry per code");
    if (ew.rows != embedding_dim_ || ew.cols != num_embeddings_)
        throw std::invalid_argument("VectorQuantizerEMA::set_ema_state: ema_w must match the codebook shape");
    cluster_size_ = cs.clone();
    ema_w_ = ew.clone();
}

Tensor VectorQuantizerEMA::forward(const Tensor& input) {
    last_z_e_ = input.clone();
    last_z_q_ = vq_quantize(input, codebook_);
    indices_ = vq_nearest_indices(input, codebook_);

    commitment_loss_ = vq_commitment_loss(input, last_z_q_, beta_);
    perplexity_ = vq_perplexity(indices_, num_embeddings_);

    used_codes_.clear();
    dead_codes_.clear();
    std::vector<bool> used(num_embeddings_, false);
    for (size_t k : indices_) if (k < num_embeddings_) used[k] = true;
    for (size_t k = 0; k < num_embeddings_; ++k)
        (used[k] ? used_codes_ : dead_codes_).push_back(k);

    return last_z_q_.clone();
}

Tensor VectorQuantizerEMA::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    (void)grad_output;
    // D6: NO straight-through identity here, and the dictionary term is
    // dropped. The codebook is assigned by the EMA, never gradient-updated,
    // so the only gradient that acts on anything is the commitment one.
    return vq_commitment_gradient(last_z_e_, last_z_q_, beta_);
}

void VectorQuantizerEMA::update_weights(double learning_rate, bool training) {
    // `learning_rate` is deliberately UNUSED. This is the documented advantage
    // of the EMA variant: the update is independent of the optimizer, so a
    // caller can pass Adam's lr and it makes no difference.
    (void)learning_rate;
    if (!training) return;   // evaluation must not pollute the statistics

    Tensor next = vq_ema_step(last_z_e_, codebook_, cluster_size_, ema_w_,
                              decay_, epsilon_);

    // Advance the statistics. `vq_ema_step` is pure, so it is recomputed here
    // from the pre-update state to keep the update atomic: read the new code
    // out of `next` only after the statistics have been advanced consistently.
    std::vector<size_t> idx = vq_nearest_indices(last_z_e_, codebook_);
    const size_t M = last_z_e_.rows, D = embedding_dim_, K = num_embeddings_;
    std::vector<double> new_n(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        double count = 0.0;
        for (size_t i = 0; i < M; ++i) if (idx[i] == k) count += 1.0;
        new_n[k] = decay_ * cluster_size_(0, k) + (1.0 - decay_) * count;
    }
    Tensor new_dw(D, K);
    for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < M; ++i) if (idx[i] == k) acc += last_z_e_(i, d);
            new_dw(d, k) = decay_ * ema_w_(d, k) + (1.0 - decay_) * acc;
        }
    for (size_t k = 0; k < K; ++k) cluster_size_(0, k) = new_n[k];
    ema_w_ = new_dw;
    codebook_ = next;
}

void VectorQuantizerEMA::zero_grad() {
    // No gradients are accumulated in this variant -- the codebook is assigned.
}

// ===========================================================================
// VQVAE
// ===========================================================================

VQVAE::VQVAE(size_t input_dim, size_t hidden_dim, size_t embedding_dim,
             size_t num_embeddings, double beta)
    : input_dim_(input_dim), hidden_dim_(hidden_dim),
      embedding_dim_(embedding_dim), num_embeddings_(num_embeddings),
      encoder_(input_dim, embedding_dim),
      decoder_hidden_(embedding_dim, hidden_dim),
      decoder_out_(hidden_dim, input_dim),
      quantizer_(embedding_dim, num_embeddings, beta) {
    if (input_dim == 0 || hidden_dim == 0)
        throw std::invalid_argument("VQVAE: input and hidden dims must be non-zero");
}

Tensor VQVAE::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Encode.
    last_z_e_ = encoder_.forward(input);
    // QUANTIZE. The decoder below receives z_q, never z_e -- that is the
    // entire difference between this and a plain autoencoder.
    Tensor z_q = quantizer_.forward(last_z_e_);
    // Decode.
    last_h_dec_ = decoder_hidden_.forward(z_q);
    last_h_dec_ = last_h_dec_.apply([](double v) { return std::max(0.0, v); });   // ReLU
    return decoder_out_.forward(last_h_dec_);
}

Tensor VQVAE::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    // dL/d(h_dec) through the output Dense.
    Tensor g = decoder_out_.backward(grad_output, 0.0);
    // ReLU backward: gate with the cached POST-activation (h > 0). Using the
    // pre-activation would need a second forward pass; the cached value is
    // exact because ReLU(x) > 0 iff x > 0 for all x except x == 0, where the
    // subgradient is taken as 0 (the repo's convention everywhere).
    for (size_t i = 0; i < g.rows && i < last_h_dec_.rows; ++i)
        for (size_t j = 0; j < g.cols && j < last_h_dec_.cols; ++j)
            if (last_h_dec_(i, j) <= 0.0) g(i, j) = 0.0;

    // dL/d(z_q) through the hidden Dense.
    Tensor gz = decoder_hidden_.backward(g, 0.0);
    // Into the quantizer: returns grad_output (straight-through) PLUS the
    // beta-weighted commitment gradient.
    Tensor gz_e = quantizer_.backward(gz, 0.0);
    // dL/d(input) through the encoder.
    return encoder_.backward(gz_e, 0.0);
}

void VQVAE::update_weights(double learning_rate) {
    encoder_.update_weights(learning_rate);
    decoder_hidden_.update_weights(learning_rate);
    decoder_out_.update_weights(learning_rate);
    quantizer_.update_weights(learning_rate);
}

void VQVAE::zero_grad() {
    encoder_.zero_grad();
    decoder_hidden_.zero_grad();
    decoder_out_.zero_grad();
    quantizer_.zero_grad();
}

std::vector<Tensor*> VQVAE::parameters() {
    std::vector<Tensor*> out;
    for (Tensor* p : encoder_.parameters()) out.push_back(p);
    for (Tensor* p : decoder_hidden_.parameters()) out.push_back(p);
    for (Tensor* p : decoder_out_.parameters()) out.push_back(p);
    for (Tensor* p : quantizer_.parameters()) out.push_back(p);
    return out;
}

std::vector<Tensor*> VQVAE::gradients() {
    std::vector<Tensor*> out;
    for (Tensor* g : encoder_.gradients()) out.push_back(g);
    for (Tensor* g : decoder_hidden_.gradients()) out.push_back(g);
    for (Tensor* g : decoder_out_.gradients()) out.push_back(g);
    for (Tensor* g : quantizer_.gradients()) out.push_back(g);
    return out;
}