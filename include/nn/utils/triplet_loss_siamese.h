#ifndef TRIPLET_LOSS_SIAMESE_H
#define TRIPLET_LOSS_SIAMESE_H

#include "../core/layer.h"
#include <vector>
#include <random>

// L2-normalize embeddings so Euclidean distance等价于 cosine distance
// Critical for triplet/contrastive loss to work properly
class EmbeddingNormalizer : public Layer {
public:
    EmbeddingNormalizer() {}

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "EmbeddingNormalizer"; }
};

// Siamese Network: two (or more) identical networks sharing weights
// Learns to output similar embeddings for similar inputs
class SiameseNetwork : public Layer {
public:
    SiameseNetwork() {}

    void set_base_network(Layer* base_network);
    std::pair<Tensor, Tensor> forward_pair(const Tensor& x1, const Tensor& x2);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void init_weights() {}
    void update_weights(double learning_rate) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "SiameseNetwork"; }

private:
    Layer* base_network_ = nullptr;
};

// Triplet Loss: margin-based loss between anchor, positive, negative
// Paper: https://arxiv.org/abs/1503.03832
// L = max(0, d(a,p) - d(a,n) + margin)
// IMPORTANT: embeddings should be L2-normalized before computing loss
class TripletLoss {
public:
    TripletLoss(float margin = 0.2f, bool normalize = true)
        : margin_(margin), normalize_(normalize), rng_(std::random_device{}()) {}

    // Compute triplet loss. If normalize=true, L2-normalizes embeddings first.
    float forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative);

    // Normalize embeddings to unit sphere before computing loss
    Tensor normalize_embeddings(const Tensor& embeddings) const;

    float margin() const { return margin_; }

private:
    float margin_;
    bool normalize_;
    std::mt19937 rng_;
};

// Contrastive Loss (Hadsell et al.)
// L = y * d^2 + (1-y) * max(0, margin - d)^2
// y=1 for similar, y=0 for dissimilar
class ContrastiveLoss {
public:
    ContrastiveLoss(float margin = 1.0f, bool normalize = true)
        : margin_(margin), normalize_(normalize), rng_(std::random_device{}()) {}

    float forward(const Tensor& emb1, const Tensor& emb2, float label);
    float euclidean_distance(const Tensor& a, const Tensor& b);
    Tensor normalize_embeddings(const Tensor& embeddings) const;

private:
    float margin_;
    bool normalize_;
    std::mt19937 rng_;
};

// Online hard triplet mining
struct TripletBatch {
    std::vector<int> anchors;
    std::vector<int> positives;
    std::vector<int> negatives;
};

class HardTripletMiner {
public:
    HardTripletMiner(float margin = 0.2f) : margin_(margin), rng_(std::random_device{}()) {}

    TripletBatch mine(const std::vector<Tensor>& embeddings,
                      const std::vector<int>& labels);

private:
    float margin_;
    std::mt19937 rng_;
};

#endif
