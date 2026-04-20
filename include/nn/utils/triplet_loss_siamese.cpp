#include "triplet_loss_siamese.h"
#include <cmath>
#include <algorithm>

Tensor SiameseNetwork::forward(const Tensor& x) {
    if (base_network_ == nullptr) return x;
    return base_network_->forward(x);
}

Tensor SiameseNetwork::backward(const Tensor& grad_output, double /* learning_rate */) {
    return grad_output;
}

void SiameseNetwork::set_base_network(Layer* base_network) {
    base_network_ = base_network;
}

std::pair<Tensor, Tensor> SiameseNetwork::forward_pair(const Tensor& x1, const Tensor& x2) {
    if (base_network_ == nullptr) return {x1, x2};
    Tensor e1 = base_network_->forward(x1);
    Tensor e2 = base_network_->forward(x2);
    return {e1, e2};
}

float TripletLoss::forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative) {
    int batch = (int)anchor.rows;
    int embedding_dim = (int)anchor.cols;

    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        float d_ap = 0.0f, d_an = 0.0f;
        for (int i = 0; i < embedding_dim; i++) {
            float diff_p = anchor[b][i] - positive[b][i];
            float diff_n = anchor[b][i] - negative[b][i];
            d_ap += diff_p * diff_p;
            d_an += diff_n * diff_n;
        }
        d_ap = std::sqrt(d_ap);
        d_an = std::sqrt(d_an);
        total_loss += std::max(0.0f, d_ap - d_an + margin_);
    }
    return total_loss / batch;
}

float ContrastiveLoss::euclidean_distance(const Tensor& a, const Tensor& b) {
    int batch = (int)a.rows;
    int emb_dim = (int)a.cols;
    float dist = 0.0f;
    for (int i = 0; i < emb_dim; i++) {
        float diff = a[0][i] - b[0][i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

float ContrastiveLoss::forward(const Tensor& emb1, const Tensor& emb2, float label) {
    float d = euclidean_distance(emb1, emb2);
    if (label > 0.5f) {
        return d * d;
    } else {
        return std::max(0.0f, margin_ - d) * std::max(0.0f, margin_ - d);
    }
}

TripletBatch HardTripletMiner::mine(const std::vector<Tensor>& embeddings,
                                     const std::vector<int>& labels) {
    TripletBatch batch;
    int n = (int)embeddings.size();
    int emb_dim = n > 0 ? (int)embeddings[0].cols : 0;

    std::vector<float> dists(n * n, 0.0f);

    // Compute all pairwise distances (each embedding is one row)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float d = 0.0f;
            for (int k = 0; k < emb_dim; k++) {
                float diff = embeddings[i][0][k] - embeddings[j][0][k];
                d += diff * diff;
            }
            dists[i * n + j] = std::sqrt(d);
        }
    }

    // For each anchor, find hardest positive and hardest negative
    for (int a = 0; a < n; a++) {
        // Find hardest positive (same label, furthest distance)
        float max_d_pos = -1.0f;
        int hardest_pos = -1;
        for (int p = 0; p < n; p++) {
            if (labels[p] == labels[a] && p != a) {
                if (dists[a * n + p] > max_d_pos) {
                    max_d_pos = dists[a * n + p];
                    hardest_pos = p;
                }
            }
        }

        // Find hardest negative (different label, closest distance)
        float min_d_neg = 1e9f;
        int hardest_neg = -1;
        for (int neg = 0; neg < n; neg++) {
            if (labels[neg] != labels[a]) {
                if (dists[a * n + neg] < min_d_neg) {
                    min_d_neg = dists[a * n + neg];
                    hardest_neg = neg;
                }
            }
        }

        if (hardest_pos >= 0 && hardest_neg >= 0) {
            float loss = max_d_pos - min_d_neg + margin_;
            if (loss > 0) {
                batch.anchors.push_back(a);
                batch.positives.push_back(hardest_pos);
                batch.negatives.push_back(hardest_neg);
            }
        }
    }

    return batch;
}
