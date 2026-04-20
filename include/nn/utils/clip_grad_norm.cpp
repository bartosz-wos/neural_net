#include "clip_grad_norm.h"

double clip_grad_norm_(const std::vector<Tensor*>& params, double max_norm) {
    double total_norm_sq = 0.0;

    // Sum squared norms
    for (Tensor* p : params) {
        if (!p) continue;
        for (size_t i = 0; i < p->rows; ++i)
            for (size_t j = 0; j < p->cols; ++j)
                total_norm_sq += (*p)[i][j] * (*p)[i][j];
    }

    double total_norm = std::sqrt(total_norm_sq);

    // Clip if needed
    if (total_norm > max_norm) {
        double scale = max_norm / total_norm;
        for (Tensor* p : params) {
            if (!p) continue;
            for (size_t i = 0; i < p->rows; ++i)
                for (size_t j = 0; j < p->cols; ++j)
                    (*p)[i][j] *= scale;
        }
        return max_norm;
    }

    return total_norm;
}