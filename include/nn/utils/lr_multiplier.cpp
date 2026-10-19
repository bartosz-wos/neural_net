#include "lr_multiplier.h"
#include "../core/model.h"
#include "../core/layer.h"

void lr_multiplier_snapshot(Model& model, std::map<Layer*, std::vector<Tensor>>& out) {
    out.clear();
    for (auto& layer : model.layers) {
        Layer* lp = layer.get();
        auto params = lp->parameters();
        if (params.empty()) continue;
        std::vector<Tensor> snap;
        snap.reserve(params.size());
        for (Tensor* p : params) snap.push_back(*p);
        out[lp] = std::move(snap);
    }
}

void lr_multiplier_apply_snapshot(Model& model, const LrMultiplier& mult,
                                  std::map<Layer*, std::vector<Tensor>>& before) {
    for (auto& layer : model.layers) {
        Layer* lp = layer.get();
        auto it = before.find(lp);
        // A layer that was absent from the snapshot moved by an unknown
        // amount, so its delta is undefined. Leaving it alone is the only
        // safe choice: silently treating "no snapshot" as "zero delta" would
        // mean discarding a real update.
        if (it == before.end()) continue;
        auto params = lp->parameters();
        if (params.size() != it->second.size()) continue;  // shape changed under us

        const std::string name = lp->name();
        for (size_t i = 0; i < params.size(); ++i) {
            const double s = mult.resolve(name, i);
            // Fast path: scale 1.0 leaves the tensor bit-identical. Skipping
            // the arithmetic entirely is what makes D1 an EXACT no-op rather
            // than a no-op that accumulates rounding error.
            if (s == 1.0) continue;
            Tensor& w = *params[i];
            const Tensor& b = it->second[i];
            const size_t n = w.rows * w.cols;
            if (b.rows != w.rows || b.cols != w.cols) continue;
            for (size_t k = 0; k < n; ++k)
                w.data[k] = b.data[k] + s * (w.data[k] - b.data[k]);
        }
    }
}

void lr_multiplier_apply(Model& model, const LrMultiplier& mult,
                         std::map<Layer*, std::vector<Tensor>>* before) {
    // Without an externally-supplied snapshot we cannot recover a delta, so
    // this two-argument form is only meaningful as the correction half of a
    // step the caller performed. Take the snapshot first when the caller does
    // not have one -- which makes the correction a no-op relative to the
    // current weights, and is the safe behaviour rather than a silent wrong
    // answer. See the header's CONTRACT note.
    if (before) {
        lr_multiplier_apply_snapshot(model, mult, *before);
        return;
    }
    std::map<Layer*, std::vector<Tensor>> snap;
    lr_multiplier_snapshot(model, snap);
    lr_multiplier_apply_snapshot(model, mult, snap);
}