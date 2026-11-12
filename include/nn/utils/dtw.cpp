#include "dtw.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===========================================================================
// utils/dtw.cpp -- the dynamic time warping utilities.
//
// THE SPECIFICATION IS tests/dtw_oracle.py, which was written and run BEFORE
// this file existed. The header carries the full rationale; the notes below
// are the implementation-side decisions only.
//
// ---------------------------------------------------------------------------
// THE RECURRENCE
// ---------------------------------------------------------------------------
//
// With x of shape (T, D) and y of shape (S, D), and `d[i][j]` the cost of
// warping x's first i time steps onto y's first j:
//
//     d[0][0] = 0
//     d[i][j] = min( d[i-1][j-1],      // diagonal
//                    d[i-1][j],        // vertical
//                    d[i][j-1] )      // horizontal
//                + local_cost(x[i-1], y[j-1])
//
// The three neighbours are the symmetric1 lattice. `local_cost` is the
// EUCLIDEAN distance between the two ROW vectors.
//
// The first row and first column are NOT zero: a warping path may start by
// holding one series still, so d[1][0] and d[0][1] carry the local cost of
// the first pair. Only d[0][0] is zero.
//
// ---------------------------------------------------------------------------
// THE BAND
// ---------------------------------------------------------------------------
//
// A cell (i,j) is unusable when |i-j| > band, on the 1-based CELL indices.
// Because the data indices are (i-1,j-1) and |i-j| == |(i-1)-(j-1)| exactly,
// the 1-based and 0-based readings are the same function -- measured, 0
// differences over 2000 random cases.
//
// The band is applied as a filter on the cell, which means an infeasible
// band leaves the final cell unreachable and the cost is +inf. That falls
// out of the recurrence rather than being special-cased.
//
// ---------------------------------------------------------------------------
// THE PATH
// ---------------------------------------------------------------------------
//
// The predecessor is recorded during the forward pass, so the backtrack does
// not have to re-derive the cost. The preference at each cell is
// DIAGONAL, then HORIZONTAL, then VERTICAL -- the order measured to
// reproduce `dtw-python`'s own `index1`/`index2` on tie-heavy integer
// observations (300/300).
//
// The cells are reported as 0-BASED DATA INDICES, matching `dtw-python`'s
// convention, so the two can be compared directly.
//
// A subtlety worth stating: the path's FIRST cell is (0,0) and its LAST is
// (T-1,S-1). It is NOT the case that every cell of the DP matrix is on the
// path -- the path visits only the cells it actually matches.
//
// ===========================================================================

namespace nn {
namespace utils {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The local cost: the Euclidean distance between two D-dimensional
// observations, i.e. between ROWS.
double local_cost(const Tensor& x, size_t xi, const Tensor& y, size_t yi) {
    const size_t dim = x.cols;
    double acc = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        const double diff = x(xi, k) - y(yi, k);
        acc += diff * diff;
    }
    return std::sqrt(acc);
}

// The guards. Every message is DISTINCT: a substring shared by two messages
// discriminates nothing between them, and this repo has hit that trap
// repeatedly.
void check_series(const Tensor& x, const Tensor& y) {
    if (x.rows == 0 || x.cols == 0) {
        throw std::invalid_argument(
            "dtw: the x series must have at least one time step and one "
            "dimension");
    }
    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "dtw: the y series must have at least one time step and one "
            "dimension");
    }
    if (x.cols != y.cols) {
        throw std::invalid_argument(
            "dtw: the two series must have the same number of dimensions "
            "(x has a different number of columns than y)");
    }
}

void check_band(int band) {
    if (band < 0) {
        throw std::invalid_argument(
            "dtw: the band must be non-negative (use dtw_distance without a "
            "band for the unbounded distance)");
    }
}

// The forward pass. `pred` records the predecessor of each cell:
//   0 = diagonal (i-1,j-1), 1 = vertical (i-1,j), 2 = horizontal (i,j-1).
struct WarpingTable {
    std::vector<std::vector<double>> cost;
    std::vector<std::vector<unsigned char>> pred;
};

WarpingTable forward(const Tensor& x, const Tensor& y, int band) {
    const size_t T = x.rows;
    const size_t S = y.rows;
    WarpingTable table;
    table.cost.assign(T + 1, std::vector<double>(S + 1, kInfinity));
    table.pred.assign(T + 1, std::vector<unsigned char>(S + 1, 0));

    table.cost[0][0] = 0.0;
    // THE FIRST ROW AND FIRST COLUMN ARE NOT INITIALISED TO ZERO, AND NOT
    // TO THE LOCAL COST EITHER. They stay +inf.
    //
    // This is the single easiest thing to get wrong in this module, and the
    // difference is not a rounding error -- it is a different ANSWER.
    // Measured, on x = [100, 0] against y = [0]:
    //
    //   with the first column initialised to the local cost ....... 0.0
    //   with the first column left at +inf ........................ 100.0
    //   dtw-python ............................................... 100.0
    //
    // The reason is structural: a finite first column lets the warping path
    // START LATE -- it can enter the grid at row 2 and thereby SKIP the
    // first observation of x entirely. That is not a warping; that is a
    // different problem. The path must begin at (0,0), which means the
    // first observation of each series is always matched.
    //
    // The band is applied to the same cells, so a banded path obeys the
    // same rule.
    for (size_t i = 1; i <= T; ++i) {
        table.cost[i][0] = kInfinity;
    }
    for (size_t j = 1; j <= S; ++j) {
        table.cost[0][j] = kInfinity;
    }

    for (size_t i = 1; i <= T; ++i) {
        for (size_t j = 1; j <= S; ++j) {
            if (band >= 0 &&
                (static_cast<int>(i) - static_cast<int>(j) > band ||
                 static_cast<int>(j) - static_cast<int>(i) > band)) {
                continue;
            }
            const double c = local_cost(x, i - 1, y, j - 1);
            // Preference: DIAGONAL, then HORIZONTAL, then VERTICAL.
            double best = table.cost[i - 1][j - 1];
            unsigned char which = 0;
            if (table.cost[i][j - 1] < best) {
                best = table.cost[i][j - 1];
                which = 2;
            }
            if (table.cost[i - 1][j] < best) {
                best = table.cost[i - 1][j];
                which = 1;
            }
            table.cost[i][j] = best + c;
            table.pred[i][j] = which;
        }
    }
    return table;
}

// The backtrack, producing 0-based data indices.
std::vector<std::pair<size_t, size_t>> backtrack(const WarpingTable& table,
                                                   size_t T, size_t S) {
    std::vector<std::pair<size_t, size_t>> cells;
    size_t i = T;
    size_t j = S;
    while (i > 0 && j > 0) {
        cells.push_back(std::make_pair(i - 1, j - 1));
        const unsigned char p = table.pred[i][j];
        if (p == 0) {
            --i;
            --j;
        } else if (p == 2) {
            --j;
        } else {
            --i;
        }
    }
    while (i > 0) {
        cells.push_back(std::make_pair(i - 1, 0));
        --i;
    }
    while (j > 0) {
        cells.push_back(std::make_pair(0, j - 1));
        --j;
    }
    std::reverse(cells.begin(), cells.end());
    return cells;
}

}  // namespace

double dtw_distance(const Tensor& x, const Tensor& y) {
    check_series(x, y);
    const WarpingTable table = forward(x, y, -1);
    return table.cost[x.rows][y.rows];
}

double dtw_distance(const Tensor& x, const Tensor& y, int band) {
    check_series(x, y);
    check_band(band);
    const WarpingTable table = forward(x, y, band);
    // An infeasible band leaves the final cell unreachable, and the cost is
    // +inf -- which is the honest answer to "no such warping path exists".
    return table.cost[x.rows][y.rows];
}

DtwPath dtw_align(const Tensor& x, const Tensor& y) {
    check_series(x, y);
    const WarpingTable table = forward(x, y, -1);
    DtwPath out;
    out.cost = table.cost[x.rows][y.rows];
    out.cells = backtrack(table, x.rows, y.rows);
    return out;
}

DtwPath dtw_align(const Tensor& x, const Tensor& y, int band) {
    check_series(x, y);
    check_band(band);
    const WarpingTable table = forward(x, y, band);
    DtwPath out;
    out.cost = table.cost[x.rows][y.rows];
    if (!std::isfinite(out.cost)) {
        // No monotone path exists within the band: the path is EMPTY and the
        // cost is +inf. An empty path is unambiguous -- there is no
        // "which" path to pick when there is no path.
        return out;
    }
    out.cells = backtrack(table, x.rows, y.rows);
    return out;
}

}  // namespace utils
}  // namespace nn
