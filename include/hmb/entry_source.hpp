// move-hmatrix: an entry oracle, the common currency of the entry route.
#pragma once
#include <functional>

namespace hmb {

// get(M, N, rows, cols, out): out (M x N, column-major, leading dimension M)
// = A(rows, cols), with ORIGINAL (user) 0-based indices.
// thread_safe tells destinations whether get() may be called concurrently.
struct EntrySource {
    int n_rows = 0, n_cols = 0;
    bool thread_safe = false;
    std::function<void(int M, int N, const int *rows, const int *cols, double *out)> get;
};

} // namespace hmb
