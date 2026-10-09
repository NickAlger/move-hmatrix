// hmatrix-bridges: any neutral view as a BlackBox or an EntrySource.
//
// With these, the output of any exporter can feed the rebuild routes (an importer that
// takes entries or matvecs) even when no block-copy importer exists for the destination.
// Both use the reference kernels of src/neutral.c (hmb_matvec, hmb_entries); they are
// simple and correct, not tuned.
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include "hmb/neutral.h"
#include <memory>

namespace hmb {

// keepalive: whatever owns the storage behind A (a shared_ptr<NeutralStore>, a handle to
// the source library's matrix, ...). It is copied into the closure.
inline BlackBox blackbox_from_neutral(const hmb_hmatrix *A, std::shared_ptr<const void> keepalive = nullptr,
                                      bool symmetric = false) {
    BlackBox b;
    b.n = A->n_rows;
    b.symmetric = symmetric;
    b.apply = [A, keepalive](char trans, int k, const double *X, double *Y) {
        const bool t = (trans == 'T' || trans == 't');
        hmb_matvec(A, t ? 'T' : 'N', k, X, t ? A->n_rows : A->n_cols, Y, t ? A->n_cols : A->n_rows);
    };
    return b;
}

inline EntrySource entries_from_neutral(const hmb_hmatrix *A, std::shared_ptr<const void> keepalive = nullptr) {
    EntrySource e;
    e.n_rows = A->n_rows;
    e.n_cols = A->n_cols;
    e.thread_safe = true; // hmb_entries is reentrant and read-only on A
    e.get = [A, keepalive](int M, int N, const int *rows, const int *cols, double *out) {
        hmb_entries(A, M, rows, N, cols, out, M);
    };
    return e;
}

} // namespace hmb
