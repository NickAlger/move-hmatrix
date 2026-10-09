// hmatrix-bridges: owning container behind an hmb_hmatrix view (C++ exporters).
// Exporters that can alias the source's storage only fill cl / bl / sons; exporters
// that must copy (a transpose, a stream they parsed) allocate leaf data with alloc().
#pragma once
#include "hmb/neutral.h"
#include <cstddef>
#include <deque>
#include <vector>

namespace hmb {

struct NeutralStore {
    std::vector<hmb_cluster> cl;
    std::vector<hmb_block> bl;
    std::vector<int> sons;
    std::vector<int> row_perm, col_perm;    // used when the source's arrays cannot be aliased
    std::deque<std::vector<double>> data;   // owned leaf data; deque keeps addresses stable
    hmb_hmatrix A{};

    double *alloc(std::size_t n) {
        data.emplace_back(n);
        return data.back().data();
    }
    // Point the view at the vectors. Call again after any vector grows.
    const hmb_hmatrix *finalize() {
        A.ncl  = (int)cl.size();
        A.cl   = cl.data();
        A.nbl  = (int)bl.size();
        A.bl   = bl.data();
        A.sons = sons.data();
        if (!row_perm.empty()) A.row_perm = row_perm.data();
        if (!col_perm.empty()) A.col_perm = col_perm.data();
        return &A;
    }
};

} // namespace hmb
