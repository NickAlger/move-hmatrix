// hmatrix-bridges: Htool-DDM adapter, library-neutral interface.
//
// Includes no Htool header, so it can be used in the same translation unit as any other
// adapter (Htool's BLAS prototypes clash with H2Opus's; see GOTCHAS.md). The Htool types
// live in htool_native.hpp, for code that already works with htool::HMatrix.
//
// Tested against Htool-DDM commit 0ee4cac (header-only), sequential (no MPI).
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include <memory>

namespace hmb {

struct HtoolMatrix; // cluster tree(s) + htool::HMatrix<double>; defined in htool_native.hpp
using HtoolHandle = std::shared_ptr<HtoolMatrix>;

struct HtoolOptions {
    double epsilon = 1e-6;  // compression tolerance (Htool's relative per-block ACA tolerance)
    double eta = 10;        // admissibility parameter (Htool's default Rjasanow-Steinbach rule)
    int leaf_size = 64;     // maximal leaf cluster size
    char symmetry = 'N';    // 'N' general; 'S' symmetric (stores the lower triangle only)
    // Htool's default (true) refines the block tree so blocks line up across levels. With
    // false, dense leaves can sit on non-leaf clusters: such a tree is not "strict", and
    // H2Lib's H -> H2 conversion silently gives a wrong result on it (GOTCHAS.md).
    bool block_tree_consistency = true;
    bool parallel = false;  // build with Htool's parallel execution policy
};

struct HtoolStats {
    double build_seconds = 0;
    long long stored_values = 0; // scalars in dense leaves + U and V factors
    int max_rank = 0;
    int n_dense_leaves = 0, n_lowrank_leaves = 0;
};

// Compresses an n x n matrix given by its entries. coords: dim x n, point-major
// (coords[dim * i + d]), one point per row/column index in the ORIGINAL ordering; rows and
// columns share one cluster tree. With opt.parallel the generator is called concurrently;
// a source that is not thread_safe is serialized (see htool_from_entries.hpp).
HtoolHandle htool_build(const EntrySource &entries, int dim, const double *coords, const HtoolOptions &opt = {},
                        HtoolStats *stats = nullptr);

// The H-matrix as a matvec operator in the ORIGINAL ordering (Htool permutes internally).
// The BlackBox keeps the matrix alive. symmetric is true for symmetry == 'S'.
BlackBox htool_blackbox(const HtoolHandle &A);

HtoolStats htool_stats(const HtoolHandle &A);
int htool_size(const HtoolHandle &A);

} // namespace hmb
