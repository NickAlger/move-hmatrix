// hmatrix-bridges: hmat-oss adapter, library-neutral interface.
//
// What it does
//   - hmatoss_build:    assemble an hmat-oss H-matrix from any EntrySource (ACA+).
//   - hmatoss_entries:  an hmat-oss matrix as an EntrySource (entry route), over get_values.
//   - hmatoss_blackbox: an hmat-oss matrix as a BlackBox (matvec route), over gemv.
//   - hmatoss_export / hmatoss_import: block copy between hmat-oss and the neutral view
//     (include/hmb/neutral.h), through hmat-oss's serialization stream; lossless.
// Includes no hmat-oss header; the hmat-oss types live in hmatoss_native.hpp (for code that
// already holds hmat_interface_t / hmat_matrix_t objects, see hmatoss_wrap there).
//
// Conventions
//   - EntrySource and BlackBox use the ORIGINAL (user) ordering and 0-based indices, like the
//     rest of the package. hmat-oss's get_values takes 1-based indices while its assembly
//     callbacks are 0-based; the adapter converts (GOTCHA(index-base) in hmb_hmatoss.cpp).
//   - hmat-oss permutations are internal -> original (hmat's "indices", i2e), the same
//     direction as the neutral row_perm / col_perm: copied as they are.
//   - hmat-oss low-rank leaves are A * B^T with B cols x rank: exported as U = A, V = B,
//     vtrans = 1; on import a vtrans = 0 factor (V rank x cols) is transposed into B.
//   - Symmetric storage (hmat-oss "lower symmetric": only the lower block triangle is
//     stored; upper after hmat-oss's transpose): get_values leaves the other triangle
//     unwritten with a success code, so hmatoss_entries serves those entries from the stored
//     triangle by symmetry, and hmatoss_export mirrors the missing blocks so the view is the
//     full matrix (GOTCHA(symmetric-storage)).
//   - Real double precision only. hmat-oss picks the scalar type at run time and reads and
//     writes values through void*, so a single-precision or complex matrix would silently
//     produce garbage in double buffers: every entry point refuses anything but
//     HMAT_DOUBLE_PRECISION.
//   - Factorized matrices are refused (GOTCHA(factorized-state)): after LU, get_values
//     returns factor entries without an error; after LLt, gemv applies the factor L instead
//     of A (by design); export would copy the factors. Extract, multiply or export before
//     factorizing. The state is read from the structure stream (write_struct, O(n + blocks))
//     when an EntrySource, BlackBox or export is made, not on every call.
//
// Errors: std::invalid_argument / std::runtime_error, message starting with the function name.
//
// Lifetimes: the HmatossHandle owns the hmat-oss matrix (and the cluster tree it was built
// on). EntrySource and BlackBox objects keep the handle alive. The exported view is an
// aliasing shared_ptr whose control block owns copies of everything (hmat-oss's public API
// exposes no stable pointer into its leaves: the stream callback only sees a buffer during
// the call), so the source may be destroyed after export. The imported matrix owns its own
// copies; the view may be released after import.
//
// Thread safety: hmatoss_entries is thread-safe: get_values only reads the matrix and keeps
// its state per call (checked with Helgrind and DRD in the study, and bitwise against
// sequential calls from 4 OpenMP threads by tests/test_hmatoss_entries.cpp). This relies on
// hmat-oss being built without HMAT_SCALAR_ARRAY_ORTHO (its CMake never sets it); with it,
// reading a dense leaf resets an orthogonality flag, a benign write race. hmatoss_build,
// hmatoss_blackbox, export and import are sequential (hmat-oss sets OpenMP and MKL to one
// thread inside its operations and restores the previous setting afterwards). hmat-oss's
// global settings (hmat_set_parameters) are process-wide and apply to every build.
//
// Limits: square matrices for hmatoss_build and hmatoss_blackbox (rectangular for export,
// import and hmatoss_entries); 32-bit indices (hmat-oss API). hmat-oss 45a2fdb leaks 16 bytes
// per block-tree node (GOTCHA(hmat-node-leak)); tests/lsan.supp silences LeakSanitizer.
//
// Tested against hmat-oss commit 45a2fdb (1.11.1 + 23, sequential default engine).
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include "hmb/neutral.h"
#include <memory>

namespace hmb {

struct HmatossMatrix; // interface + hmat_matrix_t (+ cluster tree); defined in hmatoss_native.hpp
using HmatossHandle = std::shared_ptr<HmatossMatrix>;

struct HmatossOptions {
    // ACA+ tolerance, also the matrix's low-rank epsilon: hmat-oss truncates every compressed
    // block at the matrix's epsilon (default 1e-4), not at the compressor's, so the adapter
    // sets both (GOTCHA(hmat-epsilon)).
    double epsilon = 1e-6;
    double eta = 2.0;             // standard (Hackbusch) admissibility: min(diam) <= eta * dist
    int leaf_size = 64;           // median clustering, at most leaf_size points per leaf
    bool lower_symmetric = false; // store the lower block triangle only (the source must be symmetric)
};

struct HmatossStats {
    int n_rows = 0, n_cols = 0;
    double build_seconds = 0;          // hmatoss_build only (cluster tree + assembly)
    long long stored_values = 0;       // scalars in dense leaves + A and B factors (stored blocks only)
    long long dense_values = 0;        // of which in dense leaves
    long long n_dense_leaves = 0, n_lowrank_leaves = 0;
    char storage = 'N';                // 'N' full, 'L' lower symmetric, 'U' upper symmetric (after transpose)
};

// Assembles an n x n hmat-oss matrix from entries. coords: dim x n, point-major
// (coords[dim * i + d]), one point per row/column in the ORIGINAL ordering; rows and columns
// share one cluster tree. hmat-oss calls the source sequentially (it need not be thread-safe),
// with 0-based indices, for whole blocks and single rows and columns. With
// opt.lower_symmetric only entries (i, j) of the lower triangle in hmat's ordering are
// requested.
HmatossHandle hmatoss_build(const EntrySource &entries, int dim, const double *coords, const HmatossOptions &opt = {},
                            HmatossStats *stats = nullptr);

// The matrix as an EntrySource (thread_safe = true). Upper-triangle entries of symmetric
// storage are read from the lower triangle. Indices are bounds-checked (std::invalid_argument).
EntrySource hmatoss_entries(const HmatossHandle &A);

// The matrix as a matvec operator in the ORIGINAL ordering (square only). symmetric is true
// for symmetric storage. apply() copies X (hmat-oss permutes its input vector in place,
// GOTCHA(gemv-mutates-input)) and is not meant to be called concurrently.
BlackBox hmatoss_blackbox(const HmatossHandle &A);

// Sizes, storage and hmat-oss's leaf statistics (get_info); works on factorized matrices too.
HmatossStats hmatoss_stats(const HmatossHandle &A);

// ---- Block copy through hmat-oss's serialization stream (write_struct / write_data) ----

struct HmatossExportStats {
    double seconds = 0;
    long long stream_values = 0;     // scalars copied out of hmat-oss's data stream (= stored values)
    long long transposed_values = 0; // dense values copied again to mirror symmetric storage
    long long mirrored_leaves = 0;   // leaves of the unstored triangle (low-rank ones alias the stored factors)
    char storage = 'N';
};

// Copies an hmat-oss matrix into a neutral view (validated). The result is an aliasing
// shared_ptr: its control block owns the copied leaves, trees and permutations, so the view
// outlives the source. Rows and columns share one cluster tree in the view when hmat-oss's
// two trees are identical. Bounding boxes come from hmat-oss's point coordinates. Clusters
// with a single son and 1 x 1 block splits are collapsed (they cover the same range). Zero
// blocks (rank-0 or null leaves) become rank-0 low-rank leaves.
std::shared_ptr<const hmb_hmatrix> hmatoss_export(const HmatossHandle &A, HmatossExportStats *stats = nullptr);

struct HmatossImportOptions {
    // Low-rank epsilon stored in every block: hmat-oss's later recompressions (LU, gemm,
    // axpy, truncate) use it. The import itself does not recompress.
    double epsilon = 1e-6;
    // Optional point coordinates (dim x n, point-major, ORIGINAL ordering) stored in the
    // imported cluster trees. Without them the trees carry zero coordinates: fine for every
    // operation on the imported matrix, but the trees must not be reused to build new
    // matrices by admissibility. col_coords defaults to row_coords for square matrices.
    int dim = 0;
    const double *row_coords = nullptr;
    const double *col_coords = nullptr;
};

// Builds an hmat-oss matrix (full storage) holding exactly the blocks of A: same cluster
// trees and permutations, same block tree, every dense leaf and low-rank factor copied
// (no recompression). Any valid view is representable (hmat-oss needs every block's children
// to be all pairings of its row and column children, which hmb_validate already enforces);
// invalid views are refused with hmb_validate's message. A chain of single sons becomes one
// hmat-oss cluster (it covers one range), so blocks that keep a cluster and blocks that
// descend into its single son are the same to hmat-oss; 1 x 1 block splits are collapsed.
HmatossHandle hmatoss_import(const hmb_hmatrix &A, const HmatossImportOptions &opt = {});

} // namespace hmb
