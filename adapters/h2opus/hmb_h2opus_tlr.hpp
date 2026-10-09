// hmatrix-bridges: H2Opus tile low-rank (TLR) adapter, library-neutral interface.
//
// What it does
//   - h2opus_tlr_import:       neutral view of a uniform tile partition -> H2Opus TTLR_Matrix
//                              (CPU, double), copying every tile; lossless.
//   - h2opus_tlr_from_entries: H2Opus's own TLR construction (construct_tlr_matrix) from any
//                              EntrySource (entry route into H2Opus TLR).
//   - h2opus_tlr_export:       TTLR_Matrix -> neutral view, zero-copy.
//   - h2opus_tlr_matvec / h2opus_tlr_blackbox: tlr_gemv in the ORIGINAL ordering, A and A^T.
// Includes no H2Opus header (its BLAS prototypes clash with Htool's, GOTCHA(header-clash));
// the H2Opus types live in h2opus_tlr_native.hpp, for code that already holds a TTLR_Matrix.
//
// Shapes H2Opus TLR holds, and what the import accepts
//   One square matrix, one ordering for rows and columns, the root split ONCE into p x p tiles
//   of block_size rows and columns, except the last tile row/column, which may be smaller
//   (H2Opus pads the matrix to p * block_size). Diagonal tiles are dense, off-diagonal tiles
//   are U * V^T with rank <= block_size. The import accepts exactly the views whose root block
//   splits the row and column roots into the same p tiles (sizes block_size, ..., block_size,
//   remainder; the same on both sides), whose p x p children are all leaves, and whose row and
//   column permutations are equal. Leaf kinds are converted losslessly where needed:
//     - a low-rank diagonal tile is expanded to dense (U * V^T by BLAS: rounding only);
//     - a dense off-diagonal tile is stored exactly as D * I (rank = its column count);
//     - an off-diagonal tile of rank > block_size is stored as its product, likewise.
//   Anything else (rectangular, nested blocks, unequal tiles, different row and column
//   orderings) is refused with std::invalid_argument naming the reason.
//
// Conventions
//   - TTLR_Matrix::index_map[k] is the ORIGINAL index at TLR position k: the same direction as
//     the neutral row_perm (GOTCHA(perm-direction)); tlr_gemv works in TLR order and pads its
//     vectors to p * block_size; h2opus_tlr_matvec permutes and pads for you, and calls
//     tlr_gemv with one parallel block column when it multiplies several vectors (with more,
//     tlr_gemv's accumulation buffers overlap: GOTCHA(h2opus-tlr-gemv-multi)).
//   - Tile (i, j) = U_ij V_ij^T, both block_size x rank (padded rows), column-major, ld =
//     block_size: exported as vtrans = 1 with ldU = ldV = block_size (padded leading
//     dimensions); a vtrans = 0 source is transposed on import (GOTCHA(lowrank-convention)).
//   - Padding written by the import: zero rows in U and V, and in the last diagonal tile zeros
//     except ones on the padded diagonal (H2Opus's own generator's convention, which keeps the
//     padded matrix nonsingular for tlr_potrf).
//   - Symmetric storage (H2OpusTLR_Symmetric, the type H2Opus's Cholesky uses): only tiles
//     below the diagonal are stored and the tiles above alias them with U and V swapped
//     (GOTCHA(symmetric-storage)); tlr_gemv answers A^T with A.
//   - Real double only: the adapter instantiates TTLR_Matrix<double>; no complex type exists in
//     H2Opus (GOTCHA(scalar-type)).
//   - Imported U factors are copied as given, not orthonormalized (H2Opus's construction
//     produces orthonormal U; tlr_gemv does not need it).
//
// Errors: std::invalid_argument / std::runtime_error, message starting with the function name
// ("hmb::h2opus_tlr_import: ..."). An exception thrown by the EntrySource of
// h2opus_tlr_from_entries must not unwind through H2Opus's OpenMP loops: the first one is
// kept (the remaining entries read as 0 and the source is not called again) and rethrown as
// std::runtime_error once construct_tlr_matrix returns.
//
// Lifetimes: H2opusTlrHandle owns the TTLR_Matrix and an H2Opus handle (tlr_gemv's workspace).
// The import copies, so the view may be released afterwards. The exported view is an aliasing
// shared_ptr whose control block owns the view's trees and keeps the handle alive (the leaves
// point into the TTLR_Matrix). BlackBoxes keep the handle alive.
//
// Thread safety: products on one handle are serialized by a mutex (the H2Opus handle's
// workspace is shared); different handles are independent. from_entries calls the
// EntrySource from H2Opus's OpenMP loops, concurrently if entries.thread_safe (otherwise
// serialized).
//
// Limits: 32-bit indices; CPU memory only (H2OPUS_HWTYPE_CPU); tile allocation
// (H2OpusTLRTile); a single TLR level (no nested tiles).
//
// Tested against H2Opus commit cbc7985 (CPU build, thrust 1.17.2, OpenMP, OpenBLAS).
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include "hmb/neutral.h"

#include <memory>

namespace hmb {

struct H2opusTlrMatrix; // TTLR_Matrix<double, CPU> + H2Opus handle; defined in h2opus_tlr_native.hpp
using H2opusTlrHandle = std::shared_ptr<H2opusTlrMatrix>;

struct H2opusTlrStats {
    int n = 0, block_size = 0, n_block = 0, padded_n = 0, max_rank = 0;
    bool symmetric = false;
    long long stored_values = 0;      // values H2Opus holds: padded diagonal tiles + padded U and V of stored tiles
    long long copied_values = 0;      // import: values read from the view; from_entries: entries requested
    int lowrank_diagonal_tiles = 0;   // import: diagonal tiles given low-rank, expanded to dense
    int dense_offdiagonal_tiles = 0;  // import: off-diagonal tiles stored as D * I
    int cleared_diagonal_ranks = 0;   // from_entries / wrap: stale diagonal ranks set to 0 (GOTCHA(h2opus-tlr-diag-rank))
    double seconds = 0;               // import / construction time
};

struct H2opusTlrImportOptions {
    // Store the lower tiles only (H2OpusTLR_Symmetric). The tiles above the diagonal of the
    // view are not read: the caller asserts that the view is symmetric.
    bool symmetric = false;
};

// Copies a uniform tile view (see above) into a new TTLR_Matrix.
H2opusTlrHandle h2opus_tlr_import(const hmb_hmatrix &A, const H2opusTlrImportOptions &opt = {},
                                  H2opusTlrStats *stats = nullptr);

struct H2opusTlrOptions {
    int block_size = 256;
    // H2Opus's ABSOLUTE truncation threshold: each tile keeps the leading columns of its
    // pivoted QR while |R_kk| * sqrt(block_size - k) > eps (batch_wrappers_cpu.inc), so the
    // Frobenius error of a tile is about eps and that of the matrix about n_block * eps.
    double eps = 1e-6;
    bool symmetric = false; // H2OpusTLR_Symmetric: entries of the lower tiles and diagonal only
};

// H2Opus's construct_tlr_matrix on an n x n EntrySource, tiles over the order perm
// (perm[k] = original index at position k; nullptr = identity; a clustering permutation
// makes tiles compressible). Every entry of the stored tiles is requested once, one at a
// time. For the non-symmetric type, the stale diagonal ranks construct_tlr_matrix leaves
// behind are cleared (GOTCHA(h2opus-tlr-diag-rank)).
H2opusTlrHandle h2opus_tlr_from_entries(const EntrySource &entries, const int *perm, const H2opusTlrOptions &opt = {},
                                        H2opusTlrStats *stats = nullptr);

// Zero-copy neutral view: root cluster [0, n) split into the n_block tiles (rows and columns
// share the tree), dense diagonal leaves (ldD = block_size), low-rank off-diagonal leaves
// (vtrans = 1, ldU = ldV = block_size), row_perm = col_perm = index_map (nullptr when the
// identity). The result is an aliasing shared_ptr: its control block owns the view's trees and
// keeps A alive, so the view stays valid for as long as the pointer, or a copy of it, lives.
std::shared_ptr<const hmb_hmatrix> h2opus_tlr_export(const H2opusTlrHandle &A);

// Y = op(A) X (trans 'N' or 'T'), X and Y in the ORIGINAL ordering, n x nrhs.
void h2opus_tlr_matvec(const H2opusTlrHandle &A, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy);

// The matrix as a BlackBox (symmetric for the symmetric type). Keeps the matrix alive.
BlackBox h2opus_tlr_blackbox(const H2opusTlrHandle &A);

H2opusTlrStats h2opus_tlr_stats(const H2opusTlrHandle &A);

} // namespace hmb
