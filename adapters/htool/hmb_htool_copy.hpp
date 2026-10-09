// hmatrix-bridges: Htool-DDM block copy. Exports an Htool H-matrix to the neutral view and
// imports a neutral view into Htool, both lossless (no recompression).
//
// Includes no Htool header: Htool's types appear in the native headers htool_export.hpp and
// htool_import.hpp. Tested against Htool-DDM commit 0ee4cac (header-only), sequential.
//
// Export (Htool -> neutral view)
//   - Zero-copy where the layouts agree. Dense leaves alias Htool's column-major blocks (ld =
//     rows). Low-rank leaves alias U (rows x rank) and V (rank x cols) with vtrans = 0: Htool
//     stores U*V with V rank x cols (GOTCHA(lowrank-convention)). The permutations alias the
//     trees' arrays: Htool's perm[k] = user index at tree position k, as in the neutral form.
//   - Symmetric storage (symmetry 'S', UPLO 'L' or 'U'): Htool stores one triangle of blocks
//     and its products apply the missing ones as transposes. The view is the full matrix
//     (GOTCHA(symmetric-storage)): every leaf Htool's product mirrors gets a mirror leaf.
//     Copied, and nothing else: V^T of each mirrored low-rank leaf (rank x cols scalars; the
//     mirror aliases the stored U as its V, vtrans = 1), the transpose of each mirrored dense
//     leaf, and each symmetric dense diagonal leaf, whose unreferenced triangle is rebuilt from
//     the stored one (Htool multiplies those with symv/symm and never reads the other half).
//   - Block tree: Htool's own tree, in the mirror image where symmetric storage left a block
//     out. Where an Htool node's children skip a cluster level (partition clusters below the
//     root when the partition is deeper than one level), intermediate blocks are added. The
//     tree may be non-strict (GOTCHA(strict-tree)): Htool makes dense leaves above the leaf
//     level where ACA fails, and one-sided splits with block_tree_consistency off.
//   - Not exported (std::invalid_argument): an MPI-local block row (target cluster smaller
//     than its tree), symmetric storage over two different trees, symmetric storage built with
//     block_tree_consistency off (GOTCHA(htool-sym-inconsistent): Htool only logs an error,
//     and its own products then omit the missing triangle: 67% error at N = 2,000), leaves
//     without data, and block trees whose leaves do not tile the matrix exactly once.
//
// Import (neutral view -> Htool)
//   - Builds Htool cluster trees (one shared tree when the view's rows and columns share their
//     clusters and permutation, otherwise two) and an htool::HMatrix<double> in full storage
//     ('N') with exactly the view's block tree: one-sided splits, more than two sons, single
//     sons and leaves on any level are kept as they are (Htool's HMatrix takes any children).
//   - Copies all leaf data (Htool owns its leaves): dense blocks and U with their leading
//     dimension removed, V transposed when the view stores it as cols x rank (vtrans = 1).
//   - The root's sons become Htool's partition clusters (rank = son index), as Htool's
//     ClusterTreeBuilder makes them; a view whose root cluster is a leaf gets one partition
//     cluster below the root with the same range, on which the matrix is then built.
//     GOTCHA(htool-partition): Htool's user-numbering products index the partition list
//     without checking it is non-empty.
//   - Cluster balls are the balls around the view's bounding boxes (radius = half diagonal).
//   - Empty clusters, and the blocks on them, are left out (they hold no data).
//     GOTCHA(htool-empty-cluster): Htool's matrix products pass ld = 0 to BLAS for a block
//     with no rows, an illegal argument (reference BLAS reports it and skips the call).
//   - GOTCHA(htool-user-transpose): when the imported row and column trees differ, Htool's
//     user-numbering products (add_hmatrix_vector_product / add_hmatrix_matrix_product)
//     permute with the wrong tree for op = 'T'. hmb::htool_blackbox permutes correctly
//     (blackbox_from_htool in htool_native.hpp); code calling Htool's products directly must
//     permute the input with the target tree and the output with the source tree around
//     internal_add_hmatrix_matrix_product for 'T'.
//
// Thread safety: export and import only read their input; neither is reentrant on the same
// output object. Lifetimes are stated per function.
#pragma once
#include "hmb/neutral.h"
#include "hmb_htool.hpp"

#include <memory>

namespace hmb {

struct HtoolExportInfo {
    long long aliased_values = 0; // leaf scalars the view reads in place from Htool's storage
    long long copied_values = 0;  // leaf scalars copied for symmetric storage (see above)
    int mirrored_leaves = 0;      // leaves added as mirror images of stored ones
    int added_blocks = 0;         // blocks of the view that are neither an Htool node nor its mirror
};

// The full matrix A as a neutral view: an aliasing shared_ptr whose control block owns the
// view's tree arrays, the copies listed above, and A itself (hence Htool's leaves and
// trees), so the view stays valid for as long as the pointer, or a copy of it, lives.
std::shared_ptr<const hmb_hmatrix> htool_export(const HtoolHandle &A, HtoolExportInfo *info = nullptr);

struct HtoolImportOptions {
    // Recorded in the Htool matrix for Htool's own later arithmetic (recompression, products,
    // factorizations); the import itself compresses nothing.
    double epsilon = 1e-6;
    double eta = 10;
};

struct HtoolImportInfo {
    long long copied_values = 0;     // leaf scalars copied (all of them)
    long long transposed_values = 0; // of which copied with a transpose (V of vtrans = 1 leaves)
};

// A new Htool matrix holding exactly the view's blocks (validated first; std::invalid_argument
// if the view is invalid or the matrix is empty). Nothing is borrowed: the view may be freed
// afterwards. The handle's symmetry is 'N'.
HtoolHandle htool_import(const hmb_hmatrix &A, const HtoolImportOptions &opt = {}, HtoolImportInfo *info = nullptr);

} // namespace hmb
