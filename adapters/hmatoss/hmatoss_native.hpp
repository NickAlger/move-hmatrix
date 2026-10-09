// hmatrix-bridges: hmat-oss adapter, native side (includes hmat/hmat.h).
//
// For code that already holds hmat-oss objects: hmatoss_wrap turns an hmat_interface_t +
// hmat_matrix_t pair into an HmatossHandle usable with every function of hmb_hmatoss.hpp,
// and HmatossMatrix gives access to the hmat-oss objects behind a handle. Code that only
// moves matrices between libraries should include hmb_hmatoss.hpp instead. hmat.h declares
// no BLAS prototypes and defines no short macros, but keep the one-library-per-translation-
// unit rule of LAYOUT.md anyway.
//
// Tested against hmat-oss commit 45a2fdb.
#pragma once
#include "hmb_hmatoss.hpp"

#include <hmat/hmat.h>

namespace hmb {

struct HmatossMatrix {
    hmat_interface_t hi{};               // value_type == HMAT_DOUBLE_PRECISION (checked)
    hmat_matrix_t *A = nullptr;
    hmat_cluster_tree_t *tree = nullptr; // cluster tree owned by the handle (hmatoss_build), else null
    bool owns_matrix = true;             // destroy A with the handle
    double build_seconds = 0;

    HmatossMatrix() = default;
    HmatossMatrix(const HmatossMatrix &) = delete;
    HmatossMatrix &operator=(const HmatossMatrix &) = delete;
    ~HmatossMatrix() {
        if (A && owns_matrix) hi.destroy(A);
        if (tree) hmat_delete_cluster_tree(tree);
    }
};

// Wraps a matrix created with the interface iface (copied into the handle). With
// take_ownership the handle destroys the matrix; the matrix's cluster trees must outlive the
// handle unless the matrix owns them (hmat-oss own_cluster_trees, or a matrix from
// read_struct). Throws std::invalid_argument unless iface.value_type is HMAT_DOUBLE_PRECISION
// (GOTCHA(scalar-type)). A factorized matrix may be wrapped, but hmatoss_entries,
// hmatoss_blackbox and hmatoss_export refuse it.
HmatossHandle hmatoss_wrap(const hmat_interface_t &iface, hmat_matrix_t *A, bool take_ownership = false);

} // namespace hmb
