// hmatrix-bridges: Htool-DDM adapter, native side (includes Htool headers).
//
// Use this header in code that already builds or consumes htool::HMatrix objects; code
// that only moves matrices between libraries should include hmb_htool.hpp instead. Do not
// include it in a translation unit that also includes H2Opus (conflicting BLAS prototypes)
// or H2Lib (macros named I, real, ...).
#pragma once
#include "hmb_htool.hpp"
#include "htool_from_entries.hpp"

#include <htool/clustering/cluster_node.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_matrix_product.hpp>
#include <htool/matrix/matrix_view.hpp>
#include <memory>

namespace hmb {

struct HtoolMatrix {
    std::unique_ptr<htool::Cluster<double>> row_tree;
    std::unique_ptr<htool::Cluster<double>> col_tree; // null when rows and columns share row_tree
    std::unique_ptr<htool::HMatrix<double>> A;
    char symmetry = 'N';
    double build_seconds = 0;

    const htool::Cluster<double> &rows() const { return *row_tree; }
    const htool::Cluster<double> &cols() const { return col_tree ? *col_tree : *row_tree; }
};

// Takes ownership of an H-matrix built elsewhere and of the cluster tree(s) it refers to
// (HMatrix keeps references to its trees, so they must live as long as it does).
inline HtoolHandle htool_wrap(std::unique_ptr<htool::Cluster<double>> row_tree,
                              std::unique_ptr<htool::Cluster<double>> col_tree,
                              std::unique_ptr<htool::HMatrix<double>> A, char symmetry = 'N') {
    auto h = std::make_shared<HtoolMatrix>();
    h->row_tree = std::move(row_tree);
    h->col_tree = std::move(col_tree);
    h->A = std::move(A);
    h->symmetry = symmetry;
    return h;
}

// Matvec with any htool::HMatrix in the ORIGINAL ordering. keepalive is copied into the
// closure (pass whatever owns A and its trees).
inline BlackBox blackbox_from_htool(const htool::HMatrix<double> &A, bool symmetric,
                                    std::shared_ptr<const void> keepalive = nullptr) {
    BlackBox bb;
    bb.n = A.nb_rows();
    bb.symmetric = symmetric;
    const htool::HMatrix<double> *Ap = &A;
    bb.apply = [Ap, keepalive](char trans, int k, const double *X, double *Y) {
        const int m = Ap->nb_rows(), n = Ap->nb_cols();
        const bool t = (trans == 'T' || trans == 't');
        htool::MatrixView<const double> Xv(t ? m : n, k, X);
        htool::MatrixView<double> Yv(t ? n : m, k, Y);
        // public API in user numbering: Htool permutes to and from its cluster ordering
        htool::add_hmatrix_matrix_product(t ? 'T' : 'N', 'N', 1.0, *Ap, Xv, 0.0, Yv);
    };
    return bb;
}

} // namespace hmb
