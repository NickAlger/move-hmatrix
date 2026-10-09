// move-hmatrix: H2Lib adapter, C++ conveniences over hmb_h2lib.h: handles (shared
// ownership), exceptions, BlackBox. Includes no H2Lib header. Conventions: hmb_h2lib.h.
//
// Ownership: an H2libHandle owns the cluster trees, index arrays and hmatrix of one import,
// plus the keepalive it was given; an H2libH2Handle owns an H2 matrix and keeps the
// H2libHandle it was built from alive (the H2 matrix uses its clusters); a view from
// h2lib_export keeps its H2libHandle alive. Thread safety: as hmb_h2lib.h.
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb_h2lib.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace hmb {

using H2libHandle = std::shared_ptr<hmb_h2lib_hm>;       // imported trees + hmatrix
using H2libH2Handle = std::shared_ptr<struct _h2matrix>; // H2 matrix (keeps its H2libHandle alive)

// hmb_h2lib_import; throws std::invalid_argument("hmb::h2lib_import: ..."). A borrowing
// import (HMB_H2LIB_BORROW) reads the view's leaf data in place: pass what owns that data as
// keepalive (e.g. the shared_ptr an exporter returned), and the handle keeps it alive.
inline H2libHandle h2lib_import(const hmb_hmatrix &A, int mode = HMB_H2LIB_COPY, hmb_h2lib_import_info *info = nullptr,
                                std::shared_ptr<const void> keepalive = nullptr) {
    hmb_h2lib_hm *M = nullptr;
    char msg[512] = "";
    if (hmb_h2lib_import(&A, mode, &M, info, msg, sizeof msg))
        throw std::invalid_argument(std::string("hmb::h2lib_import: ") + msg);
    // the deleter drops keepalive itself: a deleter lives as long as weak_ptrs to the handle
    return H2libHandle(M, [keepalive](hmb_h2lib_hm *p) mutable {
        hmb_h2lib_free(p);
        keepalive.reset();
    });
}

// hmb_h2lib_export of an imported matrix: the view, as an aliasing shared_ptr whose control
// block owns the view's tree arrays and keeps M alive (zero-copy: the view reads M's leaves).
inline std::shared_ptr<const hmb_hmatrix> h2lib_export(const H2libHandle &M) {
    struct Held {
        H2libHandle keep; // declared first, destroyed last
        hmb_cstore store;
        ~Held() { hmb_cstore_free(&store); }
    };
    auto h = std::make_shared<Held>();
    h->keep = M;
    hmb_cstore_init(&h->store);
    char msg[512] = "";
    if (hmb_h2lib_export(hmb_h2lib_hmatrix(M.get()), &h->store, msg, sizeof msg))
        throw std::invalid_argument(std::string("hmb::h2lib_export: ") + msg);
    return std::shared_ptr<const hmb_hmatrix>(h, &h->store.A);
}

// hmb_h2lib_to_h2; throws std::invalid_argument("hmb::h2lib_to_h2: ...") for a non-strict
// tree (GOTCHA(strict-tree)). The result keeps M alive.
inline H2libH2Handle h2lib_to_h2(const H2libHandle &M, double eps, hmb_h2lib_h2_info *info = nullptr) {
    struct _h2matrix *H2 = nullptr;
    char msg[512] = "";
    if (hmb_h2lib_to_h2(hmb_h2lib_hmatrix(M.get()), eps, &H2, info, msg, sizeof msg))
        throw std::invalid_argument(std::string("hmb::h2lib_to_h2: ") + msg);
    H2libHandle keep = M;
    return H2libH2Handle(H2, [keep](struct _h2matrix *p) mutable {
        hmb_h2lib_free_h2(p); // before the clusters it uses
        keep.reset();
    });
}

// Square matrices as BlackBoxes (ORIGINAL ordering); the closures keep the matrices alive.
inline BlackBox h2lib_blackbox(const H2libHandle &M) {
    int rows, cols;
    hmb_h2lib_size(hmb_h2lib_hmatrix(M.get()), &rows, &cols);
    if (rows != cols) throw std::invalid_argument("hmb::h2lib_blackbox: square matrices only");
    BlackBox b;
    b.n = rows;
    b.apply = [M, rows](char trans, int k, const double *X, double *Y) {
        hmb_h2lib_matvec(hmb_h2lib_hmatrix(M.get()), trans, k, X, rows, Y, rows);
    };
    return b;
}

inline BlackBox h2lib_h2_blackbox(const H2libH2Handle &H) {
    int rows, cols;
    hmb_h2lib_h2_size(H.get(), &rows, &cols);
    if (rows != cols) throw std::invalid_argument("hmb::h2lib_h2_blackbox: square matrices only");
    BlackBox b;
    b.n = rows;
    b.apply = [H, rows](char trans, int k, const double *X, double *Y) {
        hmb_h2lib_h2_matvec(H.get(), trans, k, X, rows, Y, rows);
    };
    return b;
}

} // namespace hmb
