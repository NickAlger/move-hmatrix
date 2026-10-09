// move-hmatrix: STARS-H adapter, C++ conveniences over hmb_starsh.h: a handle with shared
// ownership, exceptions, an EntrySource route in and a BlackBox route out. Includes no STARS-H
// header. Conventions, partitions and limits: hmb_starsh.h. Tested, like hmb_starsh.h, against
// STARS-H commit c115e75 (OpenMP backend).
//
//   starsh_import        neutral view -> new STARS-H matrix (format 'N'), every leaf copied
//   starsh_from_entries  EntrySource -> STARS-H's own tile low-rank approximation
//   starsh_export        STARS-H matrix -> neutral view, zero-copy for stored blocks
//   starsh_blackbox      the matrix as a BlackBox, ORIGINAL ordering
//
// Products: STARS-H only computes A X (starsh_blrm__dmml). starsh_blackbox answers 'N' with
// it, and 'T' as follows. A symmetric format ('S': one triangle stored, BlackBox::symmetric
// true) is its own transpose, so 'T' runs the same STARS-H product. Any other matrix answers
// 'T' with hmb_matvec on the handle's own zero-copy view of the same leaves: exact, but the
// package's reference kernel, not STARS-H code and not tuned.
//
// Ownership: a StarshHandle owns the STARS-H problem, clusters, format and matrix of one
// import or approximation; hmb_starsh_free runs when the last copy goes. A view from
// starsh_export is an aliasing shared_ptr whose control block owns the view's trees and the
// blocks the export copies, and keeps the handle alive. A BlackBox keeps its handle alive.
//
// Errors: std::invalid_argument("hmb::starsh_<function>: <reason>") for everything the C
// interface refuses. An exception thrown by the EntrySource of starsh_from_entries must not
// unwind through STARS-H (C code, OpenMP loops): the first one is kept, the remaining entries
// read as 0 without calling the source again, and it is rethrown as std::runtime_error once
// the approximation returns.
//
// Thread safety: as hmb_starsh.h. BlackBox::apply only reads the matrix and may run
// concurrently on one handle.
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include "hmb/neutral.h"
#include "hmb_starsh.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace hmb {

using StarshHandle = std::shared_ptr<hmb_starsh>; // problem + clusters + format + matrix

namespace detail {

// The C interface's messages start with its own function name ("hmb_starsh_import: ..."):
// keep the reason, after the C++ function's name.
[[noreturn]] inline void starsh_throw(const char *fn, const char *msg) {
    const char *why = msg;
    if (std::strncmp(msg, "hmb_starsh_", 11) == 0) {
        const char *colon = std::strstr(msg, ": ");
        if (colon) why = colon + 2;
    }
    throw std::invalid_argument(std::string("hmb::") + fn + ": " + why);
}

// Context of the entry callback: the source, and the first exception it threw.
struct StarshEntries {
    const EntrySource *src;
    std::atomic<bool> failed{false};
    std::mutex lock; // guards error
    std::exception_ptr error;
    explicit StarshEntries(const EntrySource *s) : src(s) {}
};

// hmb_starsh_entry_fn over an EntrySource: out (m x n, leading dimension ldo) = A(rows, cols),
// ORIGINAL indices. EntrySource::get writes with leading dimension m: a larger ldo goes
// through a buffer. Never throws: see "Errors" above.
inline void starsh_entries_cb(int m, int n, const int *rows, const int *cols, double *out, int ldo, void *ctx) {
    auto *e = static_cast<StarshEntries *>(ctx);
    if (!e->failed.load(std::memory_order_relaxed)) {
        try {
            if (ldo == m) {
                e->src->get(m, n, rows, cols, out);
            } else {
                std::vector<double> buf(size_t(m) * n);
                e->src->get(m, n, rows, cols, buf.data());
                for (int j = 0; j < n; j++)
                    std::copy(buf.data() + size_t(j) * m, buf.data() + size_t(j + 1) * m, out + size_t(j) * ldo);
            }
            return;
        } catch (...) {
            std::lock_guard<std::mutex> g(e->lock);
            if (!e->error) e->error = std::current_exception();
            e->failed = true;
        }
    }
    for (int j = 0; j < n; j++) std::fill(out + size_t(j) * ldo, out + size_t(j) * ldo + m, 0.0);
}

} // namespace detail

// hmb_starsh_import: a new STARS-H matrix (format 'N') holding a copy of every leaf of A, for
// any valid partition (uniform tiles, HODLR, the leaves of a general H-matrix). A may be
// released afterwards.
inline StarshHandle starsh_import(const hmb_hmatrix &A) {
    hmb_starsh *M = nullptr;
    char msg[512] = "";
    if (hmb_starsh_import(&A, &M, msg, sizeof msg)) detail::starsh_throw("starsh_import", msg);
    return StarshHandle(M, hmb_starsh_free);
}

// hmb_starsh_from_entries: STARS-H's TLR approximation of the n x n matrix A (n = A.n_rows =
// A.n_cols) on uniform tiles over the order perm (perm[k] = ORIGINAL index at position k;
// nullptr = identity; a clustering order makes the tiles compressible). opt as in
// hmb_starsh.h, except that STARS-H calls A.get concurrently only if both A.thread_safe and
// opt.thread_safe are set (otherwise one call at a time). A.get is used during the call only.
inline StarshHandle starsh_from_entries(const EntrySource &A, const int *perm, const hmb_starsh_tlr_opts &opt) {
    if (A.n_rows != A.n_cols)
        throw std::invalid_argument("hmb::starsh_from_entries: need a square EntrySource (it is " +
                                    std::to_string(A.n_rows) + " x " + std::to_string(A.n_cols) + ")");
    if (!A.get) throw std::invalid_argument("hmb::starsh_from_entries: the EntrySource has no get function");
    detail::StarshEntries ctx(&A);
    hmb_starsh_tlr_opts o = opt;
    o.thread_safe = opt.thread_safe && A.thread_safe;
    hmb_starsh *M = nullptr;
    char msg[512] = "";
    const int rc = hmb_starsh_from_entries(A.n_rows, perm, detail::starsh_entries_cb, &ctx, &o, &M, msg, sizeof msg);
    StarshHandle h(M, hmb_starsh_free); // releases M on every path below
    if (ctx.error) {
        h.reset();
        try {
            std::rethrow_exception(ctx.error);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("hmb::starsh_from_entries: the EntrySource threw: ") + e.what());
        } catch (...) {
            throw std::runtime_error("hmb::starsh_from_entries: the EntrySource threw a non-standard exception");
        }
    }
    if (rc) detail::starsh_throw("starsh_from_entries", msg);
    return h;
}

// hmb_starsh_export of M's matrix as an aliasing shared_ptr (see "Ownership"): stored blocks
// are read in place; the mirrored dense blocks of a symmetric format are copied into the
// control block. The cluster and block trees are rebuilt from the block ranges
// (hmb_starsh.h). stats may be null.
inline std::shared_ptr<const hmb_hmatrix> starsh_export(const StarshHandle &M, hmb_starsh_export_stats *stats = nullptr) {
    if (!M) throw std::invalid_argument("hmb::starsh_export: null handle");
    struct Held {
        StarshHandle keep; // declared first, destroyed last
        hmb_cstore store;
        Held() { hmb_cstore_init(&store); }
        ~Held() { hmb_cstore_free(&store); }
        Held(const Held &) = delete;
        Held &operator=(const Held &) = delete;
    };
    auto h = std::make_shared<Held>();
    h->keep = M;
    char msg[512] = "";
    if (hmb_starsh_export(hmb_starsh_blrm(M.get()), &h->store, stats, msg, sizeof msg))
        detail::starsh_throw("starsh_export", msg);
    return std::shared_ptr<const hmb_hmatrix>(h, &h->store.A);
}

// The (square) matrix of M as a BlackBox in the ORIGINAL ordering; 'T' as described under
// "Products" above; symmetric is true for a symmetric format. The closure keeps M alive.
inline BlackBox starsh_blackbox(const StarshHandle &M) {
    if (!M) throw std::invalid_argument("hmb::starsh_blackbox: null handle");
    const hmb_hmatrix *V = hmb_starsh_view(M.get());
    if (V->n_rows != V->n_cols)
        throw std::invalid_argument("hmb::starsh_blackbox: square matrices only (the matrix is " +
                                    std::to_string(V->n_rows) + " x " + std::to_string(V->n_cols) + ")");
    const int n = V->n_rows;
    const bool sym = hmb_starsh_symm(hmb_starsh_blrm(M.get())) == 'S';
    BlackBox b;
    b.n = n;
    b.symmetric = sym;
    b.apply = [M, n, sym](char trans, int k, const double *X, double *Y) {
        const bool t = trans == 'T' || trans == 't';
        if (!t && trans != 'N' && trans != 'n')
            throw std::invalid_argument(std::string("hmb::starsh_blackbox: trans must be 'N' or 'T', not '") + trans +
                                        "'");
        if (t && !sym) { // no A^T X in STARS-H: the same leaves through the reference product
            hmb_matvec(hmb_starsh_view(M.get()), 'T', k, X, n, Y, n);
            return;
        }
        char msg[512] = "";
        if (hmb_starsh_matvec(hmb_starsh_blrm(M.get()), k, X, n, Y, n, msg, sizeof msg))
            detail::starsh_throw("starsh_blackbox", msg);
    };
    return b;
}

} // namespace hmb
