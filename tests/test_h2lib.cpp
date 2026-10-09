// H2Lib adapter (hmb_h2lib.h / hmb_h2lib.hpp): import in each mode, export, strict
// refinement, H -> H2 conversion, products in the original ordering.
//
//  - H-matrices built by H2Lib itself (tests/test_h2lib_native.c: adaptive and regular
//    clustering, H2Lib's strict and non-strict block trees, ACA): the export is zero-copy,
//    validates, multiplies like H2Lib ('N' and 'T') and keeps H2Lib's strictness counts;
//    regular clustering makes clusters with a single son. Re-imports multiply like the
//    original; a borrowing import of an exported view copies nothing.
//  - The conversion refuses the non-strict tree; after a strict import it succeeds, and the H2
//    matrix is within 10 eps of the H-matrix (products and H2Lib's spectral estimate).
//  - Hand-made views (rectangular, different row and column trees and permutations, 3-way and
//    one-sided splits, leaves above the leaf level, single sons, empty clusters, rank 0, V in
//    both layouts, padded leading dimensions with NaN in the padding): every mode multiplies
//    like hmb_matvec on the view, strict modes leave no violation, borrowing aliases what the
//    layouts allow, the conversion refuses exactly the non-strict trees, export(import(view))
//    is the view's matrix and tree.
//  - The C++ handles (hmb_h2lib.hpp) keep what they use alive until the last one is released.
#include "hmb/neutral_bridge.hpp"
#include "hmb/neutral_store.hpp"
#include "hmb_h2lib.hpp"
#include "hmb_test.hpp"

#include <cfloat>
#include <limits>
#include <memory>
#include <omp.h>
#include <stdexcept>

extern "C" {
void *test_h2lib_kernel_new(int n, const double *xyz, int leaf, double eta, double eps, int regular, int strict);
struct _hmatrix *test_h2lib_kernel_hmatrix(void *K);
void test_h2lib_kernel_free(void *K);
void test_h2lib_aliasing(const struct _hmatrix *A, const hmb_hmatrix *V, long *same, long *other);
long test_h2lib_single_sons(const struct _hmatrix *A);
}

using namespace hmb;
using namespace hmb::test;

namespace {

const double kRound = 512 * DBL_EPSILON; // lossless transfers agree at rounding level
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const char *kModeName[4] = {"copy", "borrow", "strict", "strict+borrow"};

// ||H2Lib(A) - hmb_matvec(V)|| / ||hmb_matvec(V)|| for op = t, 2 right-hand sides, padded ld.
double diff_vs_view(const struct _hmatrix *A, const hmb_hmatrix &V, char t) {
    const int k = 2, nin = t == 'N' ? V.n_cols : V.n_rows, nout = t == 'N' ? V.n_rows : V.n_cols;
    const int ldx = nin + 3, ldy = nout + 1;
    std::vector<double> X = randvec(size_t(ldx) * k, 41), Y1(size_t(ldy) * k), Y2(Y1.size());
    hmb_h2lib_matvec(A, t, k, X.data(), ldx, Y1.data(), ldy);
    hmb_matvec(&V, t, k, X.data(), ldx, Y2.data(), ldy);
    for (int c = 0; c < k; c++) Y1[nout + size_t(c) * ldy] = Y2[nout + size_t(c) * ldy] = 0; // padding
    return relerr(Y1, Y2);
}

double diff_h_h2(const struct _hmatrix *A, const struct _h2matrix *B, char t) {
    int m, n;
    hmb_h2lib_size(A, &m, &n);
    const int k = 2, nin = t == 'N' ? n : m, nout = t == 'N' ? m : n;
    std::vector<double> X = randvec(size_t(nin) * k, 43), Y1(size_t(nout) * k), Y2(Y1.size());
    hmb_h2lib_matvec(A, t, k, X.data(), nin, Y1.data(), nout);
    hmb_h2lib_h2_matvec(B, t, k, X.data(), nin, Y2.data(), nout);
    return relerr(Y2, Y1);
}

// Leaf values of V outside empty blocks (an empty block holds nothing and is imported as a
// rank-0 leaf); with colfactor_only, the V^T of vtrans = 0 leaves (what BORROW copies).
long long transferred(const hmb_hmatrix &V, bool colfactor_only) {
    long long s = 0;
    for (int b = 0; b < V.nbl; b++) {
        const hmb_block &x = V.bl[b];
        const long long m = V.cl[x.rc].size, n = V.cl[x.cc].size;
        if (m == 0 || n == 0) continue;
        if (x.kind == HMB_LOWRANK && (!colfactor_only || !x.vtrans)) s += x.rank * (colfactor_only ? n : m + n);
        if (x.kind == HMB_DENSE && !colfactor_only) s += m * n;
    }
    return s;
}

// Imports V in `mode` and checks the products, the counts and (strict modes) strictness.
H2libHandle check_mode(Checker &C, const char *tag, const hmb_hmatrix &V, int mode) {
    char what[160];
    hmb_h2lib_import_info info;
    H2libHandle M = h2lib_import(V, mode, &info);
    const struct _hmatrix *A = hmb_h2lib_hmatrix(M.get());
    const char *mn = kModeName[mode];
    for (char t : {'N', 'T'}) {
        std::snprintf(what, sizeof what, "%s %s: H2Lib %c vs view", tag, mn, t);
        C.check_le(diff_vs_view(A, V, t), kRound, what);
    }
    const long long stored = transferred(V, false);
    long v[2];
    hmb_h2lib_strict_violations(A, v);
    if (mode & HMB_H2LIB_STRICT) {
        std::snprintf(what, sizeof what, "%s %s: no strict violation", tag, mn);
        C.check(v[0] == 0 && v[1] == 0, what, "(%ld, %ld)", v[0], v[1]);
        std::snprintf(what, sizeof what, "%s %s: all values transferred", tag, mn);
        C.check((long long)(info.copied + info.borrowed) >= stored, what, "%zu + %zu >= %lld", info.copied, info.borrowed,
                stored);
    } else {
        long w[2];
        hmb_strict_violations(&V, w);
        std::snprintf(what, sizeof what, "%s %s: H2Lib's tree is the view's", tag, mn);
        C.check(v[0] == w[0] && v[1] == w[1], what, "(%ld, %ld) == (%ld, %ld)", v[0], v[1], w[0], w[1]);
        const long long copies = mode & HMB_H2LIB_BORROW ? transferred(V, true) : stored;
        std::snprintf(what, sizeof what, "%s %s: copied %s", tag, mn, mode ? "only V^T" : "everything");
        C.check((long long)info.copied == copies && (long long)(info.copied + info.borrowed) == stored, what,
                "copied %zu (expected %lld), borrowed %zu", info.copied, copies, info.borrowed);
        long same, other;
        test_h2lib_aliasing(A, &V, &same, &other);
        if (mode & HMB_H2LIB_BORROW) {
            std::snprintf(what, sizeof what, "%s %s: leaves alias the view", tag, mn);
            C.check(other == 0, what, "%ld aliased, %ld not", same, other);
        }
    }
    return M;
}

// H -> H2 of an imported matrix: refused for a non-strict tree, else within 10 eps.
void check_h2(Checker &C, const char *tag, const H2libHandle &M, bool strict, double eps) {
    char what[160], msg[512] = "";
    struct _h2matrix *H2 = nullptr;
    hmb_h2lib_h2_info info{};
    const int rc = hmb_h2lib_to_h2(hmb_h2lib_hmatrix(M.get()), eps, &H2, &info, msg, sizeof msg);
    if (!strict) {
        std::snprintf(what, sizeof what, "%s: H -> H2 refuses the non-strict tree", tag);
        C.check(rc == HMB_H2LIB_ENOTSTRICT && H2 == nullptr && std::strstr(msg, "HMB_H2LIB_STRICT"), what, "%s", msg);
        return;
    }
    std::snprintf(what, sizeof what, "%s: H -> H2 succeeds", tag);
    if (!C.check(rc == 0 && H2, what, "%s", msg)) return;
    const struct _hmatrix *A = hmb_h2lib_hmatrix(M.get());
    for (char t : {'N', 'T'}) {
        std::snprintf(what, sizeof what, "%s: H2 %c vs H (eps %.0e)", tag, t, eps);
        C.check_le(diff_h_h2(A, H2, t), 10 * eps, what);
    }
    std::snprintf(what, sizeof what, "%s: spectral ||H - H2|| / ||H||", tag);
    C.check_le(hmb_h2lib_h2_reldiff(A, H2), 10 * eps, what);
    std::printf("  %s: H2 %zu bytes, rank sums %u/%u, max ranks %u/%u, %.3f s\n", tag, info.bytes, info.rank_sum_row,
                info.rank_sum_col, info.rank_max_row, info.rank_max_col,
                info.rowbasis_s + info.colbasis_s + info.projection_s);
    hmb_h2lib_free_h2(H2);
}

// ---------------------------------------------------------------------------------------
// H2Lib's own H-matrices
// ---------------------------------------------------------------------------------------
void native_cases(Checker &C, int n) {
    const double eps = 1e-6;
    Problem sphere(n, Kernel::Sym), plane(n, Kernel::Sym);
    for (int i = 0; i < n; i++) plane.xyz[3 * size_t(i) + 2] = 0.0; // the unit disk
    struct Case {
        const char *tag;
        int regular, strict;
        const Problem *P;
    };
    for (Case c : {Case{"[H2Lib adaptive]", 0, 1, &sphere}, Case{"[H2Lib regular]", 1, 1, &sphere},
                   Case{"[H2Lib regular, planar]", 1, 1, &plane}, Case{"[H2Lib non-strict]", 0, 0, &sphere}}) {
        const Problem &P = *c.P;
        void *K = test_h2lib_kernel_new(n, P.xyz.data(), 32, 2.0, eps, c.regular, c.strict);
        const struct _hmatrix *A = test_h2lib_kernel_hmatrix(K);
        char what[160], msg[512] = "";
        long hv[2];
        hmb_h2lib_strict_violations(A, hv);
        std::snprintf(what, sizeof what, "%s H2Lib's strictness", c.tag);
        C.check(c.strict ? hv[0] + hv[1] == 0 : hv[0] > 0, what, "(%ld, %ld)", hv[0], hv[1]);

        hmb_cstore s;
        hmb_cstore_init(&s);
        int rc = hmb_h2lib_export(A, &s, msg, sizeof msg);
        std::snprintf(what, sizeof what, "%s export", c.tag);
        if (!C.check(rc == 0, what, "%s", msg)) {
            test_h2lib_kernel_free(K);
            continue;
        }
        const hmb_hmatrix &V = s.A;
        long same, other, vv[2];
        test_h2lib_aliasing(A, &V, &same, &other);
        std::snprintf(what, sizeof what, "%s export is zero-copy", c.tag);
        C.check(other == 0, what, "%ld leaves aliased, %ld not", same, other);
        hmb_strict_violations(&V, vv);
        std::snprintf(what, sizeof what, "%s view keeps H2Lib's strictness", c.tag);
        C.check(vv[0] == hv[0] && vv[1] == hv[1], what, "(%ld, %ld)", vv[0], vv[1]);
        for (char t : {'N', 'T'}) {
            std::snprintf(what, sizeof what, "%s view %c vs H2Lib", c.tag, t);
            C.check_le(diff_vs_view(A, V, t), kRound, what);
        }
        BlackBox bb = blackbox_from_neutral(&V), exact = P.exact_blackbox();
        std::snprintf(what, sizeof what, "%s view vs kernel (ACA eps %.0e)", c.tag, eps);
        C.check_le(blackbox_relerr(bb, exact, 'N'), 10 * eps, what);
        std::printf("  %s: %d clusters (%ld with a single son), %d blocks, strict violations (%ld, %ld)\n", c.tag, V.ncl,
                    test_h2lib_single_sons(A), V.nbl, hv[0], hv[1]);
        if (c.P == &plane) // a flat coordinate gives H2Lib's regular clustering single sons
            C.check(test_h2lib_single_sons(A) > 0, "[H2Lib regular, planar] has single sons", "%ld",
                    test_h2lib_single_sons(A));

        check_h2(C, c.tag, h2lib_import(V, HMB_H2LIB_COPY), c.strict, eps);
        for (int mode : {HMB_H2LIB_COPY, HMB_H2LIB_BORROW, HMB_H2LIB_STRICT}) {
            char tag[96];
            std::snprintf(tag, sizeof tag, "%s re-import", c.tag);
            H2libHandle M = check_mode(C, tag, V, mode);
            if (mode == HMB_H2LIB_STRICT && !c.strict) {
                std::snprintf(tag, sizeof tag, "%s strict re-import", c.tag);
                check_h2(C, tag, M, true, eps);
            }
        }
        hmb_cstore_free(&s);
        test_h2lib_kernel_free(K);
    }
}

// ---------------------------------------------------------------------------------------
// Hand-made views (a compact version of the generator of test_neutral.cpp)
// ---------------------------------------------------------------------------------------
struct GenOpts {
    int m = 300, n = 300;
    bool shared = true; // rows and columns share one cluster tree (m == n)
    int perms = 1;      // 0: NULL, 1: one random permutation, 2: rows and columns differ
    int max_sons = 2, leaf = 16, pad = 0, kmax = 4;
    bool one_son = false, empty = false;
    double p_early = 0, p_onesided = 0;
    unsigned seed = 1;
};

struct RandomView {
    GenOpts o;
    NeutralStore S;
    std::mt19937 g;
    std::normal_distribution<double> nd;
    explicit RandomView(const GenOpts &opts) : o(opts), g(opts.seed) {
        int rroot = tree(o.m), croot = o.shared ? rroot : tree(o.n);
        if (o.perms > 0) {
            S.row_perm.resize(o.m);
            S.col_perm.resize(o.n);
            std::iota(S.row_perm.begin(), S.row_perm.end(), 0);
            std::iota(S.col_perm.begin(), S.col_perm.end(), 0);
            std::shuffle(S.row_perm.begin(), S.row_perm.end(), g);
            if (o.perms == 1 && o.m == o.n) S.col_perm = S.row_perm;
            else std::shuffle(S.col_perm.begin(), S.col_perm.end(), g);
        }
        S.A.dim = 3, S.A.n_rows = o.m, S.A.n_cols = o.n, S.A.row_root = rroot, S.A.col_root = croot;
        block(rroot, croot, 0);
        S.finalize();
    }
    double u() { return std::uniform_real_distribution<double>(0, 1)(g); }
    int ri(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
    int add(int off, int size) {
        double lo = 0.01 * off, hi = lo + 0.01 * size + 0.5;
        S.cl.push_back(hmb_cluster{off, size, 0, 0, {lo, -hi, 0.1 * lo}, {hi, -lo, 0.1 * hi}});
        return (int)S.cl.size() - 1;
    }
    int tree(int size) {
        int id = add(0, size);
        expand(id);
        return id;
    }
    void expand(int id) {
        const int size = S.cl[id].size, off = S.cl[id].offset;
        if (size <= o.leaf) return;
        std::vector<int> sizes;
        if (o.one_son && u() < 0.1) sizes = {size};
        else {
            int ns = std::min(ri(2, o.max_sons), size), left = size;
            for (int s = 0; s < ns - 1; s++) {
                int sz = (o.empty && u() < 0.2) ? 0 : ri(1, std::max(1, left - (ns - 1 - s)));
                sizes.push_back(sz);
                left -= sz;
            }
            sizes.push_back(left);
        }
        int s0 = (int)S.cl.size(), at = off;
        for (int sz : sizes) add(at, sz), at += sz;
        S.cl[id].nsons = (int)sizes.size(), S.cl[id].son0 = s0;
        for (int s = 0; s < (int)sizes.size(); s++) {
            if (sizes[s] < size) expand(s0 + s);
            else { // a son with the parent's range: split it in two
                const int sid = s0 + s, h = size / 2, c0 = (int)S.cl.size();
                add(off, h), add(off + h, size - h);
                S.cl[sid].nsons = 2, S.cl[sid].son0 = c0;
                expand(c0), expand(c0 + 1);
            }
        }
    }
    double *filled(int rows, int cols, int ld) {
        double *p = S.alloc(size_t(ld) * cols);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < ld; i++) p[i + size_t(j) * ld] = i < rows ? nd(g) : kNaN;
        return p;
    }
    int block(int rc, int cc, int depth) {
        const int id = (int)S.bl.size();
        S.bl.push_back(hmb_block{});
        S.bl[id].rc = rc, S.bl[id].cc = cc;
        const hmb_cluster r = S.cl[rc], c = S.cl[cc];
        const bool rleaf = r.nsons == 0, cleaf = c.nsons == 0;
        const int m = r.size, n = c.size;
        if ((rleaf && cleaf) || (depth > 0 && u() < o.p_early)) {
            if (u() < 0.4) {
                const int ld = std::max(1, m + o.pad);
                S.bl[id].kind = HMB_DENSE, S.bl[id].ldD = ld;
                S.bl[id].D = filled(m, n, ld);
                return id;
            }
            const int k = ri(0, o.kmax), vt = ri(0, 1), ldU = std::max(1, m + o.pad), ldV = std::max(1, (vt ? n : k) + o.pad);
            S.bl[id].kind = HMB_LOWRANK, S.bl[id].rank = k, S.bl[id].vtrans = vt;
            S.bl[id].ldU = ldU, S.bl[id].ldV = ldV;
            S.bl[id].U = filled(m, k, ldU);
            S.bl[id].V = vt ? filled(n, k, ldV) : filled(k, n, ldV);
            return id;
        }
        bool rs = !rleaf, cs = !cleaf;
        if (rs && cs && u() < o.p_onesided) (u() < 0.5 ? rs : cs) = false;
        const int rsons = rs ? r.nsons : 1, csons = cs ? c.nsons : 1, s0 = (int)S.sons.size();
        S.bl[id].kind = HMB_SUPER, S.bl[id].rsons = rsons, S.bl[id].csons = csons, S.bl[id].son0 = s0;
        S.sons.resize(s0 + size_t(rsons) * csons, -1);
        for (int j = 0; j < csons; j++)
            for (int i = 0; i < rsons; i++) {
                int ch = block(rs ? r.son0 + i : rc, cs ? c.son0 + j : cc, depth + 1);
                S.sons[s0 + i + size_t(j) * rsons] = ch;
            }
        return id;
    }
};

void hand_made_case(Checker &C, const char *tag, const GenOpts &o) {
    RandomView R(o);
    const hmb_hmatrix &V = R.S.A;
    char what[160], msg[256] = "";
    std::snprintf(what, sizeof what, "%s view validates", tag);
    if (!C.check(hmb_validate(&V, msg, sizeof msg) == 0, what, "%s", msg)) return;
    long vv[2];
    hmb_strict_violations(&V, vv);
    std::printf("  %s: %d clusters, %d blocks, strict violations (%ld, %ld)\n", tag, V.ncl, V.nbl, vv[0], vv[1]);
    const double eps = 1e-6;
    for (int mode = 0; mode < 4; mode++) {
        H2libHandle M = check_mode(C, tag, V, mode);
        char t2[96];
        std::snprintf(t2, sizeof t2, "%s %s", tag, kModeName[mode]);
        if (mode == HMB_H2LIB_COPY) {
            check_h2(C, t2, M, vv[0] + vv[1] == 0, eps);
            auto W = h2lib_export(M); // export(import(view))
            std::snprintf(what, sizeof what, "%s export(import(view)) validates", tag);
            if (!C.check(hmb_validate(W.get(), msg, sizeof msg) == 0, what, "%s", msg)) continue;
            long same, other, wv[2];
            test_h2lib_aliasing(hmb_h2lib_hmatrix(M.get()), W.get(), &same, &other);
            hmb_strict_violations(W.get(), wv);
            std::snprintf(what, sizeof what, "%s export(import(view)): zero-copy, same tree", tag);
            C.check(other == 0 && W->nbl == V.nbl && wv[0] == vv[0] && wv[1] == vv[1], what,
                    "%ld not aliased, %d blocks, (%ld, %ld)", other, W->nbl, wv[0], wv[1]);
            std::snprintf(what, sizeof what, "%s export(import(view)) N vs view", tag);
            C.check_le(diff_vs_view(hmb_h2lib_hmatrix(M.get()), *W, 'N'), kRound, what);
        }
        if (mode == HMB_H2LIB_STRICT) check_h2(C, t2, M, true, eps);
    }
}

void hand_made_cases(Checker &C) {
    {
        GenOpts o; // one tree, two permutations, 3-way and one-sided splits, early leaves
        o.perms = 2, o.max_sons = 3, o.p_early = 0.15, o.p_onesided = 0.3, o.pad = 2, o.seed = 12;
        hand_made_case(C, "[3-way, 2 perms]", o);
    }
    {
        GenOpts o; // rectangular, separate trees
        o.m = 260, o.n = 170, o.shared = false, o.perms = 2, o.max_sons = 3, o.p_early = 0.2, o.p_onesided = 0.4;
        o.pad = 3, o.seed = 13;
        hand_made_case(C, "[rect]", o);
    }
    {
        GenOpts o; // single sons, empty clusters, identity permutations
        o.m = 200, o.n = 240, o.shared = false, o.perms = 0, o.one_son = true, o.empty = true, o.p_early = 0.1;
        o.p_onesided = 0.2, o.pad = 1, o.seed = 14;
        hand_made_case(C, "[1-son, empty]", o);
    }
    {
        GenOpts o; // strict: no early leaf, no one-sided split of two non-leaf clusters
        o.perms = 1, o.max_sons = 3, o.pad = 1, o.seed = 15;
        hand_made_case(C, "[strict view]", o);
    }
}

// The C++ handles keep what they use alive: a borrowing import its keepalive (the view's
// storage), an exported view and an H2 matrix the H2libHandle they come from. Products after
// every other reference is gone (a use after free would show under AddressSanitizer), then
// everything is released with the last handle.
void handle_lifetimes(Checker &C) {
    GenOpts o; // strict tree, so that H -> H2 applies
    o.perms = 2, o.pad = 2, o.seed = 21;
    auto R = std::make_shared<RandomView>(o);
    const std::weak_ptr<RandomView> view_alive = R;
    const int n = R->S.A.n_rows;
    std::vector<double> X = randvec(size_t(n), 51), Y0(n), Y1(n), Y2(n);
    hmb_matvec(&R->S.A, 'N', 1, X.data(), n, Y0.data(), n);
    H2libHandle M = h2lib_import(R->S.A, HMB_H2LIB_BORROW, nullptr, R);
    const std::weak_ptr<hmb_h2lib_hm> import_alive = M;
    R.reset();
    std::shared_ptr<const hmb_hmatrix> W = h2lib_export(M);
    H2libH2Handle H = h2lib_to_h2(M, 1e-6);
    M.reset();
    hmb_matvec(W.get(), 'N', 1, X.data(), n, Y1.data(), n);
    hmb_h2lib_h2_matvec(H.get(), 'N', 1, X.data(), n, Y2.data(), n);
    const bool held = !view_alive.expired() && !import_alive.expired();
    W.reset();
    const bool held_by_h2 = !view_alive.expired() && !import_alive.expired();
    H.reset();
    const bool released = view_alive.expired() && import_alive.expired(); // weak_ptrs still exist
    C.check(held && held_by_h2 && released, "handles keep their sources alive until the last is released",
            "%s, %s, %s", held ? "held" : "lost", held_by_h2 ? "held by H2" : "lost", released ? "released" : "leaked");
    C.check_le(relerr(Y1, Y0), kRound, "exported view after its handle was dropped");
    C.check_le(relerr(Y2, Y0), 10 * 1e-6, "H2 matrix after its H handle was dropped (eps 1e-06)");
}

void error_cases(Checker &C) {
    char msg[512] = "";
    hmb_h2lib_hm *M = nullptr;
    NeutralStore S;
    S.cl = {hmb_cluster{0, 4, 0, 0, {}, {}}};
    hmb_block b{};
    b.kind = HMB_DENSE, b.ldD = 4;
    double D[16] = {};
    b.D = D;
    S.bl = {b};
    S.A.n_rows = S.A.n_cols = 4;
    S.finalize();
    int rc = hmb_h2lib_import(&S.A, 8, &M, nullptr, msg, sizeof msg);
    C.check(rc == HMB_H2LIB_EINVAL && M == nullptr && std::strstr(msg, "unknown mode"), "import refuses mode 8", "%s", msg);
    S.bl[0].ldD = 3;
    S.finalize();
    rc = hmb_h2lib_import(&S.A, HMB_H2LIB_COPY, &M, nullptr, msg, sizeof msg);
    C.check(rc == HMB_H2LIB_EINVAL && std::strstr(msg, "invalid view: dense block 0: ldD = 3"), "import refuses an invalid view",
            "%s", msg);
    std::string why;
    try {
        h2lib_import(S.A);
    } catch (const std::invalid_argument &e) {
        why = e.what();
    }
    C.check(why.rfind("hmb::h2lib_import: invalid view", 0) == 0, "h2lib_import throws", "%s", why.c_str());
    S.bl[0].ldD = 4;
    S.finalize();
    H2libHandle A = h2lib_import(S.A);
    struct _h2matrix *H2 = nullptr;
    rc = hmb_h2lib_to_h2(hmb_h2lib_hmatrix(A.get()), 0.0, &H2, nullptr, msg, sizeof msg);
    C.check(rc == HMB_H2LIB_EINVAL && H2 == nullptr, "H -> H2 refuses eps = 0", "%s", msg);
    hmb_cstore s;
    hmb_cstore_init(&s);
    hmb_cstore_add_clusters(&s, 1);
    rc = hmb_h2lib_export(hmb_h2lib_hmatrix(A.get()), &s, msg, sizeof msg);
    C.check(rc == HMB_H2LIB_EINVAL && std::strstr(msg, "must be empty"), "export refuses a non-empty store", "%s", msg);
    hmb_cstore_free(&s);
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_h2lib");
    const int n = arg_int(argc, argv, "--n", 2000);
    const int levels = omp_get_max_active_levels();
    C.check(hmb_h2lib_init(0) == 0 && omp_get_max_active_levels() == levels, "init keeps the OpenMP nesting", "%d levels",
            omp_get_max_active_levels());
    native_cases(C, n);
    hand_made_cases(C);
    handle_lifetimes(C);
    error_cases(C);
    return C.finish("n=%d", n);
}
