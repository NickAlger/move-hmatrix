// Core utilities (src/neutral.c, src/cstore.c, neutral_store.hpp, neutral_bridge.hpp) against
// a dense reference built independently from the same leaves.
//
// Random views cover: rows and columns sharing one cluster tree or not, row and column
// permutations that differ (or NULL = identity), rectangular matrices, 2- and 3-way splits,
// clusters with a single son and empty clusters, one-sided splits, leaves on non-leaf
// clusters, low-rank leaves with both vtrans values and rank 0, and padded leading
// dimensions whose padding holds NaN (any read of padding poisons the result).
#include "hmb/cstore.h"
#include "hmb/neutral.h"
#include "hmb/neutral_bridge.hpp"
#include "hmb/neutral_store.hpp"
#include "hmb_test.hpp"

#include <cfloat>
#include <limits>
#include <memory>

using namespace hmb;
using namespace hmb::test;

namespace {

// Lossless operations are compared with the dense reference at rounding level.
const double kRound = 512 * DBL_EPSILON; // ~1.1e-13
const double kNaN = std::numeric_limits<double>::quiet_NaN();

struct GenOpts {
    int m = 240, n = 240;
    bool shared = true;      // rows and columns share one cluster tree (m == n)
    int perms = 1;           // 0: NULL (identity), 1: one random permutation, 2: rows and columns differ
    int max_sons = 2;        // sons per non-leaf cluster: 2 .. max_sons
    int leaf = 12;           // maximal leaf cluster size
    bool one_son = false;    // sometimes give a cluster a single son with the same range
    bool empty = false;      // sometimes give a cluster an empty son
    double p_early = 0.0;    // probability of a leaf block on non-leaf clusters
    double p_onesided = 0.0; // probability of splitting only one of two splittable sides
    int pad = 0;             // leading dimension padding
    int kmax = 5;
    unsigned seed = 1;
};

// A random view and its dense m x n reference (original ordering, column-major).
struct RandomView {
    GenOpts o;
    NeutralStore S;
    std::vector<double> dense;
    std::vector<int> cover; // how many leaves cover each entry (must be 1)
    size_t stored = 0;
    long viol[2] = {0, 0}; // strictness violations, counted while generating
    std::mt19937 g;
    std::normal_distribution<double> nd;

    explicit RandomView(const GenOpts &opts) : o(opts), g(opts.seed) {
        const int m = o.m, n = o.n;
        dense.assign(size_t(m) * n, 0.0);
        cover.assign(size_t(m) * n, 0);
        int rroot = cluster_tree(m), croot = o.shared ? rroot : cluster_tree(n);
        std::vector<int> rp(m), cp(n);
        std::iota(rp.begin(), rp.end(), 0);
        std::iota(cp.begin(), cp.end(), 0);
        if (o.perms > 0) {
            std::shuffle(rp.begin(), rp.end(), g);
            if (o.perms == 1 && m == n) cp = rp;
            else std::shuffle(cp.begin(), cp.end(), g);
            S.row_perm = rp;
            S.col_perm = cp;
        }
        S.A.dim = 3;
        S.A.n_rows = m;
        S.A.n_cols = n;
        S.A.row_root = rroot;
        S.A.col_root = croot;
        block(rroot, croot, 0, rp, cp);
        S.finalize();
    }
    const hmb_hmatrix *view() const { return &S.A; }
    double u() { return std::uniform_real_distribution<double>(0, 1)(g); }
    int ri(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }

    int cluster_tree(int size) {
        int id = (int)S.cl.size();
        S.cl.push_back(hmb_cluster{0, size, 0, 0, {-1, -1, -1}, {1, 1, 1}});
        expand(id);
        return id;
    }
    // Gives cluster id its sons (consecutive ids) and recurses.
    void expand(int id) {
        const int size = S.cl[id].size, off = S.cl[id].offset;
        if (size <= o.leaf) return;
        std::vector<int> sizes;
        if (o.one_son && u() < 0.1) sizes = {size};
        else {
            int ns = std::min(ri(2, o.max_sons), size);
            int left = size;
            for (int s = 0; s < ns - 1; s++) {
                int sz = (o.empty && u() < 0.2) ? 0 : ri(1, std::max(1, left - (ns - 1 - s)));
                sizes.push_back(sz);
                left -= sz;
            }
            sizes.push_back(left);
        }
        int s0 = (int)S.cl.size(), at = off;
        for (int sz : sizes) {
            S.cl.push_back(hmb_cluster{at, sz, 0, 0, {-1, -1, -1}, {1, 1, 1}});
            at += sz;
        }
        S.cl[id].nsons = (int)sizes.size();
        S.cl[id].son0 = s0;
        for (int s = 0; s < (int)sizes.size(); s++)
            if (sizes[s] < size) expand(s0 + s);
            else if (sizes[s] > o.leaf) { // single son with the parent's range: split it
                expand_forced(s0 + s);
            }
    }
    void expand_forced(int id) {
        const int size = S.cl[id].size, off = S.cl[id].offset, h = size / 2;
        int s0 = (int)S.cl.size();
        S.cl.push_back(hmb_cluster{off, h, 0, 0, {-1, -1, -1}, {1, 1, 1}});
        S.cl.push_back(hmb_cluster{off + h, size - h, 0, 0, {-1, -1, -1}, {1, 1, 1}});
        S.cl[id].nsons = 2;
        S.cl[id].son0 = s0;
        expand(s0);
        expand(s0 + 1);
    }

    // Fills ld x cols storage: rows x cols random values, NaN in the padding rows.
    double *filled(int rows, int cols, int ld) {
        double *p = S.alloc(size_t(ld) * cols);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < ld; i++) p[i + size_t(j) * ld] = i < rows ? nd(g) : kNaN;
        return p;
    }
    void put(const std::vector<int> &rp, const std::vector<int> &cp, int ro, int co, int i, int j, double v) {
        size_t at = size_t(rp[ro + i]) + size_t(o.m) * cp[co + j];
        dense[at] = v;
        cover[at]++;
    }

    int block(int rc, int cc, int depth, const std::vector<int> &rp, const std::vector<int> &cp) {
        const int id = (int)S.bl.size();
        S.bl.push_back(hmb_block{});
        S.bl[id].rc = rc;
        S.bl[id].cc = cc;
        const hmb_cluster r = S.cl[rc], c = S.cl[cc];
        const bool rleaf = r.nsons == 0, cleaf = c.nsons == 0;
        const int m = r.size, n = c.size;
        if ((rleaf && cleaf) || (depth > 0 && u() < o.p_early)) {
            if ((rleaf && cleaf) ? u() < 0.5 : u() < 0.3) { // dense
                const int ld = std::max(1, m + o.pad);
                double *D = filled(m, n, ld);
                S.bl[id].kind = HMB_DENSE;
                S.bl[id].D = D;
                S.bl[id].ldD = ld;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++) put(rp, cp, r.offset, c.offset, i, j, D[i + size_t(j) * ld]);
                stored += size_t(m) * n;
                if (!rleaf || !cleaf) viol[0]++;
                return id;
            }
            const int k = ri(0, o.kmax), vt = ri(0, 1);
            hmb_block &b = S.bl[id];
            b.kind = HMB_LOWRANK;
            b.rank = k;
            b.vtrans = vt;
            if (k == 0 && u() < 0.5) { // rank 0 with NULL factors and ld 0
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++) put(rp, cp, r.offset, c.offset, i, j, 0.0);
                return id;
            }
            const int ldU = std::max(1, m + o.pad), ldV = std::max(1, (vt ? n : k) + o.pad);
            const double *U = filled(m, k, ldU), *V = vt ? filled(n, k, ldV) : filled(k, n, ldV);
            S.bl[id].U = U;
            S.bl[id].ldU = ldU;
            S.bl[id].V = V;
            S.bl[id].ldV = ldV;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++) {
                    double s = 0;
                    for (int l = 0; l < k; l++) s += U[i + size_t(l) * ldU] * (vt ? V[j + size_t(l) * ldV] : V[l + size_t(j) * ldV]);
                    put(rp, cp, r.offset, c.offset, i, j, s);
                }
            stored += size_t(k) * (m + n);
            return id;
        }
        bool rs = !rleaf, cs = !cleaf;
        if (rs && cs && u() < o.p_onesided) (u() < 0.5 ? rs : cs) = false;
        if ((!rleaf && !rs) || (!cleaf && !cs)) viol[1]++;
        const int rsons = rs ? r.nsons : 1, csons = cs ? c.nsons : 1, s0 = (int)S.sons.size();
        S.bl[id].kind = HMB_SUPER;
        S.bl[id].rsons = rsons;
        S.bl[id].csons = csons;
        S.bl[id].son0 = s0;
        S.sons.resize(s0 + size_t(rsons) * csons, -1);
        for (int j = 0; j < csons; j++)
            for (int i = 0; i < rsons; i++) {
                int ch = block(rs ? r.son0 + i : rc, cs ? c.son0 + j : cc, depth + 1, rp, cp);
                S.sons[s0 + i + size_t(j) * rsons] = ch;
            }
        return id;
    }
};

// Y = op(dense) X, X with leading dimension ldx; Y has the exact output length per column.
std::vector<double> dense_apply(const std::vector<double> &D, int m, int n, char t, int nrhs, const double *X, int ldx) {
    const int nout = t == 'N' ? m : n, nin = t == 'N' ? n : m;
    std::vector<double> Y(size_t(nout) * nrhs, 0.0);
    for (int c = 0; c < nrhs; c++)
        for (int o = 0; o < nout; o++) {
            long double s = 0;
            for (int i = 0; i < nin; i++) s += (long double)(t == 'N' ? D[o + size_t(m) * i] : D[i + size_t(m) * o]) * X[i + size_t(c) * ldx];
            Y[o + size_t(c) * nout] = (double)s;
        }
    return Y;
}

double max_abs(const std::vector<double> &v) {
    double a = 0;
    for (double x : v) a = std::max(a, std::fabs(x));
    return a;
}

// hmb_matvec with padded ldx/ldy; checks the result and that Y's padding is untouched.
void check_matvec(Checker &C, const char *tag, const RandomView &V, char t, int nrhs) {
    const hmb_hmatrix *A = V.view();
    const int nin = t == 'N' ? A->n_cols : A->n_rows, nout = t == 'N' ? A->n_rows : A->n_cols;
    const int ldx = nin + 2, ldy = nout + 3;
    std::vector<double> X = randvec(size_t(ldx) * nrhs, 11 + nrhs), Y(size_t(ldy) * nrhs, -7.25);
    hmb_matvec(A, t, nrhs, X.data(), ldx, Y.data(), ldy);
    std::vector<double> R = dense_apply(V.dense, A->n_rows, A->n_cols, t, nrhs, X.data(), ldx), Yc(R.size());
    bool pad_ok = true;
    for (int c = 0; c < nrhs; c++)
        for (int i = 0; i < ldy; i++) {
            if (i < nout) Yc[i + size_t(c) * nout] = Y[i + size_t(c) * ldy];
            else pad_ok = pad_ok && Y[i + size_t(c) * ldy] == -7.25;
        }
    char what[128];
    std::snprintf(what, sizeof what, "%s matvec %c nrhs=%d vs dense", tag, t, nrhs);
    C.check_le(relerr(Yc, R), kRound, what);
    std::snprintf(what, sizeof what, "%s matvec %c nrhs=%d keeps Y padding", tag, t, nrhs);
    C.check(pad_ok, what);
}

void check_entries(Checker &C, const char *tag, const RandomView &V) {
    const hmb_hmatrix *A = V.view();
    const int m = A->n_rows, n = A->n_cols;
    std::mt19937 g(5);
    // unsorted requests with duplicates
    std::vector<int> rows(37), cols(23);
    for (int &r : rows) r = std::uniform_int_distribution<int>(0, m - 1)(g);
    for (int &c : cols) c = std::uniform_int_distribution<int>(0, n - 1)(g);
    rows[5] = rows[0], rows[36] = rows[0], cols[7] = cols[3];
    const int ldo = (int)rows.size() + 4;
    std::vector<double> out(size_t(ldo) * cols.size(), -3.5);
    hmb_entries(A, (int)rows.size(), rows.data(), (int)cols.size(), cols.data(), out.data(), ldo);
    double err = 0;
    bool pad_ok = true;
    for (size_t j = 0; j < cols.size(); j++)
        for (int i = 0; i < ldo; i++) {
            double v = out[i + j * ldo];
            if (i < (int)rows.size()) err = std::max(err, std::fabs(v - V.dense[rows[i] + size_t(m) * cols[j]]));
            else pad_ok = pad_ok && v == -3.5;
        }
    const double scale = std::max(1.0, max_abs(V.dense));
    char what[128];
    std::snprintf(what, sizeof what, "%s entries (dups, unsorted) max error", tag);
    C.check_le(err / scale, kRound, what);
    std::snprintf(what, sizeof what, "%s entries keep output padding", tag);
    C.check(pad_ok, what);
    // whole matrix through precomputed inverse permutations
    std::vector<int> irow(m), icol(n), all_r(m), all_c(n);
    hmb_inverse_perms(A, irow.data(), icol.data());
    bool inv_ok = true;
    for (int k = 0; k < m; k++) inv_ok = inv_ok && irow[A->row_perm ? A->row_perm[k] : k] == k;
    for (int k = 0; k < n; k++) inv_ok = inv_ok && icol[A->col_perm ? A->col_perm[k] : k] == k;
    std::snprintf(what, sizeof what, "%s inverse permutations", tag);
    C.check(inv_ok, what);
    std::iota(all_r.begin(), all_r.end(), 0);
    std::iota(all_c.begin(), all_c.end(), 0);
    std::shuffle(all_r.begin(), all_r.end(), g);
    std::vector<double> F(size_t(m) * n);
    hmb_entries_ip(A, A->row_perm ? irow.data() : nullptr, A->col_perm ? icol.data() : nullptr, m, all_r.data(), n,
                   all_c.data(), F.data(), m);
    err = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) err = std::max(err, std::fabs(F[i + size_t(j) * m] - V.dense[all_r[i] + size_t(m) * j]));
    std::snprintf(what, sizeof what, "%s entries_ip of the whole matrix", tag);
    C.check_le(err / scale, kRound, what);
}

// Copies a view into an hmb_cstore through the C API (same leading dimensions).
const hmb_hmatrix *copy_to_cstore(hmb_cstore *s, const hmb_hmatrix *A, bool *api_ok) {
    hmb_cstore_init(s);
    bool ok = hmb_cstore_add_clusters(s, 0) == 0 && hmb_cstore_add_sons(s, 0) == 0;
    int c0 = hmb_cstore_add_clusters(s, A->ncl);
    for (int c = 0; c < A->ncl; c++) {
        ok = ok && s->cl[c0 + c].offset == 0 && s->cl[c0 + c].nsons == 0; // zero-initialised
        s->cl[c0 + c] = A->cl[c];
    }
    int s0 = hmb_cstore_add_sons(s, A->sons_len);
    for (int k = 0; k < A->sons_len; k++) {
        ok = ok && s->sons[s0 + k] == -1;
        s->sons[s0 + k] = A->sons[k];
    }
    for (int k = 0; k < A->nbl; k++) {
        hmb_block b = A->bl[k];
        const int n = A->cl[b.cc].size;
        auto dup = [&](const double *src, size_t len) {
            double *p = hmb_cstore_alloc(s, len);
            for (size_t i = 0; i < len; i++) ok = ok && p[i] == 0.0;
            std::copy(src, src + len, p);
            return p;
        };
        if (b.kind == HMB_DENSE && b.D) b.D = dup(b.D, size_t(b.ldD) * n);
        if (b.kind == HMB_LOWRANK && b.U) b.U = dup(b.U, size_t(b.ldU) * b.rank);
        if (b.kind == HMB_LOWRANK && b.V) b.V = dup(b.V, size_t(b.ldV) * (b.vtrans ? b.rank : n));
        ok = ok && hmb_cstore_add_block(s, &b) == k;
    }
    s->A = *A; // sizes, roots, dim; the array pointers are replaced by finalize
    s->A.row_perm = s->A.col_perm = nullptr;
    if (A->row_perm) {
        int *p = hmb_cstore_perm(s, 0, A->n_rows);
        for (int k = 0; k < A->n_rows; k++) ok = ok && p[k] == k;
        std::copy(A->row_perm, A->row_perm + A->n_rows, p);
    }
    if (A->col_perm) {
        int *p = hmb_cstore_perm(s, 1, A->n_cols);
        std::copy(A->col_perm, A->col_perm + A->n_cols, p);
    }
    *api_ok = ok;
    return hmb_cstore_finalize(s);
}

void check_view(Checker &C, const char *tag, const GenOpts &o) {
    RandomView V(o);
    const hmb_hmatrix *A = V.view();
    char what[128], msg[256];
    if (o.empty || o.one_son) {
        int empty = 0, single = 0;
        for (int c = 0; c < A->ncl; c++) empty += A->cl[c].size == 0, single += A->cl[c].nsons == 1;
        std::snprintf(what, sizeof what, "%s has empty clusters and single sons", tag);
        C.check((!o.empty || empty > 0) && (!o.one_son || single > 0), what, "%d, %d", empty, single);
    }
    int rc = hmb_validate(A, msg, sizeof msg);
    std::snprintf(what, sizeof what, "%s validates (%d clusters, %d blocks)", tag, A->ncl, A->nbl);
    C.check(rc == 0, what, "%s", rc ? msg : "");
    bool covered = std::all_of(V.cover.begin(), V.cover.end(), [](int c) { return c == 1; });
    std::snprintf(what, sizeof what, "%s generator tiles every entry once", tag);
    C.check(covered, what);
    std::snprintf(what, sizeof what, "%s stored values", tag);
    C.check(hmb_stored_values(A) == V.stored, what, "%zu == %zu", hmb_stored_values(A), V.stored);
    long v[2];
    hmb_strict_violations(A, v);
    std::snprintf(what, sizeof what, "%s strict violations", tag);
    C.check(v[0] == V.viol[0] && v[1] == V.viol[1], what, "(%ld, %ld) == (%ld, %ld)", v[0], v[1], V.viol[0], V.viol[1]);
    for (char t : {'N', 'T'})
        for (int nrhs : {1, 3}) check_matvec(C, tag, V, t, nrhs);
    check_entries(C, tag, V);

    hmb_cstore s;
    bool api_ok = false;
    const hmb_hmatrix *B = copy_to_cstore(&s, A, &api_ok);
    std::snprintf(what, sizeof what, "%s cstore: zeroed storage, ids, identity perms", tag);
    C.check(api_ok, what);
    rc = hmb_validate(B, msg, sizeof msg);
    std::snprintf(what, sizeof what, "%s cstore copy validates", tag);
    C.check(rc == 0, what, "%s", rc ? msg : "");
    std::vector<double> X = randvec(size_t(A->n_cols) * 2, 3), Y1(size_t(A->n_rows) * 2), Y2(Y1.size());
    hmb_matvec(A, 'N', 2, X.data(), A->n_cols, Y1.data(), A->n_rows);
    hmb_matvec(B, 'N', 2, X.data(), A->n_cols, Y2.data(), A->n_rows);
    std::snprintf(what, sizeof what, "%s cstore copy multiplies identically", tag);
    C.check(Y1 == Y2, what);
    hmb_cstore_free(&s);
    hmb_cstore_free(&s); // a freed store is empty: freeing again is harmless
}

// ---------------------------------------------------------------------------------------
// A small hand-made view for the structural checks:
//   clusters 0 = [0,8) -> 1 = [0,4), 2 = [4,8); 1 -> 3 = [0,2), 4 = [2,4); 2 -> 5 = [4,6), 6 = [6,8)
//   block 0 (0,0) super 2x2: (1,1) super 2x2 of dense leaves (blocks 1-4 below it are 5..8),
//   (2,1) low rank, vtrans 0, rank 2; (1,2) low rank, vtrans 1, rank 1; (2,2) dense on a
//   non-leaf cluster.
// ---------------------------------------------------------------------------------------
enum { B_ROOT = 0, B_11 = 1, B_21 = 2, B_12 = 3, B_22 = 4, B_33 = 5, B_43 = 6, B_34 = 7, B_44 = 8 };

struct SmallView {
    NeutralStore S;
    SmallView() {
        S.cl = {{0, 8, 2, 1, {}, {}}, {0, 4, 2, 3, {}, {}}, {4, 4, 2, 5, {}, {}}, {0, 2, 0, 0, {}, {}},
                {2, 2, 0, 0, {}, {}}, {4, 2, 0, 0, {}, {}}, {6, 2, 0, 0, {}, {}}};
        S.row_perm = {3, 6, 0, 7, 1, 4, 2, 5};
        S.col_perm = S.row_perm;
        auto blk = [](int rc, int cc, int kind) {
            hmb_block b{};
            b.rc = rc, b.cc = cc, b.kind = kind;
            return b;
        };
        S.bl.resize(9);
        S.bl[B_ROOT] = blk(0, 0, HMB_SUPER);
        S.bl[B_ROOT].rsons = S.bl[B_ROOT].csons = 2, S.bl[B_ROOT].son0 = 0;
        S.bl[B_11] = blk(1, 1, HMB_SUPER);
        S.bl[B_11].rsons = S.bl[B_11].csons = 2, S.bl[B_11].son0 = 4;
        S.bl[B_21] = blk(2, 1, HMB_LOWRANK);
        S.bl[B_12] = blk(1, 2, HMB_LOWRANK);
        S.bl[B_22] = blk(2, 2, HMB_DENSE);
        S.bl[B_33] = blk(3, 3, HMB_DENSE);
        S.bl[B_43] = blk(4, 3, HMB_DENSE);
        S.bl[B_34] = blk(3, 4, HMB_DENSE);
        S.bl[B_44] = blk(4, 4, HMB_DENSE);
        S.sons = {B_11, B_21, B_12, B_22, B_33, B_43, B_34, B_44};
        double *U = S.alloc(8), *V = S.alloc(8), *U1 = S.alloc(4), *V1 = S.alloc(4);
        for (int i = 0; i < 8; i++) U[i] = 1 + i, V[i] = 0.5 - i;
        for (int i = 0; i < 4; i++) U1[i] = 2 - i, V1[i] = 0.25 * i;
        S.bl[B_21].rank = 2, S.bl[B_21].U = U, S.bl[B_21].ldU = 4, S.bl[B_21].V = V, S.bl[B_21].ldV = 2;
        S.bl[B_12].rank = 1, S.bl[B_12].U = U1, S.bl[B_12].ldU = 4, S.bl[B_12].V = V1, S.bl[B_12].ldV = 4,
        S.bl[B_12].vtrans = 1;
        double *D = S.alloc(16);
        for (int i = 0; i < 16; i++) D[i] = i;
        S.bl[B_22].D = D, S.bl[B_22].ldD = 4;
        for (int k : {B_33, B_43, B_34, B_44}) {
            double *d = S.alloc(4);
            for (int i = 0; i < 4; i++) d[i] = k + 0.1 * i;
            S.bl[k].D = d, S.bl[k].ldD = 2;
        }
        S.A.dim = 0;
        S.A.n_rows = S.A.n_cols = 8;
        S.A.row_root = S.A.col_root = 0;
        S.finalize();
    }
};

// Applies one corruption to a fresh copy of the small view and expects hmb_validate to
// reject it with a message containing `expect`.
template <typename F>
void expect_invalid(Checker &C, const char *what, const char *expect, F corrupt) {
    SmallView V;
    corrupt(V.S);
    V.S.finalize();
    corrupt(V.S); // corruptions of the view itself (A.*) survive finalize
    char msg[256] = "";
    int rc = hmb_validate(&V.S.A, msg, sizeof msg);
    char name[96];
    std::snprintf(name, sizeof name, "validate rejects %s", what);
    C.check(rc < 0 && std::strstr(msg, expect) != nullptr, name, "rc=%d: %s", rc, msg);
}

void structural_checks(Checker &C) {
    {
        SmallView V;
        char msg[256] = "x";
        int rc = hmb_validate(&V.S.A, msg, sizeof msg);
        C.check(rc == 0 && msg[0] == 0, "small view validates", "rc=%d %s", rc, msg);
        C.check(hmb_validate(&V.S.A, nullptr, 0) == 0, "validate without a message buffer");
        long v[2];
        hmb_strict_violations(&V.S.A, v);
        C.check(v[0] == 1 && v[1] == 0, "small view: 1 dense leaf on a non-leaf cluster", "(%ld, %ld)", v[0], v[1]);
        C.check(hmb_validate(nullptr, msg, sizeof msg) < 0 && std::strstr(msg, "null view"), "validate rejects NULL", "%s", msg);
    }
    expect_invalid(C, "perm out of range", "out of range", [](NeutralStore &S) { S.row_perm[2] = 8; });
    expect_invalid(C, "perm with a repeated index", "appears twice", [](NeutralStore &S) { S.col_perm[2] = S.col_perm[5]; });
    expect_invalid(C, "root not covering the matrix", "root cluster", [](NeutralStore &S) { S.cl[0].size = 7; });
    expect_invalid(C, "sons with a gap", "starts at", [](NeutralStore &S) { S.cl[2].offset = 5, S.cl[2].size = 3; });
    expect_invalid(C, "sons not covering the parent", "sons cover", [](NeutralStore &S) { S.cl[6].size = 1; });
    expect_invalid(C, "son ids out of range", "outside [0, 7)", [](NeutralStore &S) { S.cl[1].son0 = 6; });
    expect_invalid(C, "a cluster that is its own son", "twice", [](NeutralStore &S) { S.cl[4].nsons = 1, S.cl[4].son0 = 4; });
    expect_invalid(C, "a negative cluster size", "outside", [](NeutralStore &S) {
        S.cl[5].size = -2, S.cl[6].offset = 2, S.cl[6].size = 6;
    });
    expect_invalid(C, "split count != cluster sons", "split into 3 row parts", [](NeutralStore &S) { S.bl[B_ROOT].rsons = 3; });
    expect_invalid(C, "a child on the wrong clusters", "pairs clusters", [](NeutralStore &S) { S.bl[B_21].rc = 1; });
    expect_invalid(C, "a child reached twice", "reached twice", [](NeutralStore &S) { S.sons[1] = S.sons[0]; });
    expect_invalid(C, "the root block as a child", "reached twice", [](NeutralStore &S) { S.sons[2] = 0; });
    expect_invalid(C, "a child id out of range", "is not a block id", [](NeutralStore &S) { S.sons[3] = 99; });
    expect_invalid(C, "an unreachable block", "not reachable", [](NeutralStore &S) {
        if (S.bl.size() == 9) S.bl.push_back(S.bl[B_33]);
    });
    expect_invalid(C, "zero children", "children", [](NeutralStore &S) { S.bl[B_11].csons = 0; });
    expect_invalid(C, "too few son slots", "exceed sons_len", [](NeutralStore &S) { S.A.sons_len = 6; });
    expect_invalid(C, "sons = NULL", "sons = NULL", [](NeutralStore &S) { S.A.sons = nullptr; });
    expect_invalid(C, "ldD < rows", "ldD", [](NeutralStore &S) { S.bl[B_43].ldD = 1; });
    expect_invalid(C, "ldU < rows", "ldU", [](NeutralStore &S) { S.bl[B_21].ldU = 3; });
    expect_invalid(C, "ldV < rank (vtrans 0)", "ldV = 1 < 2", [](NeutralStore &S) { S.bl[B_21].ldV = 1; });
    expect_invalid(C, "ldV < cols (vtrans 1)", "ldV = 3 < 4", [](NeutralStore &S) { S.bl[B_12].ldV = 3; });
    expect_invalid(C, "U = NULL", "NULL factor", [](NeutralStore &S) { S.bl[B_21].U = nullptr; });
    expect_invalid(C, "V = NULL", "NULL factor", [](NeutralStore &S) { S.bl[B_12].V = nullptr; });
    expect_invalid(C, "D = NULL", "D = NULL", [](NeutralStore &S) { S.bl[B_22].D = nullptr; });
    expect_invalid(C, "vtrans = 2", "vtrans", [](NeutralStore &S) { S.bl[B_21].vtrans = 2; });
    expect_invalid(C, "a negative rank", "negative rank", [](NeutralStore &S) { S.bl[B_12].rank = -1; });
    expect_invalid(C, "an unknown kind", "unknown kind", [](NeutralStore &S) { S.bl[B_44].kind = 7; });
    expect_invalid(C, "a root block on other clusters", "root block pairs", [](NeutralStore &S) { S.bl[B_ROOT].rc = 1; });
    expect_invalid(C, "a negative size", "negative size", [](NeutralStore &S) { S.A.n_rows = -1; });
    expect_invalid(C, "no clusters", "no clusters", [](NeutralStore &S) { S.A.ncl = 0; });
    expect_invalid(C, "no blocks", "no blocks", [](NeutralStore &S) { S.A.nbl = 0; });
    expect_invalid(C, "a bad root id", "not a cluster id", [](NeutralStore &S) { S.A.col_root = 42; });
    expect_invalid(C, "dim = 4", "dim = 4", [](NeutralStore &S) { S.A.dim = 4; });
    {
        // A cluster DAG (empty clusters listing the same sons twice): the original walk
        // overflowed its stack here; now it reports the repeated cluster.
        hmb_cluster cl[8] = {};
        cl[0].nsons = 7, cl[0].son0 = 1, cl[7].nsons = 7, cl[7].son0 = 1;
        hmb_block b{};
        b.kind = HMB_DENSE, b.ldD = 1;
        hmb_hmatrix A{};
        A.ncl = 8, A.cl = cl, A.nbl = 1, A.bl = &b;
        char msg[256] = "";
        int rc = hmb_validate(&A, msg, sizeof msg);
        C.check(rc < 0 && std::strstr(msg, "twice"), "validate rejects a cluster DAG (no overflow)", "%s", msg);
    }
    {
        // Rank-0 leaves may have NULL factors and zero leading dimensions.
        SmallView V;
        hmb_block &b = V.S.bl[B_21];
        b.rank = 0, b.U = b.V = nullptr, b.ldU = b.ldV = 0;
        char msg[256] = "";
        int rc = hmb_validate(&V.S.A, msg, sizeof msg);
        C.check(rc == 0, "rank-0 leaf with NULL factors validates", "%s", msg);
        std::vector<double> X(8, 1.0), Y(8);
        hmb_matvec(&V.S.A, 'T', 1, X.data(), 8, Y.data(), 8);
        C.check(std::all_of(Y.begin(), Y.end(), [](double y) { return std::isfinite(y); }), "rank-0 leaf matvec is finite");
    }
    {
        // One-sided split of a non-leaf cluster: (1,1) split by rows only.
        SmallView V;
        NeutralStore &S = V.S;
        S.bl[B_11].csons = 1;
        S.bl[B_33].cc = S.bl[B_43].cc = 1;
        S.bl[B_33].D = S.alloc(8), S.bl[B_43].D = S.alloc(8); // 2 x 4 dense blocks
        S.sons.resize(6);
        S.finalize();
        char msg[256] = "";
        int rc = hmb_validate(&S.A, msg, sizeof msg);
        C.check(rc < 0 && std::strstr(msg, "block 7 is not reachable"), "validate rejects the orphaned blocks", "%s", msg);
        S.bl.erase(S.bl.begin() + B_34, S.bl.end());
        S.finalize();
        rc = hmb_validate(&S.A, msg, sizeof msg);
        C.check(rc == 0, "one-sided split validates", "%s", msg);
        long v[2];
        hmb_strict_violations(&S.A, v);
        C.check(v[0] == 3 && v[1] == 1, "one-sided split: strict violations", "(%ld, %ld) == (3, 1)", v[0], v[1]);
    }
    {
        // A cluster with a single son: the block tree may split it (children on the son) or
        // not (children on the cluster); only the unsplit case violates strictness (in
        // variants 1 and 2 both the root and the middle block leave cluster 0 unsplit).
        //   clusters 0 = [0,4) -> 1 = [0,4) -> 2 = [0,2), 3 = [2,4)
        for (int variant = 0; variant < 3; variant++) {
            NeutralStore S;
            S.cl = {{0, 4, 1, 1, {}, {}}, {0, 4, 2, 2, {}, {}}, {0, 2, 0, 0, {}, {}}, {2, 2, 0, 0, {}, {}}};
            // variant 0: root -> (1,1); variant 1: root -> (0,1) (rows unsplit); 2: root -> (1,0)
            const int crc = variant == 1 ? 0 : 1, ccc = variant == 2 ? 0 : 1;
            hmb_block root{}, mid{};
            root.kind = HMB_SUPER, root.rsons = root.csons = 1, root.son0 = 0;
            mid.rc = crc, mid.cc = ccc, mid.kind = HMB_SUPER, mid.son0 = 1;
            mid.rsons = crc == 1 ? 2 : 1, mid.csons = ccc == 1 ? 2 : 1;
            S.bl = {root, mid};
            S.sons = {1};
            std::vector<double> D(16);
            for (int i = 0; i < 16; i++) D[i] = i + 1;
            for (int j = 0; j < mid.csons; j++)
                for (int i = 0; i < mid.rsons; i++) {
                    hmb_block l{};
                    l.kind = HMB_LOWRANK, l.rank = 1, l.vtrans = 1;
                    l.rc = mid.rsons == 2 ? 2 + i : crc, l.cc = mid.csons == 2 ? 2 + j : ccc;
                    l.U = D.data(), l.ldU = 4, l.V = D.data() + 4, l.ldV = 4;
                    S.sons.push_back((int)S.bl.size());
                    S.bl.push_back(l);
                }
            S.A.n_rows = S.A.n_cols = 4;
            S.finalize();
            char msg[256] = "", what[96];
            int rc = hmb_validate(&S.A, msg, sizeof msg);
            std::snprintf(what, sizeof what, "single-son split variant %d validates", variant);
            C.check(rc == 0, what, "%s", msg);
            long v[2];
            hmb_strict_violations(&S.A, v);
            std::snprintf(what, sizeof what, "single-son split variant %d strict violations", variant);
            C.check(v[0] == 0 && v[1] == (variant ? 2 : 0), what, "(%ld, %ld) == (0, %d)", v[0], v[1], variant ? 2 : 0);
        }
    }
    {
        // Empty and 1 x 1 matrices; nrhs = 0 leaves Y alone.
        NeutralStore S;
        S.cl = {{0, 0, 0, 0, {}, {}}};
        hmb_block b{};
        b.kind = HMB_DENSE, b.ldD = 1;
        S.bl = {b};
        S.finalize();
        char msg[256] = "";
        C.check(hmb_validate(&S.A, msg, sizeof msg) == 0, "0 x 0 matrix validates", "%s", msg);
        double x = 1, y = 5;
        hmb_matvec(&S.A, 'N', 1, &x, 1, &y, 1);
        hmb_entries(&S.A, 0, nullptr, 0, nullptr, &y, 1);
        C.check(y == 5, "0 x 0 matvec and entries are no-ops");
        NeutralStore T;
        T.cl = {{0, 1, 0, 0, {}, {}}};
        double d = 2.5;
        b.D = &d;
        T.bl = {b};
        T.A.n_rows = T.A.n_cols = 1;
        T.finalize();
        x = 3, y = 0;
        hmb_matvec(&T.A, 'T', 1, &x, 1, &y, 1);
        C.check(hmb_validate(&T.A, msg, sizeof msg) == 0 && y == 7.5, "1 x 1 matrix", "y = %g", y);
        y = 9;
        hmb_matvec(&T.A, 'N', 0, &x, 1, &y, 1);
        C.check(y == 9, "nrhs = 0 leaves Y unchanged");
    }
}

void bridge_checks(Checker &C) {
    GenOpts o;
    o.m = o.n = 160, o.perms = 2, o.max_sons = 3, o.p_early = 0.2, o.pad = 2, o.seed = 21;
    auto V = std::make_shared<RandomView>(o);
    const std::vector<double> dense = V->dense;
    const int n = o.n;
    BlackBox bb = blackbox_from_neutral(V->view(), V);
    EntrySource es = entries_from_neutral(V->view(), V);
    std::weak_ptr<RandomView> w = V;
    V.reset();
    C.check(!w.expired(), "BlackBox and EntrySource keep the store alive");
    std::vector<double> X = randvec(size_t(n) * 3, 8), Y(size_t(n) * 3);
    for (char t : {'N', 'T'}) {
        bb.apply(t, 3, X.data(), Y.data());
        char what[64];
        std::snprintf(what, sizeof what, "blackbox_from_neutral %c vs dense", t);
        C.check_le(relerr(Y, dense_apply(dense, n, n, t, 3, X.data(), n)), kRound, what);
    }
    C.check(bb.n == n && !bb.symmetric && es.n_rows == n && es.n_cols == n && es.thread_safe, "bridge metadata");
    // concurrent entry requests (thread_safe = true): every thread gets its own answer
    const int T = 4;
    std::vector<double> errs(T, 1.0);
#pragma omp parallel for num_threads(2) schedule(static, 1)
    for (int t = 0; t < T; t++) {
        std::mt19937 g(100 + t);
        std::vector<int> r(19), c(13);
        for (int &x : r) x = std::uniform_int_distribution<int>(0, n - 1)(g);
        for (int &x : c) x = std::uniform_int_distribution<int>(0, n - 1)(g);
        std::vector<double> out(r.size() * c.size());
        es.get((int)r.size(), (int)c.size(), r.data(), c.data(), out.data());
        double e = 0;
        for (size_t j = 0; j < c.size(); j++)
            for (size_t i = 0; i < r.size(); i++) e = std::max(e, std::fabs(out[i + j * r.size()] - dense[r[i] + size_t(n) * c[j]]));
        errs[t] = e;
    }
    C.check_le(*std::max_element(errs.begin(), errs.end()) / std::max(1.0, max_abs(dense)), kRound,
               "entries_from_neutral from 2 threads");
    bb = BlackBox();
    es = EntrySource();
    C.check(w.expired(), "store released with the last closure");
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_neutral");
    const int n = arg_int(argc, argv, "--n", 240);
    structural_checks(C);
    {
        GenOpts o; // square, shared tree, one permutation, binary, tight leading dimensions, strict
        o.m = o.n = n, o.seed = 1;
        check_view(C, "[strict]", o);
    }
    {
        GenOpts o; // shared clusters but different row and column permutations; 3-way splits
        o.m = o.n = n, o.perms = 2, o.max_sons = 3, o.p_early = 0.15, o.p_onesided = 0.3, o.pad = 3, o.seed = 2;
        check_view(C, "[3-way]", o);
    }
    {
        GenOpts o; // rectangular, separate trees
        o.m = n, o.n = n * 2 / 3 + 1, o.shared = false, o.perms = 2, o.max_sons = 3, o.p_early = 0.2;
        o.p_onesided = 0.4, o.pad = 2, o.kmax = 3, o.seed = 3;
        check_view(C, "[rect]", o);
    }
    {
        GenOpts o; // wide, identity permutations, single sons and empty clusters
        o.m = n / 2, o.n = n, o.shared = false, o.perms = 0, o.one_son = true, o.empty = true, o.p_early = 0.1;
        o.p_onesided = 0.2, o.pad = 1, o.seed = 4;
        check_view(C, "[1-son,empty]", o);
    }
    bridge_checks(C);
    return C.finish("n=%d", n);
}
