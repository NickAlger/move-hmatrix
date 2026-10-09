// hmat-oss adapter, block copy through the serialization stream: export -> neutral view
// (validity, matvec and entries against hmat-oss itself, symmetric storage mirrored, upper
// storage after a transpose), import(export(A)) bit for bit, and import of hand-made views
// (permuted, rectangular, padded leading dimensions, unusual and zero ranks, both factor
// layouts, one-sided and 3-way splits, 1 x 1 splits, single-son and empty clusters, blocks
// that keep a single-son cluster and blocks that descend into the son).
#include "hmatoss_native.hpp"
#include "hmb_test.hpp"
#include "hmb/neutral_bridge.hpp"
#include "hmb/neutral_store.hpp"

#include <cstring>
#include <stdexcept>

using namespace hmb;
using namespace hmb::test;

static double maxabs(const std::vector<double> &v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}
static double maxdiff(const std::vector<double> &a, const std::vector<double> &b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
// hmb_validate's verdict: "valid" or its reason.
static std::string validity(const hmb_hmatrix *A) {
    char msg[256];
    return hmb_validate(A, msg, sizeof msg) == 0 ? std::string("valid") : std::string(msg);
}
static std::vector<int> iota_vec(int n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}
// Whole matrix (original ordering) from an EntrySource.
static std::vector<double> all_entries(const EntrySource &E) {
    std::vector<int> r = iota_vec(E.n_rows), c = iota_vec(E.n_cols);
    std::vector<double> out((size_t)E.n_rows * E.n_cols);
    E.get(E.n_rows, E.n_cols, r.data(), c.data(), out.data());
    return out;
}
// op(A) X with hmat-oss's own gemv (works for rectangular matrices too).
static std::vector<double> hmat_apply(const HmatossMatrix &M, int m, int n, char trans, int k, const std::vector<double> &X) {
    const int nout = trans == 'N' ? m : n;
    std::vector<double> x = X, y((size_t)nout * k);
    double one = 1, zero = 0;
    if (M.hi.gemv(trans, &one, M.A, x.data(), &zero, y.data(), k) != 0) throw std::runtime_error("gemv failed");
    return y;
}
static std::vector<double> view_apply(const hmb_hmatrix *A, char trans, int k, const std::vector<double> &X) {
    const int nin = trans == 'N' ? A->n_cols : A->n_rows, nout = trans == 'N' ? A->n_rows : A->n_cols;
    std::vector<double> y((size_t)nout * k);
    hmb_matvec(A, trans, k, X.data(), nin, y.data(), nout);
    return y;
}

// Structure and leaf values of two views, compared exactly.
static bool same_view(const hmb_hmatrix *a, const hmb_hmatrix *b, std::string &why) {
    auto fail = [&](const char *w) {
        why = w;
        return false;
    };
    if (a->dim != b->dim || a->n_rows != b->n_rows || a->n_cols != b->n_cols || a->row_root != b->row_root ||
        a->col_root != b->col_root || a->ncl != b->ncl || a->nbl != b->nbl || a->sons_len != b->sons_len)
        return fail("header");
    if (std::memcmp(a->row_perm, b->row_perm, sizeof(int) * a->n_rows) || std::memcmp(a->col_perm, b->col_perm, sizeof(int) * a->n_cols))
        return fail("permutations");
    if (std::memcmp(a->sons, b->sons, sizeof(int) * a->sons_len)) return fail("sons");
    for (int k = 0; k < a->ncl; k++) {
        const hmb_cluster &x = a->cl[k], &y = b->cl[k];
        if (x.offset != y.offset || x.size != y.size || x.nsons != y.nsons || x.son0 != y.son0 ||
            std::memcmp(x.bmin, y.bmin, sizeof x.bmin) || std::memcmp(x.bmax, y.bmax, sizeof x.bmax))
            return fail("clusters");
    }
    for (int k = 0; k < a->nbl; k++) {
        const hmb_block &x = a->bl[k], &y = b->bl[k];
        if (x.rc != y.rc || x.cc != y.cc || x.kind != y.kind || x.rsons != y.rsons || x.csons != y.csons || x.son0 != y.son0)
            return fail("block tree");
        const int m = a->cl[x.rc].size, n = a->cl[x.cc].size;
        if (x.kind == HMB_DENSE) {
            for (int j = 0; j < n; j++)
                if (std::memcmp(x.D + (size_t)x.ldD * j, y.D + (size_t)y.ldD * j, sizeof(double) * m)) return fail("dense leaf");
        } else if (x.kind == HMB_LOWRANK) {
            if (x.rank != y.rank || x.vtrans != y.vtrans) return fail("low-rank leaf");
            if (x.rank == 0 || m == 0 || n == 0) continue;
            for (int l = 0; l < x.rank; l++)
                if (std::memcmp(x.U + (size_t)x.ldU * l, y.U + (size_t)y.ldU * l, sizeof(double) * m)) return fail("U");
            const int vr = x.vtrans ? n : x.rank, vc = x.vtrans ? x.rank : n;
            for (int l = 0; l < vc; l++)
                if (std::memcmp(x.V + (size_t)x.ldV * l, y.V + (size_t)y.ldV * l, sizeof(double) * vr)) return fail("V");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Hand-made views
// ---------------------------------------------------------------------------------------------
struct Maker {
    NeutralStore s;
    std::mt19937 g{17};
    std::normal_distribution<double> nd;

    int cluster(int offset, int size) {
        hmb_cluster c{};
        c.offset = offset;
        c.size = size;
        s.cl.push_back(c);
        return (int)s.cl.size() - 1;
    }
    // consecutive sons partitioning the parent
    void split(int parent, std::initializer_list<int> sizes) {
        int off = s.cl[parent].offset, first = -1;
        for (int sz : sizes) {
            const int id = cluster(off, sz);
            if (first < 0) first = id;
            off += sz;
        }
        s.cl[parent].son0 = first;
        s.cl[parent].nsons = (int)sizes.size();
    }
    // ld x cols buffer: rows x cols of N(0,1) values, NaN in the padding (must never be read)
    double *data(int rows, int cols, int ld) {
        double *p = s.alloc((size_t)ld * cols);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < ld; i++) p[i + (size_t)ld * j] = i < rows ? nd(g) : NAN;
        return p;
    }
    int size(int c) const { return s.cl[c].size; }
    int add(hmb_block b) {
        s.bl.push_back(b);
        return (int)s.bl.size() - 1;
    }
    int dense(int rc, int cc, int pad) {
        hmb_block b{};
        b.kind = HMB_DENSE;
        b.rc = rc;
        b.cc = cc;
        b.ldD = std::max(1, size(rc) + pad);
        b.D = size(rc) > 0 && size(cc) > 0 ? data(size(rc), size(cc), b.ldD) : nullptr;
        return add(b);
    }
    int lowrank(int rc, int cc, int rank, int vtrans, int pad) {
        hmb_block b{};
        b.kind = HMB_LOWRANK;
        b.rc = rc;
        b.cc = cc;
        b.rank = rank;
        b.vtrans = vtrans;
        const int m = size(rc), n = size(cc);
        b.ldU = std::max(1, m + pad);
        b.ldV = std::max(1, (vtrans ? n : rank) + pad);
        if (rank > 0) {
            b.U = data(m, rank, b.ldU);
            b.V = vtrans ? data(n, rank, b.ldV) : data(rank, n, b.ldV);
        }
        return add(b);
    }
    // SUPER block; children are filled in later with child()
    int super(int rc, int cc, int rsons, int csons) {
        hmb_block b{};
        b.kind = HMB_SUPER;
        b.rc = rc;
        b.cc = cc;
        b.rsons = rsons;
        b.csons = csons;
        b.son0 = (int)s.sons.size();
        s.sons.resize(s.sons.size() + (size_t)rsons * csons, -1);
        return add(b);
    }
    void child(int parent, int i, int j, int id) { s.sons[s.bl[parent].son0 + i + j * s.bl[parent].rsons] = id; }
    void perms(int nr, int nc, bool same) {
        s.row_perm = iota_vec(nr);
        std::shuffle(s.row_perm.begin(), s.row_perm.end(), g);
        if (same) {
            s.col_perm = s.row_perm;
        } else {
            s.col_perm = iota_vec(nc);
            std::shuffle(s.col_perm.begin(), s.col_perm.end(), g);
        }
    }
};

// 37 x 29, separate row and column trees.
static void make_rectangular(Maker &v) {
    const int r0 = v.cluster(0, 37);
    v.split(r0, {10, 15, 12}); // 1, 2, 3: a 3-way split
    v.split(2, {8, 7});        // 4, 5
    const int c0 = v.cluster(0, 29);
    v.split(c0, {13, 16}); // 7, 8
    v.split(7, {13});      // 9: a single son
    v.split(8, {16, 0});   // 10, 11: an empty son
    v.split(9, {6, 7});    // 12, 13
    v.perms(37, 29, false);
    hmb_hmatrix &A = v.s.A;
    A.n_rows = 37;
    A.n_cols = 29;
    A.row_root = r0;
    A.col_root = c0;
    const int root = v.super(r0, c0, 3, 2);
    const int b00 = v.super(1, 7, 1, 1); // descends into the single son 9 of cluster 7 ...
    v.child(root, 0, 0, b00);
    const int b00c = v.super(1, 9, 1, 2); // ... and splits it
    v.child(b00, 0, 0, b00c);
    v.child(b00c, 0, 0, v.dense(1, 12, 2)); // padded
    v.child(b00c, 0, 1, v.lowrank(1, 13, 2, 0, 1));
    const int b10 = v.super(2, 7, 2, 1); // one-sided split; keeps cluster 7 (single son)
    v.child(root, 1, 0, b10);
    v.child(b10, 0, 0, v.lowrank(4, 7, 9, 0, 2)); // rank 9 > min(8, 13), V rank x cols, padded
    v.child(b10, 1, 0, v.lowrank(5, 7, 0, 0, 0)); // rank 0
    v.child(root, 2, 0, v.lowrank(3, 7, 1, 1, 3));
    const int b01 = v.super(1, 8, 1, 1); // 1 x 1 split
    v.child(root, 0, 1, b01);
    v.child(b01, 0, 0, v.dense(1, 8, 0));
    const int b11 = v.super(2, 8, 2, 2);
    v.child(root, 1, 1, b11);
    v.child(b11, 0, 0, v.dense(4, 10, 0));
    v.child(b11, 1, 0, v.lowrank(5, 10, 3, 0, 0));
    v.child(b11, 0, 1, v.dense(4, 11, 0));       // 8 x 0
    v.child(b11, 1, 1, v.lowrank(5, 11, 2, 1, 0)); // 7 x 0
    v.child(root, 2, 1, v.lowrank(3, 8, 2, 1, 1));
    v.s.finalize();
}

// 64 x 64, one tree for rows and columns.
static void make_square(Maker &v) {
    const int c0 = v.cluster(0, 64);
    v.split(c0, {20, 25, 19}); // 1, 2, 3
    v.split(2, {10, 15});      // 4, 5
    v.perms(64, 64, true);
    hmb_hmatrix &A = v.s.A;
    A.n_rows = A.n_cols = 64;
    A.row_root = A.col_root = c0;
    const int root = v.super(c0, c0, 3, 3);
    v.child(root, 0, 0, v.dense(1, 1, 1));
    const int d = v.super(2, 2, 2, 2);
    v.child(root, 1, 1, d);
    v.child(d, 0, 0, v.dense(4, 4, 0));
    v.child(d, 1, 0, v.lowrank(5, 4, 2, 0, 1));
    v.child(d, 0, 1, v.lowrank(4, 5, 3, 1, 0));
    v.child(d, 1, 1, v.dense(5, 5, 3));
    v.child(root, 2, 2, v.dense(3, 3, 0));
    const int b01 = v.super(1, 2, 1, 2); // split on the columns only
    v.child(root, 0, 1, b01);
    v.child(b01, 0, 0, v.lowrank(1, 4, 4, 1, 2));
    v.child(b01, 0, 1, v.dense(1, 5, 0));
    v.child(root, 1, 0, v.lowrank(2, 1, 25, 0, 0)); // rank 25 > min(25, 20)
    v.child(root, 0, 2, v.lowrank(1, 3, 0, 1, 0));
    v.child(root, 2, 0, v.lowrank(3, 1, 5, 1, 4));
    v.child(root, 1, 2, v.lowrank(2, 3, 6, 0, 2));
    v.child(root, 2, 1, v.dense(3, 2, 2));
    v.s.finalize();
}

static void check_imported_view(Checker &C, const char *name, const hmb_hmatrix *V) {
    char what[96];
    std::snprintf(what, sizeof what, "%s: hand-made view valid", name);
    const std::string v0 = validity(V);
    C.check(v0 == "valid", what, "%s", v0.c_str());
    HmatossHandle H = hmatoss_import(*V);
    const int m = V->n_rows, n = V->n_cols;
    const std::vector<double> ref = all_entries(entries_from_neutral(V)), got = all_entries(hmatoss_entries(H));
    const double amax = maxabs(ref);
    std::snprintf(what, sizeof what, "%s: entries = view", name);
    C.check_le(maxdiff(got, ref) / amax, 1e-14, what);
    for (char t : {'N', 'T'}) {
        const int k = 3;
        const std::vector<double> X = randvec((size_t)(t == 'N' ? n : m) * k, 5);
        std::snprintf(what, sizeof what, "%s: hmat-oss gemv '%c' = view", name, t);
        C.check_le(relerr(hmat_apply(*H, m, n, t, k, X), view_apply(V, t, k, X)), 1e-14, what);
    }
    HmatossExportStats xs;
    std::shared_ptr<const hmb_hmatrix> back = hmatoss_export(H, &xs);
    std::snprintf(what, sizeof what, "%s: export of the import = view", name);
    const std::string v1 = validity(back.get());
    const double d1 = maxdiff(all_entries(entries_from_neutral(back.get())), ref) / amax;
    C.check(v1 == "valid" && d1 <= 1e-14, what, "%s, %.1e <= 1e-14; %lld values, %d blocks (view: %zu, %d)", v1.c_str(), d1,
            xs.stream_values, back->nbl, hmb_stored_values(V), V->nbl);
}

int main(int argc, char **argv) {
    Checker C("test_hmatoss_copy");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double eps = 1e-6, round = 1e-13; // round: rounding level, relative

    // ---------------------------------------------------------------------------------------
    // Export, full storage (non-symmetric kernel: a transposed factor would show)
    // ---------------------------------------------------------------------------------------
    Problem P(n, Kernel::Nonsym, Geometry::ShuffledFibonacci);
    HmatossOptions opt;
    opt.epsilon = eps;
    HmatossStats st;
    HmatossHandle S = hmatoss_build(P.entries(), 3, P.xyz.data(), opt, &st);
    HmatossExportStats xs;
    std::shared_ptr<const hmb_hmatrix> X = hmatoss_export(S, &xs);
    const hmb_hmatrix *V = X.get();
    std::printf("export n=%d: %.3fs, %lld values from the stream (hmat-oss stores %lld), %d clusters, %d blocks\n", n, xs.seconds,
                xs.stream_values, st.stored_values, V->ncl, V->nbl);
    const std::string vv = validity(V);
    C.check(vv == "valid", "export: view valid", "%s", vv.c_str());
    C.check(xs.stream_values == st.stored_values && (long long)hmb_stored_values(V) == st.stored_values && xs.mirrored_leaves == 0,
            "export: every stored value copied once", "%lld = %lld", xs.stream_values, st.stored_values);
    C.check(V->row_root == V->col_root && V->dim == 3, "export: shared tree, dimension 3");
    {
        bool vt = true, boxes = true;
        for (int b = 0; b < V->nbl; b++)
            if (V->bl[b].kind == HMB_LOWRANK && V->bl[b].rank > 0) vt = vt && V->bl[b].vtrans == 1;
        for (int c = 0; c < V->ncl; c++)
            for (int k = V->cl[c].offset; k < V->cl[c].offset + V->cl[c].size; k++)
                for (int d = 0; d < 3; d++) {
                    const double x = P.xyz[3 * (size_t)V->row_perm[k] + d];
                    boxes = boxes && x >= V->cl[c].bmin[d] && x <= V->cl[c].bmax[d];
                }
        C.check(vt, "export: low-rank leaves A*B^T as vtrans = 1");
        C.check(boxes, "export: bounding boxes contain their points");
    }
    BlackBox bbS = hmatoss_blackbox(S);
    EntrySource eS = hmatoss_entries(S);
    for (char t : {'N', 'T'}) {
        const int k = 3;
        std::vector<double> x = randvec((size_t)n * k, 3), yh((size_t)n * k);
        bbS.apply(t, k, x.data(), yh.data());
        C.check_le(relerr(view_apply(V, t, k, x), yh), round, t == 'N' ? "export: hmb_matvec 'N' = hmat-oss gemv" : "export: hmb_matvec 'T' = hmat-oss gemv");
    }
    const std::vector<double> dS = all_entries(eS);
    C.check_le(maxdiff(all_entries(entries_from_neutral(V)), dS) / maxabs(dS), round, "export: hmb_entries = get_values");

    // import(export(A)): the same matrix, bit for bit
    {
        HmatossImportOptions io;
        io.epsilon = eps;
        io.dim = 3;
        io.row_coords = P.xyz.data();
        HmatossHandle B = hmatoss_import(*V, io);
        HmatossStats sb = hmatoss_stats(B);
        C.check(sb.stored_values == st.stored_values && sb.n_dense_leaves == st.n_dense_leaves && sb.n_lowrank_leaves == st.n_lowrank_leaves,
                "round trip: same leaves", "%lld values, %lld + %lld leaves", sb.stored_values, sb.n_dense_leaves, sb.n_lowrank_leaves);
        BlackBox bbB = hmatoss_blackbox(B);
        bool same_mv = true;
        for (char t : {'N', 'T'}) {
            const int k = 4;
            std::vector<double> x = randvec((size_t)n * k, 9), y1((size_t)n * k), y2((size_t)n * k);
            bbS.apply(t, k, x.data(), y1.data());
            bbB.apply(t, k, x.data(), y2.data());
            same_mv = same_mv && std::memcmp(y1.data(), y2.data(), sizeof(double) * y1.size()) == 0;
        }
        C.check(same_mv, "round trip: gemv 'N' and 'T' bitwise identical");
        const std::vector<double> dB = all_entries(hmatoss_entries(B));
        C.check(std::memcmp(dB.data(), dS.data(), sizeof(double) * dS.size()) == 0, "round trip: all entries bitwise identical");
        std::string why;
        const bool same = same_view(hmatoss_export(B).get(), V, why);
        C.check(same, "round trip: export(import(view)) == view", "%s", same ? "identical" : why.c_str());
    }

    // ---------------------------------------------------------------------------------------
    // Export, lower-symmetric storage: the view must be the full matrix (mirrored)
    // ---------------------------------------------------------------------------------------
    {
        Problem Q(n, Kernel::Sym, Geometry::RandomSphere);
        HmatossOptions o = opt;
        o.lower_symmetric = true;
        HmatossStats sl;
        HmatossHandle L = hmatoss_build(Q.entries(), 3, Q.xyz.data(), o, &sl);
        HmatossExportStats xl;
        std::shared_ptr<const hmb_hmatrix> XL = hmatoss_export(L, &xl);
        const hmb_hmatrix *VL = XL.get();
        std::printf("export lower-symmetric: %.3fs, %lld stored values, %lld mirrored leaves, %lld dense values transposed, view holds "
                    "%zu\n",
                    xl.seconds, xl.stream_values, xl.mirrored_leaves, xl.transposed_values, hmb_stored_values(VL));
        const std::string vl = validity(VL);
        C.check(vl == "valid" && xl.storage == 'L', "symmetric export: view valid", "%s, storage %c", vl.c_str(), xl.storage);
        C.check(xl.stream_values == sl.stored_values && xl.mirrored_leaves > 0 && xl.transposed_values > 0 &&
                    (long long)hmb_stored_values(VL) > xl.stream_values,
                "symmetric export: unstored triangle mirrored");
        BlackBox bbL = hmatoss_blackbox(L);
        for (char t : {'N', 'T'}) {
            const int k = 3;
            std::vector<double> x = randvec((size_t)n * k, 4), yh((size_t)n * k);
            bbL.apply(t, k, x.data(), yh.data());
            C.check_le(relerr(view_apply(VL, t, k, x), yh), round,
                       t == 'N' ? "symmetric export: hmb_matvec 'N' = gemv" : "symmetric export: hmb_matvec 'T' = gemv");
        }
        const std::vector<double> dL = all_entries(hmatoss_entries(L)), dV = all_entries(entries_from_neutral(VL));
        C.check_le(maxdiff(dV, dL) / maxabs(dL), round, "symmetric export: hmb_entries = adapter's");
        // import gives full storage: same entries, same products up to rounding
        HmatossHandle B = hmatoss_import(*VL);
        C.check(hmatoss_stats(B).storage == 'N', "symmetric round trip: full storage");
        const std::vector<double> dB = all_entries(hmatoss_entries(B));
        C.check(std::memcmp(dB.data(), dL.data(), sizeof(double) * dL.size()) == 0, "symmetric round trip: entries bitwise");
        C.check_le(blackbox_relerr(hmatoss_blackbox(B), bbL, 'N', 3), round, "symmetric round trip: gemv 'N'");

        // upper-symmetric storage: hmat-oss's transpose of a lower-stored matrix
        hmat_matrix_t *T = L->hi.copy(L->A);
        L->hi.transpose(T);
        HmatossHandle U = hmatoss_wrap(L->hi, T, true); // shares L's cluster tree: declared after L
        HmatossExportStats xu;
        std::shared_ptr<const hmb_hmatrix> XU = hmatoss_export(U, &xu);
        const std::string vu = validity(XU.get());
        C.check(hmatoss_stats(U).storage == 'U' && xu.storage == 'U' && vu == "valid", "upper storage: export valid", "%s, storage %c",
                vu.c_str(), xu.storage);
        const std::vector<double> dU = all_entries(hmatoss_entries(U)), dXU = all_entries(entries_from_neutral(XU.get()));
        C.check(std::memcmp(dU.data(), dL.data(), sizeof(double) * dL.size()) == 0, "upper storage: entries = lower's (A = A^T)");
        C.check_le(maxdiff(dXU, dL) / maxabs(dL), round, "upper storage: exported entries");
    }

    // ---------------------------------------------------------------------------------------
    // Import of hand-made views
    // ---------------------------------------------------------------------------------------
    {
        Maker r;
        make_rectangular(r);
        check_imported_view(C, "rectangular", &r.s.A);
        Maker q;
        make_square(q);
        check_imported_view(C, "square", &q.s.A);
        HmatossHandle H = hmatoss_import(q.s.A);
        C.check_le(blackbox_relerr(hmatoss_blackbox(H), blackbox_from_neutral(&q.s.A), 'T', 2), 1e-14, "square: blackbox 'T' = view");

        // a block split into 2 row parts on a cluster with 3 sons: not all pairings
        Maker bad;
        make_square(bad);
        bad.s.bl[0].rsons = 2;
        bad.s.finalize();
        std::string what;
        try {
            hmatoss_import(bad.s.A);
        } catch (const std::invalid_argument &e) {
            what = e.what();
        }
        C.check(what.rfind("hmb::hmatoss_import: invalid view:", 0) == 0, "import: invalid view refused", "%s", what.c_str());
    }
    return C.finish("n=%d eps=%g", n, eps);
}
