// Htool-DDM block copy (hmb_htool_copy.hpp): export to the neutral view and import back,
// against Htool's own products.
//
//  - Htool matrices of the kernel: full and symmetric storage, block_tree_consistency on and
//    off, leaf size 10 (ACA fallbacks leave dense blocks above the leaf level), symmetric
//    storage of the upper triangle, a two-level partition (Htool's children skip a cluster
//    level). The export validates, multiplies like Htool ('N' and 'T', 3 right-hand sides)
//    at rounding level, is zero-copy for full storage, mirrors symmetric storage (the view is
//    then symmetric), and never reads the triangle of a symmetric diagonal leaf that Htool
//    ignores (poisoned with NaN here). import(export(A)) multiplies like A and exports to the
//    same block tree. The exported view keeps the matrix alive when no other reference is
//    left, and releases it with the view.
//  - Hand-made views Htool's builder would not produce (rectangular, different row and column
//    trees and permutations, 3-way and one-sided splits, single sons, empty clusters, a leaf
//    root, rank 0, V stored cols x rank, padded leading dimensions with NaN in the padding):
//    the import multiplies like hmb_matvec on the view and exports back to the same matrix.
#include "hmb/neutral_bridge.hpp"
#include "hmb/neutral_store.hpp"
#include "htool_export.hpp"
#include "htool_import.hpp"
#include "hmb_test.hpp"

#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>

#include <cfloat>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace hmb;
using namespace hmb::test;

namespace {

const double kRound = 512 * DBL_EPSILON; // lossless transfers agree at rounding level
const double kNaN = std::numeric_limits<double>::quiet_NaN();
using HM = htool::HMatrix<double>;

// op(A) X in the user numbering for any pair of trees. Htool's user-numbering wrappers permute
// the input with the source tree and the output with the target tree whatever op is, which is
// wrong for 'T' when the trees differ (GOTCHA(htool-user-transpose)); so permute here.
void htool_apply(const HM &A, char t, int k, const double *X, double *Y) {
    const auto &in = t == 'N' ? A.get_source_cluster() : A.get_target_cluster();
    const auto &out = t == 'N' ? A.get_target_cluster() : A.get_source_cluster();
    const int nin = in.get_size(), nout = out.get_size();
    std::vector<double> Xp(size_t(nin) * k), Yp(size_t(nout) * k, 0.0);
    for (int c = 0; c < k; c++) htool::user_to_cluster(in, X + size_t(c) * nin, Xp.data() + size_t(c) * nin);
    htool::MatrixView<const double> Xv(nin, k, Xp.data());
    htool::MatrixView<double> Yv(nout, k, Yp.data());
    htool::internal_add_hmatrix_matrix_product(t, 'N', 1.0, A, Xv, 0.0, Yv);
    for (int c = 0; c < k; c++) htool::cluster_to_user(out, Yp.data() + size_t(c) * nout, Y + size_t(c) * nout);
}

// ||hmb_matvec(V) - Htool(A)|| / ||Htool(A)|| for op = t and 3 random right-hand sides.
double diff_vs_htool(const hmb_hmatrix &V, const HM &A, char t) {
    const int k = 3, nin = t == 'N' ? V.n_cols : V.n_rows, nout = t == 'N' ? V.n_rows : V.n_cols;
    std::vector<double> X = randvec(size_t(nin) * k, 17), Y1(size_t(nout) * k), Y2(Y1.size());
    hmb_matvec(&V, t, k, X.data(), nin, Y1.data(), nout);
    htool_apply(A, t, k, X.data(), Y2.data());
    return relerr(Y1, Y2);
}

// ||op(A) X - op(B) X|| / ||op(B) X|| for two neutral views of the same size.
double diff_views(const hmb_hmatrix &A, const hmb_hmatrix &B, char t, char tb = 0) {
    const int k = 3, nin = t == 'N' ? A.n_cols : A.n_rows, nout = t == 'N' ? A.n_rows : A.n_cols;
    std::vector<double> X = randvec(size_t(nin) * k, 23), Y1(size_t(nout) * k), Y2(Y1.size());
    hmb_matvec(&A, t, k, X.data(), nin, Y1.data(), nout);
    hmb_matvec(&B, tb ? tb : t, k, X.data(), nin, Y2.data(), nout);
    return relerr(Y1, Y2);
}

bool valid(const hmb_hmatrix &A, char *msg, size_t len) { return hmb_validate(&A, msg, len) == 0; }

long long expected_copies(const hmb_hmatrix &A, long long *transposed) {
    long long s = 0;
    *transposed = 0;
    for (int b = 0; b < A.nbl; b++) {
        const hmb_block &x = A.bl[b];
        const long long m = A.cl[x.rc].size, n = A.cl[x.cc].size;
        if (m == 0 || n == 0) continue; // blocks on empty clusters are left out
        if (x.kind == HMB_DENSE) s += m * n;
        if (x.kind == HMB_LOWRANK) s += x.rank * (m + n), *transposed += x.vtrans ? x.rank * n : 0;
    }
    return s;
}

// Exports A and checks the view against Htool; returns the view.
std::shared_ptr<const hmb_hmatrix> check_export(Checker &C, const char *tag, const HtoolHandle &A, const Problem *P,
                                                 double eps) {
    HtoolExportInfo xi;
    auto V = htool_export(A, &xi);
    const hmb_hmatrix &v = *V;
    char what[128], msg[256] = "";
    bool ok = valid(v, msg, sizeof msg);
    std::snprintf(what, sizeof what, "%s export validates", tag);
    C.check(ok, what, "%s", msg);
    if (!ok) return V;
    long viol[2];
    hmb_strict_violations(&v, viol);
    const long long stored = (long long)hmb_stored_values(&v);
    std::printf("  %s: %d clusters, %d blocks, strict violations (%ld, %ld); aliased %lld, copied %lld, "
                "mirrored leaves %d, added blocks %d\n",
                tag, v.ncl, v.nbl, viol[0], viol[1], xi.aliased_values, xi.copied_values, xi.mirrored_leaves,
                xi.added_blocks);
    std::snprintf(what, sizeof what, "%s aliased + copied = stored values", tag);
    C.check(xi.aliased_values + xi.copied_values == stored, what, "%lld + %lld == %lld", xi.aliased_values,
            xi.copied_values, stored);
    if (A->symmetry == 'N') {
        std::snprintf(what, sizeof what, "%s full storage exports zero-copy", tag);
        C.check(xi.copied_values == 0 && xi.mirrored_leaves == 0 && htool_stats(A).stored_values == stored, what,
                "copied %lld, Htool stores %lld", xi.copied_values, htool_stats(A).stored_values);
    } else {
        std::snprintf(what, sizeof what, "%s symmetric storage is mirrored", tag);
        C.check(xi.mirrored_leaves > 0 && stored > htool_stats(A).stored_values, what, "%d mirrored leaves, %lld > %lld",
                xi.mirrored_leaves, stored, htool_stats(A).stored_values);
        std::snprintf(what, sizeof what, "%s view is symmetric (N vs T)", tag);
        C.check_le(diff_views(v, v, 'N', 'T'), kRound, what);
    }
    for (char t : {'N', 'T'}) {
        std::snprintf(what, sizeof what, "%s view matvec %c vs Htool", tag, t);
        C.check_le(diff_vs_htool(v, *A->A, t), kRound, what);
    }
    if (P) {
        BlackBox exact = P->exact_blackbox(), bb = blackbox_from_neutral(&v, V);
        std::snprintf(what, sizeof what, "%s view matvec vs kernel (eps %.0e)", tag, eps);
        C.check_le(blackbox_relerr(bb, exact, 'N'), 10 * eps, what);
    }
    return V;
}

// import(export(A)): same products, same block tree.
void check_round_trip(Checker &C, const char *tag, const HtoolHandle &A, const hmb_hmatrix &v) {
    char what[128];
    HtoolImportInfo ii;
    HtoolHandle B = htool_import(v, {}, &ii);
    long long tr, expect = expected_copies(v, &tr);
    std::snprintf(what, sizeof what, "%s import copies every stored value", tag);
    C.check(ii.copied_values == expect && ii.transposed_values == tr, what, "%lld == %lld, transposed %lld == %lld",
            ii.copied_values, expect, ii.transposed_values, tr);
    for (char t : {'N', 'T'}) {
        const int k = 3, n = v.n_rows;
        std::vector<double> X = randvec(size_t(n) * k, 29), Y1(size_t(n) * k), Y2(Y1.size());
        htool_apply(*A->A, t, k, X.data(), Y1.data());
        htool_apply(*B->A, t, k, X.data(), Y2.data());
        std::snprintf(what, sizeof what, "%s import(export(A)) %c vs A", tag, t);
        C.check_le(relerr(Y2, Y1), kRound, what);
    }
    auto W = htool_export(B);
    long a[2], b[2];
    hmb_strict_violations(&v, a);
    hmb_strict_violations(W.get(), b);
    std::snprintf(what, sizeof what, "%s export(import(view)) has the view's tree", tag);
    C.check(W->nbl == v.nbl && W->ncl == v.ncl && a[0] == b[0] && a[1] == b[1] &&
                hmb_stored_values(W.get()) == hmb_stored_values(&v),
            what, "%d blocks, %d clusters, (%ld, %ld)", W->nbl, W->ncl, b[0], b[1]);
}

// Sets the triangle Htool never reads of every symmetric diagonal dense leaf to NaN.
int poison_unread_triangles(const HM &A) {
    int poisoned = 0;
    for (const HM *l : htool::get_leaves_from(A).first) {
        if (!l->is_dense() || l->get_symmetry() == 'N') continue;
        auto *D = const_cast<htool::Matrix<double> *>(l->get_dense_data());
        const bool lower = l->get_UPLO() != 'U';
        for (int j = 0; j < D->nb_cols(); j++)
            for (int i = 0; i < D->nb_rows(); i++)
                if (lower ? i < j : i > j) (*D)(i, j) = kNaN;
        poisoned++;
    }
    return poisoned;
}

void kernel_cases(Checker &C, int n) {
    struct Case {
        const char *tag;
        Kernel kernel;
        char symmetry;
        bool consistent;
        int leaf;
    };
    const Case cases[] = {
        {"[N]", Kernel::Sym, 'N', true, 64},           {"[N, inconsistent]", Kernel::Nonsym, 'N', false, 64},
        {"[N, leaf 10]", Kernel::Sym, 'N', true, 10},  {"[S]", Kernel::Sym, 'S', true, 64},
        {"[S, leaf 10]", Kernel::Sym, 'S', true, 10},
    };
    for (const Case &c : cases) {
        Problem P(n, c.kernel, c.kernel == Kernel::Sym ? Geometry::RandomSphere : Geometry::ShuffledFibonacci);
        HtoolOptions o;
        o.symmetry = c.symmetry;
        o.block_tree_consistency = c.consistent;
        o.leaf_size = c.leaf;
        HtoolHandle A = htool_build(P.entries(), 3, P.xyz.data(), o);
        auto V = check_export(C, c.tag, A, &P, o.epsilon);
        if (hmb_validate(V.get(), nullptr, 0) == 0) check_round_trip(C, c.tag, A, *V);
        if (c.symmetry == 'S') {
            char what[128];
            const int poisoned = poison_unread_triangles(*A->A);
            auto W = htool_export(A);
            std::snprintf(what, sizeof what, "%s unread triangles (%d leaves NaN) ignored", c.tag, poisoned);
            C.check_le(diff_vs_htool(*W, *A->A, 'N'), kRound, what);
        }
    }
}

// Htool matrices built here with Htool's own builders (shapes htool_build does not make).
void native_cases(Checker &C, int n) {
    Problem P(n, Kernel::Sym);
    EntrySource es = P.entries();
    HtoolEntryGenerator gen(es);
    htool::ClusterTreeBuilder<double> ctb;
    ctb.set_maximal_leaf_size(64);
    {
        // symmetric storage of the upper triangle
        auto tree = std::make_unique<htool::Cluster<double>>(ctb.create_cluster_tree(n, 3, P.xyz.data(), 2, 2));
        htool::HMatrixTreeBuilder<double> tb(1e-6, 10, 'S', 'U');
        auto H = std::make_unique<HM>(tb.build(gen, *tree, *tree));
        HtoolHandle A = htool_wrap(std::move(tree), nullptr, std::move(H), 'S');
        auto V = check_export(C, "[S, upper]", A, &P, 1e-6);
        check_round_trip(C, "[S, upper]", A, *V);
        const int poisoned = poison_unread_triangles(*A->A);
        auto W = htool_export(A);
        C.check_le(diff_vs_htool(*W, *A->A, 'N'), kRound, "[S, upper] unread triangles ignored");
        C.check(poisoned > 0, "[S, upper] has symmetric diagonal leaves", "%d", poisoned);
    }
    {
        // the exported view keeps the handle alive (its aliased leaves and trees) and owns the
        // mirrored copies: multiply after every other reference to the matrix is gone (a use
        // after free would show under AddressSanitizer); all is released with the view
        HtoolOptions o;
        o.symmetry = 'S';
        HtoolHandle A = htool_build(es, 3, P.xyz.data(), o);
        const std::vector<double> X = randvec(size_t(n), 47);
        std::vector<double> Y0(n), Y1(n);
        htool_apply(*A->A, 'N', 1, X.data(), Y0.data());
        std::shared_ptr<const hmb_hmatrix> V = htool_export(A);
        const std::weak_ptr<HtoolMatrix> alive = A;
        A.reset();
        hmb_matvec(V.get(), 'N', 1, X.data(), n, Y1.data(), n);
        const bool held = !alive.expired();
        V.reset();
        C.check(held && alive.expired(), "the view keeps the handle alive until released", "%s, %s",
                held ? "held" : "lost", alive.expired() ? "released" : "leaked");
        C.check_le(relerr(Y1, Y0), kRound, "[S] view after its handle was dropped vs Htool");
    }
    {
        // symmetric storage with block_tree_consistency off: Htool's builder logs an error,
        // builds anyway, and its own products then omit the mirror of blocks stored on one
        // side only (GOTCHA(htool-sym-inconsistent)); the export refuses such a matrix
        auto tree = std::make_unique<htool::Cluster<double>>(ctb.create_cluster_tree(n, 3, P.xyz.data(), 2, 2));
        htool::HMatrixTreeBuilder<double> tb(1e-6, 10, 'S', 'L');
        tb.set_block_tree_consistency(false);
        auto H = std::make_unique<HM>(tb.build(gen, *tree, *tree));
        HtoolHandle A = htool_wrap(std::move(tree), nullptr, std::move(H), 'S');
        BlackBox bb = htool_blackbox(A), exact = P.exact_blackbox();
        std::printf("  note: [S, inconsistent] Htool's own matvec vs kernel: %.2e\n", blackbox_relerr(bb, exact, 'N'));
        std::string why;
        try {
            htool_export(A);
        } catch (const std::invalid_argument &e) {
            why = e.what();
        }
        C.check(why.find("block_tree_consistency off") != std::string::npos,
                "[S, inconsistent] export refuses the broken tree", "%s", why.c_str());
    }
    {
        // a partition of 4 (two levels) with consistency off: the root's children are
        // partition clusters two levels down, so the export adds the skipped level
        auto tree = std::make_unique<htool::Cluster<double>>(ctb.create_cluster_tree(n, 3, P.xyz.data(), 2, 4));
        htool::HMatrixTreeBuilder<double> tb(1e-6, 10, 'N', 'N');
        tb.set_block_tree_consistency(false);
        auto H = std::make_unique<HM>(tb.build(gen, *tree, *tree));
        HtoolHandle A = htool_wrap(std::move(tree), nullptr, std::move(H), 'N');
        HtoolExportInfo xi;
        auto V = htool_export(A, &xi);
        C.check(xi.added_blocks > 0, "[skipped level] export adds blocks", "%d", xi.added_blocks);
        check_export(C, "[skipped level]", A, &P, 1e-6);
        check_round_trip(C, "[skipped level]", A, *V);
    }
    {
        // an MPI-local block row (target partition 0) is refused
        auto tree = std::make_unique<htool::Cluster<double>>(ctb.create_cluster_tree(n, 3, P.xyz.data(), 2, 2));
        htool::HMatrixTreeBuilder<double> tb(1e-6, 10, 'N', 'N');
        HM H = tb.build(gen, *tree, *tree, 0);
        NeutralStore S;
        std::string why;
        try {
            htool_export(H, S);
        } catch (const std::invalid_argument &e) {
            why = e.what();
        }
        C.check(why.find("MPI-local") != std::string::npos, "export refuses an MPI-local block row", "%s", why.c_str());
        NeutralStore T;
        T.cl.push_back(hmb_cluster{});
        why.clear();
        try {
            HM full = tb.build(gen, *tree, *tree);
            htool_export(full, T);
        } catch (const std::invalid_argument &e) {
            why = e.what();
        }
        C.check(why.find("must be empty") != std::string::npos, "export refuses a non-empty store", "%s", why.c_str());
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

int tree_size(const hmb_hmatrix &A, int root) {
    int count = 1;
    for (int s = 0; s < A.cl[root].nsons; s++) count += tree_size(A, A.cl[root].son0 + s);
    return count;
}

// Clusters the import keeps: rows and columns share one tree only if they share clusters
// and permutation; otherwise each side gets its own copy of the clusters.
int imported_clusters(const hmb_hmatrix &A) {
    bool same_perm = A.n_rows == A.n_cols;
    for (int k = 0; same_perm && k < A.n_rows; k++)
        same_perm = (A.row_perm ? A.row_perm[k] : k) == (A.col_perm ? A.col_perm[k] : k);
    if (A.row_root == A.col_root && same_perm) return tree_size(A, A.row_root);
    return tree_size(A, A.row_root) + tree_size(A, A.col_root);
}

void check_import(Checker &C, const char *tag, const hmb_hmatrix &v, bool has_empty) {
    char what[128], msg[256] = "";
    std::snprintf(what, sizeof what, "%s hand-made view validates", tag);
    if (!C.check(valid(v, msg, sizeof msg), what, "%s", msg)) return;
    HtoolImportInfo ii;
    HtoolHandle B = htool_import(v, {}, &ii);
    long long tr, expect = expected_copies(v, &tr);
    std::snprintf(what, sizeof what, "%s import copies every stored value", tag);
    C.check(ii.copied_values == expect && ii.transposed_values == tr, what, "%lld == %lld, transposed %lld == %lld",
            ii.copied_values, expect, ii.transposed_values, tr);
    for (char t : {'N', 'T'}) {
        std::snprintf(what, sizeof what, "%s Htool %c vs view", tag, t);
        C.check_le(diff_vs_htool(v, *B->A, t), kRound, what);
    }
    // the package's BlackBox: right for 'N' whatever the trees
    {
        BlackBox bb = htool_blackbox(B), ref = blackbox_from_neutral(&v);
        std::vector<double> X = randvec(size_t(v.n_cols) * 2, 31), Y1(size_t(v.n_rows) * 2), Y2(Y1.size());
        bb.apply('N', 2, X.data(), Y1.data());
        ref.apply('N', 2, X.data(), Y2.data());
        std::snprintf(what, sizeof what, "%s htool_blackbox N vs view", tag);
        C.check_le(relerr(Y1, Y2), kRound, what);
        if (B->col_tree && v.n_rows == v.n_cols) { // GOTCHA(htool-user-transpose), shown not checked
            std::vector<double> Z = randvec(size_t(v.n_rows) * 2, 37), W1(Z.size()), W2(Z.size());
            bb.apply('T', 2, Z.data(), W1.data());
            ref.apply('T', 2, Z.data(), W2.data());
            const double e = relerr(W1, W2);
            std::printf("  note: htool_blackbox 'T' with two different trees: relative error %.2e%s\n", e,
                        e > kRound ? " (Htool's user-numbering product permutes with the wrong trees)" : "");
        }
    }
    HtoolExportInfo xi;
    auto W = htool_export(B, &xi);
    std::snprintf(what, sizeof what, "%s export(import(view)) validates", tag);
    if (!C.check(valid(*W, msg, sizeof msg), what, "%s", msg)) return;
    std::snprintf(what, sizeof what, "%s export(import(view)) N vs view", tag);
    C.check_le(diff_views(*W, v, 'N'), kRound, what);
    std::snprintf(what, sizeof what, "%s export(import(view)) T vs view", tag);
    C.check_le(diff_views(*W, v, 'T'), kRound, what);
    long a[2], b[2];
    hmb_strict_violations(&v, a);
    hmb_strict_violations(W.get(), b);
    std::snprintf(what, sizeof what, "%s round trip keeps the block tree", tag);
    if (has_empty) // empty clusters and their blocks are left out
        C.check(W->nbl < v.nbl && W->ncl < v.ncl, what, "%d < %d blocks, %d < %d clusters", W->nbl, v.nbl,
                W->ncl, v.ncl);
    else
        C.check(W->nbl == v.nbl && W->ncl == imported_clusters(v) && a[0] == b[0] && a[1] == b[1], what,
                "%d blocks, %d clusters (expected %d, %d), (%ld, %ld)", W->nbl, W->ncl, v.nbl,
                imported_clusters(v), b[0], b[1]);
}

void hand_made_cases(Checker &C) {
    {
        GenOpts o; // one tree, two permutations, 3-way and one-sided splits, early leaves
        o.perms = 2, o.max_sons = 3, o.p_early = 0.15, o.p_onesided = 0.3, o.pad = 2, o.seed = 2;
        RandomView V(o);
        check_import(C, "[3-way, 2 perms]", V.S.A, false);
    }
    {
        GenOpts o; // rectangular, separate trees
        o.m = 280, o.n = 190, o.shared = false, o.perms = 2, o.max_sons = 3, o.p_early = 0.2, o.p_onesided = 0.4;
        o.pad = 3, o.seed = 3;
        RandomView V(o);
        check_import(C, "[rect]", V.S.A, false);
        // the column tree on its own (native API)
        auto T = htool_import_cluster_tree(V.S.A, 1);
        const auto &perm = T->get_permutation();
        C.check(T->get_size() == o.n && std::equal(perm.begin(), perm.end(), V.S.col_perm.begin()) &&
                    T->get_clusters_on_partition().size() == size_t(V.S.cl[V.S.A.col_root].nsons),
                "htool_import_cluster_tree: size, permutation, partition", "%d sons on the partition",
                (int)T->get_clusters_on_partition().size());
    }
    {
        GenOpts o; // single sons, empty clusters, identity permutations, wide
        o.m = 150, o.n = 260, o.shared = false, o.perms = 0, o.one_son = true, o.empty = true, o.p_early = 0.1;
        o.p_onesided = 0.2, o.pad = 1, o.seed = 4;
        RandomView V(o);
        int empty = 0, single = 0;
        for (const auto &c : V.S.cl) empty += c.size == 0, single += c.nsons == 1;
        C.check(empty > 0 && single > 0, "[1-son, empty] view has empty clusters and single sons", "%d, %d", empty, single);
        check_import(C, "[1-son, empty]", V.S.A, empty > 0);
    }
    {
        // a leaf root: one dense block (Htool gets a partition cluster below the root)
        NeutralStore S;
        S.cl = {hmb_cluster{0, 5, 0, 0, {}, {}}};
        S.row_perm = {4, 2, 0, 1, 3};
        S.col_perm = S.row_perm;
        hmb_block b{};
        b.kind = HMB_DENSE, b.ldD = 7;
        double *D = S.alloc(35);
        for (int i = 0; i < 35; i++) D[i] = (i % 7) < 5 ? 1.0 + i : kNaN;
        b.D = D;
        S.bl = {b};
        S.A.n_rows = S.A.n_cols = 5;
        S.finalize();
        check_import(C, "[leaf root]", S.A, false);
    }
    {
        // errors
        std::string why;
        NeutralStore S;
        S.cl = {hmb_cluster{0, 4, 0, 0, {}, {}}};
        hmb_block b{};
        b.kind = HMB_DENSE, b.ldD = 2; // ldD < rows
        double D[16] = {};
        b.D = D;
        S.bl = {b};
        S.A.n_rows = S.A.n_cols = 4;
        S.finalize();
        try {
            htool_import(S.A);
        } catch (const std::invalid_argument &e) {
            why = e.what();
        }
        C.check(why.find("hmb::htool_import: invalid view") == 0, "import refuses an invalid view", "%s", why.c_str());
        S.cl[0].size = 0, S.bl[0].ldD = 1, S.A.n_rows = S.A.n_cols = 0;
        why.clear();
        try {
            htool_import(S.A);
        } catch (const std::invalid_argument &e) {
            why = e.what();
        }
        C.check(why.find("no rows") != std::string::npos, "import refuses an empty matrix", "%s", why.c_str());
    }
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_htool_copy");
    const int n = arg_int(argc, argv, "--n", 2000);
    kernel_cases(C, n);
    native_cases(C, n);
    hand_made_cases(C);
    return C.finish("n=%d", n);
}
