// STARS-H adapter: import of neutral views (uniform tiles, HODLR, mixed row-only/column-only
// splits; permutations, padded leading dimensions, rank-0 tiles, both vtrans values,
// rectangular) checked against hmb_matvec and the dense reference; export(import(view))
// round trip (bit-identical entries); STARS-H's own TLR approximations (symm 'N' and 'S',
// dense "false far-field" tiles, on-the-fly blocks) exported and checked against STARS-H's
// product; the get_block ordering bug; partitions the export must refuse.
#include "hmb/neutral_store.hpp"
#include "hmb_starsh.h"
#include "hmb_test.hpp"
#include "test_starsh_native.h"

#include <functional>
#include <memory>
#include <string>

using namespace hmb;
using namespace hmb::test;

namespace {

// ---- views ---------------------------------------------------------------------------
// Leaf data with padded leading dimensions; the padding holds NaN, so any read of it shows.
struct Builder {
    NeutralStore s;
    std::mt19937 g;
    explicit Builder(unsigned seed) : g(seed) {}

    int cluster(int off, int size) {
        hmb_cluster c{};
        c.offset = off;
        c.size = size;
        s.cl.push_back(c);
        return (int)s.cl.size() - 1;
    }
    double *fill(int rows, int cols, int ld) {
        double *p = s.alloc(size_t(ld) * (cols > 0 ? cols : 1));
        std::normal_distribution<double> nd;
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < ld; i++) p[i + size_t(j) * ld] = i < rows ? nd(g) : NAN;
        return p;
    }
    hmb_block dense(int rc, int cc) {
        hmb_block b{};
        const int m = s.cl[rc].size, n = s.cl[cc].size, pad = int(g() % 4);
        b.rc = rc, b.cc = cc, b.kind = HMB_DENSE, b.ldD = m + pad;
        b.D = fill(m, n, b.ldD);
        return b;
    }
    hmb_block lowrank(int rc, int cc, int rank, int vtrans) {
        hmb_block b{};
        const int m = s.cl[rc].size, n = s.cl[cc].size;
        b.rc = rc, b.cc = cc, b.kind = HMB_LOWRANK, b.rank = rank, b.vtrans = vtrans;
        b.ldU = m + int(g() % 3);
        b.U = fill(m, rank, b.ldU);
        b.ldV = (vtrans ? n : rank) + int(g() % 3) + (rank == 0 && !vtrans ? 1 : 0);
        b.V = vtrans ? fill(n, rank, b.ldV) : fill(rank, n, b.ldV);
        return b;
    }
    std::vector<int> random_perm(int n) {
        std::vector<int> p(n);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), g);
        return p;
    }
};

// m x n uniform tiles (tr rows, tc columns per tile; the last ones smaller). Diagonal tiles
// and ~15% of the others dense, the rest low rank (ranks 0..6, both vtrans).
std::unique_ptr<Builder> tlr_view(int m, int n, int tr, int tc, bool shared, bool permute, unsigned seed) {
    auto B = std::make_unique<Builder>(seed);
    const int pr = (m + tr - 1) / tr, pc = (n + tc - 1) / tc;
    shared = shared && m == n && tr == tc;
    const int rroot = B->cluster(0, m);
    B->s.cl[rroot].son0 = (int)B->s.cl.size(), B->s.cl[rroot].nsons = pr;
    for (int i = 0; i < pr; i++) B->cluster(i * tr, std::min(tr, m - i * tr));
    int croot = rroot;
    if (!shared) {
        croot = B->cluster(0, n);
        B->s.cl[croot].son0 = (int)B->s.cl.size(), B->s.cl[croot].nsons = pc;
        for (int j = 0; j < pc; j++) B->cluster(j * tc, std::min(tc, n - j * tc));
    }
    hmb_block root{};
    root.rc = rroot, root.cc = croot, root.kind = HMB_SUPER, root.rsons = pr, root.csons = pc, root.son0 = 0;
    B->s.bl.push_back(root);
    B->s.sons.resize(size_t(pr) * pc);
    for (int j = 0; j < pc; j++)
        for (int i = 0; i < pr; i++) {
            const int rc = B->s.cl[rroot].son0 + i, cc = B->s.cl[croot].son0 + j;
            const bool dense = i == j || B->g() % 100 < 15;
            B->s.sons[i + size_t(j) * pr] = (int)B->s.bl.size();
            const int rank = int(B->g() % 7);
            B->s.bl.push_back(dense ? B->dense(rc, cc) : B->lowrank(rc, cc, rank, int(B->g() % 2)));
        }
    B->s.A.n_rows = m, B->s.A.n_cols = n, B->s.A.row_root = rroot, B->s.A.col_root = croot;
    if (permute) {
        B->s.row_perm = B->random_perm(m);
        B->s.col_perm = shared ? B->s.row_perm : B->random_perm(n);
    }
    B->s.finalize();
    return B;
}

// HODLR: binary cluster tree of the given depth (uneven halves), dense diagonal leaves,
// low-rank off-diagonal blocks on every level; rows and columns share the tree.
std::unique_ptr<Builder> hodlr_view(int n, int depth, unsigned seed) {
    auto B = std::make_unique<Builder>(seed);
    // the two sons of a cluster are appended together, so their ids are consecutive
    std::function<void(int, int)> split = [&](int c, int lev) {
        if (lev >= depth) return;
        const int o = B->s.cl[c].offset, z = B->s.cl[c].size, h = z / 2 + (lev % 2);
        const int a = B->cluster(o, h);
        B->cluster(o + h, z - h);
        B->s.cl[c].son0 = a, B->s.cl[c].nsons = 2;
        split(a, lev + 1);
        split(a + 1, lev + 1);
    };
    const int root = B->cluster(0, n);
    split(root, 0);
    std::function<int(int)> blocks = [&](int c) {
        const int id = (int)B->s.bl.size();
        if (B->s.cl[c].nsons == 0) {
            B->s.bl.push_back(B->dense(c, c));
            return id;
        }
        hmb_block b{};
        b.rc = b.cc = c, b.kind = HMB_SUPER, b.rsons = b.csons = 2, b.son0 = (int)B->s.sons.size();
        B->s.bl.push_back(b);
        B->s.sons.resize(B->s.sons.size() + 4);
        const int s0 = B->s.cl[c].son0, slot = b.son0;
        const int d0 = blocks(s0);
        const int lo = (int)B->s.bl.size();
        B->s.bl.push_back(B->lowrank(s0 + 1, s0, 1 + int(B->g() % 4), 0));
        const int up = (int)B->s.bl.size();
        B->s.bl.push_back(B->lowrank(s0, s0 + 1, 1 + int(B->g() % 4), 1));
        const int d1 = blocks(s0 + 1);
        B->s.sons[slot + 0] = d0, B->s.sons[slot + 1] = lo, B->s.sons[slot + 2] = up, B->s.sons[slot + 3] = d1;
        return id;
    };
    blocks(root);
    B->s.A.n_rows = B->s.A.n_cols = n, B->s.A.row_root = B->s.A.col_root = root;
    B->s.row_perm = B->random_perm(n);
    B->s.col_perm = B->s.row_perm;
    B->s.finalize();
    return B;
}

// 30 x 25 with a root split along rows only and children split along columns only:
// rows [0,12) [12,20) [20,30) (the last two under [12,30)), columns [0,10) [10,18) [18,25).
std::unique_ptr<Builder> mixed_view(unsigned seed) {
    auto B = std::make_unique<Builder>(seed);
    const int R = B->cluster(0, 30), R0 = B->cluster(0, 12), R1 = B->cluster(12, 18);
    const int R10 = B->cluster(12, 8);
    B->cluster(20, 10);
    const int C = B->cluster(0, 25), C0 = B->cluster(0, 10);
    B->cluster(10, 8), B->cluster(18, 7);
    B->s.cl[R].son0 = R0, B->s.cl[R].nsons = 2;
    B->s.cl[R1].son0 = R10, B->s.cl[R1].nsons = 2;
    B->s.cl[C].son0 = C0, B->s.cl[C].nsons = 3;
    auto super = [&](int rc, int cc, int rs, int cs) {
        hmb_block b{};
        b.rc = rc, b.cc = cc, b.kind = HMB_SUPER, b.rsons = rs, b.csons = cs, b.son0 = (int)B->s.sons.size();
        B->s.sons.resize(B->s.sons.size() + size_t(rs) * cs, -1);
        B->s.bl.push_back(b);
        return b.son0;
    };
    auto add = [&](int slot, const hmb_block &b) {
        B->s.sons[slot] = (int)B->s.bl.size();
        B->s.bl.push_back(b);
    };
    const int top = super(R, C, 2, 1);
    B->s.sons[top] = (int)B->s.bl.size();
    const int row0 = super(R0, C, 1, 3);
    add(row0 + 0, B->dense(R0, C0));
    add(row0 + 1, B->lowrank(R0, C0 + 1, 2, 0));
    add(row0 + 2, B->lowrank(R0, C0 + 2, 0, 1));
    B->s.sons[top + 1] = (int)B->s.bl.size();
    const int row1 = super(R1, C, 2, 3);
    add(row1 + 0, B->lowrank(R10, C0, 3, 1));
    add(row1 + 1, B->dense(R10 + 1, C0));
    add(row1 + 2, B->dense(R10, C0 + 1));
    add(row1 + 3, B->lowrank(R10 + 1, C0 + 1, 2, 1));
    add(row1 + 4, B->lowrank(R10, C0 + 2, 1, 0));
    add(row1 + 5, B->dense(R10 + 1, C0 + 2));
    B->s.A.n_rows = 30, B->s.A.n_cols = 25, B->s.A.row_root = R, B->s.A.col_root = C;
    B->s.row_perm = B->random_perm(30);
    B->s.finalize();
    return B;
}

std::vector<double> dense_of(const hmb_hmatrix *A) {
    std::vector<int> r(A->n_rows), c(A->n_cols);
    std::iota(r.begin(), r.end(), 0);
    std::iota(c.begin(), c.end(), 0);
    std::vector<double> D(size_t(A->n_rows) * A->n_cols);
    hmb_entries(A, A->n_rows, r.data(), A->n_cols, c.data(), D.data(), A->n_rows);
    return D;
}

double maxabs_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

long long count_leaves(const hmb_hmatrix *A) {
    long long k = 0;
    for (int b = 0; b < A->nbl; b++)
        if (A->bl[b].kind != HMB_SUPER) k++;
    return k;
}

// STARS-H product vs hmb_matvec on the same view (ORIGINAL ordering, 3 right-hand sides)
double starsh_vs_view(struct starsh_blrm *M, const hmb_hmatrix *A, unsigned seed) {
    const int k = 3;
    std::vector<double> X = randvec(size_t(A->n_cols) * k, seed), Y1(size_t(A->n_rows) * k), Y2(Y1.size());
    hmb_matvec(A, 'N', k, X.data(), A->n_cols, Y1.data(), A->n_rows);
    char msg[256] = "";
    if (hmb_starsh_matvec(M, k, X.data(), A->n_cols, Y2.data(), A->n_rows, msg, sizeof msg)) {
        std::printf("  hmb_starsh_matvec: %s\n", msg);
        return INFINITY;
    }
    return relerr(Y2, Y1);
}

void check_import(Checker &C, const std::string &name, const hmb_hmatrix *A, unsigned seed) {
    char msg[512] = "";
    hmb_starsh *M = nullptr;
    const int rc = hmb_starsh_import(A, &M, msg, sizeof msg);
    if (!C.check(rc == 0, (name + " import").c_str(), "%s", msg)) return;
    struct starsh_blrm *bm = hmb_starsh_blrm(M);
    C.check_le(starsh_vs_view(bm, A, seed), 1e-13, (name + " STARS-H dmml vs hmb_matvec").c_str());

    std::vector<double> ref = dense_of(A), D(ref.size());
    const int dr = hmb_starsh_dense(bm, D.data(), A->n_rows, msg, sizeof msg);
    C.check(dr == 0, (name + " dense conversion").c_str(), "%s", msg);
    C.check_le(relerr(D, ref), 1e-14, (name + " STARS-H dca vs dense reference").c_str());

    C.check(ts_get_block_mismatches(bm) == 0, (name + " get_block finds every block").c_str(), "%d wrong",
            ts_get_block_mismatches(bm));
    const double kd = ts_kernel_maxdiff(bm, A, 20, seed);
    C.check(kd == 0.0, (name + " problem kernel = stored entries").c_str(), "max |diff| %.3e", kd);
    C.check_le(ts_dfe(bm), 1e-13, (name + " starsh_blrm__dfe").c_str());

    hmb_cstore S;
    hmb_cstore_init(&S);
    hmb_starsh_export_stats st;
    const int er = hmb_starsh_export(bm, &S, &st, msg, sizeof msg);
    if (C.check(er == 0, (name + " export(import(view))").c_str(), "%s", msg)) {
        const double d = maxabs_diff(dense_of(&S.A), ref);
        C.check(d == 0.0, (name + " round trip bit-identical").c_str(), "max |diff| %.3e", d);
        C.check(hmb_stored_values(&S.A) == hmb_stored_values(A) && count_leaves(&S.A) == count_leaves(A),
                (name + " round trip leaves and values").c_str(), "%lld leaves, %zu values (source %lld, %zu)",
                count_leaves(&S.A), hmb_stored_values(&S.A), count_leaves(A), hmb_stored_values(A));
        C.check(st.copied_values == 0, (name + " export is zero-copy").c_str(), "%lld values aliased, %lld copied",
                st.aliased_values, st.copied_values);
    }
    hmb_cstore_free(&S);
    hmb_starsh_free(M);
}

// ---- kernel matrices -------------------------------------------------------------------
void kernel_entries(int m, int n, const int *rows, const int *cols, double *out, int ldo, void *ctx) {
    const Problem *P = static_cast<const Problem *>(ctx);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) out[i + size_t(j) * ldo] = P->K(rows[i], cols[j]);
}

double exact_relerr(const Problem &P, const hmb_hmatrix *A, char trans) {
    const int k = 4;
    std::vector<double> X = randvec(size_t(P.n) * k, 11), Y(size_t(P.n) * k), R(Y.size());
    hmb_matvec(A, trans, k, X.data(), P.n, Y.data(), P.n);
    P.exact_apply(trans, k, X.data(), R.data());
    return relerr(Y, R);
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_starsh");
    const int n = arg_int(argc, argv, "--n", 2000), tile = arg_int(argc, argv, "--tile", 192);
    const double tol = arg_dbl(argc, argv, "--eps", 1e-6);
    char msg[512] = "";

    // ---- 1. neutral views -> STARS-H -> neutral views ------------------------------------
    {
        auto t1 = tlr_view(200, 200, 32, 32, true, true, 1);
        check_import(C, "TLR 200 shared perm", &t1->s.A, 1);
        auto t2 = tlr_view(203, 171, 29, 24, false, true, 2);
        check_import(C, "TLR 203x171 two perms", &t2->s.A, 2);
        auto t3 = tlr_view(150, 150, 40, 40, true, false, 3);
        check_import(C, "TLR 150 identity", &t3->s.A, 3);
        auto h = hodlr_view(97, 3, 4);
        check_import(C, "HODLR 97 depth 3", &h->s.A, 4);
        auto mx = mixed_view(5);
        check_import(C, "mixed row/column splits", &mx->s.A, 5);
        // an invalid view is refused with the validator's reason
        auto bad = tlr_view(64, 64, 16, 16, true, false, 6);
        for (auto &b : bad->s.bl)
            if (b.kind == HMB_DENSE) {
                b.ldD = 3;
                break;
            }
        bad->s.finalize();
        hmb_starsh *M = nullptr;
        const int rc = hmb_starsh_import(&bad->s.A, &M, msg, sizeof msg);
        C.check(rc != 0 && M == nullptr && std::strstr(msg, "ldD"), "invalid view refused", "%s", msg);
    }

    // ---- 2. STARS-H's own TLR approximation -> export ------------------------------------
    Problem P(n, Kernel::Sym, Geometry::RandomSphere);
    const std::vector<int> perm = bisection_perm(P, tile / 2);
    for (char symm : {'N', 'S'}) {
        hmb_starsh_tlr_opts o{tile, tol, tile / 2, symm, 1};
        hmb_starsh *M = nullptr;
        double t0 = now();
        const int rc = hmb_starsh_from_entries(n, perm.data(), kernel_entries, &P, &o, &M, msg, sizeof msg);
        const double tb = now() - t0;
        std::string tag = std::string("STARS-H TLR symm=") + symm;
        if (!C.check(rc == 0, (tag + " from_entries").c_str(), "%s", msg)) continue;
        struct starsh_blrm *bm = hmb_starsh_blrm(M);
        hmb_cstore S;
        hmb_cstore_init(&S);
        hmb_starsh_export_stats st;
        t0 = now();
        const int er = hmb_starsh_export(bm, &S, &st, msg, sizeof msg);
        const double te = now() - t0;
        if (C.check(er == 0, (tag + " export").c_str(), "%s", msg)) {
            std::printf("  n=%d tile=%d tol=%.0e: %lld far + %lld near blocks (%lld mirrored), max rank %d, %lld values "
                        "aliased, %lld copied; build %.2fs, export %.4fs\n",
                        n, tile, tol, st.far_blocks, st.near_blocks, st.mirrored_blocks, st.max_rank, st.aliased_values,
                        st.copied_values, tb, te);
            C.check_le(starsh_vs_view(bm, &S.A, 21), 1e-13, (tag + " hmb_matvec(export) vs dmml").c_str());
            C.check_le(exact_relerr(P, &S.A, 'N'), 10 * tol, (tag + " vs exact kernel (<= 10 eps)").c_str());
            C.check(st.near_blocks > 0, (tag + " dense (false far-field) tiles exported").c_str(), "%lld",
                    st.near_blocks);
            if (symm == 'S') {
                C.check(st.mirrored_blocks > 0, (tag + " lower triangle mirrored").c_str(),
                        "%lld mirrored, %lld values copied", st.mirrored_blocks, st.copied_values);
                std::vector<double> X = randvec(size_t(n) * 2, 5), YN(size_t(n) * 2), YT(YN.size());
                hmb_matvec(&S.A, 'N', 2, X.data(), n, YN.data(), n);
                hmb_matvec(&S.A, 'T', 2, X.data(), n, YT.data(), n);
                C.check_le(relerr(YT, YN), 1e-14, (tag + " exported view symmetric").c_str());
            }
            const double kd = ts_kernel_maxdiff(bm, &S.A, 10, 3);
            C.check(kd == 0.0, (tag + " kernel reads stored matrix after build").c_str(), "max |diff| %.3e", kd);
        }
        hmb_cstore_free(&S);
        hmb_starsh_free(M);
    }

    // ---- 3. matrices built with STARS-H's API directly -----------------------------------
    {   // on-the-fly dense blocks (onfly = 1): the export evaluates them through the kernel
        const int n2 = 600, t2 = 64;
        Problem Q(n2, Kernel::Sym, Geometry::RandomSphere, 0, 9);
        std::vector<int> p2 = bisection_perm(Q, t2 / 2);
        ts_matrix *T = ts_kernel_tlr(n2, Q.xyz.data(), p2.data(), t2, tol, 24, 'N', 1);
        if (C.check(T != nullptr && ts_onfly(ts_blrm(T)) == 1, "native onfly TLR built")) {
            hmb_cstore S;
            hmb_cstore_init(&S);
            hmb_starsh_export_stats st;
            const int er = hmb_starsh_export(ts_blrm(T), &S, &st, msg, sizeof msg);
            if (C.check(er == 0 && st.copied_values > 0, "onfly export evaluates near blocks", "%s%s%lld values copied",
                        msg, *msg ? "; " : "", st.copied_values))
                C.check_le(starsh_vs_view(ts_blrm(T), &S.A, 31), 1e-13, "onfly hmb_matvec(export) vs dmml");
            hmb_cstore_free(&S);
        }
        ts_free(T);
        // symmetric format with dense off-diagonal tiles (small maxrank): transposed copies
        T = ts_kernel_tlr(n2, Q.xyz.data(), p2.data(), t2, tol, 8, 'S', 0);
        if (C.check(T != nullptr, "native symmetric TLR built")) {
            hmb_cstore S;
            hmb_cstore_init(&S);
            hmb_starsh_export_stats st;
            const int er = hmb_starsh_export(ts_blrm(T), &S, &st, msg, sizeof msg);
            if (C.check(er == 0 && st.copied_values > 0 && st.mirrored_blocks > 0, "symmetric dense tiles mirrored",
                        "%s%s%lld mirrored, %lld values copied", msg, *msg ? "; " : "", st.mirrored_blocks,
                        st.copied_values))
                C.check_le(starsh_vs_view(ts_blrm(T), &S.A, 33), 1e-13, "symmetric hmb_matvec(export) vs dmml");
            hmb_cstore_free(&S);
        }
        ts_free(T);
    }
    {   // block list in column order: STARS-H's get_block returns other blocks' factors,
        // the export (which reads the lists directly) is unaffected (GOTCHA(starsh-get-block))
        ts_matrix *T = ts_random_tlr_colmajor(300, 64, 4, 7);
        const int bad = ts_get_block_mismatches(ts_blrm(T));
        C.check(bad > 0, "column-ordered list: get_block bug reproduced", "%d of 25 blocks wrong", bad);
        hmb_cstore S;
        hmb_cstore_init(&S);
        const int er = hmb_starsh_export(ts_blrm(T), &S, nullptr, msg, sizeof msg);
        if (C.check(er == 0, "column-ordered list export", "%s", msg))
            C.check_le(starsh_vs_view(ts_blrm(T), &S.A, 41), 1e-13, "column-ordered hmb_matvec(export) vs dmml");
        hmb_cstore_free(&S);
        ts_free(T);
    }
    {   // partitions no block tree describes are refused with a reason
        const char *want[3] = {"overlaps", "covered by no block", "without nesting"};
        for (int kind = 0; kind < 3; kind++) {
            ts_matrix *T = ts_broken(kind);
            hmb_cstore S;
            hmb_cstore_init(&S);
            const int er = hmb_starsh_export(ts_blrm(T), &S, nullptr, msg, sizeof msg);
            char what[64];
            std::snprintf(what, sizeof what, "broken partition %d refused", kind);
            C.check(er != 0 && std::strstr(msg, want[kind]), what, "%s", msg);
            hmb_cstore_free(&S);
            ts_free(T);
        }
    }
    return C.finish("n=%d tile=%d eps=%.0e", n, tile, tol);
}
