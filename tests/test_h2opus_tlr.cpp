// H2Opus TLR adapter: import of uniform tile views (padding when n is not a multiple of the
// tile size, permutations, padded leading dimensions, rank-0 tiles, both vtrans values,
// low-rank diagonal tiles, dense and high-rank off-diagonal tiles, symmetric storage)
// checked against hmb_matvec for A and A^T; export(import(view)) round trip; H2Opus's own
// construction (construct_tlr_matrix, both types) exported and checked against tlr_gemv,
// including its stale diagonal ranks; shapes the import must refuse; the exported view
// outliving the handle; an EntrySource that throws inside H2Opus's OpenMP loops.
#include "hmb/neutral_store.hpp"
#include "hmb_h2opus_tlr.hpp"
#include "hmb_test.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

using namespace hmb;
using namespace hmb::test;

namespace {

// Uniform tile views. Leaf data have padded leading dimensions filled with NaN beyond the
// valid rows, so a read of the padding shows up in the products.
struct TileView {
    NeutralStore s;
    std::mt19937 g;
    int n = 0, bs = 0, p = 0;
    TileView(int n_, int bs_, unsigned seed) : g(seed), n(n_), bs(bs_), p((n_ + bs_ - 1) / bs_) {}

    double *fill(int rows, int cols, int ld) {
        double *a = s.alloc(size_t(ld) * (cols > 0 ? cols : 1));
        std::normal_distribution<double> nd;
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < ld; i++) a[i + size_t(j) * ld] = i < rows ? nd(g) : NAN;
        return a;
    }
    int dim(int k) const { return k == p - 1 ? n - k * bs : bs; }
    hmb_block dense(int i, int j) {
        hmb_block b{};
        b.rc = 1 + i, b.cc = 1 + j, b.kind = HMB_DENSE, b.ldD = dim(i) + int(g() % 4);
        b.D = fill(dim(i), dim(j), b.ldD);
        return b;
    }
    hmb_block lowrank(int i, int j, int rank, int vtrans) {
        hmb_block b{};
        const int m = dim(i), c = dim(j);
        b.rc = 1 + i, b.cc = 1 + j, b.kind = HMB_LOWRANK, b.rank = rank, b.vtrans = vtrans;
        b.ldU = m + int(g() % 3);
        b.U = fill(m, rank, b.ldU);
        b.ldV = (vtrans ? c : std::max(rank, 1)) + int(g() % 3);
        b.V = vtrans ? fill(c, rank, b.ldV) : fill(rank, c, b.ldV);
        return b;
    }
    // tile (i, j) for every i, j given by make(i, j); clusters: root 0, tiles 1 .. p
    void build(const std::function<hmb_block(int, int)> &make, bool permute) {
        hmb_cluster root{};
        root.size = n, root.son0 = 1, root.nsons = p;
        s.cl.push_back(root);
        for (int k = 0; k < p; k++) {
            hmb_cluster c{};
            c.offset = k * bs, c.size = dim(k);
            s.cl.push_back(c);
        }
        hmb_block top{};
        top.kind = HMB_SUPER, top.rsons = top.csons = p;
        s.bl.push_back(top);
        s.sons.resize(size_t(p) * p);
        for (int j = 0; j < p; j++)
            for (int i = 0; i < p; i++) {
                s.sons[i + size_t(j) * p] = (int)s.bl.size();
                s.bl.push_back(make(i, j));
            }
        s.A.n_rows = s.A.n_cols = n;
        if (permute) {
            s.row_perm.resize(n);
            std::iota(s.row_perm.begin(), s.row_perm.end(), 0);
            std::shuffle(s.row_perm.begin(), s.row_perm.end(), g);
            s.col_perm = s.row_perm;
        }
        s.finalize();
    }
};

// general tiles: diagonal dense (one low-rank), off-diagonal low-rank ranks 0..6 with ~15%
// dense and one of rank bs + 3 (stored through its product)
std::unique_ptr<TileView> general_view(int n, int bs, bool permute, unsigned seed) {
    auto V = std::make_unique<TileView>(n, bs, seed);
    TileView &t = *V;
    t.build(
        [&](int i, int j) {
            if (i == j) return i == 1 ? t.lowrank(i, j, 5, 0) : t.dense(i, j);
            if (i == 2 && j == 0) return t.lowrank(i, j, bs + 3, 1);
            if (t.g() % 100 < 15) return t.dense(i, j);
            return t.lowrank(i, j, int(t.g() % 7), int(t.g() % 2));
        },
        permute);
    return V;
}

// symmetric tiles: (j, i) = (i, j)^T (low-rank: U and V swapped; dense: transposed copy),
// symmetric dense diagonal tiles
std::unique_ptr<TileView> symmetric_view(int n, int bs, unsigned seed) {
    auto V = std::make_unique<TileView>(n, bs, seed);
    TileView &t = *V;
    std::vector<hmb_block> lower(size_t(t.p) * t.p);
    t.build(
        [&](int i, int j) {
            hmb_block b{};
            if (i == j) {
                b = t.dense(i, j);
                double *D = const_cast<double *>(b.D);
                for (int c = 0; c < t.dim(j); c++)
                    for (int r = 0; r < c; r++) D[r + size_t(c) * b.ldD] = D[c + size_t(r) * b.ldD];
            } else if (i > j) {
                b = t.g() % 100 < 20 ? t.dense(i, j) : t.lowrank(i, j, int(t.g() % 6), 1);
                lower[i + size_t(j) * t.p] = b;
            } else {
                const hmb_block &l = lower[j + size_t(i) * t.p]; // made earlier: column i < column j
                b = l;
                b.rc = 1 + i, b.cc = 1 + j;
                if (l.kind == HMB_LOWRANK) {
                    std::swap(b.U, b.V);
                    std::swap(b.ldU, b.ldV);
                } else {
                    const int m = t.dim(i), c = t.dim(j);
                    double *T = t.s.alloc(size_t(m) * c);
                    for (int a = 0; a < m; a++)
                        for (int q = 0; q < c; q++) T[a + size_t(q) * m] = l.D[q + size_t(a) * l.ldD];
                    b.D = T, b.ldD = m;
                }
            }
            return b;
        },
        true);
    return V;
}

std::vector<double> dense_of(const hmb_hmatrix *A) {
    std::vector<int> r(A->n_rows), c(A->n_cols);
    std::iota(r.begin(), r.end(), 0);
    std::iota(c.begin(), c.end(), 0);
    std::vector<double> D(size_t(A->n_rows) * A->n_cols);
    hmb_entries(A, A->n_rows, r.data(), A->n_cols, c.data(), D.data(), A->n_rows);
    return D;
}

// relative difference of h2opus_tlr_matvec and hmb_matvec on A (3 right-hand sides)
double tlr_vs_view(const H2opusTlrHandle &h, const hmb_hmatrix *A, char trans, unsigned seed) {
    const int n = A->n_rows, k = 3;
    std::vector<double> X = randvec(size_t(n) * k, seed), Y1(size_t(n) * k), Y2(Y1.size());
    hmb_matvec(A, trans, k, X.data(), n, Y1.data(), n);
    h2opus_tlr_matvec(h, trans, k, X.data(), n, Y2.data(), n);
    return relerr(Y2, Y1);
}

void check_import(Checker &C, const std::string &name, const hmb_hmatrix *A, bool symmetric) {
    H2opusTlrHandle h;
    H2opusTlrStats st;
    try {
        h = h2opus_tlr_import(*A, H2opusTlrImportOptions{symmetric}, &st);
    } catch (const std::exception &e) {
        C.check(false, (name + " import").c_str(), "%s", e.what());
        return;
    }
    C.check(true, (name + " import").c_str(),
            "n=%d bs=%d tiles=%d padded=%d max rank %d, %lld values copied, %lld stored; %d low-rank diagonal, %d "
            "dense off-diagonal tiles",
            st.n, st.block_size, st.n_block, st.padded_n, st.max_rank, st.copied_values, st.stored_values,
            st.lowrank_diagonal_tiles, st.dense_offdiagonal_tiles);
    C.check_le(tlr_vs_view(h, A, 'N', 1), 1e-13, (name + " tlr_gemv N vs hmb_matvec").c_str());
    C.check_le(tlr_vs_view(h, A, 'T', 2), 1e-13, (name + " tlr_gemv T vs hmb_matvec").c_str());
    BlackBox bb = h2opus_tlr_blackbox(h);
    C.check(bb.symmetric == symmetric && bb.n == A->n_rows, (name + " blackbox").c_str());

    std::shared_ptr<const hmb_hmatrix> E = h2opus_tlr_export(h);
    const std::vector<double> ref = dense_of(A), got = dense_of(E.get());
    double amax = 0, dmax = 0;
    for (size_t q = 0; q < ref.size(); q++) {
        amax = std::max(amax, std::fabs(ref[q]));
        dmax = std::max(dmax, std::fabs(got[q] - ref[q]));
    }
    // exact except for low-rank diagonal tiles and high-rank tiles, which BLAS expands
    C.check_le(dmax / amax, 1e-14, (name + " export(import(view)) entries").c_str());
    C.check(E->bl[1].kind == HMB_DENSE && E->bl[1].ldD == st.block_size,
            (name + " export aliases padded tiles (ld = block size)").c_str(), "ldD %d, block size %d", E->bl[1].ldD,
            st.block_size);
}

// dense tiles with the given row and column tile sizes (separate row and column trees)
std::unique_ptr<NeutralStore> dense_tiles(const std::vector<int> &rs, const std::vector<int> &cs) {
    auto s = std::make_unique<NeutralStore>();
    auto tree = [&](const std::vector<int> &sz) {
        int total = 0;
        for (int z : sz) total += z;
        const int root = (int)s->cl.size();
        hmb_cluster r{};
        r.size = total, r.son0 = root + 1, r.nsons = (int)sz.size();
        s->cl.push_back(r);
        for (size_t k = 0, off = 0; k < sz.size(); off += sz[k++]) {
            hmb_cluster c{};
            c.offset = (int)off, c.size = sz[k];
            s->cl.push_back(c);
        }
        return root;
    };
    const int rr = tree(rs), cr = tree(cs);
    hmb_block top{};
    top.rc = rr, top.cc = cr, top.kind = HMB_SUPER, top.rsons = (int)rs.size(), top.csons = (int)cs.size();
    s->bl.push_back(top);
    for (size_t j = 0; j < cs.size(); j++)
        for (size_t i = 0; i < rs.size(); i++) {
            hmb_block d{};
            d.rc = rr + 1 + (int)i, d.cc = cr + 1 + (int)j, d.kind = HMB_DENSE, d.ldD = rs[i];
            d.D = s->alloc(size_t(rs[i]) * cs[j]);
            s->sons.push_back((int)s->bl.size());
            s->bl.push_back(d);
        }
    s->A.n_rows = s->cl[rr].size, s->A.n_cols = s->cl[cr].size, s->A.row_root = rr, s->A.col_root = cr;
    s->finalize();
    return s;
}

void check_refused(Checker &C, const char *what, const hmb_hmatrix *A, const char *word) {
    std::string msg = "(accepted)";
    try {
        h2opus_tlr_import(*A);
    } catch (const std::invalid_argument &e) {
        msg = e.what();
    }
    C.check(msg.find(word) != std::string::npos, what, "%s", msg.c_str());
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_h2opus_tlr");
    const int n = arg_int(argc, argv, "--n", 2001), bs = arg_int(argc, argv, "--bs", 256);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6);

    // ---- 1. neutral tile views -> H2Opus TLR -> neutral views -----------------------------
    auto v1 = general_view(200, 32, true, 1); // 6 x 32 + 8: padded last tile
    check_import(C, "tiles 200/32 perm", &v1->s.A, false);
    auto v2 = general_view(192, 32, false, 2); // no padding, identity ordering
    check_import(C, "tiles 192/32 identity", &v2->s.A, false);
    auto v3 = symmetric_view(150, 40, 3); // symmetric storage, padded
    check_import(C, "symmetric tiles 150/40", &v3->s.A, true);
    {
        H2opusTlrStats sn, ss;
        h2opus_tlr_import(v3->s.A, {}, &sn);
        h2opus_tlr_import(v3->s.A, H2opusTlrImportOptions{true}, &ss);
        C.check(ss.stored_values < sn.stored_values, "symmetric type stores the lower tiles only",
                "%lld values (non-symmetric %lld)", ss.stored_values, sn.stored_values);
    }

    // ---- 2. shapes H2Opus TLR cannot hold --------------------------------------------------
    {
        auto bad = general_view(100, 25, true, 4);
        bad->s.col_perm.assign(bad->s.row_perm.rbegin(), bad->s.row_perm.rend());
        bad->s.finalize();
        check_refused(C, "different row and column orderings refused", &bad->s.A, "permutations differ");
    }
    {
        auto uneven = dense_tiles({25, 23, 27, 25}, {25, 23, 27, 25});
        check_refused(C, "unequal tiles refused", &uneven->A, "uniform tile partition");
        auto rect = dense_tiles({32, 32}, {24, 24});
        check_refused(C, "rectangular view refused", &rect->A, "square");
        auto rows_only = dense_tiles({32, 32}, {64}); // block rows, one block column
        check_refused(C, "row-only split refused", &rows_only->A, "same number of tiles");
    }
    {   // HODLR-like: the diagonal tiles of a 2 x 2 split are split again
        NeutralStore s;
        auto cl = [&](int off, int size, int son0, int nsons) {
            hmb_cluster c{};
            c.offset = off, c.size = size, c.son0 = son0, c.nsons = nsons;
            s.cl.push_back(c);
        };
        cl(0, 40, 1, 2), cl(0, 20, 3, 2), cl(20, 20, 5, 2), cl(0, 10, 0, 0), cl(10, 10, 0, 0), cl(20, 10, 0, 0),
            cl(30, 10, 0, 0);
        auto dense = [&](int rc, int cc) {
            hmb_block b{};
            b.rc = rc, b.cc = cc, b.kind = HMB_DENSE, b.ldD = s.cl[rc].size;
            b.D = s.alloc(size_t(s.cl[rc].size) * s.cl[cc].size);
            return b;
        };
        auto super = [&](int rc) {
            hmb_block b{};
            b.rc = b.cc = rc, b.kind = HMB_SUPER, b.rsons = b.csons = 2, b.son0 = (int)s.sons.size();
            s.sons.resize(s.sons.size() + 4);
            return b;
        };
        s.bl.push_back(super(0));
        for (int q = 0; q < 4; q++) {
            const int i = q % 2, j = q / 2;
            s.sons[q] = (int)s.bl.size();
            if (i == j) {
                hmb_block d = super(1 + i);
                s.bl.push_back(d);
                for (int r = 0; r < 4; r++) {
                    s.sons[d.son0 + r] = (int)s.bl.size();
                    s.bl.push_back(dense(3 + 2 * i + r % 2, 3 + 2 * i + r / 2));
                }
            } else {
                s.bl.push_back(dense(1 + i, 1 + j));
            }
        }
        s.A.n_rows = s.A.n_cols = 40;
        s.finalize();
        check_refused(C, "nested (HODLR) partition refused", &s.A, "subdivided");
    }

    // ---- 3. H2Opus's own construction -> export -------------------------------------------
    Problem P(n, Kernel::Sym, Geometry::RandomSphere);
    const std::vector<int> perm = bisection_perm(P, bs / 2);
    double normF = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) normF += P.K(i, j) * P.K(i, j);
    normF = std::sqrt(normF);
    for (bool sym : {false, true}) {
        H2opusTlrStats st;
        H2opusTlrOptions o;
        o.block_size = bs, o.eps = eps, o.symmetric = sym;
        H2opusTlrHandle h = h2opus_tlr_from_entries(P.entries(), perm.data(), o, &st);
        const std::string tag = sym ? "H2Opus TLR symmetric" : "H2Opus TLR non-symmetric";
        std::printf("  %s: n=%d bs=%d tiles=%d (padded to %d) max rank %d, %lld stored values, %lld entries, %.2fs\n",
                    tag.c_str(), st.n, st.block_size, st.n_block, st.padded_n, st.max_rank, st.stored_values,
                    st.copied_values, st.seconds);
        if (sym) C.check(st.cleared_diagonal_ranks == 0, (tag + ": no diagonal ranks").c_str());
        else
            C.check(st.cleared_diagonal_ranks > 0, (tag + ": stale diagonal ranks cleared").c_str(), "%d of %d tiles",
                    st.cleared_diagonal_ranks, st.n_block);
        std::shared_ptr<const hmb_hmatrix> E = h2opus_tlr_export(h);
        C.check_le(tlr_vs_view(h, E.get(), 'N', 3), 1e-13, (tag + " hmb_matvec(export) vs tlr_gemv N").c_str());
        C.check_le(tlr_vs_view(h, E.get(), 'T', 4), 1e-13, (tag + " hmb_matvec(export) vs tlr_gemv T").c_str());
        // each tile's Frobenius error is at most eps (absolute), the matrix's n_block * eps
        const double bound = 10 * st.n_block * eps / normF;
        BlackBox bb = h2opus_tlr_blackbox(h), ex = P.exact_blackbox();
        char what[96];
        std::snprintf(what, sizeof what, "%s vs exact kernel (<= 10 n_block eps / |A|_F)", tag.c_str());
        C.check_le(blackbox_relerr(bb, ex, 'N'), bound, what);
    }

    // ---- 4. lifetimes; an EntrySource that throws -----------------------------------------
    {   // the exported view keeps the matrix alive after the last handle is gone
        auto v = general_view(200, 32, true, 5);
        const int m = 200, k = 2;
        std::vector<double> X = randvec(size_t(m) * k, 8), Y1(size_t(m) * k), Y2(Y1.size());
        std::shared_ptr<const hmb_hmatrix> E;
        std::weak_ptr<H2opusTlrMatrix> w;
        {
            H2opusTlrHandle h = h2opus_tlr_import(v->s.A);
            w = h;
            E = h2opus_tlr_export(h);
            h2opus_tlr_matvec(h, 'N', k, X.data(), m, Y1.data(), m);
        }
        const bool alive = !w.expired();
        hmb_matvec(E.get(), 'N', k, X.data(), m, Y2.data(), m); // reads the H2Opus tiles
        const double d = relerr(Y2, Y1);
        E.reset();
        C.check(alive && w.expired() && d <= 1e-13, "exported view keeps the matrix alive",
                "alive without a handle: %s, freed with the view: %s, view product vs tlr_gemv %.3e <= 1e-13",
                alive ? "yes" : "no", w.expired() ? "yes" : "no", d);
    }
    {   // the exception must not unwind through H2Opus's OpenMP loops (std::terminate)
        Problem Q(300, Kernel::Sym, Geometry::RandomSphere, 0, 4);
        EntrySource src = Q.entries();
        auto get = src.get;
        src.get = [get](int M, int N, const int *r, const int *c, double *out) {
            if (r[0] == 123 && c[0] == 45) throw std::runtime_error("no entry (123, 45)");
            get(M, N, r, c, out);
        };
        H2opusTlrOptions o;
        o.block_size = 64;
        std::string msg = "(no exception)";
        try {
            h2opus_tlr_from_entries(src, nullptr, o);
        } catch (const std::runtime_error &e) {
            msg = e.what();
        }
        C.check(msg == "hmb::h2opus_tlr_from_entries: the EntrySource threw: no entry (123, 45)",
                "throwing EntrySource -> std::runtime_error", "%s", msg.c_str());
    }
    return C.finish("n=%d bs=%d eps=%.0e", n, bs, eps);
}
