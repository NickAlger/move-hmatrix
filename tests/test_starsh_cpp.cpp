// STARS-H adapter, C++ interface (hmb_starsh.hpp): starsh_from_entries from an EntrySource
// (symm 'N' and 'S'), starsh_export, starsh_import of the exported view (round trip),
// starsh_blackbox ('N' is STARS-H's product; 'T' goes through the view, or through STARS-H's
// product for the symmetric format); the exported view outliving the handle; the entry
// callback with a padded leading dimension; a source that is not thread-safe (its calls must
// be serialized) and one that throws; the errors.
#include "hmb_starsh.hpp" // first: the header must compile on its own
#include "hmb/neutral_store.hpp"
#include "hmb_test.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

using namespace hmb;
using namespace hmb::test;

namespace {

std::vector<double> dense_of(const hmb_hmatrix *A) {
    std::vector<int> r(A->n_rows), c(A->n_cols);
    std::iota(r.begin(), r.end(), 0);
    std::iota(c.begin(), c.end(), 0);
    std::vector<double> D(size_t(A->n_rows) * A->n_cols);
    hmb_entries(A, A->n_rows, r.data(), A->n_cols, c.data(), D.data(), A->n_rows);
    return D;
}

// relative difference between op(A) X from the BlackBox and from hmb_matvec on the view V
double bb_vs_view(const BlackBox &B, const hmb_hmatrix *V, char trans, unsigned seed) {
    const int n = B.n, k = 3;
    std::vector<double> X = randvec(size_t(n) * k, seed), Y1(size_t(n) * k), Y2(Y1.size());
    hmb_matvec(V, trans, k, X.data(), n, Y1.data(), n);
    B.apply(trans, k, X.data(), Y2.data());
    return relerr(Y2, Y1);
}

// the message of the exception of type E that f throws, or "(no exception)"
template <class E, class F> std::string thrown(F f) {
    try {
        f();
    } catch (const E &e) {
        return e.what();
    }
    return "(no exception)";
}

bool starts_with(const std::string &s, const char *p) { return s.rfind(p, 0) == 0; }

} // namespace

int main(int argc, char **argv) {
    Checker C("test_starsh_cpp");
    const int n = arg_int(argc, argv, "--n", 800), tile = arg_int(argc, argv, "--tile", 100);
    const double tol = arg_dbl(argc, argv, "--eps", 1e-6);
    Problem P(n, Kernel::Sym, Geometry::RandomSphere, 0, 3);
    const std::vector<int> perm = bisection_perm(P, tile / 2);
    const BlackBox exact = P.exact_blackbox();
    const hmb_starsh_tlr_opts opts_n{tile, tol, tile / 2, 'N', 1};

    // ---- 1. EntrySource -> STARS-H TLR -> view and BlackBox -> re-import --------------------
    for (char symm : {'N', 'S'}) {
        const std::string tag = std::string("symm ") + symm + ": ";
        hmb_starsh_tlr_opts o = opts_n;
        o.symm = symm;
        const double t0 = now();
        StarshHandle S = starsh_from_entries(P.entries(), perm.data(), o);
        const double tb = now() - t0;
        hmb_starsh_export_stats st;
        std::shared_ptr<const hmb_hmatrix> V = starsh_export(S, &st);
        C.check(V && V->n_rows == n && V->n_cols == n && st.far_blocks > 0, (tag + "from_entries, export").c_str(),
                "%lld far + %lld near blocks (%lld mirrored), max rank %d, %lld values aliased, %lld copied; %.2fs",
                st.far_blocks, st.near_blocks, st.mirrored_blocks, st.max_rank, st.aliased_values, st.copied_values,
                tb);
        const BlackBox B = starsh_blackbox(S);
        C.check(B.n == n && B.symmetric == (symm == 'S'), (tag + "BlackBox size and symmetry").c_str(),
                "n %d, symmetric %s", B.n, B.symmetric ? "true" : "false");
        C.check_le(bb_vs_view(B, V.get(), 'N', 1), 1e-13, (tag + "BlackBox N (dmml) vs hmb_matvec(view)").c_str());
        C.check_le(bb_vs_view(B, V.get(), 'T', 2), 1e-13, (tag + "BlackBox T vs hmb_matvec(view, T)").c_str());
        C.check_le(blackbox_relerr(B, exact, 'N'), 10 * tol, (tag + "BlackBox vs exact kernel (<= 10 eps)").c_str());

        // round trip through the C++ import: the same entries, bit for bit
        StarshHandle S2 = starsh_import(*V);
        std::shared_ptr<const hmb_hmatrix> V2 = starsh_export(S2);
        const std::vector<double> d1 = dense_of(V.get()), d2 = dense_of(V2.get());
        double dmax = 0;
        for (size_t q = 0; q < d1.size(); q++) dmax = std::max(dmax, std::fabs(d1[q] - d2[q]));
        C.check(dmax == 0.0, (tag + "export(import(export)) bit-identical").c_str(), "max |diff| %.3e (must be 0)",
                dmax);
        const BlackBox B2 = starsh_blackbox(S2);
        C.check(!B2.symmetric, (tag + "the import stores every block ('N')").c_str(), "symmetric %s",
                B2.symmetric ? "true" : "false");
        C.check_le(blackbox_relerr(B2, B, 'N'), 1e-13, (tag + "re-imported product vs source product").c_str());
    }

    // ---- 2. the exported view keeps the matrix alive ---------------------------------------
    {
        StarshHandle S = starsh_from_entries(P.entries(), perm.data(), opts_n);
        std::weak_ptr<hmb_starsh> w = S;
        std::vector<double> X = randvec(size_t(n) * 2, 4), Y1(size_t(n) * 2), Y2(Y1.size());
        BlackBox B = starsh_blackbox(S);
        B.apply('N', 2, X.data(), Y1.data());
        std::shared_ptr<const hmb_hmatrix> V = starsh_export(S);
        S.reset();
        B = BlackBox(); // drops the closure's copy of the handle
        const bool alive = !w.expired();
        hmb_matvec(V.get(), 'N', 2, X.data(), n, Y2.data(), n); // reads STARS-H's factors
        const double d = relerr(Y2, Y1);
        V.reset();
        C.check(alive && w.expired() && d <= 1e-13, "exported view keeps the matrix alive",
                "alive without a handle: %s, freed with the view: %s, view product vs STARS-H %.3e <= 1e-13",
                alive ? "yes" : "no", w.expired() ? "yes" : "no", d);
    }

    // ---- 3. the EntrySource ---------------------------------------------------------------
    {   // the entry callback with a padded leading dimension (STARS-H may pass ld > m)
        const int m = 5, q = 4, ld = 8;
        const int rows[m] = {3, 0, 7, 7, 2}, cols[q] = {1, 6, 0, 5};
        std::vector<double> out(size_t(ld) * q, -7.0);
        const EntrySource src = P.entries();
        detail::StarshEntries ctx(&src);
        detail::starsh_entries_cb(m, q, rows, cols, out.data(), ld, &ctx);
        int bad = 0;
        for (int j = 0; j < q; j++)
            for (int i = 0; i < ld; i++) bad += out[i + size_t(j) * ld] != (i < m ? P.K(rows[i], cols[j]) : -7.0);
        C.check(bad == 0, "entry callback, ld > m: entries placed, padding kept", "%d of %d values wrong", bad, ld * q);
    }
    {   // a source that is not thread-safe: STARS-H must call it one at a time
        std::atomic<int> active{0}, most{0};
        std::atomic<long> calls{0};
        EntrySource src = P.entries();
        const auto get = src.get;
        src.thread_safe = false; // opts_n.thread_safe is 1: the source decides
        src.get = [&](int M, int N, const int *r, const int *c, double *out) {
            const int a = ++active;
            int m0 = most.load();
            while (a > m0 && !most.compare_exchange_weak(m0, a)) {
            }
            calls++;
            get(M, N, r, c, out);
            active--;
        };
        StarshHandle S = starsh_from_entries(src, perm.data(), opts_n);
        const char *threads = std::getenv("OMP_NUM_THREADS");
        C.check(S && most.load() == 1, "source not thread-safe: calls serialized",
                "%ld calls, at most %d at a time (must be 1; OMP_NUM_THREADS=%s)", calls.load(), most.load(),
                threads ? threads : "unset");
    }
    {   // an exception from the source must not unwind through STARS-H (C, OpenMP)
        EntrySource src = P.entries();
        const auto get = src.get;
        src.get = [get](int M, int N, const int *r, const int *c, double *out) {
            for (int i = 0; i < M; i++)
                if (r[i] == 123) throw std::runtime_error("no row 123");
            get(M, N, r, c, out);
        };
        const std::string msg = thrown<std::runtime_error>([&] { starsh_from_entries(src, perm.data(), opts_n); });
        C.check(msg == "hmb::starsh_from_entries: the EntrySource threw: no row 123",
                "throwing EntrySource -> std::runtime_error", "%s", msg.c_str());
    }

    // ---- 4. errors --------------------------------------------------------------------------
    {
        StarshHandle S = starsh_from_entries(P.entries(), perm.data(), opts_n);
        std::shared_ptr<const hmb_hmatrix> V = starsh_export(S);
        // an invalid view: one dense leaf with a leading dimension below its row count
        std::vector<hmb_block> bl(V->bl, V->bl + V->nbl);
        for (auto &b : bl)
            if (b.kind == HMB_DENSE) {
                b.ldD = 1;
                break;
            }
        hmb_hmatrix bad = *V;
        bad.bl = bl.data();
        std::string msg = thrown<std::invalid_argument>([&] { starsh_import(bad); });
        C.check(starts_with(msg, "hmb::starsh_import: invalid view: "), "invalid view refused", "%s", msg.c_str());

        hmb_starsh_tlr_opts z = opts_n;
        z.tile_size = 0;
        msg = thrown<std::invalid_argument>([&] { starsh_from_entries(P.entries(), perm.data(), z); });
        C.check(starts_with(msg, "hmb::starsh_from_entries: need tile_size > 0"), "bad options refused", "%s",
                msg.c_str());

        EntrySource rect = P.entries();
        rect.n_cols = n - 1;
        msg = thrown<std::invalid_argument>([&] { starsh_from_entries(rect, nullptr, opts_n); });
        C.check(starts_with(msg, "hmb::starsh_from_entries: need a square EntrySource"),
                "rectangular EntrySource refused", "%s", msg.c_str());

        // a rectangular view imports (STARS-H holds it), but a BlackBox is square
        NeutralStore s;
        hmb_cluster rc{}, cc{};
        rc.size = 3, cc.size = 2;
        s.cl = {rc, cc};
        hmb_block d{};
        d.rc = 0, d.cc = 1, d.kind = HMB_DENSE, d.ldD = 3, d.D = s.alloc(6);
        s.bl.push_back(d);
        s.A.n_rows = 3, s.A.n_cols = 2, s.A.row_root = 0, s.A.col_root = 1;
        s.finalize();
        StarshHandle R = starsh_import(s.A);
        msg = thrown<std::invalid_argument>([&] { starsh_blackbox(R); });
        C.check(starts_with(msg, "hmb::starsh_blackbox: square matrices only"), "rectangular BlackBox refused", "%s",
                msg.c_str());

        const BlackBox B = starsh_blackbox(S);
        std::vector<double> x(n, 1.0), y(n);
        msg = thrown<std::invalid_argument>([&] { B.apply('C', 1, x.data(), y.data()); });
        C.check(starts_with(msg, "hmb::starsh_blackbox: trans must be"), "BlackBox trans 'C' refused", "%s",
                msg.c_str());
    }
    return C.finish("n=%d tile=%d eps=%.0e", n, tile, tol);
}
