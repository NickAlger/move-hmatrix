// ButterflyPACK adapter, factorization and solves (bpack_factor, bpack_solve_blackbox): HODLR
// and H, rebuilt from matvecs and from entries of the symmetric kernel with a diagonal shift of
// 10 (K + 10 I: symmetric positive definite, smallest eigenvalue at least 10).
//  - Residual ||K (S x) - x|| / ||x|| of the solve S against the exact kernel product K (the
//    dense source, not ButterflyPACK's product), bounded through the rebuilt matrix's own
//    forward error; and ||B (S x) - x|| / ||x|| against the rebuilt matrix B (the factors are
//    consistent with what ButterflyPACK stores).
//  - GOTCHA(bpack-errsol): with keep_forward (the default) B multiplies after the
//    factorization as before; with keep_forward = false a HODLR's products are refused after
//    it (ButterflyPACK deleted its blocks) while its solves work, and an H matrix (which then
//    has no forward operator at all) factors and solves.
//  - 'T': HODLR solves with A^T; H refuses (ButterflyPACK has no transposed H-LU solve).
//  - Statistics; a second bpack_factor does nothing; bpack_solve_blackbox factors on demand
//    and keeps the matrix alive; the thread count survives (GOTCHA(bpack-global-state));
//    argument checks.
#include "hmb_bpack.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <omp.h>

using namespace hmb;
using namespace hmb::test;

namespace {

bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

template <class E, class F> bool throws(F f, const char *prefix) {
    try {
        f();
    } catch (const E &e) {
        return starts_with(e.what(), prefix);
    }
    return false;
}

// ||op(R) (op(S) x) - x||_F / ||x||_F for k random vectors x: the residual of the solve S
// against the operator R.
double residual(const BlackBox &S, const BlackBox &R, char trans = 'N', int k = 4) {
    const size_t len = size_t(S.n) * k;
    std::vector<double> x = randvec(len, 11), y(len), r(len);
    S.apply(trans, k, x.data(), y.data());
    R.apply(trans, k, y.data(), r.data());
    return relerr(r, x);
}

// op(A) on three fixed random vectors
std::vector<double> products(const BlackBox &A, char trans) {
    std::vector<double> x = randvec(size_t(A.n) * 3, 5), y(x.size());
    A.apply(trans, 3, x.data(), y.data());
    return y;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_bpack_solve");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6);
    Problem P(n, Kernel::Sym, Geometry::RandomSphere, /*shift=*/10.0);
    const BlackBox D = dense_kernel_blackbox(P); // K, exact to round-off
    // Bounds. K (S x) - x = (K - B) (S x) + (B S - I) x. The first term is the forward error e
    // of B (measured: ||(B - K) X|| / ||K X|| on random X) amplified because S x leans toward
    // K's small eigenvalues, where B's error weighs more: 2-3.3x at n = 2,000 (growing with
    // n: 3.4 at 4,000); the bound is 20 (e + tol). The second term is the factorization's own
    // error: 1e-15 (HODLR) and 8e-7 (H-LU at tolerance 1e-6) at n = 2,000; bound 10 tol.
    auto residual_bound = [tol](double e) { return 20 * (e + tol); };
    const double consistency_bound = 10 * tol;
    const int threads = omp_get_max_threads();
    char what[128];

    // ---- both routes, both formats, keep_forward ------------------------------------------
    for (bool entries : {false, true}) {
        for (BpackFormat f : {BpackFormat::HODLR, BpackFormat::H}) {
            const bool hodlr = f == BpackFormat::HODLR;
            const char *route = entries ? "entries" : "matvec";
            char name[48];
            std::snprintf(name, sizeof name, "%s %s", bpack_format_name(f), route);
            BpackOptions o;
            o.format = f;
            o.tolerance = tol;
            BpackStats st;
            BpackHandle h = entries ? bpack_from_entries(P.entries(), 3, P.xyz.data(), o, &st)
                                    : bpack_from_matvec(D, 3, P.xyz.data(), o, &st);
            const BlackBox B = bpack_blackbox(h);
            const double e = blackbox_relerr(B, D), e_T = blackbox_relerr(B, D, 'T');
            const std::vector<double> yN = products(B, 'N'), yT = products(B, 'T');
            std::snprintf(what, sizeof what, "%s: not factored after construction", name);
            C.check(!bpack_factor_stats(h).factored, what);

            BpackFactorStats fs;
            bpack_factor(h, &fs);
            std::snprintf(what, sizeof what, "%s: factor statistics", name);
            C.check(fs.factored && fs.seconds > 0 && fs.rank_max > 0 && fs.bpack_mem_factor_mb > 0, what,
                    "%.3fs, rank %d, %.1f MB", fs.seconds, fs.rank_max, fs.bpack_mem_factor_mb);
            BpackFactorStats again;
            bpack_factor(h, &again);
            std::snprintf(what, sizeof what, "%s: second bpack_factor does nothing", name);
            C.check(again.seconds == fs.seconds && bpack_factor_stats(h).seconds == fs.seconds, what);

            // GOTCHA(bpack-errsol): keep_forward keeps the forward operator through the factorization
            const double dN = relerr(products(B, 'N'), yN), dT = relerr(products(B, 'T'), yT);
            std::snprintf(what, sizeof what, "%s: multiplies after the factorization as before", name);
            C.check(dN <= 1e-14 && dT <= 1e-14, what, "change 'N' %.1e, 'T' %.1e", dN, dT);

            const BlackBox S = bpack_solve_blackbox(h);
            std::snprintf(what, sizeof what, "%s: solve BlackBox size and flags", name);
            C.check(S.n == n && !S.symmetric && S.apply, what);
            const double r = residual(S, D), rB = residual(S, B);
            std::snprintf(what, sizeof what, "%s: residual vs K <= 20 (e + tol)", name);
            C.check_le(r, residual_bound(e), what);
            std::snprintf(what, sizeof what, "%s: residual vs rebuilt matrix <= 10 tol", name);
            C.check_le(rB, consistency_bound, what);
            double r_T = -1;
            if (hodlr) {
                r_T = residual(S, D, 'T');
                std::snprintf(what, sizeof what, "%s: transposed solve, residual vs K", name);
                C.check_le(r_T, residual_bound(e_T), what);
            } else {
                std::vector<double> x(n, 1.0), y(n);
                std::snprintf(what, sizeof what, "%s: transposed solve refused", name);
                C.check(throws<std::logic_error>([&] { S.apply('T', 1, x.data(), y.data()); },
                                                 "hmb::bpack_solve_blackbox:"),
                        what);
            }
            std::snprintf(what, sizeof what, "%s: thread count kept", name);
            C.check(omp_get_max_threads() == threads, what, "%d -> %d", threads, omp_get_max_threads());
            std::printf("  %s: build %.2fs (rank %d), factor %.3fs (rank %d, Mem_Factor %.1f MB); forward error %.2e; "
                        "residual vs K %.2e (%.1f x the forward error)%s, vs B %.1e\n",
                        name, st.seconds_init + st.seconds_build, st.rank_max, fs.seconds, fs.rank_max,
                        fs.bpack_mem_factor_mb, e, r, r / e, hodlr ? "" : ", 'T' refused", rB);
            if (hodlr) std::printf("  %s: transposed solve residual vs K %.2e\n", name, r_T);
        }
    }

    // ---- bpack_solve_blackbox factors on demand and keeps the matrix alive ----------------
    {
        BpackHandle h = bpack_from_entries(P.entries(), 3, P.xyz.data());
        const BlackBox S = bpack_solve_blackbox(h);
        C.check(bpack_factor_stats(h).factored, "bpack_solve_blackbox factored the matrix");
        const std::vector<double> y1 = products(S, 'N');
        h.reset();
        C.check(relerr(products(S, 'N'), y1) <= 1e-14, "solve works after the handle is released");
    }

    // ---- GOTCHA(bpack-errsol): keep_forward = false ------------------------------------------
    {
        BpackOptions o;
        o.keep_forward = false;
        BpackHandle h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        const BlackBox B = bpack_blackbox(h);
        const double e = blackbox_relerr(B, D);
        bpack_factor(h);
        std::vector<double> x(n, 1.0), y(n);
        C.check(throws<std::logic_error>([&] { B.apply('N', 1, x.data(), y.data()); }, "hmb::bpack_blackbox:") &&
                    throws<std::logic_error>([&] { bpack_blackbox(h); }, "hmb::bpack_blackbox:"),
                "HODLR without keep_forward: products refused after the factorization");
        C.check_le(residual(bpack_solve_blackbox(h), D), residual_bound(e),
                   "HODLR without keep_forward: solve, residual vs K");

        // H without keep_forward never had a forward operator: e unknown, but the entry route
        // stays within its tolerance of the kernel (hmb_bpack.hpp), so e = tol in the bound
        o.format = BpackFormat::H;
        h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        C.check(throws<std::logic_error>([&] { bpack_blackbox(h); }, "hmb::bpack_blackbox:"),
                "H without keep_forward: products refused");
        BpackFactorStats fs;
        bpack_factor(h, &fs);
        C.check(fs.factored, "H without keep_forward: factors");
        C.check_le(residual(bpack_solve_blackbox(h), D), residual_bound(tol), "H without keep_forward: solve, residual vs K");
    }

    // ---- argument checks ---------------------------------------------------------------------
    {
        C.check(throws<std::invalid_argument>([] { bpack_factor(nullptr); }, "hmb::bpack_factor:") &&
                    throws<std::invalid_argument>([] { bpack_solve_blackbox(nullptr); }, "hmb::bpack_solve_blackbox:") &&
                    throws<std::invalid_argument>([] { bpack_factor_stats(nullptr); }, "hmb::bpack_factor_stats:"),
                "null handles refused");
        const BlackBox S = bpack_solve_blackbox(bpack_from_entries(P.entries(), 3, P.xyz.data()));
        std::vector<double> x(n, 1.0), y(n);
        C.check(throws<std::invalid_argument>([&] { S.apply('X', 1, x.data(), y.data()); }, "hmb::bpack_solve_blackbox:"),
                "trans other than N/T refused");
    }
    return C.finish("n=%d tol=%g", n, tol);
}
