// ButterflyPACK adapter, matvec route: bpack_from_matvec for HODLR and H from an exact
// (dense) source, symmetric and nonsymmetric, then the rebuilt matrix's products in the
// original ordering ('N' and 'T'; the matvec after construction is the ErrSol regression,
// GOTCHA(bpack-errsol)). Checks the transpose handling (GOTCHA(bpack-transpose)),
// the statistics against an independent count, the ordering (GOTCHA(index-base),
// GOTCHA(perm-direction)) through the ranks, and that the thread count survives
// (GOTCHA(bpack-global-state)). The Htool-source variant is test_bpack_htool.
#include "hmb_bpack.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <omp.h>

using namespace hmb;
using namespace hmb::test;

int main(int argc, char **argv) {
    Checker C("test_bpack_matvec");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6);
    // Thresholds (GOTCHA(tolerance-stacking)): HODLR stayed within 1-2x the tolerance at this
    // size; H had a median of 4x, a 90th percentile of 8x and 4 of 164 runs above 15x (worst
    // 56x), so its bound is 100x. Ordering or transpose mistakes give errors of 1e-2 or more.
    const double hodlr_bound = 20 * tol, h_bound = 100 * tol;
    for (Kernel kern : {Kernel::Sym, Kernel::Nonsym}) {
        const bool sym = kern == Kernel::Sym;
        // the nonsymmetric problem also uses a shuffled user numbering of its points
        Problem P(n, kern, sym ? Geometry::RandomSphere : Geometry::ShuffledFibonacci);
        const BlackBox D = dense_kernel_blackbox(P);
        const char *kname = sym ? "sym" : "nonsym";
        if (!sym) {
            // how far from symmetric the source is: answering A^T requests with A would show
            std::vector<double> X = randvec(size_t(n) * 2, 3), YN(size_t(n) * 2), YT(size_t(n) * 2);
            D.apply('N', 2, X.data(), YN.data());
            D.apply('T', 2, X.data(), YT.data());
            std::printf("nonsym source: |A x - A^T x| / |A x| = %.2f\n", relerr(YT, YN));
        }
        for (BpackFormat f : {BpackFormat::HODLR, BpackFormat::H}) {
            const char *fname = bpack_format_name(f);
            const double err_bound = f == BpackFormat::H ? h_bound : hodlr_bound;
            char what[128];
            BpackOptions o;
            o.format = f;
            o.tolerance = tol;
            auto cnt = std::make_shared<CallCounts>();
            const int threads_before = omp_get_max_threads();
            BpackStats st;
            const double t0 = now();
            BpackHandle h = bpack_from_matvec(counted(D, cnt), 3, P.xyz.data(), o, &st);
            const double t_build = now() - t0;
            BlackBox B = bpack_blackbox(h);
            std::printf("%s %s: %.2fs, %ld calls (%ld for A^T), %lld vectors (%lld for A^T), rank %d, "
                        "source time %.2fs, callback %.2fs\n",
                        kname, fname, t_build, st.calls, st.calls_transposed, st.vectors, st.vectors_transposed,
                        st.rank_max, st.seconds_in_source, st.seconds_callback);

            const double err_N = blackbox_relerr(B, D, 'N');
            std::snprintf(what, sizeof what, "%s %s matvec 'N' after construction", kname, fname);
            C.check_le(err_N, err_bound, what);
            std::snprintf(what, sizeof what, "%s %s matvec 'T' after construction", kname, fname);
            C.check_le(blackbox_relerr(B, D, 'T'), err_bound, what);
            // the adapter's own estimate (other random vectors) agrees with the measured error
            std::snprintf(what, sizeof what, "%s %s check_error agrees (within 5x)", kname, fname);
            C.check(st.check_error > err_N / 5 && st.check_error < err_N * 5, what, "check_error %.2e, measured %.2e",
                    st.check_error, err_N);

            // ButterflyPACK's calls, plus the one call of the check
            std::snprintf(what, sizeof what, "%s %s stats count every call", kname, fname);
            C.check(st.calls + 1 == cnt->calls && st.vectors + o.check_vectors == cnt->vectors, what,
                    "adapter %ld calls / %lld vectors (+1 / +%d for the check), source %ld / %lld", st.calls, st.vectors,
                    o.check_vectors, cnt->calls.load(), cnt->vectors.load());
            std::snprintf(what, sizeof what, "%s %s ButterflyPACK asked for A^T", kname, fname);
            C.check(st.calls_transposed > 0 && st.vectors_transposed > 0, what, "%ld calls, %lld vectors",
                    st.calls_transposed, st.vectors_transposed);
            // symmetric source: A^T requests answered with apply('N'); otherwise forwarded as 'T'
            std::snprintf(what, sizeof what, "%s %s source got 'T' %s", kname, fname, sym ? "never" : "for each A^T request");
            C.check(cnt->calls_transposed == (sym ? 0 : st.calls_transposed), what, "source 'T' calls %ld, A^T requests %ld",
                    cnt->calls_transposed.load(), st.calls_transposed);
            std::snprintf(what, sizeof what, "%s %s source called from one thread at a time", kname, fname);
            C.check(cnt->max_in_flight == 1, what, "max in flight %d", cnt->max_in_flight.load());

            // Ordering: the errors above stay small even when the points do not match the
            // indices (a consistent permutation mistake), but then the ranks blow up
            // (calibrated at n = 2000: HODLR 924 instead of about 200, H 236 instead of 19).
            const int rank_bound = f == BpackFormat::HODLR ? n / 4 : o.leaf_size;
            std::snprintf(what, sizeof what, "%s %s ranks show a matching ordering", kname, fname);
            C.check(st.rank_max > 0 && st.rank_max <= rank_bound, what, "rank_max %d <= %d", st.rank_max, rank_bound);

            std::snprintf(what, sizeof what, "%s %s thread count restored", kname, fname);
            C.check(omp_get_max_threads() == threads_before, what, "%d -> %d", threads_before, omp_get_max_threads());
            C.check(B.n == n && bpack_size(h) == n && !B.symmetric, "handle size and BlackBox flags");
            C.check(bpack_stats(h).calls == st.calls && bpack_stats(h).seconds_build == st.seconds_build,
                    "bpack_stats returns the construction's statistics");
        }
    }
    return C.finish("n=%d tol=%g", n, tol);
}
