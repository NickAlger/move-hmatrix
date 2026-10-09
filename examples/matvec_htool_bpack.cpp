// Example, matvec route: an Htool-DDM H-matrix of a kernel -> hmb::BlackBox -> ButterflyPACK
// HODLR and H (hmb::bpack_from_matvec), with ButterflyPACK's own construction from the
// kernel's entries (hmb::bpack_from_entries) as the direct baseline. Port of experiment E2's
// driver drv_bp; the defaults reproduce its N = 4,000 runs, --n 16000 the large ones
// (E2 ran one thread: OMP_NUM_THREADS=1).
//
//   matvec_htool_bpack [--n 4000] [--eps 1e-6] [--tol 1e-6] [--leaf 64] [--eta 10]
//                      [--reps 1] [--no-entries] [--no-exact]
//
// Source: Htool, symmetric storage ('S', so a symmetric BlackBox), tolerance eps, eta, leaf.
// Destinations, all with ButterflyPACK leaf size `leaf`, near_para 2, rank0 64, rankrate 2
// (E2 gave H the source's largest rank, 19, as rank0; its rank0 = 64 run matched):
//   HODLR from matvecs at tol; H from matvecs at tol, and at tol / 10 to show
//   GOTCHA(tolerance-stacking); HODLR and H from the kernel's entries at tol (baselines).
// For each: callback calls and sampling vectors (and the A^T share), time, cost in source
// matvecs = construction time / time of ONE SINGLE-VECTOR source matvec (as E2; the blocked
// cost per vector is printed too), memory (retained heap, peak RSS growth, ButterflyPACK's own
// count), accuracy against the source and the kernel. Times are medians over --reps runs.
// Checks: error against the source <= 20 x tol (HODLR) or 100 x tol (H, whose misses have a
// heavy tail); every A^T request answered with 'N' (symmetric source); the entry route within
// 10 x tol of the kernel.
#include "hmb_bpack.hpp"
#include "hmb_htool.hpp"
#include "hmb_test.hpp"

#ifdef __GLIBC__
#include <malloc.h>
#endif

using namespace hmb;
using namespace hmb::test;

namespace {

double heap_mb() {
#ifdef __GLIBC__
    malloc_trim(0);
    struct mallinfo2 mi = mallinfo2();
    return (mi.uordblks + mi.hblkhd) / 1048576.0;
#else
    return 0;
#endif
}

// Time of one call of S.apply('N', k, ...): median over 3 batches of max(1, 32 / k) calls.
double time_matvec(const BlackBox &S, int k) {
    std::vector<double> x = randvec(size_t(S.n) * k, 11), y(size_t(S.n) * k);
    const int inner = std::max(1, 32 / k);
    S.apply('N', k, x.data(), y.data()); // warm-up
    std::vector<double> t;
    for (int r = 0; r < 3; r++) {
        const double t0 = now();
        for (int i = 0; i < inner; i++) S.apply('N', k, x.data(), y.data());
        t.push_back((now() - t0) / inner);
    }
    return median(t);
}

struct Run {
    std::string name;
    double tol = 0;
    bool from_entries = false;
    BpackStats st;
    double seconds = 0, peak_mb = 0, heap_mb = 0, err_src = 0, err_kernel = -1;
    long source_T_calls = 0;
};

} // namespace

int main(int argc, char **argv) {
    Checker C("matvec_htool_bpack");
    const int n = arg_int(argc, argv, "--n", 4000);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6), tol = arg_dbl(argc, argv, "--tol", 1e-6);
    const int leaf = arg_int(argc, argv, "--leaf", 64), reps = std::max(1, arg_int(argc, argv, "--reps", 1));
    const double eta = arg_dbl(argc, argv, "--eta", 10);
    const bool entries = !arg_flag(argc, argv, "--no-entries"), exact = !arg_flag(argc, argv, "--no-exact");
    std::printf("== matvec_htool_bpack n=%d eps=%g tol=%g leaf=%d eta=%g reps=%d OMP_NUM_THREADS=%s\n", n, eps, tol, leaf,
                eta, reps, std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "(unset)");

    // ---- source: Htool H-matrix of the kernel, symmetric storage -------------------------
    Problem P(n);
    HtoolOptions ho;
    ho.epsilon = eps;
    ho.eta = eta;
    ho.leaf_size = leaf;
    ho.symmetry = 'S';
    HtoolStats hs;
    HtoolHandle src = htool_build(P.entries(), 3, P.xyz.data(), ho, &hs);
    const BlackBox S = htool_blackbox(src);
    const double t_mv1 = time_matvec(S, 1), t_mv64 = time_matvec(S, 64) / 64;
    std::printf("[source] Htool build %.3fs, %lld stored values (%.1f MB, %.3f of n^2), max rank %d\n", hs.build_seconds,
                hs.stored_values, hs.stored_values * 8e-6, hs.stored_values / (double(n) * n), hs.max_rank);
    std::printf("[source] one matvec: %.5fs single vector, %.5fs per vector in blocks of 64\n", t_mv1, t_mv64);

    const int kt = 3; // accuracy vectors
    std::vector<double> xt = randvec(size_t(n) * kt, 7), y_src(size_t(n) * kt), y_ex;
    S.apply('N', kt, xt.data(), y_src.data());
    double src_err = -1;
    if (exact) {
        y_ex.resize(y_src.size());
        P.exact_apply('N', kt, xt.data(), y_ex.data());
        src_err = relerr(y_src, y_ex);
        std::printf("[source] error against the kernel %.3e\n", src_err);
    }

    // ---- constructions ------------------------------------------------------------------------
    struct Plan {
        const char *name;
        BpackFormat format;
        double tol;
        bool from_entries;
    };
    std::vector<Plan> plans = {{"HODLR matvec", BpackFormat::HODLR, tol, false},
                               {"H matvec", BpackFormat::H, tol, false},
                               {"H matvec tol/10", BpackFormat::H, tol / 10, false}};
    if (entries) {
        plans.push_back({"HODLR entries", BpackFormat::HODLR, tol, true});
        plans.push_back({"H entries", BpackFormat::H, tol, true});
    }
    std::vector<Run> runs;
    for (const Plan &p : plans) {
        Run run;
        run.name = p.name;
        run.tol = p.tol;
        run.from_entries = p.from_entries;
        BpackOptions o;
        o.format = p.format;
        o.tolerance = p.tol;
        o.leaf_size = leaf;
        std::vector<double> times;
        for (int r = 0; r < reps; r++) {
            long tcalls = 0;
            BlackBox counted = S; // counts the source's 'T' calls (expected 0: symmetric source)
            counted.apply = [&S, &tcalls](char t, int k, const double *X, double *Y) {
                if (t == 'T' || t == 't') tcalls++;
                S.apply(t, k, X, Y);
            };
            const double h0 = heap_mb();
            const long rss0 = rss_kb();
            reset_hwm();
            BpackStats st;
            BpackHandle h = p.from_entries ? bpack_from_entries(P.entries(), 3, P.xyz.data(), o, &st)
                                           : bpack_from_matvec(counted, 3, P.xyz.data(), o, &st);
            const long hwm = hwm_kb();
            times.push_back(st.seconds_build);
            if (r == reps - 1) {
                run.st = st;
                run.source_T_calls = tcalls;
                run.peak_mb = (hwm - rss0) / 1024.0;
                run.heap_mb = heap_mb() - h0;
                std::vector<double> y(size_t(n) * kt);
                bpack_blackbox(h).apply('N', kt, xt.data(), y.data());
                run.err_src = relerr(y, y_src);
                if (exact) run.err_kernel = relerr(y, y_ex);
            }
        }
        run.seconds = median(times);
        const BpackStats &st = run.st;
        if (p.from_entries)
            std::printf("[%s tol %g] %.3fs (init %.3fs) = %.0f source matvecs; %lld entries (%.2f n^2) in %ld calls; "
                        "rank %d; heap %.1f MB, peak +%.1f MB, ButterflyPACK Mem_Fill %.1f MB; error vs source %.2e, "
                        "vs kernel %.2e\n",
                        p.name, p.tol, run.seconds, st.seconds_init, run.seconds / t_mv1, st.entries,
                        st.entries / (double(n) * n), st.entry_calls, st.rank_max, run.heap_mb, run.peak_mb,
                        st.bpack_mem_fill_mb, run.err_src, run.err_kernel);
        else
            std::printf("[%s tol %g] %.3fs (init %.3fs) = %.0f source matvecs; %ld calls (%ld A^T, %.0f%%), %lld vectors "
                        "(%.2f n; %lld A^T); in source %.3fs (= %.2f x vectors x blocked cost), adapter copies %.3fs; "
                        "rank %d; heap %.1f MB, peak +%.1f MB, ButterflyPACK Mem_Fill %.1f MB (no low-rank blocks for H); "
                        "error vs source %.2e (adapter's check_error %.2e), vs kernel %.2e\n",
                        p.name, p.tol, run.seconds, st.seconds_init, run.seconds / t_mv1, st.calls, st.calls_transposed,
                        100.0 * st.calls_transposed / std::max(1L, st.calls), st.vectors, double(st.vectors) / n,
                        st.vectors_transposed, st.seconds_in_source, st.seconds_in_source / (st.vectors * t_mv64),
                        st.seconds_callback - st.seconds_in_source, st.rank_max, run.heap_mb, run.peak_mb,
                        st.bpack_mem_fill_mb, run.err_src, st.check_error, run.err_kernel);
        std::fflush(stdout);
        runs.push_back(run);
    }

    // ---- summary and checks ----------------------------------------------------------------------
    std::printf("\nRESULT n=%d eps=%g tol=%g t_src_mv1=%.5f t_src_mv64_per_vector=%.5f src_build=%.3f src_err=%.2e\n", n, eps,
                tol, t_mv1, t_mv64, hs.build_seconds, src_err);
    std::printf("%-16s %8s %9s %9s %9s %8s %9s %9s %8s %9s %9s\n", "construction", "tol", "calls(T)", "vectors", "seconds",
                "x mv1", "err_src", "err_kern", "rank", "heap MB", "peak MB");
    for (const Run &r : runs) {
        char calls[32];
        if (r.from_entries) std::snprintf(calls, sizeof calls, "-");
        else std::snprintf(calls, sizeof calls, "%ld(%ld)", r.st.calls, r.st.calls_transposed);
        std::printf("%-16s %8.0e %9s %9lld %9.3f %8.0f %9.2e %9.2e %8d %9.1f %9.1f\n", r.name.c_str(), r.tol, calls,
                    r.from_entries ? r.st.entries : r.st.vectors, r.seconds, r.seconds / t_mv1, r.err_src, r.err_kernel,
                    r.st.rank_max, r.heap_mb, r.peak_mb);
    }
    std::printf("(vectors column: sampling vectors for the matvec route, entries for the entry route; x mv1: "
                "construction time / one single-vector source matvec)\n\n");

    for (const Run &r : runs) {
        char what[96];
        if (!r.from_entries) {
            // GOTCHA(tolerance-stacking): HODLR stays within a few times its tolerance; H
            // usually 4-8x, with rare misses up to about 60x (tests/test_bpack_matvec.cpp)
            const double factor = r.name.rfind("HODLR", 0) == 0 ? 20 : 100;
            std::snprintf(what, sizeof what, "%s: error vs source <= %.0f x tol", r.name.c_str(), factor);
            C.check_le(r.err_src, factor * r.tol, what);
            std::snprintf(what, sizeof what, "%s: A^T asked, answered with 'N' (symmetric)", r.name.c_str());
            C.check(r.st.calls_transposed > 0 && r.source_T_calls == 0, what, "%ld requests, %ld 'T' calls to Htool",
                    r.st.calls_transposed, r.source_T_calls);
        } else if (exact) {
            std::snprintf(what, sizeof what, "%s: error vs kernel <= 10 x tol", r.name.c_str());
            C.check_le(r.err_kernel, 10 * r.tol, what);
        }
    }
    // GOTCHA(tolerance-stacking): H from matvecs misses tol; asking for tol / 10 is what gets
    // near it (E2: 6.1e-6 and 1.1e-6 at N = 4,000). Printed, not checked: single runs vary.
    const Run &h1 = runs[1], &h2 = runs[2];
    std::printf("tolerance stacking: H at tol %g reached %.2e (%.1f x tol); at tol %g, %.2e (%.1f x the first tol)\n", h1.tol,
                h1.err_src, h1.err_src / h1.tol, h2.tol, h2.err_src, h2.err_src / h1.tol);
    return C.finish("n=%d tol=%g", n, tol);
}
