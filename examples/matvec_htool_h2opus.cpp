// Example, matvec route: an Htool-DDM H-matrix of a kernel -> hmb::BlackBox -> H2Opus H^2
// built by HARA (hmb::h2opus_from_matvec). Port of experiment E2's driver drv_h2 (HARA part;
// H2Opus's own Chebyshev construction from the kernel is not a transfer and is left out).
// The defaults reproduce E2's N = 4,000 run, --n 16000 the large one (E2 ran one thread:
// OMP_NUM_THREADS=1).
//
//   matvec_htool_h2opus [--n 4000] [--eps 1e-6] [--tol 1e-6] [--leaf 64] [--eta 10]
//                       [--h2eta 1] [--max-samples 512] [--reps 1] [--no-exact]
//
// Source: Htool, symmetric storage ('S': a symmetric BlackBox, which HARA requires),
// tolerance eps, admissibility eta, leaf size leaf. Destination: H2Opus H^2 on a KD-tree of
// the same points (leaf size leaf, box-center admissibility h2eta), HARA at relative
// tolerance tol (absolute tolerance from a sampled norm), at most max-samples per level.
// Reports calls and sampling vectors (with the norm estimate's share; HARA never asks for
// A^T), time split, cost in source matvecs = construction time / time of ONE SINGLE-VECTOR
// source matvec (as E2), memory (H2Opus's count, retained heap, peak RSS growth), accuracy
// against the source and the kernel. Times are medians over --reps runs. HARA's many short
// OpenMP regions make its time very sensitive to other load on the cores (hmb_h2opus.hpp).
// Checks: error against the source <= 20 x tol (and against the kernel, plus twice the
// source's own error); no A^T request.
#include "hmb_h2opus.hpp"
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

} // namespace

int main(int argc, char **argv) {
    Checker C("matvec_htool_h2opus");
    const int n = arg_int(argc, argv, "--n", 4000);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6), tol = arg_dbl(argc, argv, "--tol", 1e-6);
    const int leaf = arg_int(argc, argv, "--leaf", 64), reps = std::max(1, arg_int(argc, argv, "--reps", 1));
    const double eta = arg_dbl(argc, argv, "--eta", 10), h2eta = arg_dbl(argc, argv, "--h2eta", 1.0);
    const int max_samples = arg_int(argc, argv, "--max-samples", 512);
    const bool exact = !arg_flag(argc, argv, "--no-exact");
    std::printf("== matvec_htool_h2opus n=%d eps=%g tol=%g leaf=%d eta=%g h2eta=%g max_samples=%d reps=%d "
                "OMP_NUM_THREADS=%s\n",
                n, eps, tol, leaf, eta, h2eta, max_samples, reps,
                std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "(unset)");

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
    const double t_mv1 = time_matvec(S, 1), t_mv32 = time_matvec(S, 32) / 32;
    std::printf("[source] Htool build %.3fs, %lld stored values (%.1f MB), max rank %d\n", hs.build_seconds,
                hs.stored_values, hs.stored_values * 8e-6, hs.max_rank);
    std::printf("[source] one matvec: %.5fs single vector, %.5fs per vector in blocks of 32 (HARA's block)\n", t_mv1,
                t_mv32);

    const int kt = 3;
    std::vector<double> xt = randvec(size_t(n) * kt, 7), y_src(size_t(n) * kt), y_ex;
    S.apply('N', kt, xt.data(), y_src.data());
    double src_err = -1;
    if (exact) {
        y_ex.resize(y_src.size());
        P.exact_apply('N', kt, xt.data(), y_ex.data());
        src_err = relerr(y_src, y_ex);
        std::printf("[source] error against the kernel %.3e\n", src_err);
    }

    // ---- HARA ---------------------------------------------------------------------------------
    H2opusOptions o;
    o.tolerance = tol;
    o.leaf_size = leaf;
    o.eta = h2eta;
    o.max_samples = max_samples;
    std::vector<double> times;
    H2opusStats st;
    double heap = 0, peak = 0, err_src = 0, err_kernel = -1;
    long source_T_calls = 0;
    for (int r = 0; r < reps; r++) {
        long tcalls = 0;
        BlackBox counted = S;
        counted.apply = [&S, &tcalls](char t, int k, const double *X, double *Y) {
            if (t == 'T' || t == 't') tcalls++;
            S.apply(t, k, X, Y);
        };
        const double h0 = heap_mb();
        const long rss0 = rss_kb();
        reset_hwm();
        H2opusStats s;
        H2opusHandle h = h2opus_from_matvec(counted, 3, P.xyz.data(), o, &s);
        const long hwm = hwm_kb();
        times.push_back(s.seconds_structure + s.seconds_norm + s.seconds_hara);
        std::printf("[HARA run %d] structure %.3fs, norm estimate %.3fs (%ld calls), hara %.3fs; in source %.3fs; "
                    "%ld calls, %lld vectors\n",
                    r, s.seconds_structure, s.seconds_norm, s.norm_calls, s.seconds_hara, s.seconds_in_source, s.calls,
                    s.vectors);
        if (r == reps - 1) {
            st = s;
            source_T_calls = tcalls;
            peak = (hwm - rss0) / 1024.0;
            heap = heap_mb() - h0;
            std::vector<double> y(size_t(n) * kt);
            h2opus_blackbox(h).apply('N', kt, xt.data(), y.data());
            err_src = relerr(y, y_src);
            if (exact) err_kernel = relerr(y, y_ex);
        }
    }
    const double seconds = median(times);
    std::printf("[HARA tol %g] %.3fs = %.0f source matvecs; %ld calls (%ld for the norm), %lld vectors (%.2f n); in "
                "source %.3fs (%.0f%% of the time, %.2f x vectors x blocked cost), adapter copies %.3fs; norm estimate "
                "%.4e, absolute tolerance %.3e; depth %d, max rank %d; H2Opus memory %.1f MB, retained heap %.1f MB, "
                "peak +%.1f MB; error vs source %.2e (adapter's check_error %.2e), vs kernel %.2e\n",
                tol, seconds, seconds / t_mv1, st.calls, st.norm_calls, st.vectors, double(st.vectors) / n,
                st.seconds_in_source, 100 * st.seconds_in_source / seconds, st.seconds_in_source / (st.vectors * t_mv32),
                st.seconds_callback - st.seconds_in_source, st.norm_estimate, st.abs_tolerance, st.depth, st.max_rank,
                st.memory_mb, heap, peak, err_src, st.check_error, err_kernel);
    std::printf("\nRESULT route=HARA n=%d eps=%g tol=%g calls=%ld vectors=%lld vec_per_n=%.4f seconds=%.3f "
                "t_src_mv1=%.5f ratio_over_srcmv1=%.0f err_vs_src=%.3e err_vs_kernel=%.3e src_err=%.3e memory_mb=%.2f "
                "heap_mb=%.1f peak_mb=%.1f\n\n",
                n, eps, tol, st.calls, st.vectors, double(st.vectors) / n, seconds, t_mv1, seconds / t_mv1, err_src,
                err_kernel, src_err, st.memory_mb, heap, peak);

    C.check_le(err_src, 20 * tol, "HARA: error vs source <= 20 x tol");
    if (exact) C.check_le(err_kernel, 20 * tol + 2 * src_err, "HARA: error vs kernel <= 20 x tol + 2 x source error");
    C.check(source_T_calls == 0, "HARA never asked for A^T", "%ld", source_T_calls);
    C.check(st.vectors > 0 && st.calls > st.norm_calls, "HARA sampled the source", "%ld calls", st.calls);
    return C.finish("n=%d tol=%g", n, tol);
}
