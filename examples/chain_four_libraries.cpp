// Example: four libraries in a chain (the study's extreme example). An H-matrix built by
// Htool-DDM travels through H2Lib and ButterflyPACK to H2Opus, which ends up holding the
// INVERSE of the matrix as an H^2 matrix:
//
//   1. Htool-DDM builds an H-matrix of K(x, y) = 1 / (0.1 + |x - y|) + 10 [x == y] on N random
//      points of the unit sphere (ACA on the kernel's entries, tolerance eps, full storage).
//   2. Block copy into H2Lib: hmb::htool_export (zero-copy neutral view), then
//      hmb::h2lib_import(view, HMB_H2LIB_STRICT) (copies every leaf; refines the block tree
//      into an H2Lib-strict one where needed, losslessly).
//   3. H -> H2 inside H2Lib (hmb::h2lib_to_h2, tolerance eps).
//   4. ButterflyPACK rebuilds a HODLR from the H2 matrix's products (hmb::h2lib_h2_blackbox ->
//      hmb::bpack_from_matvec, tolerance tol; the H2 product is not exactly symmetric, so
//      ButterflyPACK's A^T requests are answered with H2Lib's transposed product).
//   5. ButterflyPACK factors it (hmb::bpack_factor): the algebra of the chain.
//   6. H2Opus rebuilds the inverse as an H^2 matrix from ButterflyPACK's solves
//      (hmb::bpack_solve_blackbox -> hmb::h2opus_from_matvec, HARA at tolerance tol), and
//      applies it: the residual ||K (H2Opus x) - x|| / ||x||, with the exact kernel product,
//      is the chain's result.
//
// HARA needs a symmetric operator, and the solve S is not symmetric: the HODLR is symmetric
// only up to its errors, which the inversion amplifies (asymmetry ||(S - S^T) x|| / ||S x||
// 1.9-2.0e-5 at N = 4,000, tolerance 1e-6). The exact inverse is symmetric, so the skew part
// is error; the example samples T = (S + S^T) / 2 (ButterflyPACK's transposed HODLR solve,
// two solves per sample), which HARA then rebuilds well within its tolerance (6e-8), and whose
// residual is lower than S's own. --assume-symmetric samples T = S with
// H2opusOptions::assume_symmetric instead: HARA cannot fit the skew part, its result is off by
// about the asymmetry, and the final residual was 4.5 times larger (5.0-5.3e-5 against
// 1.1-1.2e-5 at N = 4,000), with larger ranks (225 against 170-174) and more memory (129
// against 109 MB).
//
//   chain_four_libraries [--n 4000] [--eps 1e-6] [--tol 1e-6] [--assume-symmetric]
//
// Per hop: time; cost in units of one source matvec (one single-vector product of the Htool
// matrix, timed at the start and at the end of the run, their mean); what crossed into the
// hop (entries evaluated, values copied, or sampling vectors); memory as the library counts
// it; accuracy against the previous hop and against the kernel K (exact products of the
// kernel on fixed random vectors). Hops 5 and 6 hold A^-1: their accuracy against K is the
// residual ||K (Y x) - x|| / ||x||, against the previous hop the residual with the HODLR
// (hop 5) and the product error against ButterflyPACK's solve (hop 6).
//
// Measured at N = 4,000 on one thread (two threads: 46 s): 67 s, 85% of it in hop 6, mostly
// HARA's own recompression (the inverse's ranks reach 170; K's H2 basis ranks are 31); the
// HODLR from H2Lib's products (one vector at a time) takes 9 s. Final residual 1.2e-5.
// Applying the H2Opus inverse costs 3 source matvecs per vector, ButterflyPACK's solve 1.1.
//
// Checks, relative to the tolerances. Copies: lossless (512 rounding units). Compressions
// and rebuilds: within 10 eps (Htool, H2Lib) or 20 tol (ButterflyPACK, H2Opus) of what they
// copied or sampled (H2Opus with --assume-symmetric: 20 tol + twice the asymmetry).
// Residuals: for the solve S of the HODLR B, K S x - x = (K - B) S x + (B S x - x), B's error
// against K amplified by a = (||K X|| / ||X||) (||S X|| / ||X||) plus the factorization's own
// error; for the H2Opus matrix M, K M x - x = K (M - T) x + (K T x - x), HARA's error against T
// amplified by a plus T's residual. a is measured on random vectors (4 at N = 4,000, growing
// with N: S x leans toward K's small eigenvalues, where an absolute error weighs more);
// each residual must stay within 10 times its two terms. The asymmetry, an error of S, must
// stay within 5 times S's expected error against the exact inverse (a e4 + tol).
//
// Only neutral-facing headers (GOTCHA(header-clash): Htool's and H2Opus's BLAS prototypes
// clash; H2Lib's macros I and real would break the others). Times are for OMP_NUM_THREADS
// threads (the study's runs: 1) with H2Lib serial, and one BLAS thread
// (OPENBLAS_NUM_THREADS=1, which tests/run_tests.sh sets).
#include "hmb_bpack.hpp"
#include "hmb_h2lib.hpp"
#include "hmb_h2opus.hpp"
#include "hmb_htool_copy.hpp"
#include "hmb_test.hpp"

#include <atomic>
#include <cfloat>
#include <string>

using namespace hmb;
using namespace hmb::test;

namespace {

// Time of one single-vector product: median of 3 batches of 10, after a warm-up.
double time_matvec(const BlackBox &A) {
    std::vector<double> x = randvec(A.n, 11), y(A.n);
    A.apply('N', 1, x.data(), y.data());
    std::vector<double> t;
    for (int r = 0; r < 3; r++) {
        const double t0 = now();
        for (int i = 0; i < 10; i++) A.apply('N', 1, x.data(), y.data());
        t.push_back((now() - t0) / 10);
    }
    return median(t);
}

double norm(const std::vector<double> &v) {
    double s = 0;
    for (double t : v) s += t * t;
    return std::sqrt(s);
}

std::vector<double> apply(const BlackBox &A, const std::vector<double> &X, char trans = 'N') {
    std::vector<double> Y(X.size());
    A.apply(trans, int(X.size() / A.n), X.data(), Y.data());
    return Y;
}

// (S + S^T) / 2, from S's products with 'N' and 'T': symmetric to rounding.
BlackBox symmetric_part(const BlackBox &S) {
    BlackBox b;
    b.n = S.n;
    b.symmetric = true;
    b.apply = [S](char, int k, const double *X, double *Y) {
        std::vector<double> T(size_t(S.n) * k);
        S.apply('N', k, X, Y);
        S.apply('T', k, X, T.data());
        for (size_t i = 0; i < T.size(); i++) Y[i] = 0.5 * (Y[i] + T[i]);
    };
    return b;
}

struct Hop {
    std::string what, moved;
    double seconds = 0, memory_mb = 0, vs_previous = -1, vs_kernel = -1;
};

} // namespace

int main(int argc, char **argv) {
    Checker C("chain_four_libraries");
    const int n = arg_int(argc, argv, "--n", 4000);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6), tol = arg_dbl(argc, argv, "--tol", 1e-6);
    const bool assume_symmetric = arg_flag(argc, argv, "--assume-symmetric");
    const double shift = 10, round_off = 512 * DBL_EPSILON;
    auto env = [](const char *k) { return std::getenv(k) ? std::getenv(k) : "(unset)"; };
    std::printf("== chain_four_libraries n=%d eps=%g tol=%g shift=%g%s OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s\n", n,
                eps, tol, shift, assume_symmetric ? " assume_symmetric" : "", env("OMP_NUM_THREADS"),
                env("OPENBLAS_NUM_THREADS"));
    hmb_h2lib_init(0); // H2Lib's own loops serial, as in the study

    Problem P(n, Kernel::Sym, Geometry::RandomSphere, shift);
    const BlackBox K = P.exact_blackbox();
    const int kv = 4; // reference vectors
    const std::vector<double> X = randvec(size_t(n) * kv, 7), KX = apply(K, X);
    std::vector<Hop> hops;
    char buf[160];

    // ---- 1. Htool-DDM: H-matrix from the kernel's entries ---------------------------------
    std::atomic<long long> entries{0};
    EntrySource E = P.entries();
    E.get = [inner = E.get, &entries](int M, int N, const int *rows, const int *cols, double *out) {
        entries += (long long)M * N;
        inner(M, N, rows, cols, out);
    };
    HtoolOptions ho;
    ho.epsilon = eps; // full storage ('N'): the export then aliases every leaf
    HtoolStats hs;
    double t0 = now();
    const HtoolHandle A1 = htool_build(E, 3, P.xyz.data(), ho, &hs);
    Hop h1{"Htool-DDM: ACA from the kernel's entries"};
    h1.seconds = now() - t0;
    const BlackBox H1 = htool_blackbox(A1);
    const double t_mv_start = time_matvec(H1);
    const std::vector<double> Y1 = apply(H1, X);
    std::snprintf(buf, sizeof buf, "%lld entries (%.2f N^2)", entries.load(), entries / (double(n) * n));
    h1.moved = buf;
    h1.memory_mb = hs.stored_values * 8e-6;
    h1.vs_kernel = relerr(Y1, KX);
    hops.push_back(h1);
    std::printf("[1] Htool: %.3fs, %lld stored values, %d dense + %d low-rank leaves, rank <= %d; one matvec %.5fs\n",
                h1.seconds, hs.stored_values, hs.n_dense_leaves, hs.n_lowrank_leaves, hs.max_rank, t_mv_start);

    // ---- 2. block copy into H2Lib ------------------------------------------------------------
    HtoolExportInfo xi;
    hmb_h2lib_import_info ii;
    t0 = now();
    std::shared_ptr<const hmb_hmatrix> view = htool_export(A1, &xi);
    const H2libHandle A2 = h2lib_import(*view, HMB_H2LIB_STRICT, &ii);
    Hop h2{"block copy Htool -> H2Lib (strict)"};
    h2.seconds = now() - t0;
    long viol[2];
    hmb_strict_violations(view.get(), viol);
    view.reset(); // the import copied everything
    const BlackBox H2 = h2lib_blackbox(A2);
    const std::vector<double> Y2 = apply(H2, X);
    std::snprintf(buf, sizeof buf, "%zu values copied (%.2f x Htool's)", ii.copied, ii.copied / double(hs.stored_values));
    h2.moved = buf;
    h2.memory_mb = hmb_h2lib_bytes(hmb_h2lib_hmatrix(A2.get())) * 1e-6;
    h2.vs_previous = relerr(Y2, Y1);
    h2.vs_kernel = relerr(Y2, KX);
    hops.push_back(h2);
    std::printf("[2] export: %lld values aliased, %lld copied; strictness violations of the view %ld + %ld; H2Lib: %ld "
                "dense + %ld low-rank leaves\n",
                xi.aliased_values, xi.copied_values, viol[0], viol[1], ii.dense_leaves, ii.lowrank_leaves);

    // ---- 3. H -> H2 inside H2Lib -----------------------------------------------------------------
    hmb_h2lib_h2_info h2i;
    t0 = now();
    const H2libH2Handle A3 = h2lib_to_h2(A2, eps, &h2i);
    Hop h3{"H -> H2 inside H2Lib"};
    h3.seconds = now() - t0;
    const BlackBox H3 = h2lib_h2_blackbox(A3);
    const std::vector<double> Y3 = apply(H3, X);
    std::snprintf(buf, sizeof buf, "- (basis ranks <= %u)", std::max(h2i.rank_max_row, h2i.rank_max_col));
    h3.moved = buf;
    h3.memory_mb = h2i.bytes * 1e-6;
    h3.vs_previous = relerr(Y3, Y2);
    h3.vs_kernel = relerr(Y3, KX);
    hops.push_back(h3);
    std::printf("[3] H2: bases %.3fs + %.3fs, projection %.3fs; %.1f MB (near %.1f, far %.1f); one matvec %.5fs\n",
                h2i.rowbasis_s, h2i.colbasis_s, h2i.projection_s, h2i.bytes * 1e-6, h2i.near_bytes * 1e-6,
                h2i.far_bytes * 1e-6, time_matvec(H3));

    // ---- 4. ButterflyPACK HODLR from the H2 matrix's products ----------------------------------
    BpackOptions bo;
    bo.format = BpackFormat::HODLR;
    bo.tolerance = tol;
    BpackStats bs;
    long h2_T_calls = 0; // H2 products with 'T' (the H2 BlackBox is not flagged symmetric)
    BlackBox H3counted = H3;
    H3counted.apply = [&H3, &h2_T_calls](char t, int k, const double *Xi, double *Yi) {
        if (t == 'T' || t == 't') h2_T_calls++;
        H3.apply(t, k, Xi, Yi);
    };
    t0 = now();
    const BpackHandle A4 = bpack_from_matvec(H3counted, 3, P.xyz.data(), bo, &bs);
    Hop h4{"matvecs H2Lib H2 -> ButterflyPACK HODLR"};
    h4.seconds = now() - t0;
    const BlackBox B4 = bpack_blackbox(A4);
    const std::vector<double> Y4 = apply(B4, X);
    std::snprintf(buf, sizeof buf, "%lld vectors (%lld A^T), %ld calls", bs.vectors, bs.vectors_transposed, bs.calls);
    h4.moved = buf;
    h4.memory_mb = bs.bpack_mem_fill_mb;
    h4.vs_previous = relerr(Y4, Y3);
    h4.vs_kernel = relerr(Y4, KX);
    hops.push_back(h4);
    std::printf("[4] HODLR: init %.3fs, construction %.3fs (in H2Lib's products %.3fs), rank %d; check_error %.2e\n",
                bs.seconds_init, bs.seconds_build, bs.seconds_in_source, bs.rank_max, bs.check_error);

    // ---- 5. ButterflyPACK factorization ---------------------------------------------------------
    BpackFactorStats fs;
    t0 = now();
    bpack_factor(A4, &fs);
    Hop h5{"ButterflyPACK factorization (HODLR)"};
    h5.seconds = now() - t0;
    const BlackBox S5 = bpack_solve_blackbox(A4);
    const std::vector<double> x = randvec(size_t(n) * kv, 13), y5 = apply(S5, x);
    h5.moved = "-";
    h5.memory_mb = fs.bpack_mem_factor_mb;
    h5.vs_previous = relerr(apply(B4, y5), x); // residual with the HODLR
    h5.vs_kernel = relerr(apply(K, y5), x);
    hops.push_back(h5);
    // a: how much an inverse's residual amplifies a forward error (see the header)
    const double a = norm(KX) / norm(X) * norm(y5) / norm(x);
    // The solve is not symmetric (see the header); HODLR has a transposed solve, so the
    // asymmetry is measured on products directly.
    const double asym = relerr(apply(S5, x, 'T'), y5);
    std::printf("[5] factorization: rank %d, %.1f MB; asymmetry ||(S - S^T) x|| / ||S x|| = %.2e; amplification a = "
                "%.2f\n",
                fs.rank_max, fs.bpack_mem_factor_mb, asym, a);

    // ---- 6. H2Opus H^2 of the inverse from ButterflyPACK's solves ----------------------------------
    // GOTCHA(inverse-asymmetry): HARA samples only T X and builds a symmetric H^2 (one basis
    // tree), so T must be symmetric: T = (S + S^T) / 2 by default, S itself with
    // --assume-symmetric (see the header).
    const BlackBox T6 = assume_symmetric ? S5 : symmetric_part(S5);
    const std::vector<double> yT = assume_symmetric ? y5 : apply(T6, x);
    const double rT = assume_symmetric ? h5.vs_kernel : relerr(apply(K, yT), x); // T's residual vs K
    H2opusOptions so;
    so.tolerance = tol;
    so.assume_symmetric = assume_symmetric;
    H2opusStats ss;
    t0 = now();
    const H2opusHandle A6 = h2opus_from_matvec(T6, 3, P.xyz.data(), so, &ss);
    Hop h6{assume_symmetric ? "solves -> H2Opus H^2 (HARA, assume_symmetric)" : "solves -> H2Opus H^2 (HARA on (S+S^T)/2)"};
    h6.seconds = now() - t0;
    const BlackBox M6 = h2opus_blackbox(A6);
    const std::vector<double> y6 = apply(M6, x);
    const double e6T = relerr(y6, yT); // HARA's error against what it sampled
    std::snprintf(buf, sizeof buf, "%lld vectors (%ld for the norm), %ld calls", ss.vectors, ss.norm_calls, ss.calls);
    h6.moved = buf;
    h6.memory_mb = ss.memory_mb;
    h6.vs_previous = relerr(y6, y5); // H2Opus's A^-1 against ButterflyPACK's solve
    h6.vs_kernel = relerr(apply(K, y6), x);
    hops.push_back(h6);
    std::printf("[6] HARA: norm estimate %.4e (absolute tolerance %.2e), structure %.3fs, norm %.3fs, hara %.3fs (in "
                "ButterflyPACK's solves %.3fs); depth %d, rank <= %d; error against the sampled operator %.2e; "
                "residual vs K of the sampled operator %.2e\n",
                ss.norm_estimate, ss.abs_tolerance, ss.seconds_structure, ss.seconds_norm, ss.seconds_hara,
                ss.seconds_in_source, ss.depth, ss.max_rank, e6T, rT);

    // ---- report ----------------------------------------------------------------------------------
    const double t_mv_end = time_matvec(H1), t_mv = 0.5 * (t_mv_start + t_mv_end);
    const double t_apply_M = time_matvec(M6), t_apply_S = time_matvec(S5);
    double total = 0;
    std::printf("\nsource matvec (one vector, Htool) %.5fs (start %.5fs, end %.5fs)\n", t_mv, t_mv_start, t_mv_end);
    std::printf("%-2s %-46s %8s %8s %-40s %7s %9s %9s\n", "", "hop", "seconds", "x mv", "into the hop", "MB", "vs prev",
                "vs kernel");
    for (size_t i = 0; i < hops.size(); i++) {
        const Hop &h = hops[i];
        total += h.seconds;
        char prev[16] = "-";
        if (h.vs_previous >= 0) std::snprintf(prev, sizeof prev, "%.2e", h.vs_previous);
        std::printf("%-2zu %-46s %8.3f %8.1f %-40s %7.1f %9s %9.2e\n", i + 1, h.what.c_str(), h.seconds, h.seconds / t_mv,
                    h.moved.c_str(), h.memory_mb, prev, h.vs_kernel);
    }
    std::printf("(hops 5 and 6 hold A^-1: 'vs kernel' is the residual ||K (Y x) - x|| / ||x||; 'vs prev' the residual "
                "with the HODLR (5) and the error against ButterflyPACK's solve (6))\n");
    std::printf("chain %.2fs = %.0f source matvecs; apply A^-1 to one vector: H2Opus %.5fs = %.1f source matvecs, "
                "ButterflyPACK %.5fs = %.1f\n",
                total, total / t_mv, t_apply_M, t_apply_M / t_mv, t_apply_S, t_apply_S / t_mv);
    std::printf("\nRESULT n=%d eps=%g tol=%g sampled=%s final_residual=%.3e solve_residual=%.3e sampled_residual=%.3e "
                "asymmetry=%.2e amplification=%.2f hara_rank=%d chain_seconds=%.2f chain_cost=%.0f t_src_mv1=%.5f\n\n",
                n, eps, tol, assume_symmetric ? "S" : "(S+S^T)/2", h6.vs_kernel, h5.vs_kernel, rT, asym, a, ss.max_rank,
                total, total / t_mv, t_mv);

    // ---- checks ------------------------------------------------------------------------------------
    C.check_le(h1.vs_kernel, 10 * eps, "1 Htool vs kernel <= 10 eps");
    C.check(xi.copied_values == 0, "2 export is zero-copy", "%lld values copied", xi.copied_values);
    C.check_le(h2.vs_previous, round_off, "2 H2Lib copy vs Htool: lossless");
    C.check_le(h3.vs_previous, 10 * eps, "3 H2 vs H <= 10 eps");
    C.check_le(h4.vs_previous, 20 * tol, "4 HODLR vs H2 <= 20 tol");
    C.check(bs.calls_transposed > 0 && h2_T_calls == bs.calls_transposed, "4 A^T requests answered by H2Lib's 'T'",
            "%ld requests, %ld 'T' products", bs.calls_transposed, h2_T_calls);
    C.check_le(h5.vs_previous, 10 * tol, "5 solve: residual vs HODLR <= 10 tol");
    C.check_le(h5.vs_kernel, 10 * (a * h4.vs_kernel + h5.vs_previous), "5 solve: residual vs K <= 10 (a e4 + r_B)");
    C.check_le(asym, 5 * (a * h4.vs_kernel + tol), "5 asymmetry of the solve <= 5 (a e4 + tol)");
    if (assume_symmetric)
        C.check_le(e6T, 20 * tol + 2 * asym, "6 H2Opus vs S <= 20 tol + 2 asymmetry");
    else
        C.check_le(e6T, 20 * tol, "6 H2Opus vs (S + S^T) / 2 <= 20 tol");
    C.check_le(h6.vs_kernel, 10 * (rT + a * e6T), "6 final residual <= 10 (r_T + a e6)");
    return C.finish("n=%d eps=%g tol=%g: final residual %.2e", n, eps, tol, h6.vs_kernel);
}
