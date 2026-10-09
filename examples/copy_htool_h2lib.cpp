// Example: Htool-DDM -> H2Lib block copy, then H -> H2 inside H2Lib (the study's route E1).
//
// Builds an Htool H-matrix of K(x, y) = 1 / (0.1 + |x - y|) + shift * [x == y] on N random
// points of the unit sphere, exports it to the neutral view (zero-copy), imports it into H2Lib
// (copy, borrow and strict modes), checks the products, converts it to H2 inside H2Lib and
// checks again. Costs are in units of one source matvec: the median time of one sequential
// Htool matrix-vector product (Htool's add_hmatrix_vector_product, the unit of the study),
// timed at the start and at the end of the run (their mean: the product is memory-bound and
// sensitive to other load on the machine). Values copied, memory, and the accuracy against
// Htool and against the kernel are printed next to them.
//
//   copy_htool_h2lib [--n 4000] [--eps 1e-6] [--eta 10] [--leaf 64] [--inconsistent]
//                    [--mode all|copy|borrow|strict] [--h2eps EPS] [--shift 10]
//                    [--reps 3] [--mvreps 7] [--pardepth 0]
//
// Study runs (results of E1): defaults at --n 4000 and --n 20000; --leaf 10 (Htool's ACA
// falls back to dense blocks above the leaf level); --inconsistent (one-sided splits). With
// a non-strict tree the conversion runs on the strict import, and the conversion of the copy
// import must be refused (GOTCHA(strict-tree)). Everything timed runs on one thread: Htool's
// sequential build and product, H2Lib with --pardepth 0, as in the study; OMP_NUM_THREADS
// only speeds up the exact reference products. Run with one BLAS thread, as the study did
// (OPENBLAS_NUM_THREADS=1, which tests/run_tests.sh sets).
#include "hmb/neutral_bridge.hpp"
#include "hmb_h2lib.hpp"
#include "hmb_htool_copy.hpp"
#include "hmb_test.hpp"
#include "htool_native.hpp" // Htool's own sequential product, the cost unit (no H2Lib header here)

#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>

#include <cfloat>
#include <functional>
#include <string>
#ifdef __GLIBC__
#include <malloc.h>
#endif

using namespace hmb;
using namespace hmb::test;

namespace {

double median_time(int reps, const std::function<void()> &f) {
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        const double t0 = now();
        f();
        t.push_back(now() - t0);
    }
    return median(t);
}

long peak_kb = 0; // process peak RSS across the resets below

// RSS before a measurement, with freed heap memory returned to the system first.
long settled_rss_kb() {
    peak_kb = std::max(peak_kb, hwm_kb());
#ifdef __GLIBC__
    malloc_trim(0);
#endif
    reset_hwm();
    return rss_kb();
}

// Products of the source and of the kernel with two fixed random vectors, computed once.
struct Reference {
    int n;
    std::vector<double> X, Ysource, Yexact;
    Reference(int n_, const BlackBox &source, const BlackBox &exact)
        : n(n_), X(randvec(size_t(n_) * 2, 5)), Ysource(X.size()), Yexact(X.size()) {
        source.apply('N', 2, X.data(), Ysource.data());
        exact.apply('N', 2, X.data(), Yexact.data());
    }
    // relative errors of A against the source and against the kernel
    std::pair<double, double> errors(const BlackBox &A) const {
        std::vector<double> Y(X.size());
        A.apply('N', 2, X.data(), Y.data());
        return {relerr(Y, Ysource), relerr(Y, Yexact)};
    }
};

} // namespace

int main(int argc, char **argv) {
    Checker C("copy_htool_h2lib");
    const int n = arg_int(argc, argv, "--n", 4000), leaf = arg_int(argc, argv, "--leaf", 64);
    const int reps = arg_int(argc, argv, "--reps", 3), mvreps = arg_int(argc, argv, "--mvreps", 7);
    const int pardepth = arg_int(argc, argv, "--pardepth", 0);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6), eta = arg_dbl(argc, argv, "--eta", 10);
    const double h2eps = arg_dbl(argc, argv, "--h2eps", eps), shift = arg_dbl(argc, argv, "--shift", 10);
    const bool inconsistent = arg_flag(argc, argv, "--inconsistent");
    const std::string mode_arg = arg_str(argc, argv, "--mode", "all");
    const double kRound = 512 * DBL_EPSILON; // lossless transfers agree at rounding level
    auto env = [](const char *name) { return std::getenv(name) ? std::getenv(name) : "(unset)"; };
    std::printf("# N=%d eps=%g eta=%g leaf=%d consistent_tree=%d h2eps=%g shift=%g reps=%d mvreps=%d pardepth=%d "
                "OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s\n",
                n, eps, eta, leaf, !inconsistent, h2eps, shift, reps, mvreps, pardepth, env("OMP_NUM_THREADS"),
                env("OPENBLAS_NUM_THREADS"));
    hmb_h2lib_init(pardepth);

    // ---- source: Htool from the kernel
    Problem P(n, Kernel::Sym, Geometry::RandomSphere, shift);
    HtoolOptions o;
    o.epsilon = eps, o.eta = eta, o.leaf_size = leaf, o.block_tree_consistency = !inconsistent;
    HtoolHandle A;
    HtoolStats st;
    const double t_build = median_time(reps, [&] { A = htool_build(P.entries(), 3, P.xyz.data(), o, &st); });
    std::vector<double> x = randvec(n, 3), y(n);
    // the cost unit, timed now and again at the end (the machine's load may change in between)
    auto source_matvec = [&] {
        return median_time(mvreps, [&] {
            htool::add_hmatrix_vector_product(exec_compat::seq, 'N', 1.0, *A->A, x.data(), 0.0, y.data());
        });
    };
    const double t_mv = source_matvec();
    const BlackBox source = htool_blackbox(A);
    const Reference ref(n, source, P.exact_blackbox());
    std::printf("htool_build_s = %.4f\nhtool_matvec_seq_s = %.6f\nhtool_stored_values = %lld (%.2f N^2)\n"
                "htool_leaves dense=%d lowrank=%d rank_max=%d\nhtool_err_vs_kernel = %.3e\n",
                t_build, t_mv, st.stored_values, st.stored_values / (double(n) * n), st.n_dense_leaves,
                st.n_lowrank_leaves, st.max_rank, relerr(ref.Ysource, ref.Yexact));

    // ---- export (zero-copy)
    std::shared_ptr<const hmb_hmatrix> V; // keeps A alive
    HtoolExportInfo xi;
    const double t_exp = median_time(reps, [&] { V = htool_export(A, &xi); });
    const hmb_hmatrix &view = *V;
    char msg[512] = "";
    C.check(hmb_validate(&view, msg, sizeof msg) == 0, "exported view validates", "%s", msg);
    long viol[2];
    hmb_strict_violations(&view, viol);
    const bool strict = viol[0] + viol[1] == 0;
    const long long stored = (long long)hmb_stored_values(&view);
    std::printf("export_s = %.6f\nview clusters = %d blocks = %d\nview_values = %lld (aliased %lld, copied %lld)\n"
                "strict_violations dense_nonleaf = %ld unsplit_nonleaf = %ld\n",
                t_exp, view.ncl, view.nbl, stored, xi.aliased_values, xi.copied_values, viol[0], viol[1]);
    std::vector<std::pair<std::string, double>> costs{{"export", t_exp}};
    C.check(xi.copied_values == 0, "export is zero-copy", "%lld values copied", xi.copied_values);

    // ---- import into H2Lib
    struct Mode {
        const char *name;
        int flags;
    };
    const Mode modes[] = {{"copy", HMB_H2LIB_COPY}, {"borrow", HMB_H2LIB_BORROW}, {"strict", HMB_H2LIB_STRICT}};
    H2libHandle for_h2, copy_import;
    for (const Mode &m : modes) {
        if (mode_arg != "all" && mode_arg != m.name) continue;
        H2libHandle M;
        hmb_h2lib_import_info info;
        const double t_imp = median_time(reps, [&] {
            M.reset();
            M = h2lib_import(view, m.flags, &info, V);
        });
        M.reset();
        const long rss0 = settled_rss_kb();
        M = h2lib_import(view, m.flags, &info, V);
        const long rss1 = rss_kb(), hwm = hwm_kb();
        const BlackBox dst = h2lib_blackbox(M);
        std::vector<double> yy(n);
        const double t_mv2 = median_time(mvreps, [&] { dst.apply('N', 1, x.data(), yy.data()); });
        const auto e = ref.errors(dst);
        const size_t bytes = hmb_h2lib_bytes(hmb_h2lib_hmatrix(M.get()));
        costs.emplace_back(std::string("export + import (") + m.name + ")", t_exp + t_imp);
        std::printf("[%s] import_s = %.6f\n"
                    "[%s] values_copied = %zu (%.3f x source), borrowed = %zu\n[%s] h2lib_bytes = %zu (%.3f x source "
                    "bytes)\n[%s] rss_extra_MB = %.1f peak_extra_MB = %.1f\n[%s] leaves dense=%ld lowrank=%ld\n"
                    "[%s] h2lib_matvec_s = %.6f\n[%s] h2lib_vs_htool = %.3e\n[%s] h2lib_vs_kernel = %.3e\n",
                    m.name, t_imp, m.name, info.copied, double(info.copied) / stored,
                    info.borrowed, m.name, bytes, bytes / (8.0 * stored), m.name, (rss1 - rss0) / 1024.0,
                    (hwm - rss0) / 1024.0, m.name, info.dense_leaves, info.lowrank_leaves, m.name, t_mv2, m.name,
                    e.first, m.name, e.second);
        char what[96];
        std::snprintf(what, sizeof what, "[%s] H2Lib vs Htool (lossless)", m.name);
        C.check_le(e.first, kRound, what);
        std::snprintf(what, sizeof what, "[%s] H2Lib vs kernel", m.name);
        C.check_le(e.second, 10 * eps, what);
        if (m.flags & HMB_H2LIB_STRICT) {
            long v[2];
            hmb_h2lib_strict_violations(hmb_h2lib_hmatrix(M.get()), v);
            C.check(v[0] + v[1] == 0, "[strict] H2Lib tree is strict", "(%ld, %ld)", v[0], v[1]);
        }
        if (m.flags == HMB_H2LIB_COPY) copy_import = M;
        if ((m.flags == HMB_H2LIB_COPY && strict) || (m.flags & HMB_H2LIB_STRICT && !strict)) for_h2 = M;
    }

    // ---- H -> H2 inside H2Lib
    if (copy_import && !strict) { // GOTCHA(strict-tree): refused instead of silently wrong
        struct _h2matrix *H2 = nullptr;
        const int rc = hmb_h2lib_to_h2(hmb_h2lib_hmatrix(copy_import.get()), h2eps, &H2, nullptr, msg, sizeof msg);
        std::printf("{copy} h2_convert refused: %s\n", msg);
        C.check(rc == HMB_H2LIB_ENOTSTRICT, "[copy] H -> H2 refuses the non-strict tree", "rc = %d", rc);
        hmb_h2lib_free_h2(H2);
    }
    copy_import.reset();
    if (for_h2) {
        const char *from = strict ? "copy" : "strict";
        H2libH2Handle H2;
        hmb_h2lib_h2_info hi{};
        const double t_conv = median_time(reps, [&] {
            H2.reset();
            H2 = h2lib_to_h2(for_h2, h2eps, &hi);
        });
        const BlackBox h2 = h2lib_h2_blackbox(H2);
        std::vector<double> yy(n);
        const double t_mvh2 = median_time(mvreps, [&] { h2.apply('N', 1, x.data(), yy.data()); });
        const auto e = ref.errors(h2);
        costs.emplace_back(std::string("H -> H2 (from the ") + from + " import)", t_conv);
        const double t_n = now();
        const double spec = hmb_h2lib_h2_reldiff(hmb_h2lib_hmatrix(for_h2.get()), H2.get());
        std::printf("{%s h2eps=%g} h2_convert_s = %.4f\n{%s h2eps=%g} h2_convert_parts_s "
                    "rowbasis=%.3f colbasis=%.3f projection=%.3f\n{%s h2eps=%g} h2_bytes = %zu (near %zu far %zu) = "
                    "%.3f x H source bytes\n{%s h2eps=%g} h2_rank_sum row=%u col=%u kmax row=%u col=%u\n"
                    "{%s h2eps=%g} h2_matvec_s = %.6f\n{%s h2eps=%g} h2_vs_htool = %.3e\n{%s h2eps=%g} h2_vs_kernel = "
                    "%.3e\n{%s h2eps=%g} h2_spectral_rel_diff_vs_H = %.3e (power iteration, %.2f s)\n",
                    from, h2eps, t_conv, from, h2eps, hi.rowbasis_s, hi.colbasis_s, hi.projection_s, from,
                    h2eps, hi.bytes, hi.near_bytes, hi.far_bytes, hi.bytes / (8.0 * stored), from, h2eps, hi.rank_sum_row,
                    hi.rank_sum_col, hi.rank_max_row, hi.rank_max_col, from, h2eps, t_mvh2, from, h2eps, e.first, from,
                    h2eps, e.second, from, h2eps, spec, now() - t_n);
        C.check_le(e.first, 10 * h2eps, "H2 vs Htool");
        C.check_le(e.second, 10 * std::max(eps, h2eps), "H2 vs kernel");
        C.check_le(spec, 10 * h2eps, "H2 spectral difference from H");
    }
    std::printf("process_peak_rss_MB = %.1f\n", std::max(peak_kb, hwm_kb()) / 1024.0);

    // ---- costs in source matvecs
    const double t_mv_end = source_matvec(), unit = 0.5 * (t_mv + t_mv_end);
    std::printf("source_matvec_s = %.6f (start %.6f, end %.6f)\n", unit, t_mv, t_mv_end);
    double copy_min = 1e300, copy_max = 0, h2 = 0;
    for (const auto &c : costs) {
        std::printf("cost %-34s = %8.2f source matvecs (%.4f s)\n", c.first.c_str(), c.second / unit, c.second);
        if (c.first.rfind("export + import", 0) == 0)
            copy_min = std::min(copy_min, c.second / unit), copy_max = std::max(copy_max, c.second / unit);
        if (c.first.rfind("H -> H2", 0) == 0) h2 = c.second / unit;
    }
    // short enough for the detail column of run_tests.sh (source matvecs)
    return C.finish("n=%d: copy %.1f-%.1f, H->H2 %.0f matvecs", n, copy_min > copy_max ? 0.0 : copy_min, copy_max, h2);
}
