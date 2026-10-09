// Rebuild from entries: hmat-oss -> Htool-DDM (the study's experiment E3, through the package).
//
// An hmat-oss H-matrix (the source) is exposed as an EntrySource over get_values
// (hmb::hmatoss_entries) and recompressed by Htool-DDM's ACA (hmb::htool_build). No matvec of
// the source is used. Reported: entries requested against values stored in the source, time
// against Htool's direct build from the kernel and in source matvecs, accuracy against the
// source and the kernel, memory, and the effect of the destination tolerance
// (GOTCHA(tolerance-stacking)).
//
// Usage: entries_hmatoss_htool [--n 4000] [--eps-src 1e-6] [--eps-dst 1e-6] [--reps 3]
//        [--leaf 64] [--eta-hmat 2] [--eta-htool 10] [--no-tolerance]
// E3's runs: --n 5000 and --n 20000 (other options at their defaults), OMP_NUM_THREADS=2.
#include "hmb_hmatoss.hpp"
#include "hmb_htool.hpp"
#include "hmb_test.hpp"

#include <atomic>
#include <cstring>
#include <omp.h>

using namespace hmb;
using namespace hmb::test;

// Counts what a destination asks of a source; thread-safe if the source is.
struct Counter {
    std::atomic<long long> calls_rc{0}, calls_blk{0}, ent_rc{0}, ent_blk{0}, ns_rc{0}, ns_blk{0};
    std::atomic<int> inflight{0}, maxinflight{0};
    long long entries() const { return ent_rc + ent_blk; }
};
static EntrySource counted(const EntrySource &src, Counter &c) {
    EntrySource e = src;
    e.get = [src, &c](int M, int N, const int *rows, const int *cols, double *out) {
        const int k = ++c.inflight;
        for (int m = c.maxinflight.load(); k > m && !c.maxinflight.compare_exchange_weak(m, k);) {
        }
        const double t0 = now();
        src.get(M, N, rows, cols, out);
        const long long ns = (long long)(1e9 * (now() - t0));
        --c.inflight;
        if (M == 1 || N == 1) c.calls_rc++, c.ent_rc += (long long)M * N, c.ns_rc += ns;
        else c.calls_blk++, c.ent_blk += (long long)M * N, c.ns_blk += ns;
    };
    return e;
}

static double time_matvec(const BlackBox &A, int n) {
    std::vector<double> x = randvec(n, 1), y(n);
    std::vector<double> t;
    for (int r = 0; r < 3; r++) {
        const double t0 = now();
        A.apply('N', 1, x.data(), y.data());
        t.push_back(now() - t0);
    }
    return median(t);
}

// Y = op(A) X for nv random vectors (fixed seed), one thread (deterministic).
static std::vector<double> apply(const BlackBox &A, int nv) {
    const int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    std::vector<double> X = randvec((size_t)A.n * nv, 7), Y((size_t)A.n * nv);
    A.apply('N', nv, X.data(), Y.data());
    omp_set_num_threads(nt);
    return Y;
}

int main(int argc, char **argv) {
    const int n = arg_int(argc, argv, "--n", 4000);
    const double eps_src = arg_dbl(argc, argv, "--eps-src", 1e-6), eps_dst = arg_dbl(argc, argv, "--eps-dst", 1e-6);
    const int reps = arg_int(argc, argv, "--reps", 3), leaf = arg_int(argc, argv, "--leaf", 64);
    const double eta_hmat = arg_dbl(argc, argv, "--eta-hmat", 2.0), eta_htool = arg_dbl(argc, argv, "--eta-htool", 10.0);
    const bool tolerance = !arg_flag(argc, argv, "--no-tolerance");
    const int maxthr = omp_get_max_threads(), nv = 3;
    Checker C("entries_hmatoss_htool");
    std::printf("== N=%d eps_src=%g eps_dst=%g reps=%d leaf=%d eta_hmat=%g eta_htool=%g max_threads=%d\n", n, eps_src, eps_dst, reps,
                leaf, eta_hmat, eta_htool, maxthr);

    // Non-symmetric kernel on a Fibonacci sphere whose user numbering is shuffled (E3's problem).
    Problem P(n, Kernel::Nonsym, Geometry::ShuffledFibonacci);
    std::vector<double> y_exact((size_t)n * nv);
    {
        const std::vector<double> X = randvec((size_t)n * nv, 7); // the vectors apply() uses
        P.exact_apply('N', nv, X.data(), y_exact.data());
    }

    // ---- source: hmat-oss, full storage, ACA+ ------------------------------------------------
    HmatossOptions so;
    so.epsilon = eps_src;
    so.eta = eta_hmat;
    so.leaf_size = leaf;
    HmatossStats ss;
    HmatossHandle S = hmatoss_build(P.entries(), 3, P.xyz.data(), so, &ss);
    const double src_stored = (double)ss.stored_values;
    BlackBox src_mv = hmatoss_blackbox(S);
    const double t_mv = time_matvec(src_mv, n);
    std::printf("[src] hmat-oss build (sequential) %.3f s; stored values %lld (%.2f MB, %.2f%% of N^2); %lld low-rank + %lld dense "
                "leaves; matvec %.4f s\n",
                ss.build_seconds, ss.stored_values, 8e-6 * src_stored, 100. * src_stored / ((double)n * n), ss.n_lowrank_leaves,
                ss.n_dense_leaves, t_mv);
    EntrySource src = hmatoss_entries(S); // GOTCHA(index-base): the adapter adds the +1

    HtoolOptions ho;
    ho.epsilon = eps_dst;
    ho.eta = eta_htool;
    ho.leaf_size = leaf;
    ho.parallel = true; // Htool's OpenMP build; the source must be (and is) thread-safe

    // ---- time: direct build from the kernel vs transfer from the source -----------------------
    std::vector<int> thr_list = {1};
    if (maxthr > 1) thr_list.push_back(maxthr);
    double transfer_matvecs = 0;
    for (int nt : thr_list) {
        omp_set_num_threads(nt);
        std::vector<double> tm, td, tt;
        for (int r = 0; r < reps; r++) {
            tm.push_back(time_matvec(src_mv, n)); // one source matvec, timed next to the builds
            HtoolStats st;
            htool_build(P.entries(), 3, P.xyz.data(), ho, &st);
            td.push_back(st.build_seconds);
            htool_build(src, 3, P.xyz.data(), ho, &st);
            tt.push_back(st.build_seconds);
        }
        const double m = median(tm), d = median(td), t = median(tt);
        if (nt == 1) transfer_matvecs = t / m;
        std::printf("[time] threads=%d: src matvec %.4f s | direct %.3f s = %.1f matvecs | transfer %.3f s = %.1f matvecs = %.2fx "
                    "direct = %.2fx src build\n",
                    nt, m, d, d / m, t, t / m, t / d, t / ss.build_seconds);
    }
    omp_set_num_threads(maxthr);

    // ---- counts, memory, accuracy (one instrumented build of each) ----------------------------
    Counter cd, ct;
    HtoolStats sd, st;
    HtoolHandle D = htool_build(counted(P.entries(), cd), 3, P.xyz.data(), ho, &sd);
    reset_hwm();
    const long rss0 = rss_kb();
    HtoolHandle T = htool_build(counted(src, ct), 3, P.xyz.data(), ho, &st);
    const long rss1 = rss_kb(), hwm = hwm_kb();
    for (int w = 0; w < 2; w++) {
        const Counter &c = w ? ct : cd;
        const HtoolStats &s = w ? st : sd;
        std::printf("[count] %-8s calls row/col %lld, block %lld | entries %lld (%.2f%% of N^2, %.3fx src stored, %.3fx dst stored) "
                    "| ns per entry row/col %.1f, block %.1f | up to %d concurrent calls\n",
                    w ? "transfer" : "direct", c.calls_rc.load(), c.calls_blk.load(), c.entries(),
                    100. * c.entries() / ((double)n * n), c.entries() / src_stored, c.entries() / (double)s.stored_values,
                    (double)c.ns_rc / std::max(1LL, c.ent_rc.load()), (double)c.ns_blk / std::max(1LL, c.ent_blk.load()),
                    c.maxinflight.load());
        std::printf("[dst] %-8s %d low-rank + %d dense leaves, max rank %d, stored %lld (%.3fx src)\n", w ? "transfer" : "direct",
                    s.n_lowrank_leaves, s.n_dense_leaves, s.max_rank, s.stored_values, s.stored_values / src_stored);
    }
    std::printf("[mem] transfer build: RSS +%ld kB, peak +%ld kB; destination %.0f kB, source %.0f kB\n", rss1 - rss0, hwm - rss0,
                8e-3 * st.stored_values, 8e-3 * src_stored);
    C.check(ct.entries() < (long long)n * n, "transfer requests fewer than N^2 entries", "%.1f%% of N^2",
            100. * ct.entries() / ((double)n * n));
    C.check(ct.maxinflight > 1 || maxthr == 1, "source served concurrent requests", "up to %d", ct.maxinflight.load());

    const std::vector<double> y_src = apply(src_mv, nv), y_t = apply(htool_blackbox(T), nv), y_d = apply(htool_blackbox(D), nv);
    const double e_ts = relerr(y_t, y_src), e_tx = relerr(y_t, y_exact), e_sx = relerr(y_src, y_exact), e_dx = relerr(y_d, y_exact);
    std::printf("[acc] ||y_a - y_b|| / ||y_b||, %d random vectors: transfer vs src %.2e | transfer vs exact %.2e | src vs exact %.2e | "
                "direct vs exact %.2e | transfer vs direct %.2e\n",
                nv, e_ts, e_tx, e_sx, e_dx, relerr(y_t, y_d));
    C.check_le(e_ts, 10 * eps_dst, "transfer vs source <= 10 eps_dst");
    C.check_le(e_tx, 10 * eps_dst, "transfer vs kernel <= 10 eps_dst");
    C.check_le(e_sx, 10 * eps_src, "source vs kernel <= 10 eps_src");
    C.check_le(e_dx, 10 * eps_dst, "direct vs kernel <= 10 eps_dst");
    {
        // concurrent get_values: the multi-threaded transfer equals a one-thread transfer
        omp_set_num_threads(1);
        HtoolHandle T1 = htool_build(src, 3, P.xyz.data(), ho);
        omp_set_num_threads(maxthr);
        const std::vector<double> y1 = apply(htool_blackbox(T1), nv);
        C.check(std::memcmp(y1.data(), y_t.data(), sizeof(double) * y1.size()) == 0, "1-thread and multi-thread transfer identical",
                "%d threads", maxthr);
    }

    // ---- GOTCHA(tolerance-stacking): the rebuilt matrix carries both tolerances ---------------
    // The rebuild's error against the kernel combines the source's error and the destination's
    // recompression error. Three settings: the same tolerance on both sides; a source 100x
    // tighter than the destination (the advice: destination 10-100x looser than the source);
    // a destination 100x tighter than the source (it then compresses the source's error).
    if (tolerance) {
        struct Row {
            double es, ed, err;
            long long entries, stored;
            double seconds;
        };
        auto transfer = [&](const EntrySource &s, double es, double ed) {
            HtoolOptions o = ho;
            o.epsilon = ed;
            Counter c;
            HtoolStats hs;
            HtoolHandle H = htool_build(counted(s, c), 3, P.xyz.data(), o, &hs);
            return Row{es, ed, relerr(apply(htool_blackbox(H), nv), y_exact), c.entries(), hs.stored_values, hs.build_seconds};
        };
        HmatossOptions tight = so;
        tight.epsilon = eps_src / 100;
        HmatossHandle S2 = hmatoss_build(P.entries(), 3, P.xyz.data(), tight);
        const Row same{eps_src, eps_dst, e_tx, ct.entries(), st.stored_values, st.build_seconds};
        const Row looser = transfer(hmatoss_entries(S2), tight.epsilon, eps_dst);
        const Row tighter = transfer(src, eps_src, eps_src / 100);
        const Row direct{0, eps_dst, e_dx, cd.entries(), sd.stored_values, sd.build_seconds};
        std::printf("[tol] eps_src -> eps_dst: error vs kernel | entries requested (x src stored) | dst stored | build time\n");
        const char *label[] = {"same tolerance", "source 100x tighter", "destination 100x tighter", "direct build from the kernel"};
        const Row *rows[] = {&same, &looser, &tighter, &direct};
        for (int k = 0; k < 4; k++) {
            char es[16] = "kernel";
            if (rows[k]->es > 0) std::snprintf(es, sizeof es, "%.0e", rows[k]->es);
            std::printf("[tol] %6s -> %.0e: %.2e | %lld (%.2fx) | %lld | %.3f s   (%s)\n", es, rows[k]->ed, rows[k]->err,
                        rows[k]->entries, rows[k]->entries / src_stored, rows[k]->stored, rows[k]->seconds, label[k]);
        }
        C.check(same.err > e_sx, "same tolerance: less accurate than the source", "%.2e vs source %.2e (%.2fx)", same.err, e_sx,
                same.err / e_sx);
        C.check(std::fabs(looser.err - e_dx) <= 0.5 * e_dx && std::fabs((double)looser.entries / direct.entries - 1) <= 0.05,
                "source 100x tighter: same as the direct build", "error %.2e vs %.2e, entries %.3fx", looser.err, e_dx,
                (double)looser.entries / direct.entries);
        C.check(tighter.entries > 1.5 * same.entries && tighter.err > 0.5 * e_sx, "destination 100x tighter: dearer, no gain",
                "%.2fx the entries, error %.2e (source %.2e)", (double)tighter.entries / same.entries, tighter.err, e_sx);
    }

    // ---- lifetime: the rebuilt matrix does not need the source ------------------------------
    std::weak_ptr<HmatossMatrix> alive = S;
    src = EntrySource{};
    src_mv = BlackBox{};
    S.reset();
    const std::vector<double> y_after = apply(htool_blackbox(T), nv);
    C.check(alive.expired() && std::memcmp(y_after.data(), y_t.data(), sizeof(double) * y_t.size()) == 0,
            "transfer independent of the source", "hmat-oss matrix destroyed: %s", alive.expired() ? "yes" : "no");
    return C.finish("N=%d transfer %.1f source matvecs, error %.2e vs source", n, transfer_matvecs, e_ts);
}
