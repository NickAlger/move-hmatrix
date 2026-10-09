// H2Opus adapter (HARA): h2opus_from_matvec from an exact (dense) source, then the rebuilt
// H^2 matrix's products in the original ordering. Checks accuracy against the source,
// the statistics against an independent count, that HARA never asks for A^T, the norm
// estimate behind the absolute tolerance, the permutation and the ranks (GOTCHA(perm-direction);
// with points that do not match the indices HARA's ranks reach 250 instead of about 30 and
// its error 6e-2, silently), exact symmetry of the result, the refusal of a nonsymmetric
// BlackBox (GOTCHA(symmetric-storage); forced through, HARA returns a 144% error), argument
// checks, an exception from the source, and concurrent products. Native header included for
// the permutation and the symmetry check. The Htool-source variant is test_h2opus_hara_htool.
#include "h2opus_native.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <thread>
#ifdef __GLIBC__
#include <malloc.h>
#endif

using namespace hmb;
using namespace hmb::test;

namespace {

bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

double heap_mb() { // bytes in use, after returning free pages (glibc); -1 elsewhere
#ifdef __GLIBC__
    malloc_trim(0);
    struct mallinfo2 mi = mallinfo2();
    return (mi.uordblks + mi.hblkhd) / 1048576.0;
#else
    return -1;
#endif
}

template <class E, class F> bool throws(F f, const char *prefix) {
    try {
        f();
    } catch (const E &e) {
        return starts_with(e.what(), prefix);
    }
    return false;
}

double power_norm(const BlackBox &A, int iterations) {
    std::vector<double> v = randvec(A.n, 9), w(A.n);
    double norm = 0;
    for (int it = 0; it < iterations; it++) {
        double s = 0;
        for (double t : v) s += t * t;
        s = std::sqrt(s);
        for (double &t : v) t /= s;
        A.apply('N', 1, v.data(), w.data());
        double s2 = 0;
        for (double t : w) s2 += t * t;
        norm = std::sqrt(s2);
        v.swap(w);
    }
    return norm;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_h2opus_hara");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6);
    // E2 measured 2.0-4.6e-6 against the source at 1e-6 (N = 4,000-16,000).
    const double err_bound = 20 * tol;

    // ---- symmetric kernel, random points --------------------------------------------------
    Problem P(n);
    const BlackBox D = dense_kernel_blackbox(P);
    {
        H2opusOptions o;
        o.tolerance = tol;
        auto cnt = std::make_shared<CallCounts>();
        H2opusStats st;
        const BlackBox source = counted(D, cnt);
        const double heap0 = heap_mb(), t0 = now();
        H2opusHandle h = h2opus_from_matvec(source, 3, P.xyz.data(), o, &st);
        const double t_build = now() - t0, retained = heap_mb() - heap0;
        BlackBox B = h2opus_blackbox(h);
        std::printf("HARA: %.2fs (structure %.3fs, norm %.3fs, hara %.2fs, source %.2fs), %ld calls (%ld for the norm), "
                    "%lld vectors, depth %d, max rank %d, %.1f MB\n",
                    t_build, st.seconds_structure, st.seconds_norm, st.seconds_hara, st.seconds_in_source, st.calls,
                    st.norm_calls, st.vectors, st.depth, st.max_rank, st.memory_mb);
        // GOTCHA(h2opus-workspace): without the adapter's compaction, HARA's workspace and spare
        // capacity stay allocated (203 MB for a 27 MB matrix at n = 4,000)
        if (heap0 >= 0)
            C.check_le(retained, 1.25 * st.memory_mb + 2, "retained memory close to H2Opus's count (MB)");
        const double err_N = blackbox_relerr(B, D, 'N');
        C.check_le(err_N, err_bound, "matvec 'N' against the source");
        C.check_le(blackbox_relerr(B, D, 'T'), err_bound, "matvec 'T' against the source");
        C.check(st.check_error > err_N / 5 && st.check_error < err_N * 5, "check_error agrees (within 5x)",
                "check_error %.2e, measured %.2e", st.check_error, err_N);
        C.check(st.calls + 1 == cnt->calls && st.vectors + o.check_vectors == cnt->vectors, "stats count every call",
                "adapter %ld / %lld (+1 / +%d for the check), source %ld / %lld", st.calls, st.vectors, o.check_vectors,
                cnt->calls.load(), cnt->vectors.load());
        C.check(cnt->calls_transposed == 0, "HARA never asks for A^T", "%ld 'T' calls", cnt->calls_transposed.load());
        C.check(cnt->max_in_flight == 1, "source called from one thread at a time", "max in flight %d",
                cnt->max_in_flight.load());
        C.check(st.norm_calls >= 2 && st.norm_calls <= 2 * o.norm_iterations, "norm estimate takes 2..2*norm_iterations calls",
                "%ld", st.norm_calls);
        const double ref = power_norm(D, 60);
        C.check_le(std::fabs(st.norm_estimate - ref) / ref, 1e-2, "norm estimate against 60 power iterations");
        C.check(st.abs_tolerance == tol * st.norm_estimate, "HARA's absolute tolerance = tolerance * norm estimate",
                "%.4e", st.abs_tolerance);

        bool identity = true;
        for (int i = 0; i < n; i++) identity = identity && h->new2old[i] == i;
        C.check(!identity && h->H.sym, "H2Opus permutes (new2old not the identity), symmetric storage");
        // ranks: a scrambled ordering gives 250 at n = 2000 (calibrated), a matching one about 30
        C.check(st.max_rank > 0 && st.max_rank <= o.leaf_size, "ranks show a matching ordering", "max rank %d <= %d",
                st.max_rank, o.leaf_size);
        // exact symmetry of the stored matrix (one basis tree): H^T x == H x
        const int k = 3;
        std::vector<double> x = randvec(size_t(n) * k, 3), yN(x.size()), yT(x.size());
        hgemv(H2Opus_NoTrans, 1.0, h->H, x.data(), n, 0.0, yN.data(), n, k, h->handle);
        hgemv(H2Opus_Trans, 1.0, h->H, x.data(), n, 0.0, yT.data(), n, k, h->handle);
        C.check_le(relerr(yT, yN), 1e-14, "stored matrix exactly symmetric (BlackBox symmetric == true)");
        C.check(B.symmetric && B.n == n && h2opus_size(h) == n && h2opus_stats(h).calls == st.calls,
                "BlackBox flags, size, stored statistics");

        // concurrent products on one matrix are serialized (shared H2Opus workspace)
        std::vector<double> ref1(size_t(n) * k), ref2(size_t(n) * k), c1(size_t(n) * k), c2(size_t(n) * k);
        std::vector<double> x2 = randvec(size_t(n) * k, 4);
        B.apply('N', k, x.data(), ref1.data());
        B.apply('N', k, x2.data(), ref2.data());
        std::thread t1([&] { for (int r = 0; r < 5; r++) B.apply('N', k, x.data(), c1.data()); });
        std::thread t2([&] { for (int r = 0; r < 5; r++) B.apply('N', k, x2.data(), c2.data()); });
        t1.join();
        t2.join();
        C.check(c1 == ref1 && c2 == ref2, "concurrent products from two threads give the sequential results");
    }

    // ---- shuffled numbering of the points ----------------------------------------------------
    {
        Problem Q(n, Kernel::Sym, Geometry::ShuffledFibonacci);
        const BlackBox DQ = dense_kernel_blackbox(Q);
        H2opusStats st;
        H2opusHandle h = h2opus_from_matvec(DQ, 3, Q.xyz.data(), H2opusOptions(), &st);
        C.check_le(blackbox_relerr(h2opus_blackbox(h), DQ), err_bound, "shuffled numbering: matvec against the source");
        C.check(st.max_rank <= 64, "shuffled numbering: ranks", "max rank %d", st.max_rank);
    }

    // ---- symmetry is required (GOTCHA(symmetric-storage)) --------------------------------------
    {
        const int m = 1000;
        Problem Q(m, Kernel::Nonsym, Geometry::ShuffledFibonacci);
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(dense_kernel_blackbox(Q), 3, Q.xyz.data()); },
                                              "hmb::h2opus_from_matvec:"),
                "nonsymmetric BlackBox refused");
        Problem S(m);
        BlackBox unflagged = dense_kernel_blackbox(S);
        unflagged.symmetric = false;
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(unflagged, 3, S.xyz.data()); },
                                              "hmb::h2opus_from_matvec:"),
                "symmetric operator not flagged symmetric refused");
        H2opusOptions o;
        o.assume_symmetric = true;
        H2opusHandle h = h2opus_from_matvec(unflagged, 3, S.xyz.data(), o);
        C.check_le(blackbox_relerr(h2opus_blackbox(h), unflagged), err_bound, "accepted with assume_symmetric");
    }

    // ---- argument checks -----------------------------------------------------------------------
    {
        H2opusOptions o;
        o.block_size = 24;
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(D, 3, P.xyz.data(), o); }, "hmb::h2opus_from_matvec:"),
                "block_size 24 refused (H2Opus would abort)");
        o = H2opusOptions();
        o.tolerance = 0;
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(D, 3, P.xyz.data(), o); }, "hmb::h2opus_from_matvec:"),
                "tolerance 0 refused");
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(D, 3, nullptr); }, "hmb::h2opus_from_matvec:"),
                "missing points refused");
        C.check(throws<std::invalid_argument>([&] { h2opus_from_matvec(BlackBox(), 3, P.xyz.data()); },
                                              "hmb::h2opus_from_matvec:"),
                "BlackBox without apply refused");
        C.check(throws<std::invalid_argument>([&] { h2opus_blackbox(nullptr); }, "hmb::h2opus_blackbox:"),
                "null handle refused");
    }

    // ---- an exception from the source, in the norm estimate and inside HARA -----------------------
    for (long fail_at : {3L, 25L}) { // 3: norm estimate; 25 > 2 * norm_iterations: inside HARA
        auto cnt = std::make_shared<CallCounts>();
        std::string m = "no exception";
        try {
            h2opus_from_matvec(counted(D, cnt, fail_at), 3, P.xyz.data());
        } catch (const std::runtime_error &e) {
            m = e.what();
        }
        char what[96];
        std::snprintf(what, sizeof what, "source exception at call %ld comes back as runtime_error", fail_at);
        C.check(starts_with(m, "hmb::h2opus_from_matvec: the source BlackBox threw: deliberate failure"), what, "%s",
                m.c_str());
        std::snprintf(what, sizeof what, "source not called after call %ld threw", fail_at);
        C.check(cnt->calls == fail_at, what, "%ld calls", cnt->calls.load());
    }
    return C.finish("n=%d tol=%g", n, tol);
}
