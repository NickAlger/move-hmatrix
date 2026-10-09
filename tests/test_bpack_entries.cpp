// ButterflyPACK adapter, entry route: bpack_from_entries for HODLR and H from the kernel's
// entries (symmetric and nonsymmetric kernels; one source declared not thread-safe). Checks
// the products of the rebuilt matrix against the exact dense kernel ('N' and 'T': the matvec
// after construction, GOTCHA(bpack-errsol)), that every index reaching the source is a valid
// 0-based ORIGINAL index (ButterflyPACK's callbacks are 1-based, GOTCHA(index-base)), the
// entry counts, that calls never overlap, and the ordering through the ranks.
#include "hmb_bpack.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <omp.h>

using namespace hmb;
using namespace hmb::test;

namespace {

struct EntryCounts {
    std::atomic<long> calls{0}, bad_indices{0};
    std::atomic<long long> entries{0};
    std::atomic<int> in_flight{0}, max_in_flight{0};
};

EntrySource counted_entries(const EntrySource &E, std::shared_ptr<EntryCounts> c, bool thread_safe) {
    EntrySource s = E;
    s.thread_safe = thread_safe;
    auto inner = E.get;
    const int nr = E.n_rows, nc = E.n_cols;
    s.get = [inner, c, nr, nc](int M, int N, const int *rows, const int *cols, double *out) {
        const int f = ++c->in_flight;
        int m = c->max_in_flight.load();
        while (f > m && !c->max_in_flight.compare_exchange_weak(m, f)) {
        }
        c->calls++;
        c->entries += (long long)M * N;
        for (int i = 0; i < M; i++) c->bad_indices += (rows[i] < 0 || rows[i] >= nr);
        for (int j = 0; j < N; j++) c->bad_indices += (cols[j] < 0 || cols[j] >= nc);
        inner(M, N, rows, cols, out);
        c->in_flight--;
    };
    return s;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_bpack_entries");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6);
    // E2: 2.4e-6 (HODLR) and 4e-7 (H) at 1e-6 against the kernel; the entry route meets its tolerance.
    const double err_bound = 10 * tol;
    for (Kernel kern : {Kernel::Sym, Kernel::Nonsym}) {
        const bool sym = kern == Kernel::Sym;
        Problem P(n, kern, sym ? Geometry::RandomSphere : Geometry::ShuffledFibonacci);
        const BlackBox D = dense_kernel_blackbox(P);
        const char *kname = sym ? "sym" : "nonsym";
        for (BpackFormat f : {BpackFormat::HODLR, BpackFormat::H}) {
            const char *fname = bpack_format_name(f);
            char what[128];
            BpackOptions o;
            o.format = f;
            o.tolerance = tol;
            auto cnt = std::make_shared<EntryCounts>();
            const bool declared_safe = sym; // the nonsymmetric source claims not to be thread-safe
            const int threads_before = omp_get_max_threads();
            BpackStats st;
            const double t0 = now();
            BpackHandle h = bpack_from_entries(counted_entries(P.entries(), cnt, declared_safe), 3, P.xyz.data(), o, &st);
            const double t_build = now() - t0;
            BlackBox B = bpack_blackbox(h);
            std::printf("%s %s: %.2fs, %ld block calls, %lld entries (%.2f n^2), rank %d, source time %.2fs\n", kname,
                        fname, t_build, st.entry_calls, st.entries, st.entries / (double(n) * n), st.rank_max,
                        st.seconds_in_source);

            std::snprintf(what, sizeof what, "%s %s matvec 'N' after construction", kname, fname);
            C.check_le(blackbox_relerr(B, D, 'N'), err_bound, what);
            std::snprintf(what, sizeof what, "%s %s matvec 'T' after construction", kname, fname);
            C.check_le(blackbox_relerr(B, D, 'T'), err_bound, what);
            std::snprintf(what, sizeof what, "%s %s indices reach the source 0-based", kname, fname);
            C.check(cnt->bad_indices == 0, what, "%ld out of [0, n)", cnt->bad_indices.load());
            std::snprintf(what, sizeof what, "%s %s stats count every entry", kname, fname);
            C.check(st.entries == cnt->entries && st.entry_calls == cnt->calls, what, "adapter %lld / %ld, source %lld / %ld",
                    st.entries, st.entry_calls, cnt->entries.load(), cnt->calls.load());
            std::snprintf(what, sizeof what, "%s %s entries between n and 4 n^2", kname, fname);
            C.check(st.entries >= n && st.entries <= 4LL * n * n, what, "%lld", st.entries);
            std::snprintf(what, sizeof what, "%s %s calls never overlap%s", kname, fname,
                          declared_safe ? "" : " (source not thread-safe)");
            C.check(cnt->max_in_flight == 1, what, "max in flight %d", cnt->max_in_flight.load());
            const int rank_bound = f == BpackFormat::HODLR ? n / 4 : o.leaf_size;
            std::snprintf(what, sizeof what, "%s %s ranks show a matching ordering", kname, fname);
            C.check(st.rank_max > 0 && st.rank_max <= rank_bound, what, "rank_max %d <= %d", st.rank_max, rank_bound);
            std::snprintf(what, sizeof what, "%s %s thread count restored", kname, fname);
            C.check(omp_get_max_threads() == threads_before, what, "%d -> %d", threads_before, omp_get_max_threads());
        }
    }
    // a rectangular source is refused
    {
        EntrySource E;
        E.n_rows = 10;
        E.n_cols = 12;
        E.get = [](int, int, const int *, const int *, double *) {};
        std::vector<double> xyz(36, 0.0);
        bool refused = false;
        try {
            bpack_from_entries(E, 3, xyz.data());
        } catch (const std::invalid_argument &e) {
            refused = std::string(e.what()).rfind("hmb::bpack_from_entries:", 0) == 0;
        }
        C.check(refused, "rectangular EntrySource refused with invalid_argument");
    }
    return C.finish("n=%d tol=%g", n, tol);
}
