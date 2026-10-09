// ButterflyPACK adapter: the process-level hazards and the error paths.
//  - GOTCHA(bpack-global-state): ButterflyPACK's own construct_init resets the OpenMP thread
//    count (shown with a raw call); through the adapter the caller's count survives
//    constructions and products.
//  - GOTCHA(bpack-errsol): keep_forward = false makes bpack_blackbox refuse H (instead of a
//    segfault) while HODLR still multiplies (before any factorization); with the default
//    keep_forward = true, a HODLR factored by bpack_factor still multiplies afterwards, and
//    solves through the native handles (d_c_bpack_solve in ButterflyPACK's ordering).
//    The adapter's own solves: test_bpack_solve.
//  - An exception thrown by the source never crosses ButterflyPACK: the construction ends,
//    the source is not called again, and the exception comes back as std::runtime_error.
//  - A source that itself calls ButterflyPACK (an H rebuilt from a HODLR's products).
//  - Argument validation.
#include "bpack_native.hpp"
#include "test_bpack_h2opus_common.hpp"

#include <omp.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif

using namespace hmb;
using namespace hmb::test;

extern "C" {
static void no_distance(int *, int *, double *, C2Fptr) {}
static void no_nearfar(int *, int *, int *, C2Fptr) {}
}

namespace {

bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

double heap_mb() {
#ifdef __GLIBC__
    malloc_trim(0);
    struct mallinfo2 mi = mallinfo2();
    return (mi.uordblks + mi.hblkhd) / 1048576.0;
#else
    return 0;
#endif
}

// Raw ButterflyPACK setup, as an application would write it (no adapter); returns the thread
// count right after construct_init.
int raw_construct_init_threads(const Problem &P) {
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    F2Cptr bmat = nullptr, option = nullptr, stats = nullptr, msh = nullptr, ker = nullptr, ptree = nullptr;
    int one = 1, groups[1] = {0}, n = P.n, dim = 3, nlevel = 0, tree[1] = {P.n}, nloc = 0;
    MPI_Fint comm = 0;
    std::vector<int> perm(n);
    std::vector<double> xyz = P.xyz;
    d_c_bpack_createptree(&one, groups, &comm, &ptree);
    d_c_bpack_createoption(&option);
    d_c_bpack_createstats(&stats);
    d_c_bpack_set_I_option(&option, "verbosity", -1);
    d_c_bpack_construct_init(&n, &dim, xyz.data(), nullptr, &nlevel, tree, perm.data(), &nloc, &bmat, &option, &stats,
                             &msh, &ker, &ptree, &no_distance, &no_nearfar, nullptr);
    const int threads = omp_get_max_threads();
    d_c_bpack_delete(&bmat);
    d_c_bpack_deletekernelquant(&ker);
    d_c_bpack_deletemesh(&msh);
    d_c_bpack_deletestats(&stats);
    d_c_bpack_deleteproctree(&ptree);
    d_c_bpack_deleteoption(&option);
    return threads;
}

template <class F> std::string exception_message(F f, bool &threw_runtime) {
    threw_runtime = false;
    try {
        f();
    } catch (const std::runtime_error &e) {
        threw_runtime = true;
        return e.what();
    } catch (const std::exception &e) {
        return std::string("other exception: ") + e.what();
    }
    return "no exception";
}

template <class E, class F> bool throws(F f, const char *prefix) {
    try {
        f();
    } catch (const E &e) {
        return starts_with(e.what(), prefix);
    }
    return false;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_bpack_safety");
    const int n = arg_int(argc, argv, "--n", 1500);
    const double tol = 1e-6;
    Problem P(n, Kernel::Sym, Geometry::RandomSphere, /*shift=*/10.0);
    const BlackBox D = dense_kernel_blackbox(P);

    // ---- GOTCHA(bpack-global-state): thread count -------------------------------------
    {
        const char *env = std::getenv("OMP_NUM_THREADS");
        const int env_threads = env && std::atoi(env) > 0 ? std::atoi(env) : 1;
        const int mine = env_threads + 1;
        omp_set_num_threads(mine);
        const int raw = raw_construct_init_threads(Problem(300));
        C.check(raw == env_threads, "raw construct_init resets the thread count", "%d -> %d ($OMP_NUM_THREADS or 1: %d)",
                mine, raw, env_threads);
        omp_set_num_threads(mine);
        BpackOptions o;
        BpackHandle hm = bpack_from_matvec(D, 3, P.xyz.data(), o);
        C.check(omp_get_max_threads() == mine, "thread count kept by bpack_from_matvec", "%d -> %d", mine,
                omp_get_max_threads());
        BpackHandle he = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        C.check(omp_get_max_threads() == mine, "thread count kept by bpack_from_entries", "%d -> %d", mine,
                omp_get_max_threads());
        const double e = blackbox_relerr(bpack_blackbox(hm), D);
        C.check(omp_get_max_threads() == mine, "thread count kept by products", "%d -> %d", mine, omp_get_max_threads());
        C.check_le(e, 20 * tol, "product with the changed thread count");
        hm.reset();
        he.reset();
        C.check(omp_get_max_threads() == mine, "thread count kept by destruction", "%d -> %d", mine, omp_get_max_threads());
        omp_set_num_threads(env_threads);
    }

    // ---- GOTCHA(bpack-errsol): keep_forward -------------------------------------------
    {
        BpackOptions o;
        o.format = BpackFormat::H;
        o.keep_forward = false;
        double h0 = heap_mb();
        BpackHandle h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        const double mb_without = heap_mb() - h0;
        C.check(throws<std::logic_error>([&] { bpack_blackbox(h); }, "hmb::bpack_blackbox:"),
                "H with keep_forward = false: bpack_blackbox refuses (no segfault)");
        h.reset();
        o.keep_forward = true;
        h0 = heap_mb();
        h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        const double mb_with = heap_mb() - h0;
        C.check_le(blackbox_relerr(bpack_blackbox(h), D), 10 * tol, "H with keep_forward = true multiplies");
        if (mb_with > 0) // glibc's malloc statistics (not available under AddressSanitizer)
            std::printf("  H retained heap: %.1f MB with keep_forward, %.1f MB without\n", mb_with, mb_without);
        h.reset();

        o.format = BpackFormat::HODLR;
        o.keep_forward = false;
        h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        C.check_le(blackbox_relerr(bpack_blackbox(h), D), 10 * tol, "HODLR with keep_forward = false still multiplies");
        h.reset();

        // default keep_forward: factor (adapter), then multiply, and solve with the native handles
        o.keep_forward = true;
        h = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        BlackBox B = bpack_blackbox(h);
        bpack_factor(h);
        C.check_le(blackbox_relerr(B, D), 10 * tol, "HODLR multiplies after bpack_factor (keep_forward)");
        // solve B x = b in ButterflyPACK's order (rows permuted by new2old)
        const int nrhs = 2;
        std::vector<double> x = randvec(size_t(n) * nrhs, 5), b(size_t(n) * nrhs), bi(b.size()), xi(b.size()), xs(b.size());
        B.apply('N', nrhs, x.data(), b.data());
        for (int v = 0; v < nrhs; v++)
            for (int i = 0; i < n; i++) bi[i + size_t(v) * n] = b[h->new2old[i] + size_t(v) * n];
        {
            std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
            int nloc = h->nloc, k = nrhs;
            d_c_bpack_solve(xi.data(), bi.data(), &nloc, &k, &h->bmat, &h->option, &h->stats, &h->ptree);
        }
        for (int v = 0; v < nrhs; v++)
            for (int i = 0; i < n; i++) xs[h->new2old[i] + size_t(v) * n] = xi[i + size_t(v) * n];
        // x = B^-1 (B x): exact up to the factorization's own error (test_bpack_solve's bound)
        C.check_le(relerr(xs, x), 10 * tol, "native solve through new2old recovers x");
    }

    // ---- exceptions from the source ------------------------------------------------------
    {
        auto cnt = std::make_shared<CallCounts>();
        bool rt = false;
        std::string m = exception_message([&] { bpack_from_matvec(counted(D, cnt, 5), 3, P.xyz.data()); }, rt);
        C.check(rt && starts_with(m, "hmb::bpack_from_matvec: the source BlackBox threw: deliberate failure"),
                "BlackBox exception comes back as runtime_error", "%s", m.c_str());
        C.check(cnt->calls == 5, "source not called after it threw", "%ld calls", cnt->calls.load());

        long calls = 0;
        EntrySource E = P.entries();
        auto inner = E.get;
        E.get = [inner, &calls](int M, int N, const int *r, const int *c, double *out) {
            if (++calls == 3) throw std::runtime_error("deliberate entry failure");
            inner(M, N, r, c, out);
        };
        m = exception_message([&] { bpack_from_entries(E, 3, P.xyz.data()); }, rt);
        C.check(rt && starts_with(m, "hmb::bpack_from_entries: the source EntrySource threw: deliberate entry failure"),
                "EntrySource exception comes back as runtime_error", "%s", m.c_str());
        C.check(calls == 3, "entry source not called after it threw", "%ld calls", calls);

        BpackHandle h = bpack_from_matvec(D, 3, P.xyz.data());
        C.check_le(blackbox_relerr(bpack_blackbox(h), D), 20 * tol, "construction works after the failures");
    }

    // ---- a source that calls ButterflyPACK itself ------------------------------------------
    {
        BpackOptions o;
        BpackHandle h1 = bpack_from_entries(P.entries(), 3, P.xyz.data(), o);
        BlackBox B1 = bpack_blackbox(h1);
        o.format = BpackFormat::H;
        BpackStats st;
        BpackHandle h2 = bpack_from_matvec(B1, 3, P.xyz.data(), o, &st);
        // H's bound: 100x (GOTCHA(tolerance-stacking), see test_bpack_matvec)
        C.check_le(blackbox_relerr(bpack_blackbox(h2), B1), 100 * tol, "H rebuilt from a ButterflyPACK HODLR's products");
        std::printf("  nested: %ld calls, %lld vectors, check_error %.2e\n", st.calls, st.vectors, st.check_error);
    }

    // ---- argument validation ---------------------------------------------------------------
    {
        BpackOptions bad;
        bad.rank0 = 0;
        C.check(throws<std::invalid_argument>([&] { bpack_from_matvec(D, 3, P.xyz.data(), bad); }, "hmb::bpack_from_matvec:"),
                "rank0 = 0 refused");
        bad = BpackOptions();
        bad.tolerance = 0;
        C.check(throws<std::invalid_argument>([&] { bpack_from_entries(P.entries(), 3, P.xyz.data(), bad); },
                                              "hmb::bpack_from_entries:"),
                "tolerance = 0 refused");
        C.check(throws<std::invalid_argument>([&] { bpack_from_matvec(D, 3, nullptr); }, "hmb::bpack_from_matvec:"),
                "missing points refused");
        C.check(throws<std::invalid_argument>([&] { bpack_from_matvec(BlackBox(), 3, P.xyz.data()); },
                                              "hmb::bpack_from_matvec:"),
                "BlackBox without apply refused");
        C.check(throws<std::invalid_argument>([&] { bpack_blackbox(nullptr); }, "hmb::bpack_blackbox:"), "null handle refused");
        BpackHandle h = bpack_from_entries(P.entries(), 3, P.xyz.data());
        BlackBox B = bpack_blackbox(h);
        std::vector<double> x(n, 1.0), y(n);
        C.check(throws<std::invalid_argument>([&] { B.apply('X', 1, x.data(), y.data()); }, "hmb::bpack_blackbox:"),
                "trans other than N/T refused");
    }
    return C.finish("n=%d", n);
}
