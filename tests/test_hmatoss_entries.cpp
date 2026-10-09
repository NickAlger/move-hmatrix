// hmat-oss adapter: build from an EntrySource, entries over get_values (the 1-based fix,
// symmetric storage, concurrent requests), matvec, and the refusals (non-double interface,
// factorized matrix). The test also drives hmat-oss directly to show what each fix prevents.
#include "hmatoss_native.hpp"
#include "hmb_test.hpp"

#include <atomic>
#include <cstring>
#include <omp.h>
#include <stdexcept>

using namespace hmb;
using namespace hmb::test;

// Dense matrix in the ORIGINAL ordering through hmat-oss's get_block: an independent hmat-oss
// code path (whole blocks in hmat's ordering, mirrored for symmetric storage).
static std::vector<double> dense_via_get_block(const HmatossMatrix &M, int n) {
    std::vector<double> blk((size_t)n * n), out((size_t)n * n);
    hmat_get_values_context_t g;
    std::memset(&g, 0, sizeof g);
    g.matrix = M.A;
    g.values = blk.data();
    g.row_size = g.col_size = n;
    if (M.hi.get_block(&g) != 0) throw std::runtime_error("get_block failed");
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) out[g.row_indices[i] + (size_t)n * g.col_indices[j]] = blk[i + (size_t)n * j];
    return out;
}

static double maxabs(const std::vector<double> &v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// max |E(rows, cols) - R(rows, cols)| over a request
static double request_diff(const EntrySource &E, const std::vector<double> &R, int n, const std::vector<int> &rows,
                           const std::vector<int> &cols) {
    const int M = (int)rows.size(), N = (int)cols.size();
    std::vector<double> out((size_t)M * N);
    E.get(M, N, rows.data(), cols.data(), out.data());
    double d = 0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) d = std::max(d, std::fabs(out[i + (size_t)M * j] - R[rows[i] + (size_t)n * cols[j]]));
    return d;
}

// Random requests: single rows, single columns and blocks, unsorted, with duplicates.
struct Request {
    std::vector<int> rows, cols;
};
static std::vector<Request> random_requests(int count, int n, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<Request> rq(count);
    for (int q = 0; q < count; q++) {
        const int shape = q % 3, m = shape == 0 ? 1 : 1 + (int)(g() % 100), k = shape == 1 ? 1 : 1 + (int)(g() % 100);
        for (int i = 0; i < m; i++) rq[q].rows.push_back((int)(g() % n));
        for (int j = 0; j < k; j++) rq[q].cols.push_back((int)(g() % n));
    }
    return rq;
}

static double frob_rel(const std::vector<double> &a, const std::vector<double> &ref) { return relerr(a, ref); }

// A small hmat-oss matrix built directly with the C API (kernel through simple_compute, which
// receives 0-based indices and always returns double, whatever the matrix's scalar type).
static const Problem *g_problem = nullptr;
static void kernel_cb(void *, int i, int j, void *res) { *(double *)res = g_problem->K(i, j); }
struct NativeMatrix {
    hmat_interface_t hi;
    hmat_cluster_tree_t *tree = nullptr;
    hmat_matrix_t *A = nullptr;
    // hodlr: HODLR admissibility and lower-symmetric storage (for the hodlrsym factorization)
    NativeMatrix(const Problem &P, hmat_value_t type, double eps, bool hodlr = false) {
        g_problem = &P;
        hmat_init_default_interface(&hi, type);
        hi.init();
        std::vector<double> xyz = P.xyz;
        hmat_clustering_algorithm_t *med = hmat_create_clustering_median(), *algo = hmat_create_clustering_max_dof(med, 64);
        tree = hmat_create_cluster_tree(xyz.data(), 3, P.n, algo);
        hmat_delete_clustering(algo);
        hmat_delete_clustering(med);
        hmat_admissibility_t *adm = hodlr ? hmat_create_admissibility_hodlr() : hmat_create_admissibility_standard(2.0);
        A = hi.create_empty_hmatrix_admissibility(tree, tree, hodlr ? 1 : 0, adm);
        hi.set_low_rank_epsilon(A, eps);
        hmat_assemble_context_t ac;
        hmat_assemble_context_init(&ac);
        ac.compression = hmat_create_compression_aca_plus(eps);
        ac.simple_compute = kernel_cb;
        ac.lower_symmetric = hodlr ? 1 : 0;
        ac.progress = nullptr;
        hi.assemble_generic(A, &ac);
        hmat_delete_compression(ac.compression);
        hmat_delete_admissibility(adm);
    }
    ~NativeMatrix() {
        if (A) hi.destroy(A);
        hmat_delete_cluster_tree(tree);
    }
};

template <class F> static bool throws_with(F f, const char *needle, std::string *what = nullptr) {
    try {
        f();
    } catch (const std::exception &e) {
        if (what) *what = e.what();
        return std::strstr(e.what(), needle) != nullptr;
    }
    return false;
}

int main(int argc, char **argv) {
    Checker C("test_hmatoss_entries");
    const int n = arg_int(argc, argv, "--n", 1500);
    const double eps = 1e-6;
    const double round = 1e-12; // rounding-level threshold, relative to max |A|

    // ---------------------------------------------------------------------------------------
    // Non-symmetric kernel, full storage
    // ---------------------------------------------------------------------------------------
    Problem P(n, Kernel::Nonsym, Geometry::ShuffledFibonacci);
    HmatossOptions opt;
    opt.epsilon = eps;
    HmatossStats st;
    HmatossHandle S = hmatoss_build(P.entries(), 3, P.xyz.data(), opt, &st);
    std::printf("hmat-oss build n=%d eps=%g: %.2fs, stored %lld values (%.1f%% of dense; %lld in %lld dense leaves, %lld low-rank "
                "leaves)\n",
                n, eps, st.build_seconds, st.stored_values, 100.0 * st.stored_values / ((double)n * n), st.dense_values,
                st.n_dense_leaves, st.n_lowrank_leaves);
    C.check(st.n_rows == n && st.n_cols == n && st.storage == 'N' && st.n_lowrank_leaves > 0, "build: stats",
            "%d x %d storage %c", st.n_rows, st.n_cols, st.storage);

    const std::vector<double> R = dense_via_get_block(*S, n); // hmat-oss's own values
    std::vector<double> K((size_t)n * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) K[i + (size_t)n * j] = P.K(i, j);
    const double amax = maxabs(R);
    C.check_le(frob_rel(R, K), 10 * eps, "build: ||A - K||_F / ||K||_F <= 10 eps");

    EntrySource E = hmatoss_entries(S);
    C.check(E.n_rows == n && E.n_cols == n && E.thread_safe, "entries: size and thread_safe flag");
    {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        C.check_le(request_diff(E, R, n, all, all) / amax, round, "entries: whole matrix = get_block (rel. max)");
        double d = 0;
        for (const Request &q : random_requests(300, n, 1)) d = std::max(d, request_diff(E, R, n, q.rows, q.cols));
        C.check_le(d / amax, round, "entries: 300 random requests = get_block");
    }

    // GOTCHA(index-base): get_values with 0-based indices returns rc 0 and the neighbouring
    // entries (indices >= 1 here: index 0 would read before hmat's permutation array).
    {
        std::mt19937 g(5);
        std::vector<int> rows(40), cols(40);
        for (int &r : rows) r = 1 + (int)(g() % (n - 1));
        for (int &c : cols) c = 1 + (int)(g() % (n - 1));
        std::vector<int> r0 = rows, c0 = cols;
        std::vector<double> wrong(rows.size() * cols.size()), right(wrong.size()), ref(wrong.size());
        hmat_get_values_context_t ctx;
        std::memset(&ctx, 0, sizeof ctx);
        ctx.matrix = S->A;
        ctx.values = wrong.data();
        ctx.row_indices = r0.data();
        ctx.row_size = (int)r0.size();
        ctx.col_indices = c0.data();
        ctx.col_size = (int)c0.size();
        const int rc = S->hi.get_values(&ctx);
        E.get((int)rows.size(), (int)cols.size(), rows.data(), cols.data(), right.data());
        for (size_t j = 0; j < cols.size(); j++)
            for (size_t i = 0; i < rows.size(); i++) ref[i + rows.size() * j] = R[rows[i] + (size_t)n * cols[j]];
        const double ew = frob_rel(wrong, ref), er = frob_rel(right, ref);
        C.check(rc == 0 && ew > 0.1, "index-base: raw 0-based get_values wrong, rc 0", "rc %d, rel. error %.2f", rc, ew);
        C.check_le(er, round, "index-base: adapter on the same indices");
    }

    // Concurrent requests from OpenMP threads: identical to the sequential results.
    {
        const std::vector<Request> rq = random_requests(96, n, 2);
        std::vector<std::vector<double>> ref(rq.size());
        for (size_t q = 0; q < rq.size(); q++) {
            ref[q].resize(rq[q].rows.size() * rq[q].cols.size());
            E.get((int)rq[q].rows.size(), (int)rq[q].cols.size(), rq[q].rows.data(), rq[q].cols.data(), ref[q].data());
        }
        std::atomic<int> inflight{0}, maxinflight{0};
        std::atomic<long> mismatches{0}, calls{0};
        const int reps = 8, nq = (int)rq.size();
#pragma omp parallel for num_threads(4) schedule(dynamic)
        for (int t = 0; t < reps * nq; t++) {
            const Request &q = rq[t % nq];
            std::vector<double> out(ref[t % nq].size(), -1.0);
            const int c = ++inflight;
            for (int m = maxinflight.load(); c > m && !maxinflight.compare_exchange_weak(m, c);) {
            }
            E.get((int)q.rows.size(), (int)q.cols.size(), q.rows.data(), q.cols.data(), out.data());
            --inflight;
            calls++;
            if (std::memcmp(out.data(), ref[t % nq].data(), sizeof(double) * out.size())) mismatches++;
        }
        C.check(mismatches == 0 && maxinflight >= 2, "entries: concurrent requests bitwise identical",
                "%ld calls on 4 threads, %ld mismatches, up to %d concurrent", calls.load(), mismatches.load(), maxinflight.load());
    }
    {
        int bad_row = n, zero = 0;
        double v;
        C.check(throws_with([&] { E.get(1, 1, &bad_row, &zero, &v); }, "hmb::hmatoss_entries: row index"),
                "entries: out-of-range index refused");
    }

    // Matvec in the original ordering.
    {
        BlackBox bb = hmatoss_blackbox(S);
        BlackBox ex = P.exact_blackbox();
        C.check(bb.n == n && !bb.symmetric, "blackbox: size and symmetric flag");
        C.check_le(blackbox_relerr(bb, ex, 'N', 3), 10 * eps, "blackbox: 'N' vs kernel <= 10 eps");
        C.check_le(blackbox_relerr(bb, ex, 'T', 3), 10 * eps, "blackbox: 'T' vs kernel <= 10 eps");
        const int k = 3;
        std::vector<double> X = randvec((size_t)n * k, 11), X0 = X, Y((size_t)n * k), Yd((size_t)n * k, 0.0);
        bb.apply('N', k, X.data(), Y.data());
        for (int v = 0; v < k; v++)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) Yd[i + (size_t)n * v] += R[i + (size_t)n * j] * X[j + (size_t)n * v];
        C.check_le(frob_rel(Y, Yd), round, "blackbox: 'N' = dense product of its entries");
        C.check(std::memcmp(X.data(), X0.data(), sizeof(double) * X.size()) == 0, "blackbox: input X left untouched");
    }

    // ---------------------------------------------------------------------------------------
    // Symmetric kernel, lower-symmetric storage (GOTCHA(symmetric-storage))
    // ---------------------------------------------------------------------------------------
    {
        Problem Q(n, Kernel::Sym, Geometry::RandomSphere);
        HmatossOptions o;
        o.epsilon = eps;
        o.lower_symmetric = true;
        HmatossStats sst;
        HmatossHandle L = hmatoss_build(Q.entries(), 3, Q.xyz.data(), o, &sst);
        std::printf("lower-symmetric build: %.2fs, stored %lld values (%.1f%% of dense)\n", sst.build_seconds, sst.stored_values,
                    100.0 * sst.stored_values / ((double)n * n));
        C.check(sst.storage == 'L', "symmetric: storage reported", "storage %c", sst.storage);

        // what hmat-oss itself does: get_values on the whole matrix leaves the upper part unwritten
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 1); // 1-based
        std::vector<int> idx2 = idx;
        std::vector<double> raw((size_t)n * n, NAN);
        hmat_get_values_context_t ctx;
        std::memset(&ctx, 0, sizeof ctx);
        ctx.matrix = L->A;
        ctx.values = raw.data();
        ctx.row_indices = idx.data();
        ctx.row_size = n;
        ctx.col_indices = idx2.data();
        ctx.col_size = n;
        const int rc = L->hi.get_values(&ctx);
        long unwritten = 0;
        for (double x : raw) unwritten += std::isnan(x);
        C.check(rc == 0 && unwritten > 0.4 * n * n, "symmetric: raw get_values skips upper part",
                "rc %d, %.1f%% unwritten", rc, 100.0 * unwritten / ((double)n * n));

        const std::vector<double> R2 = dense_via_get_block(*L, n);
        std::vector<double> K2((size_t)n * n);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) K2[i + (size_t)n * j] = Q.K(i, j);
        C.check_le(frob_rel(R2, K2), 10 * eps, "symmetric: ||A - K||_F / ||K||_F <= 10 eps");
        const double amax2 = maxabs(R2);
        EntrySource E2 = hmatoss_entries(L);
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::vector<double> full((size_t)n * n);
        E2.get(n, n, all.data(), all.data(), full.data());
        long nonfinite = 0;
        double asym = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                nonfinite += !std::isfinite(full[i + (size_t)n * j]);
                asym = std::max(asym, std::fabs(full[i + (size_t)n * j] - full[j + (size_t)n * i]));
            }
        C.check(nonfinite == 0, "symmetric: adapter writes every entry", "%ld non-finite", nonfinite);
        C.check_le(request_diff(E2, R2, n, all, all) / amax2, round, "symmetric: whole matrix = get_block");
        C.check(asym == 0.0, "symmetric: A(i,j) == A(j,i) exactly", "max diff %.1e", asym);
        double d = 0;
        for (const Request &q : random_requests(300, n, 3)) d = std::max(d, request_diff(E2, R2, n, q.rows, q.cols));
        C.check_le(d / amax2, round, "symmetric: 300 random requests = get_block");

        BlackBox bb = hmatoss_blackbox(L);
        C.check(bb.symmetric, "symmetric: blackbox symmetric flag");
        C.check_le(blackbox_relerr(bb, Q.exact_blackbox(), 'N', 3), 10 * eps, "symmetric: blackbox 'N' vs kernel");
        C.check_le(blackbox_relerr(bb, Q.exact_blackbox(), 'T', 3), 10 * eps, "symmetric: blackbox 'T' vs kernel");
    }

    // ---------------------------------------------------------------------------------------
    // Refusals
    // ---------------------------------------------------------------------------------------
    {
        Problem Q(300, Kernel::Nonsym, Geometry::ShuffledFibonacci);
        // GOTCHA(scalar-type): a single-precision matrix writes floats through void*.
        NativeMatrix F(Q, HMAT_SIMPLE_PRECISION, 1e-4);
        int r[4] = {1, 2, 3, 4};
        std::vector<int> r2(r, r + 4);
        double buf[16];
        for (double &x : buf) x = NAN;
        hmat_get_values_context_t ctx;
        std::memset(&ctx, 0, sizeof ctx);
        ctx.matrix = F.A;
        ctx.values = buf;
        ctx.row_indices = r;
        ctx.row_size = 4;
        ctx.col_indices = r2.data();
        ctx.col_size = 4;
        const int rc = F.hi.get_values(&ctx);
        int unwritten = 0, wrong = 0;
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++) {
                const double v = buf[i + 4 * j];
                if (std::isnan(v)) unwritten++;
                else if (!(std::fabs(v - Q.K(i, j)) <= 1e-3 * std::fabs(Q.K(i, j)))) wrong++;
            }
        C.check(rc == 0 && unwritten + wrong == 16, "scalar-type: float matrix into double buffer",
                "rc %d: %d of 16 values wrong, %d unwritten", rc, wrong, unwritten);
        std::string what;
        C.check(throws_with([&] { hmatoss_wrap(F.hi, F.A); }, "single precision", &what), "scalar-type: hmatoss_wrap refuses float");
        std::printf("  refused with: %s\n", what.c_str());

        // GOTCHA(factorized-state): after LU, get_values returns factor entries with rc 0.
        NativeMatrix D(Q, HMAT_DOUBLE_PRECISION, 1e-8);
        HmatossHandle W = hmatoss_wrap(D.hi, D.A); // not owned
        EntrySource EW = hmatoss_entries(W);
        std::vector<int> all(Q.n);
        std::iota(all.begin(), all.end(), 0);
        std::vector<double> before((size_t)Q.n * Q.n), after(before.size());
        EW.get(Q.n, Q.n, all.data(), all.data(), before.data());
        hmat_factorization_context_t fc;
        hmat_factorization_context_init(&fc);
        fc.factorization = hmat_factorization_lu;
        fc.progress = nullptr;
        C.check(D.hi.factorize_generic(D.A, &fc) == 0, "factorized: LU of a wrapped matrix");
        EW.get(Q.n, Q.n, all.data(), all.data(), after.data()); // an EntrySource made before: no check
        C.check(frob_rel(after, before) > 0.1, "factorized: get_values then returns the factors", "rel. change %.2f",
                frob_rel(after, before));
        C.check(throws_with([&] { hmatoss_entries(W); }, "LU-factorized", &what), "factorized: hmatoss_entries refuses");
        std::printf("  refused with: %s\n", what.c_str());
        C.check(throws_with([&] { hmatoss_blackbox(W); }, "LU-factorized"), "factorized: hmatoss_blackbox refuses");
        C.check(throws_with([&] { hmatoss_export(W); }, "LU-factorized"), "factorized: hmatoss_export refuses");

        // hodlrsym has no code in hmat-oss's stream: write_struct throws (and prints a backtrace)
        Problem Y(300, Kernel::Sym, Geometry::RandomSphere, 10.0); // shifted: SPD diagonal leaves
        NativeMatrix H(Y, HMAT_DOUBLE_PRECISION, 1e-8, true);
        fc.factorization = hmat_factorization_hodlrsym;
        C.check(H.hi.factorize_generic(H.A, &fc) == 0, "factorized: hodlrsym of a HODLR matrix");
        HmatossHandle WH = hmatoss_wrap(H.hi, H.A);
        C.check(throws_with([&] { hmatoss_entries(WH); }, "cannot serialize", &what), "factorized: hodlrsym refused");
        std::printf("  refused with: %.140s\n", what.c_str());
    }
    return C.finish("n=%d eps=%g", n, eps);
}
