// move-hmatrix: ButterflyPACK adapter (rebuild from matvecs or entries, matvec, exit
// diagnostics). Conventions and the gotchas handled here: hmb_bpack.hpp, bpack_native.hpp.
#include "bpack_native.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void hmb_bpack_flush_fortran_output(); // bpack_flush.f90

namespace hmb {
namespace {

double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string msg(const char *fn, const std::string &text) { return std::string(fn) + ": " + text; }

// GOTCHA(bpack-global-state): ButterflyPACK's construct_init calls
// omp_set_num_threads($OMP_NUM_THREADS, or 1 when unset). Every call into ButterflyPACK is
// wrapped in one of these, so the caller's thread count survives (and ButterflyPACK computes
// with it after the init).
class ThreadCountGuard {
    int saved_;

  public:
    ThreadCountGuard() : saved_(omp_get_max_threads()) {}
    ~ThreadCountGuard() { omp_set_num_threads(saved_); }
    ThreadCountGuard(const ThreadCountGuard &) = delete;
    ThreadCountGuard &operator=(const ThreadCountGuard &) = delete;
};

// ---- GOTCHA(fortran-stop): diagnosing a Fortran STOP inside ButterflyPACK -----------------
// ButterflyPACK ends the process with STOP (exit status 0) when its randomized sampling does
// not converge (constructions from matvecs) and on internal errors and unsupported
// operations (a transposed H-LU solve, which the adapter refuses beforehand). While the
// adapter is inside ButterflyPACK, a StopGuard publishes what is running; the exit handler
// (registered once) reports it, and replaces the exit status if asked to.
enum class Activity { BuildMatvec, BuildEntries, Factor, Multiply, Solve };

struct StopContext {
    const char *function = nullptr; // public function that called ButterflyPACK
    Activity activity = Activity::BuildMatvec;
    int n = 0;
    BpackOptions opt;
    const StopContext *outer = nullptr; // enclosing call (a source calling ButterflyPACK)
};

std::atomic<const StopContext *> g_running{nullptr};

void stop_handler() {
    const StopContext *c = g_running.load();
    if (!c) return;
    const BpackOptions &o = c->opt;
    const char *article = o.format == BpackFormat::H ? "an" : "a", *format = bpack_format_name(o.format);
    char what[96];
    switch (c->activity) {
    case Activity::BuildMatvec: std::snprintf(what, sizeof what, "builds %s %s matrix from matvecs", article, format); break;
    case Activity::BuildEntries: std::snprintf(what, sizeof what, "builds %s %s matrix from entries", article, format); break;
    case Activity::Factor: std::snprintf(what, sizeof what, "factors %s %s matrix", article, format); break;
    case Activity::Multiply: std::snprintf(what, sizeof what, "multiplies by %s %s matrix", article, format); break;
    case Activity::Solve: std::snprintf(what, sizeof what, "solves with %s %s matrix", article, format); break;
    }
    std::fprintf(stderr,
                 "\nmove-hmatrix: the process is exiting while ButterflyPACK %s\n"
                 "  (%s: n = %d, tolerance %g, leaf_size %d, rank0 %d, rankrate %g, itermax %d).\n",
                 what, c->function, c->n, o.tolerance, o.leaf_size, o.rank0, o.rankrate, o.itermax);
    if (c->activity == Activity::BuildMatvec)
        std::fprintf(stderr,
                     "  ButterflyPACK ends the program with a Fortran STOP (exit status 0) when the randomized\n"
                     "  sampling of a level does not converge within itermax tries (\"randomized scheme not\n"
                     "  converged\" on standard output). Raise BpackOptions::rank0 (first sample count; set it\n"
                     "  near the largest rank expected), rankrate (growth per try) or itermax (tries per level),\n"
                     "  or loosen the tolerance; also check that the source's apply('T') is the transpose.\n");
    else
        std::fprintf(stderr, "  ButterflyPACK ends the program with a Fortran STOP (exit status 0) on internal errors\n"
                             "  and unsupported operations; its own message is on standard output.\n");
    if (o.stop_exit_status != 0)
        std::fprintf(stderr, "  Ending the process with exit status %d (BpackOptions::stop_exit_status).\n",
                     o.stop_exit_status);
    else
        std::fprintf(stderr, "  Keeping ButterflyPACK's exit status (BpackOptions::stop_exit_status = 0).\n");
    std::fflush(stderr);
    if (o.stop_exit_status != 0) {
        // _Exit skips the Fortran runtime's cleanup: flush its buffered messages first.
        hmb_bpack_flush_fortran_output();
        std::fflush(nullptr);
        std::_Exit(o.stop_exit_status);
    }
}

class StopGuard {
    StopContext ctx_;

  public:
    StopGuard(const char *function, Activity activity, int n, const BpackOptions &opt) {
        static std::once_flag once;
        std::call_once(once, [] { std::atexit(stop_handler); });
        ctx_.function = function;
        ctx_.activity = activity;
        ctx_.n = n;
        ctx_.opt = opt;
        ctx_.outer = g_running.load();
        g_running.store(&ctx_);
    }
    ~StopGuard() { g_running.store(ctx_.outer); }
    StopGuard(const StopGuard &) = delete;
    StopGuard &operator=(const StopGuard &) = delete;
};

// ---- callback contexts ----------------------------------------------------------------
// An exception must not cross ButterflyPACK's Fortran frames: callbacks record the first
// one, answer the remaining requests with zeros (no further source calls), and the public
// function rethrows it once ButterflyPACK has returned.
struct MatvecContext {
    const BlackBox *A = nullptr;
    const int *new2old = nullptr;
    int n = 0;
    std::vector<double> xo, yo; // ORIGINAL ordering
    BpackStats *st = nullptr;
    std::exception_ptr error;
    bool source_threw = false;
};

struct EntryContext {
    const EntrySource *E = nullptr;
    int n = 0;
    std::mutex lock; // ButterflyPACK already serializes the block calls; kept as a guarantee
    std::vector<int> rows, cols;
    BpackStats *st = nullptr;
    std::exception_ptr error;
    bool source_threw = false;
};

void record(std::exception_ptr &slot, bool &source_threw, bool from_source) {
    if (slot) return;
    slot = std::current_exception();
    source_threw = from_source;
}

} // namespace
} // namespace hmb

extern "C" {

// construct_init's distance and near-field callbacks (used only with nogeo = 2).
static void hmb_bpack_no_distance(int *, int *, double *, C2Fptr) {}
static void hmb_bpack_no_nearfar(int *, int *, int *, C2Fptr) {}

// ButterflyPACK -> BlackBox. xin, xout: n x nvec, column-major, in ButterflyPACK's order.
static void hmb_bpack_matvec(char const *trans, int *nin, int *nout, int *nvec, double const *xin, double *xout,
                             C2Fptr ctx) {
    hmb::MatvecContext &c = *static_cast<hmb::MatvecContext *>(ctx);
    hmb::BpackStats &st = *c.st;
    const double t0 = hmb::wall();
    const int n = c.n, k = *nvec;
    const bool t = trans[0] != 'N' && trans[0] != 'n'; // 'T' (or 'C', the same for real data)
    st.calls++;
    st.vectors += k;
    if (t) {
        st.calls_transposed++;
        st.vectors_transposed += k;
    }
    const size_t len = size_t(std::max(*nout, 0)) * size_t(std::max(k, 0));
    if (!c.error && (*nin != n || *nout != n))
        c.error = std::make_exception_ptr(std::runtime_error(
            hmb::msg("hmb::bpack_from_matvec", "ButterflyPACK passed local vectors of length " + std::to_string(*nin) +
                                                   " -> " + std::to_string(*nout) + " for n = " + std::to_string(n) +
                                                   " (an MPI build? only the sequential build is supported)")));
    if (c.error || k <= 0) {
        std::fill(xout, xout + len, 0.0);
        st.seconds_callback += hmb::wall() - t0;
        return;
    }
    try {
        const size_t nk = size_t(n) * size_t(k);
        c.xo.resize(nk);
        c.yo.resize(nk);
        // GOTCHA(perm-direction): ButterflyPACK row i is original row new2old[i]
        for (int v = 0; v < k; v++) {
            const double *x = xin + size_t(v) * n;
            double *xo = c.xo.data() + size_t(v) * n;
            for (int i = 0; i < n; i++) xo[c.new2old[i]] = x[i];
        }
        const double ts = hmb::wall();
        try {
            // GOTCHA(bpack-transpose): a third to half of the calls ask for A^T * X; only a
            // symmetric source may answer them with A * X.
            c.A->apply(t && !c.A->symmetric ? 'T' : 'N', k, c.xo.data(), c.yo.data());
        } catch (...) {
            st.seconds_in_source += hmb::wall() - ts;
            hmb::record(c.error, c.source_threw, true);
            throw;
        }
        st.seconds_in_source += hmb::wall() - ts;
        for (int v = 0; v < k; v++) {
            const double *yo = c.yo.data() + size_t(v) * n;
            double *y = xout + size_t(v) * n;
            for (int i = 0; i < n; i++) y[i] = yo[c.new2old[i]];
        }
    } catch (...) {
        hmb::record(c.error, c.source_threw, false);
        std::fill(xout, xout + len, 0.0);
    }
    st.seconds_callback += hmb::wall() - t0;
}

// ButterflyPACK's single-entry callback: called only with elem_extract = 0 (one call per
// entry), which the adapter does not use. Fails loudly should a ButterflyPACK version call it.
static void hmb_bpack_entry_unused(int *, int *, double *val, C2Fptr ctx) {
    hmb::EntryContext &c = *static_cast<hmb::EntryContext *>(ctx);
    std::lock_guard<std::mutex> lock(c.lock);
    *val = 0;
    if (!c.error)
        c.error = std::make_exception_ptr(std::runtime_error(
            "hmb::bpack_from_entries: ButterflyPACK called its single-entry callback, which the adapter does not "
            "serve (it sets elem_extract = 2)"));
}

// ButterflyPACK -> EntrySource, lists of sub-blocks (elem_extract = 2). Intersection k has
// rowidx[k] rows and colidx[k] columns, listed one intersection after the other in allrows /
// allcols (1-based ORIGINAL indices); its values go to alldat_loc as a column-major block
// after the previous intersection's, the layout of a 1 x 1 process grid (pmaps).
static void hmb_bpack_entry_blocks(int *Ninter, int *Nallrows, int *Nallcols, int64_t *Nalldat_loc, int *allrows,
                                   int *allcols, double *alldat_loc, int *rowidx, int *colidx, int *pgidx, int *Npmap,
                                   int *pmaps, C2Fptr ctx) {
    (void)pgidx;
    hmb::EntryContext &c = *static_cast<hmb::EntryContext *>(ctx);
    std::lock_guard<std::mutex> lock(c.lock);
    const double t0 = hmb::wall();
    const char *fn = "hmb::bpack_from_entries";
    c.st->entry_calls++;
    const int64_t ndat = std::max<int64_t>(*Nalldat_loc, 0);
    if (!c.error) {
        try {
            const int ninter = *Ninter, npmap = *Npmap;
            for (int p = 0; ninter > 0 && p < npmap; p++)
                if (pmaps[p] != 1 || pmaps[p + npmap] != 1)
                    throw std::runtime_error(hmb::msg(fn, "ButterflyPACK asked for entries distributed over a " +
                                                              std::to_string(pmaps[p]) + " x " +
                                                              std::to_string(pmaps[p + npmap]) +
                                                              " process grid; only the sequential build is supported"));
            int64_t nr_all = 0, nc_all = 0, nd_all = 0;
            for (int k = 0; k < ninter; k++) {
                nr_all += rowidx[k];
                nc_all += colidx[k];
                nd_all += int64_t(rowidx[k]) * colidx[k];
            }
            if (nr_all != *Nallrows || nc_all != *Nallcols || nd_all != ndat)
                throw std::runtime_error(hmb::msg(fn, "unexpected entry-block layout from ButterflyPACK (" +
                                                          std::to_string(nr_all) + " rows, " + std::to_string(nc_all) +
                                                          " columns, " + std::to_string(nd_all) + " values; expected " +
                                                          std::to_string(*Nallrows) + ", " + std::to_string(*Nallcols) +
                                                          ", " + std::to_string(ndat) + ")"));
            int64_t ro = 0, co = 0, d = 0;
            for (int k = 0; k < ninter; k++) {
                const int nr = rowidx[k], nc = colidx[k];
                c.rows.resize(nr);
                c.cols.resize(nc);
                // GOTCHA(index-base): 1-based original indices
                for (int i = 0; i < nr; i++) c.rows[i] = allrows[ro + i] - 1;
                for (int j = 0; j < nc; j++) c.cols[j] = allcols[co + j] - 1;
                for (int i = 0; i < nr; i++)
                    if (c.rows[i] < 0 || c.rows[i] >= c.n)
                        throw std::runtime_error(hmb::msg(fn, "ButterflyPACK asked for row " + std::to_string(c.rows[i] + 1) +
                                                             " (1-based) of a matrix of size " + std::to_string(c.n)));
                for (int j = 0; j < nc; j++)
                    if (c.cols[j] < 0 || c.cols[j] >= c.n)
                        throw std::runtime_error(hmb::msg(fn, "ButterflyPACK asked for column " +
                                                             std::to_string(c.cols[j] + 1) +
                                                             " (1-based) of a matrix of size " + std::to_string(c.n)));
                if (nr > 0 && nc > 0) {
                    const double ts = hmb::wall();
                    try {
                        c.E->get(nr, nc, c.rows.data(), c.cols.data(), alldat_loc + d);
                    } catch (...) {
                        c.st->seconds_in_source += hmb::wall() - ts;
                        hmb::record(c.error, c.source_threw, true);
                        throw;
                    }
                    c.st->seconds_in_source += hmb::wall() - ts;
                }
                c.st->entries += int64_t(nr) * nc;
                ro += nr;
                co += nc;
                d += int64_t(nr) * nc;
            }
        } catch (...) {
            hmb::record(c.error, c.source_threw, false);
        }
    }
    if (c.error) std::fill(alldat_loc, alldat_loc + ndat, 0.0);
    c.st->seconds_callback += hmb::wall() - t0;
}

} // extern "C"

namespace hmb {
namespace {

int format_code(BpackFormat f) { return f == BpackFormat::H ? 2 : 1; } // ButterflyPACK's option "format"

void check_args(const char *fn, int n, int dim, const double *coords, const BpackOptions &o) {
    if (n < 1) throw std::invalid_argument(msg(fn, "matrix size " + std::to_string(n) + " (need at least 1)"));
    if (dim < 1 || !coords)
        throw std::invalid_argument(msg(fn, "ButterflyPACK clusters the points itself: dim >= 1 and coords are required"));
    if (o.format != BpackFormat::HODLR && o.format != BpackFormat::H)
        throw std::invalid_argument(msg(fn, "unknown BpackFormat"));
    if (!(o.tolerance > 0 && o.tolerance < 1))
        throw std::invalid_argument(msg(fn, "tolerance must be in (0, 1), got " + std::to_string(o.tolerance)));
    if (o.leaf_size < 1) throw std::invalid_argument(msg(fn, "leaf_size must be at least 1"));
    if (!(o.near_para >= 0)) throw std::invalid_argument(msg(fn, "near_para must be >= 0"));
    if (o.rank0 < 1) throw std::invalid_argument(msg(fn, "rank0 must be at least 1"));
    if (!(o.rankrate >= 1)) throw std::invalid_argument(msg(fn, "rankrate must be >= 1"));
    if (o.itermax < 1) throw std::invalid_argument(msg(fn, "itermax must be at least 1"));
    if (o.check_vectors < 0) throw std::invalid_argument(msg(fn, "check_vectors must be >= 0"));
}

[[noreturn]] void rethrow(const char *fn, const char *source, const std::exception_ptr &e, bool from_source) {
    try {
        std::rethrow_exception(e);
    } catch (const std::exception &x) {
        if (!from_source) throw; // the adapter's own error, already prefixed
        throw std::runtime_error(msg(fn, std::string("the source ") + source + " threw: " + x.what()));
    } catch (...) {
        throw std::runtime_error(msg(fn, std::string("the source ") + source + " threw a non-standard exception"));
    }
}

double get_stat(BpackMatrix &h, const char *name) {
    double v = 0;
    d_c_bpack_getstats(&h.stats, name, &v);
    return v;
}

// Options, clustering (construct_init) and the permutation. Caller holds bpack_mutex().
std::shared_ptr<BpackMatrix> init_matrix(const char *fn, int n, int dim, const double *coords,
                                         const BpackOptions &o, BpackStats &st) {
    auto h = std::make_shared<BpackMatrix>();
    h->n = n;
    h->options = o;
    std::vector<int> perm(n, 0);
    const double t0 = wall();
    {
        ThreadCountGuard threads;
        int nproc = 1, groups[1] = {0};
        MPI_Fint comm = 0; // GOTCHA(mpi-stub): the sequential build's stub ignores it
        d_c_bpack_createptree(&nproc, groups, &comm, &h->ptree);
        d_c_bpack_createoption(&h->option);
        d_c_bpack_createstats(&h->stats);
        d_c_bpack_set_I_option(&h->option, "format", format_code(o.format));
        d_c_bpack_set_I_option(&h->option, "LRlevel", 0); // low-rank blocks on every level (no butterflies)
        d_c_bpack_set_D_option(&h->option, "tol_comp", o.tolerance);
        d_c_bpack_set_D_option(&h->option, "tol_rand", o.tolerance);
        d_c_bpack_set_D_option(&h->option, "tol_Rdetect", 0.1 * o.tolerance);
        d_c_bpack_set_D_option(&h->option, "near_para", o.near_para);
        d_c_bpack_set_I_option(&h->option, "Nmin_leaf", o.leaf_size);
        d_c_bpack_set_I_option(&h->option, "xyzsort", 1); // KD-tree clustering of the points
        d_c_bpack_set_I_option(&h->option, "nogeo", 0);   // geometry given
        d_c_bpack_set_I_option(&h->option, "cpp", 1);     // C callbacks
        d_c_bpack_set_I_option(&h->option, "elem_extract", 2); // entry route: lists of blocks, no MPI inside
        d_c_bpack_set_I_option(&h->option, "rank0", o.rank0);
        d_c_bpack_set_D_option(&h->option, "rankrate", o.rankrate);
        d_c_bpack_set_I_option(&h->option, "itermax", o.itermax);
        // GOTCHA(bpack-errsol): without ErrSol = 1 (or precon > 1), H keeps no forward
        // operator after construction and a matvec segfaults.
        d_c_bpack_set_I_option(&h->option, "ErrSol", o.keep_forward ? 1 : 0);
        d_c_bpack_set_I_option(&h->option, "verbosity", o.verbosity);
        std::vector<double> xyz(coords, coords + size_t(dim) * n); // ButterflyPACK takes a non-const pointer
        int nn = n, ndim = dim, nlevel = 0, tree[1] = {n};
        d_c_bpack_construct_init(&nn, &ndim, xyz.data(), nullptr, &nlevel, tree, perm.data(), &h->nloc, &h->bmat,
                                 &h->option, &h->stats, &h->msh, &h->ker, &h->ptree, &hmb_bpack_no_distance,
                                 &hmb_bpack_no_nearfar, nullptr);
    }
    st.seconds_init = wall() - t0;
    if (h->nloc != n)
        throw std::runtime_error(msg(fn, "ButterflyPACK reports " + std::to_string(h->nloc) + " local rows of " +
                                             std::to_string(n) + " (an MPI build? only the sequential build is supported)"));
    h->new2old.resize(n);
    std::vector<char> seen(n, 0);
    for (int i = 0; i < n; i++) {
        const int orig = perm[i] - 1; // GOTCHA(index-base): ButterflyPACK's new2old is 1-based
        if (orig < 0 || orig >= n || seen[orig]++)
            throw std::runtime_error(msg(fn, "ButterflyPACK returned an invalid permutation"));
        h->new2old[i] = orig;
    }
    return h;
}

void finish(BpackMatrix &h, BpackStats &st, BpackStats *out) {
    {
        ThreadCountGuard threads;
        st.rank_max = int(get_stat(h, "Rank_max_Constr"));
        st.bpack_mem_fill_mb = get_stat(h, "Mem_Fill");
    }
    h.build_stats = st;
    if (out) *out = st;
}

// Why A's forward product is not available, or nullptr. Caller holds bpack_mutex().
const char *forward_missing(const BpackMatrix &h) {
    if (h.options.format == BpackFormat::H && !h.options.keep_forward)
        return "this H matrix was built with keep_forward = false: ButterflyPACK kept no forward operator and its "
               "product would crash (GOTCHA(bpack-errsol)); rebuild it with keep_forward = true";
    if (h.forward_deleted)
        return "this HODLR matrix was built with keep_forward = false and has been factored, which deleted its "
               "forward blocks (GOTCHA(bpack-errsol)); rebuild it with keep_forward = true";
    return nullptr;
}

// Y = op(A) X (ButterflyPACK's mult), or op(A^-1) X with inverse (inv_mult, after
// bpack_factor), in the ORIGINAL ordering.
void apply(BpackMatrix &h, bool inverse, char trans, int k, const double *X, double *Y) {
    const char *fn = inverse ? "hmb::bpack_solve_blackbox" : "hmb::bpack_blackbox";
    const bool t = (trans == 'T' || trans == 't');
    if (!t && trans != 'N' && trans != 'n')
        throw std::invalid_argument(msg(fn, std::string("trans must be 'N' or 'T', got '") + trans + "'"));
    if (k <= 0) return;
    const int n = h.n;
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    if (inverse) {
        if (!h.factor_stats.factored) throw std::logic_error(msg(fn, "the matrix is not factored (bpack_factor)"));
        if (t && h.options.format == BpackFormat::H)
            throw std::logic_error(msg(fn, "ButterflyPACK has no transposed H-LU solve (it would stop the process); "
                                           "use HODLR for A^-T, or 'N' if A is symmetric"));
    } else if (const char *why = forward_missing(h)) {
        throw std::logic_error(msg(fn, why));
    }
    std::vector<double> xi(size_t(n) * k), yi(size_t(n) * k);
    for (int v = 0; v < k; v++)
        for (int i = 0; i < n; i++) xi[i + size_t(v) * n] = X[h.new2old[i] + size_t(v) * n];
    {
        StopGuard running(fn, inverse ? Activity::Solve : Activity::Multiply, n, h.options);
        ThreadCountGuard threads;
        char tr[2] = {t ? 'T' : 'N', 0};
        int kk = k, nloc = h.nloc;
        if (inverse)
            d_c_bpack_inv_mult(tr, xi.data(), yi.data(), &nloc, &nloc, &kk, &h.bmat, &h.option, &h.stats, &h.ptree);
        else
            d_c_bpack_mult(tr, xi.data(), yi.data(), &nloc, &nloc, &kk, &h.bmat, &h.option, &h.stats, &h.ptree);
    }
    for (int v = 0; v < k; v++)
        for (int i = 0; i < n; i++) Y[h.new2old[i] + size_t(v) * n] = yi[i + size_t(v) * n];
}

// ||(B - A) X||_F / ||A X||_F for k Gaussian vectors X (fixed seed): B the rebuilt matrix, A
// the source. One source call; catches ButterflyPACK's occasional large misses
// (GOTCHA(tolerance-stacking)).
double check_against_source(const char *fn, BpackMatrix &h, const BlackBox &A, int k) {
    const size_t len = size_t(h.n) * k;
    std::mt19937 gen(12345);
    std::normal_distribution<double> normal;
    std::vector<double> X(len), YA(len), YB(len);
    for (double &x : X) x = normal(gen);
    try {
        A.apply('N', k, X.data(), YA.data());
    } catch (const std::exception &e) {
        throw std::runtime_error(msg(fn, std::string("the source BlackBox threw: ") + e.what()));
    } catch (...) {
        throw std::runtime_error(msg(fn, "the source BlackBox threw a non-standard exception"));
    }
    apply(h, false, 'N', k, X.data(), YB.data());
    double num = 0, den = 0;
    for (size_t i = 0; i < len; i++) {
        num += (YB[i] - YA[i]) * (YB[i] - YA[i]);
        den += YA[i] * YA[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

} // namespace

std::recursive_mutex &bpack_mutex() {
    static std::recursive_mutex m;
    return m;
}

BpackMatrix::~BpackMatrix() {
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    ThreadCountGuard threads;
    if (bmat) d_c_bpack_delete(&bmat);
    if (ker) d_c_bpack_deletekernelquant(&ker);
    if (msh) d_c_bpack_deletemesh(&msh);
    if (stats) d_c_bpack_deletestats(&stats);
    if (ptree) d_c_bpack_deleteproctree(&ptree);
    if (option) d_c_bpack_deleteoption(&option);
}

const char *bpack_format_name(BpackFormat f) { return f == BpackFormat::H ? "H" : "HODLR"; }

BpackHandle bpack_from_matvec(const BlackBox &A, int dim, const double *coords, const BpackOptions &opt,
                              BpackStats *stats) {
    const char *fn = "hmb::bpack_from_matvec";
    if (!A.apply) throw std::invalid_argument(msg(fn, "the BlackBox has no apply function"));
    check_args(fn, A.n, dim, coords, opt);
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    StopGuard running(fn, Activity::BuildMatvec, A.n, opt);
    BpackStats st;
    std::shared_ptr<BpackMatrix> h = init_matrix(fn, A.n, dim, coords, opt, st);
    MatvecContext ctx;
    ctx.A = &A;
    ctx.new2old = h->new2old.data();
    ctx.n = A.n;
    ctx.st = &st;
    const double t0 = wall();
    {
        ThreadCountGuard threads;
        d_c_bpack_construct_matvec_compute(&h->bmat, &h->option, &h->stats, &h->msh, &h->ker, &h->ptree,
                                           &hmb_bpack_matvec, &ctx);
    }
    st.seconds_build = wall() - t0;
    if (ctx.error) rethrow(fn, "BlackBox", ctx.error, ctx.source_threw);
    // (an H matrix without its forward operator cannot multiply, GOTCHA(bpack-errsol))
    if (opt.check_vectors > 0 && (opt.format != BpackFormat::H || opt.keep_forward))
        st.check_error = check_against_source(fn, *h, A, opt.check_vectors);
    finish(*h, st, stats);
    return h;
}

BpackHandle bpack_from_entries(const EntrySource &A, int dim, const double *coords, const BpackOptions &opt,
                               BpackStats *stats) {
    const char *fn = "hmb::bpack_from_entries";
    if (!A.get) throw std::invalid_argument(msg(fn, "the EntrySource has no get function"));
    if (A.n_rows != A.n_cols)
        throw std::invalid_argument(msg(fn, "square matrices only (got " + std::to_string(A.n_rows) + " x " +
                                                std::to_string(A.n_cols) + ")"));
    check_args(fn, A.n_rows, dim, coords, opt);
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    StopGuard running(fn, Activity::BuildEntries, A.n_rows, opt);
    BpackStats st;
    std::shared_ptr<BpackMatrix> h = init_matrix(fn, A.n_rows, dim, coords, opt, st);
    EntryContext ctx;
    ctx.E = &A;
    ctx.n = A.n_rows;
    ctx.st = &st;
    const double t0 = wall();
    {
        ThreadCountGuard threads;
        d_c_bpack_construct_element_compute(&h->bmat, &h->option, &h->stats, &h->msh, &h->ker, &h->ptree,
                                            &hmb_bpack_entry_unused, &hmb_bpack_entry_blocks, &ctx);
    }
    st.seconds_build = wall() - t0;
    if (ctx.error) rethrow(fn, "EntrySource", ctx.error, ctx.source_threw);
    finish(*h, st, stats);
    return h;
}

BlackBox bpack_blackbox(const BpackHandle &A) {
    const char *fn = "hmb::bpack_blackbox";
    if (!A) throw std::invalid_argument(msg(fn, "null handle"));
    {
        std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
        if (const char *why = forward_missing(*A)) throw std::logic_error(msg(fn, why));
    }
    BlackBox bb;
    bb.n = A->n;
    bb.symmetric = false;
    BpackHandle keep = A;
    bb.apply = [keep](char trans, int k, const double *X, double *Y) { apply(*keep, false, trans, k, X, Y); };
    return bb;
}

void bpack_factor(const BpackHandle &A, BpackFactorStats *stats) {
    const char *fn = "hmb::bpack_factor";
    if (!A) throw std::invalid_argument(msg(fn, "null handle"));
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    BpackMatrix &h = *A;
    if (!h.factor_stats.factored) {
        BpackFactorStats fs;
        StopGuard running(fn, Activity::Factor, h.n, h.options);
        const double t0 = wall();
        {
            ThreadCountGuard threads;
            d_c_bpack_factor(&h.bmat, &h.option, &h.stats, &h.ptree, &h.msh);
        }
        fs.seconds = wall() - t0;
        fs.factored = true;
        // GOTCHA(bpack-errsol): without ErrSol = 1, HODLR's factorization deletes the forward blocks
        h.forward_deleted = h.options.format == BpackFormat::HODLR && !h.options.keep_forward;
        ThreadCountGuard threads;
        fs.rank_max = int(get_stat(h, "Rank_max"));
        fs.bpack_mem_factor_mb = get_stat(h, "Mem_Factor");
        h.factor_stats = fs;
    }
    if (stats) *stats = h.factor_stats;
}

BlackBox bpack_solve_blackbox(const BpackHandle &A) {
    if (!A) throw std::invalid_argument("hmb::bpack_solve_blackbox: null handle");
    bpack_factor(A);
    BlackBox bb;
    bb.n = A->n;
    bb.symmetric = false;
    BpackHandle keep = A;
    bb.apply = [keep](char trans, int k, const double *X, double *Y) { apply(*keep, true, trans, k, X, Y); };
    return bb;
}

BpackFactorStats bpack_factor_stats(const BpackHandle &A) {
    if (!A) throw std::invalid_argument("hmb::bpack_factor_stats: null handle");
    std::lock_guard<std::recursive_mutex> lock(bpack_mutex());
    return A->factor_stats;
}

int bpack_size(const BpackHandle &A) {
    if (!A) throw std::invalid_argument("hmb::bpack_size: null handle");
    return A->n;
}

BpackStats bpack_stats(const BpackHandle &A) {
    if (!A) throw std::invalid_argument("hmb::bpack_stats: null handle");
    return A->build_stats;
}

} // namespace hmb
