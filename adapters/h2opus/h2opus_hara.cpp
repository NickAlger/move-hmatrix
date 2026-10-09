// hmatrix-bridges: H2Opus adapter, H^2 from matvecs with HARA, and hgemv in the original
// ordering. Conventions and the gotchas handled here: hmb_h2opus.hpp, h2opus_native.hpp.
#include "h2opus_native.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace hmb {
namespace {

// GOTCHA(scalar-type): H2Opus picks its precision at build time (h2opusconf.h).
static_assert(std::is_same<H2Opus_Real, double>::value,
              "hmatrix-bridges needs H2Opus built in double precision (H2OPUS_USE_DOUBLE_PRECISION)");

double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string msg(const char *fn, const std::string &text) { return std::string(fn) + ": " + text; }

// The caller's points, original order, as H2Opus's data set (read while the KD-tree is built).
class PointSet : public H2OpusDataSet<H2Opus_Real> {
    int n_, dim_;
    const double *xyz_;

  public:
    PointSet(int n, int dim, const double *xyz) : n_(n), dim_(dim), xyz_(xyz) {}
    int getDimension() const override { return dim_; }
    size_t getDataSetSize() const override { return size_t(n_); }
    H2Opus_Real getDataPoint(size_t i, int d) const override { return xyz_[size_t(dim_) * i + d]; }
};

// HARA -> BlackBox. HARA's samples are in H2Opus's order: GOTCHA(perm-direction), H2Opus
// position i is original index new2old[i]. HARA asks for A*X only, which is why the operator
// must be symmetric. An exception from the source must not unwind through HARA's
// half-updated matrix: it is recorded, the remaining samples are zeros, and
// h2opus_from_matvec rethrows it.
class BlackBoxSampler : public HMatrixSampler {
    const BlackBox &A_;
    const int *new2old_;
    int n_;
    H2opusStats &st_;
    std::vector<double> xo_, yo_; // ORIGINAL ordering

  public:
    std::exception_ptr error;

    BlackBoxSampler(const BlackBox &A, const int *new2old, H2opusStats &st)
        : A_(A), new2old_(new2old), n_(A.n), st_(st) {}

    void sample(H2Opus_Real *input, H2Opus_Real *output, int k) override {
        const double t0 = wall();
        const int n = n_;
        st_.calls++;
        st_.vectors += k;
        const size_t nk = size_t(n) * size_t(std::max(k, 0));
        if (error || k <= 0) {
            std::fill(output, output + nk, 0.0);
            st_.seconds_callback += wall() - t0;
            return;
        }
        try {
            xo_.resize(nk);
            yo_.resize(nk);
            for (int v = 0; v < k; v++) {
                const double *x = input + size_t(v) * n;
                double *xo = xo_.data() + size_t(v) * n;
                for (int i = 0; i < n; i++) xo[new2old_[i]] = x[i];
            }
            const double ts = wall();
            A_.apply('N', k, xo_.data(), yo_.data());
            st_.seconds_in_source += wall() - ts;
            for (int v = 0; v < k; v++) {
                const double *yo = yo_.data() + size_t(v) * n;
                double *y = output + size_t(v) * n;
                for (int i = 0; i < n; i++) y[i] = yo[new2old_[i]];
            }
        } catch (...) {
            error = std::current_exception();
            std::fill(output, output + nk, 0.0);
        }
        st_.seconds_callback += wall() - t0;
    }
};

[[noreturn]] void rethrow_source(const char *fn, const std::exception_ptr &e) {
    try {
        std::rethrow_exception(e);
    } catch (const std::exception &x) {
        throw std::runtime_error(msg(fn, std::string("the source BlackBox threw: ") + x.what()));
    } catch (...) {
        throw std::runtime_error(msg(fn, "the source BlackBox threw a non-standard exception"));
    }
}

void apply(H2opusMatrix &h, int k, const double *X, double *Y) {
    if (k <= 0) return;
    const int n = h.H.n;
    std::vector<double> xi(size_t(n) * k), yi(size_t(n) * k);
    for (int v = 0; v < k; v++)
        for (int i = 0; i < n; i++) xi[i + size_t(v) * n] = X[h.new2old[i] + size_t(v) * n];
    {
        std::lock_guard<std::mutex> lock(h.lock);
        hgemv(H2Opus_NoTrans, 1.0, h.H, xi.data(), n, 0.0, yi.data(), n, k, h.handle);
    }
    for (int v = 0; v < k; v++)
        for (int i = 0; i < n; i++) Y[h.new2old[i] + size_t(v) * n] = yi[i + size_t(v) * n];
}

// ||(B - A) X||_F / ||A X||_F for k Gaussian vectors X (fixed seed): B the rebuilt matrix, A
// the source. One source call.
double check_against_source(const char *fn, H2opusMatrix &h, const BlackBox &A, int k) {
    const size_t len = size_t(h.H.n) * k;
    std::mt19937 gen(12345);
    std::normal_distribution<double> normal;
    std::vector<double> X(len), YA(len), YB(len);
    for (double &x : X) x = normal(gen);
    try {
        A.apply('N', k, X.data(), YA.data());
    } catch (...) {
        rethrow_source(fn, std::current_exception());
    }
    apply(h, k, X.data(), YB.data());
    double num = 0, den = 0;
    for (size_t i = 0; i < len; i++) {
        num += (YB[i] - YA[i]) * (YB[i] - YA[i]);
        den += YA[i] * YA[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

} // namespace

H2opusHandle h2opus_from_matvec(const BlackBox &A, int dim, const double *coords, const H2opusOptions &opt,
                                H2opusStats *stats) {
    const char *fn = "hmb::h2opus_from_matvec";
    if (!A.apply) throw std::invalid_argument(msg(fn, "the BlackBox has no apply function"));
    if (A.n < 1) throw std::invalid_argument(msg(fn, "matrix size " + std::to_string(A.n) + " (need at least 1)"));
    if (dim < 1 || !coords)
        throw std::invalid_argument(msg(fn, "H2Opus clusters the points itself: dim >= 1 and coords are required"));
    // GOTCHA(symmetric-storage): HARA samples only A*X and builds a symmetric matrix.
    if (!A.symmetric && !opt.assume_symmetric)
        throw std::invalid_argument(msg(fn, "HARA rebuilds symmetric operators only (it never samples A^T), and this "
                                            "BlackBox is not flagged symmetric; set H2opusOptions::assume_symmetric "
                                            "if the operator is symmetric anyway"));
    if (!(opt.tolerance > 0 && opt.tolerance < 1))
        throw std::invalid_argument(msg(fn, "tolerance must be in (0, 1), got " + std::to_string(opt.tolerance)));
    if (opt.block_size != 16 && opt.block_size != 32)
        throw std::invalid_argument(msg(fn, "block_size must be 16 or 32 (H2Opus asserts it), got " +
                                            std::to_string(opt.block_size)));
    if (opt.max_samples < opt.block_size)
        throw std::invalid_argument(msg(fn, "max_samples must be at least block_size"));
    if (opt.leaf_size < 1 || !(opt.eta > 0) || opt.converged_samples < 1 || opt.norm_iterations < 1)
        throw std::invalid_argument(msg(fn, "leaf_size, eta, converged_samples and norm_iterations must be positive"));
    if (opt.check_vectors < 0) throw std::invalid_argument(msg(fn, "check_vectors must be >= 0"));

    const int n = A.n;
    H2opusStats st;
    auto h = std::make_shared<H2opusMatrix>(n);
    h->options = opt;

    double t0 = wall();
    {
        PointSet points(n, dim, coords);
        H2OpusBoxCenterAdmissibility admissibility(opt.eta);
        buildHMatrixStructure(h->H, &points, opt.leaf_size, admissibility);
    }
    const int *map = vec_ptr(h->H.u_basis_tree.index_map);
    h->new2old.assign(map, map + n);
    std::vector<char> seen(n, 0);
    for (int i = 0; i < n; i++)
        if (h->new2old[i] < 0 || h->new2old[i] >= n || seen[h->new2old[i]]++)
            throw std::runtime_error(msg(fn, "H2Opus's index_map is not a permutation"));
    st.seconds_structure = wall() - t0;

    BlackBoxSampler sampler(A, h->new2old.data(), st);
    t0 = wall();
    const double norm = sampler_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, n, opt.norm_iterations, h->handle);
    st.seconds_norm = wall() - t0;
    st.norm_calls = st.calls;
    if (sampler.error) rethrow_source(fn, sampler.error);
    if (!(norm > 0) || !std::isfinite(norm))
        throw std::runtime_error(msg(fn, "the norm estimate of the operator is " + std::to_string(norm) +
                                             " (a zero operator, or NaN/Inf in its products)"));
    st.norm_estimate = norm;
    st.abs_tolerance = opt.tolerance * norm; // HARA's threshold is absolute

    t0 = wall();
    hara(&sampler, h->H, opt.max_samples, opt.converged_samples, st.abs_tolerance, opt.block_size, h->handle,
         opt.verbose);
    st.seconds_hara = wall() - t0;
    if (sampler.error) rethrow_source(fn, sampler.error);
    // GOTCHA(h2opus-workspace): after HARA, the H2Opus handle keeps HARA's workspace and the
    // matrix's arrays keep the spare capacity they grew while HARA updated them: 200 MB
    // retained for a 27 MB H^2 at n = 4,000. A copy into a fresh matrix, with a fresh handle
    // (which grows only to hgemv's needs), retains 26 MB.
    auto compact = std::make_shared<H2opusMatrix>(n);
    compact->H = h->H;
    compact->new2old = std::move(h->new2old);
    compact->options = opt;
    h = compact;

    if (opt.check_vectors > 0) st.check_error = check_against_source(fn, *h, A, opt.check_vectors);
    st.memory_mb = double(h->H.getMemoryUsage()) * 1e3; // H2Opus reports GB (1e9 bytes)
    st.depth = h->H.u_basis_tree.depth;
    for (int level = 0; level < st.depth; level++) st.max_rank = std::max(st.max_rank, h->H.u_basis_tree.getLevelRank(level));
    h->build_stats = st;
    if (stats) *stats = st;
    return h;
}

BlackBox h2opus_blackbox(const H2opusHandle &A) {
    if (!A) throw std::invalid_argument("hmb::h2opus_blackbox: null handle");
    BlackBox bb;
    bb.n = A->H.n;
    bb.symmetric = true;
    H2opusHandle keep = A;
    bb.apply = [keep](char trans, int k, const double *X, double *Y) {
        if (trans != 'N' && trans != 'n' && trans != 'T' && trans != 't')
            throw std::invalid_argument(std::string("hmb::h2opus_blackbox: trans must be 'N' or 'T', got '") + trans + "'");
        apply(*keep, k, X, Y); // symmetric: A^T = A
    };
    return bb;
}

int h2opus_size(const H2opusHandle &A) {
    if (!A) throw std::invalid_argument("hmb::h2opus_size: null handle");
    return A->H.n;
}

H2opusStats h2opus_stats(const H2opusHandle &A) {
    if (!A) throw std::invalid_argument("hmb::h2opus_stats: null handle");
    return A->build_stats;
}

} // namespace hmb
