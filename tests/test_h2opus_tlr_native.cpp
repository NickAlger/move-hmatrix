// H2Opus TLR quirks on H2Opus's own objects (native header): construct_tlr_matrix leaves
// stale ranks on the diagonal tiles of a non-symmetric matrix, which tlr_gemv adds to the dense
// diagonal (GOTCHA(h2opus-tlr-diag-rank)); tlr_gemv with several vectors and two parallel
// columns overlaps its accumulation buffers (GOTCHA(h2opus-tlr-gemv-multi)). Both are shown on
// raw H2Opus calls (these checks flip if a later H2Opus fixes them), then h2opus_tlr_wrap and
// h2opus_tlr_matvec are checked to avoid them.
#include "h2opus_tlr_native.hpp"
#include "hmb_test.hpp"

#include <h2opus/core/tlr/tlr_construct.h>

using namespace hmb;
using namespace hmb::test;

namespace {

struct KernelGen { // entries in TLR order (identity index map)
    const Problem *P;
    double operator()(int i, int j) { return P->K(i, j); }
};

// raw tlr_gemv on n x k vectors in TLR order (identity map), padded by hand
std::vector<double> raw_gemv(H2opusTlr &A, h2opusHandle_t hh, const std::vector<double> &X, int k, int npc) {
    const int n = A.n, pn = A.getPaddedDim();
    std::vector<double> Xp(size_t(pn) * k, 0.0), Yp(size_t(pn) * k, 0.0), Y(size_t(n) * k);
    for (int c = 0; c < k; c++)
        for (int i = 0; i < n; i++) Xp[i + size_t(c) * pn] = X[i + size_t(c) * n];
    tlr_gemv<double, H2OPUS_HWTYPE_CPU>(H2Opus_NoTrans, 1.0, A, Xp.data(), pn, 0.0, Yp.data(), pn, k, npc, hh);
    for (int c = 0; c < k; c++)
        for (int i = 0; i < n; i++) Y[i + size_t(c) * n] = Yp[i + size_t(c) * pn];
    return Y;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_h2opus_tlr_native");
    const int n = arg_int(argc, argv, "--n", 1000), bs = arg_int(argc, argv, "--bs", 128), k = 3;
    const double eps = 1e-8;
    Problem P(n, Kernel::Sym, Geometry::RandomSphere);
    std::vector<double> X = randvec(size_t(n) * k, 3), R(size_t(n) * k);
    P.exact_apply('N', k, X.data(), R.data());

    h2opusHandle_t hh;
    h2opusCreateHandle(&hh);
    auto A = std::make_unique<H2opusTlr>(n, H2OpusTLR_NonSymmetric, bs, nullptr, H2OpusTLRTile, hh->getMainStream());
    KernelGen gen{&P};
    construct_tlr_matrix<double, KernelGen, H2OPUS_HWTYPE_CPU>(*A, gen, eps, hh->getMainStream());
    int diag = 0;
    for (int j = 0; j < A->n_block; j++) diag += A->block_ranks[j + size_t(j) * A->n_block] != 0;

    // the diagonal-rank bug: one vector, npc = 2 (H2Opus's examples)
    std::vector<double> Y1(size_t(n) * k);
    for (int c = 0; c < k; c++) {
        std::vector<double> x(X.begin() + size_t(c) * n, X.begin() + size_t(c + 1) * n);
        std::vector<double> y = raw_gemv(*A, hh, x, 1, 2);
        std::copy(y.begin(), y.end(), Y1.begin() + size_t(c) * n);
    }
    const double e_stale = relerr(Y1, R);
    C.check(diag > 0 && e_stale > 1e-3, "construct_tlr_matrix: stale diagonal ranks (bug present)",
            "%d of %d diagonal tiles, tlr_gemv error %.2e", diag, A->n_block, e_stale);

    // the multi-vector bug: same matrix, 3 vectors at once with npc = 2 vs one at a time
    const double e_multi = relerr(raw_gemv(*A, hh, X, k, 2), Y1);
    C.check(e_multi > 1e-3, "tlr_gemv 3 vectors, npc 2: wrong (bug present)", "differs from single vectors by %.2e",
            e_multi);
    const double e_npc1 = relerr(raw_gemv(*A, hh, X, k, 1), Y1);
    C.check_le(e_npc1, 1e-14, "tlr_gemv 3 vectors, npc 1 = single vectors");
    h2opusDestroyHandle(hh);

    // the adapter: wrap clears the diagonal ranks, h2opus_tlr_matvec picks npc
    H2opusTlrHandle h = h2opus_tlr_wrap(std::move(A));
    C.check(h2opus_tlr_stats(h).cleared_diagonal_ranks == diag, "wrap clears the stale ranks", "%d cleared",
            h2opus_tlr_stats(h).cleared_diagonal_ranks);
    std::vector<double> Y(size_t(n) * k);
    h2opus_tlr_matvec(h, 'N', k, X.data(), n, Y.data(), n);
    // each tile's Frobenius error is at most eps (absolute), the matrix's n_block * eps
    double normF = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) normF += P.K(i, j) * P.K(i, j);
    const double bound = 10 * h2opus_tlr_stats(h).n_block * eps / std::sqrt(normF);
    C.check_le(relerr(Y, R), bound, "wrapped, 3 vectors vs kernel (<= 10 n_block eps/|A|_F)");
    std::shared_ptr<const hmb_hmatrix> E = h2opus_tlr_export(h);
    std::vector<double> Z(size_t(n) * k);
    hmb_matvec(E.get(), 'N', k, X.data(), n, Z.data(), n);
    C.check_le(relerr(Z, Y), 1e-13, "hmb_matvec(export(wrap)) vs tlr_gemv");
    return C.finish("n=%d bs=%d", n, bs);
}
