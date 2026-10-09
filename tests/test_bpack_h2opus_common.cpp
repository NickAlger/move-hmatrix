// Dense kernel operator for the ButterflyPACK and H2Opus tests (see test_bpack_h2opus_common.hpp).
// Its own translation unit: the BLAS prototype below must not meet H2Opus's or Htool's.
#include "test_bpack_h2opus_common.hpp"

#include <memory>
#include <vector>

extern "C" void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                       const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
                       const double *beta, double *c, const int *ldc);

namespace hmb {
namespace test {

BlackBox dense_kernel_blackbox(const Problem &P) {
    const int n = P.n;
    auto K = std::make_shared<std::vector<double>>(size_t(n) * n);
    double *k = K->data();
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) k[i + size_t(j) * n] = P.K(i, j);
    BlackBox b;
    b.n = n;
    b.symmetric = P.symmetric();
    b.apply = [K, n](char trans, int nv, const double *X, double *Y) {
        const char t = (trans == 'T' || trans == 't') ? 'T' : 'N', no = 'N';
        const double one = 1, zero = 0;
        if (nv > 0) dgemm_(&t, &no, &n, &nv, &n, &one, K->data(), &n, X, &n, &zero, Y, &n);
    };
    return b;
}

} // namespace test
} // namespace hmb
