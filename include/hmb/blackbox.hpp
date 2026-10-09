// move-hmatrix: a black-box operator, the common currency of the matvec route.
#pragma once
#include <functional>

namespace hmb {

// apply(trans, k, X, Y): Y (n x k) = op(A) X (n x k), trans in {'N', 'T'}.
// Column-major, leading dimension n, ORIGINAL (user) ordering on both sides.
// If symmetric is true, a destination may answer 'T' requests with 'N'.
struct BlackBox {
    int n = 0;
    bool symmetric = false;
    std::function<void(char trans, int k, const double *X, double *Y)> apply;
};

} // namespace hmb
