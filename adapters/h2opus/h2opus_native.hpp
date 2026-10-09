// hmatrix-bridges: H2Opus adapter (HARA), native side (includes H2Opus headers).
//
// For code that goes on from a matrix built by h2opus_from_matvec to H2Opus's own
// operations (hgemv, hcompress, hlru_*, a copy to the GPU, ...). Code that only moves
// matrices between libraries should include hmb_h2opus.hpp. Do not include this header in a
// translation unit that also includes Htool (clashing BLAS prototypes, GOTCHA(header-clash));
// it also defines macros such as vec_ptr. Tested against H2Opus cbc7985, CPU build, double
// precision (hmb_h2opus.hpp).
//
// H2Opus works in its own ordering: vectors given to hgemv on H are column-major n x k with
// row i holding original index new2old[i] (GOTCHA(perm-direction)). The matrix is symmetric
// (H.sym): one basis tree serves rows and columns. The H2Opus handle holds workspace that is
// not thread-safe: lock `lock` around native calls if other threads may multiply with the
// same matrix through h2opus_blackbox.
#pragma once
#include "hmb_h2opus.hpp"

#include <h2opus.h>
#include <mutex>
#include <vector>

namespace hmb {

struct H2opusMatrix {
    HMatrix H;                       // symmetric H^2, in H2Opus's order
    h2opusHandle_t handle = nullptr; // workspace for hgemv, HARA, ...
    std::vector<int> new2old;        // H2Opus position i holds original index new2old[i] (copy of H's index_map)
    std::mutex lock;                 // serializes products with this matrix
    H2opusOptions options;           // as used for the construction
    H2opusStats build_stats;

    explicit H2opusMatrix(int n) : H(n, true) { h2opusCreateHandle(&handle); }
    ~H2opusMatrix() {
        if (handle) h2opusDestroyHandle(handle);
    }
    H2opusMatrix(const H2opusMatrix &) = delete;
    H2opusMatrix &operator=(const H2opusMatrix &) = delete;
};

} // namespace hmb
