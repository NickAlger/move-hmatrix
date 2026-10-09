// hmatrix-bridges: H2Opus TLR adapter, native side (includes H2Opus headers).
//
// For code that builds or consumes TTLR_Matrix objects itself; code that only moves
// matrices between libraries should include hmb_h2opus_tlr.hpp instead. Do not include it in
// a translation unit that also includes Htool (conflicting Fortran BLAS prototypes) or H2Lib
// (macros I and UINT_MAX, a type named real): GOTCHA(header-clash).
#pragma once
#include "hmb_h2opus_tlr.hpp"

#include <h2opus.h>
#include <h2opus/core/tlr/tlr_gemv.h>
#include <h2opus/core/tlr/tlr_struct.h>

#include <memory>
#include <mutex>

namespace hmb {

using H2opusTlr = TTLR_Matrix<double, H2OPUS_HWTYPE_CPU>; // the one TLR type the adapter handles

struct H2opusTlrMatrix {
    std::unique_ptr<H2opusTlr> A;
    h2opusHandle_t handle = nullptr; // tlr_gemv's workspace; one per matrix
    std::mutex lock;                 // serializes products (the workspace is shared): hold it
                                     // around native calls that use handle while other
                                     // threads may multiply through h2opus_tlr_blackbox
    H2opusTlrStats stats;

    H2opusTlrMatrix() { h2opusCreateHandle(&handle); }
    ~H2opusTlrMatrix() {
        A.reset();
        if (handle) h2opusDestroyHandle(handle);
    }
    H2opusTlrMatrix(const H2opusTlrMatrix &) = delete;
    H2opusTlrMatrix &operator=(const H2opusTlrMatrix &) = delete;
};

// Takes ownership of a TLR matrix built with H2Opus (CPU, H2OpusTLRTile allocation, square).
// For the non-symmetric type, ranks left on diagonal tiles are set to 0: construct_tlr_matrix
// compresses a stale buffer into them and tlr_gemv would add that to the dense diagonal
// (GOTCHA(h2opus-tlr-diag-rank)). Throws std::invalid_argument for other allocations or types.
H2opusTlrHandle h2opus_tlr_wrap(std::unique_ptr<H2opusTlr> A);

} // namespace hmb
