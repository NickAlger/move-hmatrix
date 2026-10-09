// hmatrix-bridges: ButterflyPACK adapter, native side (includes ButterflyPACK's C header).
//
// For code that goes on from a matrix built by bpack_from_matvec / bpack_from_entries to
// ButterflyPACK operations the adapter does not wrap (d_c_bpack_solve, d_c_bpack_logdet,
// ...). Code that only moves, factors or solves should include hmb_bpack.hpp.
// Tested against ButterflyPACK a8cf81b, double precision, sequential build (hmb_bpack.hpp);
// its double-precision C header is dBPACK_wrapper.h.
//
// ButterflyPACK works in its own ordering: vectors passed to d_c_bpack_* are column-major,
// n x nrhs, with row i of ButterflyPACK's order holding original index new2old[i]
// (GOTCHA(perm-direction); new2old here is already 0-based, converted from ButterflyPACK's
// 1-based array, GOTCHA(index-base)).
//
// Rules for native calls (the adapter's own calls follow them; tests/test_bpack_safety.cpp
// solves this way after bpack_factor):
//  - Hold bpack_mutex() while calling ButterflyPACK if other threads may use it.
//  - Factor with bpack_factor, not d_c_bpack_factor: the adapter records the factorization
//    (factor_stats, forward_deleted); otherwise bpack_solve_blackbox would factor a second
//    time, and GOTCHA(bpack-errsol): a HODLR built with keep_forward = false loses its forward
//    blocks in the factorization, after which a product through bpack_blackbox would crash
//    instead of throwing. keep_forward = true also makes ButterflyPACK's factor and
//    d_c_bpack_solve run one extra test solve (ErrSol = 1); inv_mult (the adapter's solves)
//    does not.
//  - ButterflyPACK has no transposed H-LU solve: d_c_bpack_inv_mult with 'T' on an H matrix
//    stops the process.
//  - GOTCHA(bpack-global-state): only ButterflyPACK's init functions reset the OpenMP thread
//    count; factor, solve and mult leave it alone.
//  - GOTCHA(mpi-stub): dBPACK_wrapper.h #defines MPI_Fint when MPI is not used; do not
//    include <mpi.h> after this header in the same translation unit.
#pragma once
#include "hmb_bpack.hpp"

#include <dBPACK_wrapper.h>
#include <mutex>
#include <vector>

namespace hmb {

struct BpackMatrix {
    // ButterflyPACK objects (pass their addresses to d_c_bpack_*, e.g. &bmat).
    F2Cptr bmat = nullptr, option = nullptr, stats = nullptr, msh = nullptr, ker = nullptr, ptree = nullptr;
    int n = 0;
    int nloc = 0;             // local size, = n in the sequential build
    std::vector<int> new2old; // ButterflyPACK position i holds original index new2old[i] (0-based)
    BpackOptions options;     // as used for the construction
    BpackStats build_stats;
    BpackFactorStats factor_stats; // factored == false until bpack_factor
    bool forward_deleted = false;  // a HODLR factored with keep_forward = false

    BpackMatrix() = default;
    BpackMatrix(const BpackMatrix &) = delete;
    BpackMatrix &operator=(const BpackMatrix &) = delete;
    ~BpackMatrix(); // frees the ButterflyPACK objects (under bpack_mutex())
};

// The process-wide lock that serializes every call into ButterflyPACK.
std::recursive_mutex &bpack_mutex();

} // namespace hmb
