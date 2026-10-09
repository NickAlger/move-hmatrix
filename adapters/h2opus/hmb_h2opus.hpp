// hmatrix-bridges: H2Opus adapter (H^2 from matvecs with HARA), library-neutral interface.
//
// Rebuilds a SYMMETRIC real operator given as a black box (hmb::BlackBox) as an H2Opus H^2
// matrix with HARA, H2Opus's hierarchical adaptive randomized approximation, on a structure
// built from the points, and exposes the result as a BlackBox. Includes no H2Opus header:
// H2Opus and Htool declare clashing BLAS prototypes (GOTCHA(header-clash)), so the two
// libraries meet only through the BlackBox. h2opus_native.hpp holds the H2Opus types.
// (The tile low-rank adapter of the same library is hmb_h2opus_tlr.hpp.)
//
// Tested against H2Opus cbc7985 (CPU build, double precision, OpenMP; thrust 1.17.2), GCC
// 13.3, OpenBLAS 0.3.26.
//
// Conventions
//  - BlackBox vectors use the ORIGINAL ordering. H2Opus builds a median-split KD-tree on the
//    points and multiplies in its own order (index_map: new2old, 0-based); the adapter
//    permutes the samples, the products and the rebuilt matrix's vectors
//    (GOTCHA(perm-direction)).
//  - coords: dim x n, point-major (coords[dim * i + d] is coordinate d of the point of
//    row/column i); read only during h2opus_from_matvec.
//
// What H2Opus requires (refused with std::invalid_argument)
//  - Symmetric operators only (GOTCHA(symmetric-storage)): HARA samples only A*X and builds
//    a symmetric H^2 (one basis tree). A BlackBox with symmetric == false is refused unless
//    H2opusOptions::assume_symmetric says that the operator is symmetric although it is not
//    flagged so (e.g. an Htool matrix stored in full, 'N', of a symmetric kernel). Fed a
//    nonsymmetric operator, HARA silently returns a wrong matrix: off by about the operator's
//    asymmetry ||(A - A^T) x|| / ||A x||, which HARA cannot fit. An operator that is symmetric
//    only up to errors above the tolerance, such as a solve with an approximate factorization
//    (GOTCHA(inverse-asymmetry)), is better sampled as (A + A^T) / 2: at N = 4,000 that gave a
//    4.5x smaller residual and 23% smaller ranks (examples/chain_four_libraries.cpp).
//  - Real double only (GOTCHA(scalar-type)): checked at compile time.
//  - block_size 16 or 32 (H2Opus asserts and aborts otherwise).
//
// Tolerance and cost: HARA takes an ABSOLUTE threshold. The adapter estimates ||A||_2 by
// power iteration on the black box (up to 2 x norm_iterations single-vector products) and
// passes tolerance * ||A||_2. Measured (smooth kernel on a sphere, Htool source at 1e-6,
// tolerance 1e-6, one thread, N = 4,000-16,000): errors against the source of 2.0-4.6e-6;
// 2,060-4,170 sampling vectors in 75-141 calls; 2,900-4,800 times one single-vector source
// matvec, most of it in HARA's own recompression (horthog + hcompress after every block of
// samples), not in the source. The per-level cap max_samples silently limits the ranks: a
// cap that is too small overshoots the tolerance without any warning; check_error (a few
// products of the source against the rebuilt matrix, measured after every construction)
// shows it.
//
// Memory: the rebuilt matrix keeps what H2Opus counts (27 MB at N = 4,000 here), but HARA
// peaks at about 9 times that while it runs (+241 MB). GOTCHA(h2opus-workspace): HARA leaves
// its workspace in the H2Opus handle and spare capacity in the matrix's arrays (203 MB
// retained at N = 4,000); the adapter copies the result into a fresh matrix and handle.
//
// Speed under other load: HARA runs many short OpenMP regions. With OMP_NUM_THREADS equal to
// the number of cores of a machine busy with other jobs it slowed from 1.3 s to 8-12 s
// (N = 2,000, 2 cores); one thread, or OMP_WAIT_POLICY=passive, took 3.3 s under that load.
//
// Ownership and threads: the handle owns the H^2 matrix and an H2Opus handle (workspace);
// the BlackBox from h2opus_blackbox keeps it alive and serializes products on one matrix
// (they share the workspace). Different matrices are independent. The source BlackBox is
// used only during h2opus_from_matvec, from one thread at a time. Exceptions thrown by the
// source are caught (the remaining samples get zeros, the source is not called again) and
// rethrown as std::runtime_error once HARA returns. One process, CPU only.
#pragma once
#include "hmb/blackbox.hpp"
#include <memory>

namespace hmb {

struct H2opusMatrix; // H2Opus HMatrix + handle + permutation; defined in h2opus_native.hpp
using H2opusHandle = std::shared_ptr<H2opusMatrix>;

struct H2opusOptions {
    double tolerance = 1e-6;       // relative to the estimated ||A||_2
    int leaf_size = 64;            // largest KD-tree leaf
    double eta = 1.0;              // admissibility: (diam(t) + diam(s)) / 2 <= eta * dist(centers of t, s)
    int max_samples = 512;         // HARA's cap on the samples of one level
    int block_size = 32;           // samples per black-box call inside HARA: 16 or 32
    int converged_samples = 10;    // HARA's r: samples below the threshold that end a level
    int norm_iterations = 10;      // power iterations of the norm estimate (2 products each)
    bool assume_symmetric = false; // accept a BlackBox not flagged symmetric (see above)
    // After HARA, compare the rebuilt matrix with the source on this many random vectors
    // (one more source call) -> H2opusStats::check_error. 0: skip.
    int check_vectors = 4;
    bool verbose = false; // HARA's per-level table on stdout
};

struct H2opusStats {
    long calls = 0;              // black-box calls, norm estimate included (not the check)
    long long vectors = 0;       // sampling vectors, norm estimate included (not the check)
    long norm_calls = 0;         // of which for the norm estimate (one vector each)
    double norm_estimate = 0;    // estimated ||A||_2
    double abs_tolerance = 0;    // tolerance * norm_estimate, as passed to HARA
    double seconds_structure = 0, seconds_norm = 0, seconds_hara = 0;
    double seconds_callback = 0;  // inside the adapter's sampler (source + permutation copies)
    double seconds_in_source = 0; // inside the source's apply
    double memory_mb = 0;         // H2Opus's count for the rebuilt matrix
    int depth = 0, max_rank = 0;  // levels of the basis tree; largest level rank
    // ||(B - A) X||_F / ||A X||_F for check_vectors Gaussian vectors X, with A the source and
    // B the rebuilt matrix; -1 when not measured.
    double check_error = -1;
};

// Rebuilds the symmetric operator A (A.n x A.n, original ordering) from products.
H2opusHandle h2opus_from_matvec(const BlackBox &A, int dim, const double *coords, const H2opusOptions &opt = {},
                                H2opusStats *stats = nullptr);

// The H^2 matrix as a matvec operator in the ORIGINAL ordering (H2Opus's hgemv). It is
// symmetric by construction (one basis tree), so symmetric == true and 'T' == 'N'.
BlackBox h2opus_blackbox(const H2opusHandle &A);

int h2opus_size(const H2opusHandle &A);
H2opusStats h2opus_stats(const H2opusHandle &A); // statistics of the construction

} // namespace hmb
