// hmatrix-bridges: ButterflyPACK adapter, library-neutral interface.
//
// Rebuilds a square real matrix in ButterflyPACK from a black-box operator (hmb::BlackBox,
// the matvec route) or from an entry oracle (hmb::EntrySource, the entry route), factors it
// there, and exposes the matrix and its inverse as BlackBoxes. Includes no ButterflyPACK
// header; bpack_native.hpp holds the ButterflyPACK handles, for ButterflyPACK's other
// operations (log-determinant, ...).
//
// Tested against ButterflyPACK a8cf81b (5.0.0+), double precision (d_c_bpack_* interface),
// sequential build with ButterflyPACK's MPI stub; GCC and gfortran 13.3, OpenBLAS 0.3.26.
//
// Formats: HODLR (ButterflyPACK format 1) and H (format 2), with low-rank (not butterfly)
// blocks. ButterflyPACK cannot build butterfly blocks inside H, HSS-BF or H2 from matvecs.
//
// Conventions
//  - Everything at this interface uses the ORIGINAL (user) ordering, 0-based. ButterflyPACK
//    clusters the points itself (KD-tree) and stores the matrix in its own order. Its
//    permutation is new2old and 1-based, and its entry callbacks receive 1-based original
//    indices; the adapter translates both ways (GOTCHA(index-base), GOTCHA(perm-direction)).
//  - coords: dim x n, point-major (coords[dim * i + d] is coordinate d of the point of
//    row/column i). The points drive the clustering and, for H, the admissibility
//    (near_para). They are copied; the caller's array is not kept.
//
// The two routes
//  - bpack_from_matvec: ButterflyPACK's randomized construction from products. It asks for
//    A*X AND A^T*X (a third of the calls for HODLR, close to half for H; GOTCHA(bpack-transpose):
//    answering them with A*X silently gives a wrong matrix unless A is symmetric). A BlackBox
//    with symmetric == false must implement apply('T'); a symmetric one gets 'N' only.
//    ButterflyPACK calls it from one thread at a time, in blocks of rank0 + 1, then
//    rankrate times more vectors per retry (HODLR up to 257 at N = 4,000 and 513 at 16,000;
//    H up to 63). Measured (smooth kernel on a sphere, Htool source at 1e-6, one thread,
//    N = 4,000-16,000): HODLR 4,100-8,100 sampling vectors, 930-1,360 times the time of one
//    single-vector source matvec; H 5,900-10,000 vectors, 1,400-2,000 times. The source's
//    products take 55-75% of that time.
//  - bpack_from_entries: ButterflyPACK's own compression from entries, the "direct"
//    baseline: 1 N^2 (HODLR) and 2.5 N^2 (H) entries at N = 4,000, 0.4 and 1 N^2 at 16,000;
//    1.6-3.7 times faster than the matvec route. The EntrySource receives lists of
//    sub-blocks (ButterflyPACK's block callback, elem_extract = 2), one call at a time (so
//    EntrySource::thread_safe does not matter): 1,000 times fewer calls than single entries,
//    with the same entries and result.
//
// Accuracy, GOTCHA(tolerance-stacking): on the matvec route ButterflyPACK accepts a level as
// soon as the rank it detects is below its number of samples, and H first tries each level
// with only the previous level's rank + 1 samples and no error test. The result misses the
// tolerance, H sometimes by far. Errors against the source at tolerance 1e-6: HODLR 1-2x at
// N = 1,500-2,000 and 4-7x at 4,000-16,000; H 6-8x at 4,000-16,000, and over 164 runs at
// N = 1,500-2,000 median 4x, 90th percentile 8x, 4 runs above 15x (worst 56x). At 1e-7, H
// reached 1.0-3.2e-6. Ask about 10x tighter than the accuracy you need and read
// BpackStats::check_error, measured after every matvec construction (check_vectors products
// of the source against the rebuilt matrix). The entry route stays close to its tolerance
// (against the kernel at 1e-6: HODLR 2-4e-6, H 4-5e-7). Either way the rebuilt matrix also
// carries the source's own error.
//
// Process-wide side effects of ButterflyPACK, and what the adapter does about them
//  - GOTCHA(fortran-stop): when the sampling of one level does not converge within itermax
//    tries, ButterflyPACK prints "randomized scheme not converged" and ends the PROCESS with
//    a Fortran STOP, exit status 0: no error code, no exception (it does the same on internal
//    errors and unsupported operations). While the adapter is inside ButterflyPACK
//    (construction, factorization, product, solve), an exit handler (registered once) prints
//    what was running, the likely cause and the options to change on stderr; then, unless
//    BpackOptions::stop_exit_status is 0, it flushes C and Fortran output and ends the
//    process with that status (default 70) instead of 0, skipping the exit handlers
//    registered before it. With ButterflyPACK's own sampling defaults (rank0
//    32, rankrate 1.2, itermax 10) a HODLR of N = 4,000 stops (its top rank is 233); this
//    adapter's defaults (64, 2, 10) reach 513 samples in 4 tries (top rank 426 at 16,000).
//  - GOTCHA(bpack-global-state): each construction sets the OpenMP thread count to
//    $OMP_NUM_THREADS (1 when unset) and reseeds Fortran's random number generator from
//    the clock and the process id. The adapter restores the caller's thread count after
//    every call into ButterflyPACK, so ButterflyPACK computes with the caller's count. The
//    seed cannot be fixed: sample counts, ranks and errors vary between identical runs, and
//    any other Fortran code in the process that uses random_number is reseeded.
//  - GOTCHA(mpi-stub): this is the sequential build. Its MPI stub defines the Fortran MPI
//    symbols (mpi_allreduce_, mpi_bcast_, ...), and ButterflyPACK's C header #defines
//    MPI_Fint, so it cannot share a program with a real MPI. Under MPI, use an MPI build of
//    ButterflyPACK (not supported by this adapter: callbacks would get distributed vectors).
//  - GOTCHA(bpack-errsol): ButterflyPACK's H keeps no forward operator after construction
//    unless ErrSol = 1 (or precon > 1); a matvec then segfaults. keep_forward (default true)
//    sets ErrSol = 1, which keeps a second copy of every H block (about doubling the
//    matrix's memory) and makes ButterflyPACK's factor and solve run one extra test solve.
//    With keep_forward = false, bpack_blackbox refuses H matrices. HODLR keeps its blocks
//    after construction either way, but factorization deletes them unless keep_forward
//    (bpack_blackbox then refuses the factored HODLR too).
//
// Factorization and solves (bpack_factor, bpack_solve_blackbox): HODLR by ButterflyPACK's
// recursive inversion, exact for the stored HODLR with low-rank blocks (no sampling); H by
// H-LU, truncated at the tolerance. Both work on matrices built from matvecs or entries. With
// keep_forward = true the forward operator is untouched: bpack_blackbox gives the same
// products, bit for bit, before and after. Measured at N = 2,000 (tolerance 1e-6, kernel with
// a diagonal shift of 10, tests/test_bpack_solve.cpp): factorization 0.05 s (HODLR) and 0.24 s
// (H), against 0.6-1.3 s for the construction; the residual ||K (A^-1 x) - x|| / ||x|| against
// the true kernel K is 2-3.3x the rebuilt matrix's own forward error (HODLR 2.4-4.5e-6, H
// 1.1-11e-6), a factor that grows with N (3.4 at N = 4,000); against the rebuilt matrix
// itself it is 1e-15 for HODLR and 8e-7 for H. Transposed solves: HODLR yes; ButterflyPACK has
// no transposed H-LU solve (it stops the process), so the adapter refuses 'T' for H.
// GOTCHA(inverse-asymmetry): the solve of a symmetric A is not symmetric. The HODLR is symmetric
// only up to its errors, and the inversion amplifies them: ||(S - S^T) x|| / ||S x|| = 2e-5 for
// a HODLR of N = 4,000 at tolerance 1e-6 (the HODLR's own asymmetry: 6e-6). A consumer that
// assumes symmetry (H2Opus's HARA) should get (S + S^T) / 2, from the transposed HODLR solve
// (examples/chain_four_libraries.cpp).
//
// Ownership and threads: a handle owns its ButterflyPACK objects; the BlackBox returned by
// bpack_blackbox or bpack_solve_blackbox keeps the matrix alive. The source (BlackBox or
// EntrySource) is used only during the construction call. All calls into ButterflyPACK
// (constructions, factorizations, products, solves, destruction) are serialized by one
// process-wide recursive mutex, because ButterflyPACK keeps global Fortran state; handles can
// be used from any thread, and a source may itself call ButterflyPACK from inside a callback
// (e.g. rebuild one ButterflyPACK matrix from another one's BlackBox).
//
// Limits: square, real double, one process, low-rank blocks only. Errors are reported by
// exceptions whose message starts with the function name; an exception thrown by the source
// is caught (it must not cross ButterflyPACK's Fortran frames), the remaining samples are
// answered with zeros, and it is rethrown as std::runtime_error when ButterflyPACK returns.
#pragma once
#include "hmb/blackbox.hpp"
#include "hmb/entry_source.hpp"
#include <memory>

namespace hmb {

struct BpackMatrix; // ButterflyPACK handles + permutation; defined in bpack_native.hpp
using BpackHandle = std::shared_ptr<BpackMatrix>;

enum class BpackFormat { HODLR, H };

struct BpackOptions {
    BpackFormat format = BpackFormat::HODLR;
    // ButterflyPACK's relative tolerance (tol_comp and tol_rand; rank detection at a tenth of
    // it). GOTCHA(tolerance-stacking): the matvec route misses it, H sometimes by far (above).
    double tolerance = 1e-6;
    int leaf_size = 64;      // largest leaf cluster (Nmin_leaf)
    double near_para = 2.0;  // H admissibility (ButterflyPACK's near_para; 2: strong); unused by HODLR
    // Matvec route: a level is first sampled with rank0 + 1 vectors; each retry multiplies
    // the count by rankrate; after itermax tries ButterflyPACK STOPs the process
    // (GOTCHA(fortran-stop)). rank0 near the largest expected rank saves retries.
    int rank0 = 64;
    double rankrate = 2.0;
    int itermax = 10;
    bool keep_forward = true; // ErrSol = 1: keep the forward operator (GOTCHA(bpack-errsol))
    // Matvec route: after the construction, compare the rebuilt matrix with the source on
    // this many random vectors (one more source call) -> BpackStats::check_error. 0: skip.
    int check_vectors = 4;
    int verbosity = -1;       // ButterflyPACK's own output: -1 none, 0 summaries, 1 details
    // Exit status if ButterflyPACK ends the process during this construction; 0 keeps
    // ButterflyPACK's own status (0) and runs the remaining exit handlers.
    int stop_exit_status = 70;
};

struct BpackStats {
    // Matvec route: ButterflyPACK's callback calls and vectors (columns), and the share that
    // asked for A^T (the check below is not included).
    long calls = 0, calls_transposed = 0;
    long long vectors = 0, vectors_transposed = 0;
    // Matvec route: ||(B - A) X||_F / ||A X||_F for check_vectors Gaussian vectors X, with A
    // the source and B the rebuilt matrix; -1 when not measured (entry route, 0 vectors, or
    // an H matrix built with keep_forward = false).
    double check_error = -1;
    // Entry route: block callbacks and entries evaluated.
    long entry_calls = 0;
    long long entries = 0;
    double seconds_init = 0;      // clustering and setup (construct_init)
    double seconds_build = 0;     // the construction itself, callbacks included
    double seconds_callback = 0;  // inside the adapter's callbacks (source + permutation copies)
    double seconds_in_source = 0; // inside the source's apply / get
    int rank_max = 0;             // largest rank, as reported by ButterflyPACK
    // ButterflyPACK's own memory count (stats "Mem_Fill"), in MB. For H built from matvecs it
    // leaves out the low-rank blocks (ButterflyPACK does not count them on that route).
    double bpack_mem_fill_mb = 0;
};

// Rebuilds the operator A (A.n x A.n, original ordering) from products. dim, coords: the
// points of the rows/columns (see Conventions). Throws std::invalid_argument for bad
// arguments, std::runtime_error if the source throws or ButterflyPACK misbehaves.
BpackHandle bpack_from_matvec(const BlackBox &A, int dim, const double *coords, const BpackOptions &opt = {},
                              BpackStats *stats = nullptr);

// Rebuilds the matrix A (square) from its entries with ButterflyPACK's own compression.
BpackHandle bpack_from_entries(const EntrySource &A, int dim, const double *coords, const BpackOptions &opt = {},
                               BpackStats *stats = nullptr);

// The rebuilt matrix as a matvec operator in the ORIGINAL ordering; 'N' and 'T' (exact
// transpose of the stored matrix, which is not exactly symmetric: symmetric == false).
// Throws std::logic_error for an H matrix built with keep_forward = false, and for a HODLR
// built so once it is factored (apply then throws too).
BlackBox bpack_blackbox(const BpackHandle &A);

int bpack_size(const BpackHandle &A);
BpackStats bpack_stats(const BpackHandle &A); // statistics of the construction
const char *bpack_format_name(BpackFormat f); // "HODLR", "H"

struct BpackFactorStats {
    bool factored = false;
    double seconds = 0;             // the factorization (ButterflyPACK's ErrSol test solve included)
    int rank_max = 0;               // largest rank of the matrix and its factors (ButterflyPACK's "Rank_max")
    double bpack_mem_factor_mb = 0; // ButterflyPACK's own count for the factors (stats "Mem_Factor")
};

// Factors A in place: HODLR by recursive inversion, H by H-LU. A second call does nothing.
// With keep_forward = true (the default) bpack_blackbox keeps multiplying by A, with the same
// results as before. With keep_forward = false the factorization deletes a HODLR's forward
// blocks (GOTCHA(bpack-errsol)): its BlackBoxes throw std::logic_error from then on.
void bpack_factor(const BpackHandle &A, BpackFactorStats *stats = nullptr);

// A^-1 as a BlackBox in the ORIGINAL ordering (ButterflyPACK's inv_mult); factors A first
// (bpack_factor) if that was not done. 'T' applies A^-T for HODLR; for H, apply('T') throws
// std::logic_error (ButterflyPACK has no transposed H-LU solve and would stop the process).
// symmetric == false, also for a symmetric A (GOTCHA(inverse-asymmetry)). The BlackBox keeps
// the matrix alive.
BlackBox bpack_solve_blackbox(const BpackHandle &A);

BpackFactorStats bpack_factor_stats(const BpackHandle &A); // factored == false before bpack_factor

} // namespace hmb
