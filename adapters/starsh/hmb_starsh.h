/* hmatrix-bridges: STARS-H adapter, library-neutral C interface.
 *
 * What it does
 *   hmb_starsh_import        neutral view (hmb_hmatrix) -> STARS-H block low-rank matrix
 *                            (STARSH_blrm), lossless copy of every leaf.
 *   hmb_starsh_from_entries  entries (a callback) -> STARS-H's own tile low-rank (TLR)
 *                            approximation (starsh_blrf_new_tlr + starsh_blrm_approximate).
 *   hmb_starsh_export        any STARSH_blrm -> neutral view; zero-copy for stored blocks.
 *   hmb_starsh_matvec        STARS-H's product starsh_blrm__dmml, in the ORIGINAL ordering.
 *   hmb_starsh_dense         STARS-H's dense conversion starsh_blrm__dca, ORIGINAL ordering.
 *   hmb_starsh_symm          whether a STARS-H matrix stores one triangle ('S') or all blocks.
 * Includes no STARS-H header: STARS-H's matrix crosses the interface as a pointer to the
 * forward-declared struct starsh_blrm, which is STARSH_blrm in starsh.h, so code holding a
 * STARSH_blrm * passes it directly. C++ code can use hmb_starsh.hpp, which wraps this
 * interface in the package's C++ conventions (StarshHandle, exceptions, EntrySource, BlackBox).
 *
 * Which partitions STARS-H can hold
 *   STARS-H's format (STARSH_blrf) is a flat list of "far-field" (low-rank) and
 *   "near-field" (dense) blocks; each block pairs one entry of the row cluster list with one
 *   of the column cluster list, and a cluster is just a contiguous range of the pivoted
 *   order. Its products, dense conversion and error routines walk the lists and never look
 *   at a hierarchy (STARSH_HIERARCHICAL, STARSH_H and STARSH_HODLR are commented out in
 *   starsh-constants.h). So it holds ANY non-nested leaf list with one permutation per side:
 *   uniform tiles (TLR), HODLR, or the leaves of a general H-matrix. hmb_starsh_import
 *   therefore accepts every valid neutral view (it flattens the block tree to its leaves),
 *   and hmb_starsh_export accepts every STARS-H matrix whose blocks tile the matrix exactly
 *   once with nested ("laminar") cluster ranges, i.e. every partition a block tree can
 *   describe. Anything else (overlapping or missing blocks, ranges that overlap without
 *   nesting) is refused with a message. Parts of STARS-H outside this adapter are TLR-only:
 *   the MPI product (starsh_blrm__dmml_mpi_tlr) and HiCMA expect uniform tiles.
 *
 * Conventions
 *   - STARS-H's pivot[k] is the ORIGINAL index at pivoted position k (its kernels receive
 *     pivot + start[i] as row indices): the same direction as the neutral row_perm/col_perm,
 *     copied as is (GOTCHA(perm-direction)). starsh_blrm__dmml and __dca work in the
 *     pivoted order; hmb_starsh_matvec / hmb_starsh_dense permute to and from the ORIGINAL
 *     ordering.
 *   - Low-rank blocks are U * V^T with U rows x rank and V cols x rank, both column-major
 *     with leading dimension = rows (cols); exported as vtrans = 1; on import a vtrans = 0
 *     factor (V rank x cols) is transposed (GOTCHA(lowrank-convention)). Rank 0 is kept as
 *     rank 0 (STARS-H itself stores exactly-zero tiles that way).
 *   - Symmetric format (STARSH_blrf symm == 'S', e.g. starsh_blrf_new_tlr(..., 'S', ...)):
 *     only blocks (i, j) of one triangle are stored and starsh_blrm__dmml also applies their
 *     transposes; the export mirrors them (low-rank: U and V swapped, zero-copy; dense: a
 *     transposed copy), so the view is the full matrix (GOTCHA(symmetric-storage)). The
 *     import always builds a non-symmetric ('N') format holding every leaf.
 *   - Real double precision only (dtype 'd', scalar kernels); anything else is refused
 *     (GOTCHA(scalar-type)).
 *   - Errors: every function returns 0 on success and a negative value otherwise, with a
 *     reason in msg (if msg != NULL).
 *
 * Ownership and lifetimes
 *   - hmb_starsh (from import or from_entries) owns the STARS-H problem, cluster(s), format
 *     and matrix; hmb_starsh_free releases all of them. The imported matrix owns copies of the
 *     leaves (STARS-H frees its factor arrays itself in starsh_blrm_free), so the source view
 *     may be released after the import.
 *   - The problem's kernel of an hmb_starsh evaluates the stored matrix itself (entries are
 *     read from the leaves), so STARS-H routines that fall back on the kernel
 *     (starsh_blrf_get_block, starsh_blrm__dfe, on-the-fly near blocks, a new approximation
 *     on the same problem) see the represented matrix. The user's entry callback is used only
 *     during hmb_starsh_from_entries. STARS-H passes its kernels the problem's row_data in
 *     some routines and the clusters' data in others; the adapter sets both
 *     (GOTCHA(starsh-kernel-data)), and so must code that builds STARS-H objects itself.
 *   - hmb_starsh_export aliases STARS-H's stored factors and dense blocks; the STARSH_blrm
 *     must outlive the view. Mirrored dense blocks of symmetric formats and near blocks of
 *     on-the-fly matrices (onfly == 1, evaluated through the kernel) are copied into the store.
 *
 * Thread safety: import, export and from_entries are not reentrant with respect to the same
 * objects; from_entries calls starsh_init() on first use (process-wide backend selection,
 * from STARS-H's environment variables). hmb_starsh_matvec and hmb_starsh_dense only read
 * the matrix and may run concurrently. The entry callback of from_entries is called
 * concurrently by STARS-H's OpenMP backend unless thread_safe == 0 (then calls are
 * serialized).
 *
 * Limits: 32-bit sizes. STARS-H's OpenMP product starsh_blrm__dmml_omp assumes ranks <= 100
 * (GOTCHA(starsh-dmml-omp)); this adapter always calls the sequential starsh_blrm__dmml.
 * starsh_blrm_get_block returns the factors of another block unless the block lists are
 * sorted by block row (GOTCHA(starsh-get-block)); the import sorts them, the export does not
 * use get_block. STARS-H's approximation leaks ~104 bytes per far-field block when some
 * tiles exceed maxrank (GOTCHA(starsh-false-far-leak); suppressed for LeakSanitizer by
 * tests/lsan.supp).
 *
 * Tested against STARS-H commit c115e75 (0.3.1), OpenMP backend, OpenBLAS + LAPACKE.
 */
#ifndef HMB_STARSH_H
#define HMB_STARSH_H

#include "hmb/cstore.h"
#include "hmb/neutral.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct starsh_blrm; /* STARSH_blrm (starsh.h) */

/* Owner of a STARS-H matrix created by this adapter (problem, clusters, format, matrix). */
typedef struct hmb_starsh hmb_starsh;

/* ---- neutral view -> STARS-H ---------------------------------------------------------- */

/* Copies every leaf of a valid neutral view into a new STARS-H matrix (format 'N', blocks
 * sorted by row cluster then column cluster: GOTCHA(starsh-get-block)). The row (column)
 * cluster list holds the distinct row (column) ranges the leaves use, sorted by offset
 * (larger first on ties); for a uniform tile partition that is exactly the layout of
 * starsh_cluster_new_plain. Rows and columns share one STARS-H cluster object when the
 * view's two trees and permutations coincide. */
int hmb_starsh_import(const hmb_hmatrix *A, hmb_starsh **M, char *msg, size_t msglen);

/* ---- entries -> STARS-H TLR approximation ---------------------------------------------- */

/* out(i, j) = A(rows[i], cols[j]) for ORIGINAL 0-based indices; out is m x n, column-major,
 * leading dimension ldo. */
typedef void hmb_starsh_entry_fn(int m, int n, const int *rows, const int *cols, double *out, int ldo,
                                 void *ctx);

typedef struct {
  int tile_size;   /* rows (and columns) per tile; the last tile holds the remainder */
  double tol;      /* STARS-H's per-tile relative Frobenius tolerance */
  int maxrank;     /* tiles needing a larger rank are stored dense ("false far-field") */
  char symm;       /* 'N' every tile; 'S' lower tiles only (A must be symmetric) */
  int thread_safe; /* nonzero: entries may be called concurrently */
} hmb_starsh_tlr_opts;

/* Builds the TLR approximation of an n x n matrix given by its entries, with STARS-H's
 * own routines: uniform tiles over the order perm (perm[k] = original index at position k;
 * NULL = identity; a clustering permutation makes the tiles compressible), every tile
 * (diagonal ones included) compressed by starsh_blrm_approximate (randomized SVD by default;
 * STARSH_BACKEND / STARSH_LRENGINE / STARSH_OVERSAMPLE select others) and dense blocks stored
 * (onfly = 0). Calls starsh_init() on first use (GOTCHA(starsh-init)). */
int hmb_starsh_from_entries(int n, const int *perm, hmb_starsh_entry_fn *entries, void *ctx,
                            const hmb_starsh_tlr_opts *opts, hmb_starsh **M, char *msg, size_t msglen);

/* The STARS-H matrix (borrowed; owned by M). */
struct starsh_blrm *hmb_starsh_blrm(const hmb_starsh *M);
/* Zero-copy neutral view of M (as hmb_starsh_export would build it; owned by M). */
const hmb_hmatrix *hmb_starsh_view(const hmb_starsh *M);
void hmb_starsh_free(hmb_starsh *M);

/* ---- STARS-H -> neutral view ----------------------------------------------------------- */

typedef struct {
  long long far_blocks, near_blocks; /* stored blocks */
  long long mirrored_blocks;         /* blocks of the unstored triangle (symmetric formats) */
  long long aliased_values;          /* values the view points to in STARS-H's storage */
  long long copied_values;           /* values copied into the store (mirrored dense, onfly) */
  int max_rank;
} hmb_starsh_export_stats;

/* Fills an initialized (empty) store with a neutral view of M and finalizes it; the view is
 * store->A. The cluster tree is rebuilt from the cluster ranges the blocks use (STARS-H keeps
 * no hierarchy); rows and columns share one tree when M uses one cluster object for both.
 * The result is checked with hmb_validate. stats may be NULL. */
int hmb_starsh_export(struct starsh_blrm *M, hmb_cstore *store, hmb_starsh_export_stats *stats, char *msg,
                      size_t msglen);

/* ---- operations in the ORIGINAL ordering --------------------------------------------- */

/* 'S' when M stores one triangle (symmetric format; starsh_blrm__dmml applies the full
 * symmetric matrix, so it is its own transpose), 'N' when it stores every block, 0 for an
 * incomplete matrix (no format). Matrices from hmb_starsh_import are always 'N'. */
char hmb_starsh_symm(const struct starsh_blrm *M);

/* Y = A X with starsh_blrm__dmml; X is n_cols x nrhs (leading dimension ldx), Y is
 * n_rows x nrhs (ldy), both in the ORIGINAL ordering. STARS-H has no transposed product
 * (use hmb_matvec on the exported view for A^T X; a symmetric format is its own transpose). */
int hmb_starsh_matvec(struct starsh_blrm *M, int nrhs, const double *X, int ldx, double *Y, int ldy, char *msg,
                      size_t msglen);

/* D = A (n_rows x n_cols, column-major, leading dimension ldd) with starsh_blrm__dca, in the
 * ORIGINAL ordering. Allocates one n_rows x n_cols temporary. */
int hmb_starsh_dense(struct starsh_blrm *M, double *D, int ldd, char *msg, size_t msglen);

#ifdef __cplusplus
}
#endif
#endif /* HMB_STARSH_H */
