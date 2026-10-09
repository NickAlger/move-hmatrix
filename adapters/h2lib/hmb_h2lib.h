/* move-hmatrix: H2Lib adapter, library-neutral C interface.
 *
 * Moves H-matrices between the neutral view (hmb/neutral.h) and H2Lib, converts them to H2
 * inside H2Lib, and multiplies H2Lib matrices in the ORIGINAL (user) ordering. Includes no
 * H2Lib header: H2Lib's matrices cross this interface as pointers to its own structs, which
 * are only forward-declared here (struct _hmatrix is H2Lib's hmatrix, phmatrix a pointer to
 * it), so this header can share a translation unit with any other neutral-facing header
 * (GOTCHA(header-clash): H2Lib defines macros named I and UINT_MAX and a type named real).
 * Code that already uses H2Lib passes its phmatrix / ph2matrix directly.
 *
 * Tested against H2Lib 920662c (community branch) built with OPT=1 USE_BLAS=1 USE_OPENMP=1,
 * i.e. real double precision. The adapter is compiled with the -D flags of libh2.a
 * (H2LIB_CPPFLAGS): struct layouts depend on them, and the adapter refuses to compile for a
 * complex or single-precision build (GOTCHA(scalar-type)).
 *
 * Conventions
 *  - H2Lib clusters hold index arrays: idx[k] = original index at position k of the cluster,
 *    sons' arrays consecutive inside their father's, so the neutral permutation is the root's
 *    idx and a cluster's offset is its position in it.
 *  - H2Lib's low-rank blocks are A * B^H with A rows x k and B cols x k (rkmatrix), that is
 *    the neutral U * V^T with vtrans = 1 for real data (GOTCHA(lowrank-convention)).
 *  - A super block's son (i, j) is son[i + j * rsons], its row cluster the father's row
 *    cluster (rows not split) or its i-th son: the neutral layout.
 *  - Matrices are column-major with a leading dimension (amatrix.ld).
 *
 * Errors: functions return 0 or a negative HMB_H2LIB_E* code, with a message in msg (if
 * msg != NULL). H2Lib itself aborts on failed asserts (GOTCHA(h2lib-omp-asserts)); the
 * adapter checks its inputs first so that H2Lib's asserts are never reached through it.
 *
 * Thread safety: the functions are reentrant for distinct objects; H2Lib's own routines
 * may use OpenMP internally (see hmb_h2lib_init). hmb_h2lib_init is not thread-safe.
 */
#ifndef HMB_H2LIB_H
#define HMB_H2LIB_H

#include <stddef.h>

#include "hmb/cstore.h"
#include "hmb/neutral.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _hmatrix;  /* H2Lib's hmatrix (phmatrix) */
struct _h2matrix; /* H2Lib's h2matrix (ph2matrix) */

/* An imported matrix: H2Lib cluster trees, their index arrays and the hmatrix. */
typedef struct hmb_h2lib_hm hmb_h2lib_hm;

enum {
  HMB_H2LIB_EINVAL = -1,      /* invalid argument or view (see msg) */
  HMB_H2LIB_ENOTSTRICT = -2,  /* H -> H2 conversion asked for a non-strict block tree */
  HMB_H2LIB_EUNSUPPORTED = -3 /* an H2Lib structure the neutral form cannot describe */
};

/* Import modes (flags; BORROW | STRICT is allowed). */
enum {
  HMB_H2LIB_COPY = 0,   /* copy every leaf: H2Lib owns all of its data */
  HMB_H2LIB_BORROW = 1, /* alias dense blocks and U, and V when the view stores it cols x
                           rank (vtrans = 1); other V are transposed into H2Lib storage. The
                           view's leaf data must outlive the result and must not change; the
                           result is read-only: H2Lib routines that modify blocks in place
                           (truncation, arithmetic, factorizations) would abort on a borrowed
                           block or write into the source library's storage */
  HMB_H2LIB_STRICT = 2  /* refine the block tree into an H2Lib-strict one while importing,
                           losslessly: dense leaves on non-leaf clusters are split down to
                           leaf clusters, super blocks that leave a non-leaf cluster unsplit
                           are split on every non-leaf side, low-rank leaves are restricted */
};

typedef struct {
  size_t copied;     /* leaf scalars copied into H2Lib storage */
  size_t borrowed;   /* leaf scalars aliased (HMB_H2LIB_BORROW) */
  size_t index_ints; /* permutation entries copied into H2Lib's index arrays */
  long dense_leaves, lowrank_leaves; /* leaves of the H2Lib matrix */
} hmb_h2lib_import_info;

/* Optional set-up of H2Lib's parallelism: calls init_h2lib() (once), then sets H2Lib's
 * recursion depth for OpenMP to pardepth (pardepth < 0 keeps H2Lib's choice: $H2_PARDEPTH,
 * else log2(number of processors) + 1; 0 runs H2Lib's own loops serially). Without this
 * call H2Lib runs serially. GOTCHA(h2lib-omp-asserts): init_h2lib() turns on nested OpenMP
 * for the whole process; the previous setting (max active levels) is restored here, so
 * H2Lib's nested parallel regions only get threads at their first level. */
int hmb_h2lib_init(int pardepth);

/* Builds H2Lib cluster trees and an hmatrix holding the view's blocks (mode above). The
 * view is validated first. Rows and columns share one H2Lib cluster tree when they share
 * clusters and permutation. Clusters get the view's bounding boxes (dimension A->dim). A
 * low-rank block with no rows or no columns holds nothing (its factors may be NULL) and
 * becomes a rank-0 leaf. The index arrays are copied (H2Lib wants uint arrays it can own). */
int hmb_h2lib_import(const hmb_hmatrix *A, int mode, hmb_h2lib_hm **out, hmb_h2lib_import_info *info, char *msg,
                     size_t msglen);
/* The imported hmatrix (owned by M). */
struct _hmatrix *hmb_h2lib_hmatrix(const hmb_h2lib_hm *M);
/* Frees M. Free anything built on M's clusters first (an H2 matrix from hmb_h2lib_to_h2). */
void hmb_h2lib_free(hmb_h2lib_hm *M);

/* Exports an H2Lib hmatrix as a neutral view in `out`, which must be initialised and empty
 * (hmb_cstore_init). Zero-copy: the view points at A's dense blocks and rkmatrix factors
 * (U = A, V = B, vtrans = 1) and at its root clusters' index arrays; the store owns only the
 * tree arrays. A and its clusters must outlive the view. A leaf with no data is a rank-0
 * block; bounding boxes of more than 3 dimensions are left out (dim 0). Returns
 * HMB_H2LIB_EUNSUPPORTED for a cluster whose sons' index arrays are not consecutive inside
 * its own, or a son block on a cluster that is neither its father's nor one of its sons. */
int hmb_h2lib_export(const struct _hmatrix *A, hmb_cstore *out, char *msg, size_t msglen);

/* H2Lib strictness of A itself (as hmb_strict_violations for a view):
 * v[0] = dense leaves on a non-leaf row or column cluster,
 * v[1] = super blocks whose sons keep a non-leaf row or column cluster. */
void hmb_h2lib_strict_violations(const struct _hmatrix *A, long v[2]);

typedef struct {
  double rowbasis_s, colbasis_s, projection_s; /* time of the three steps */
  size_t bytes, near_bytes, far_bytes;         /* H2Lib's getsize_h2matrix and parts */
  unsigned rank_sum_row, rank_sum_col;         /* sums of the cluster basis ranks */
  unsigned rank_max_row, rank_max_col;         /* largest cluster basis ranks */
} hmb_h2lib_h2_info;

/* H -> H2 inside H2Lib (compress_hmatrix_h2matrix's three steps with blockwise relative
 * Euclidean truncation at eps). GOTCHA(strict-tree): H2Lib's conversion needs a strict block
 * tree and on any other returns a wrong H2 matrix or crashes; this refuses A with
 * HMB_H2LIB_ENOTSTRICT unless both strictness counts are zero (import with HMB_H2LIB_STRICT
 * to get a strict tree losslessly). The H2 matrix uses A's clusters: free it before them.
 * A itself may be freed after the call. */
int hmb_h2lib_to_h2(const struct _hmatrix *A, double eps, struct _h2matrix **out, hmb_h2lib_h2_info *info, char *msg,
                    size_t msglen);
void hmb_h2lib_free_h2(struct _h2matrix *A);

/* Y = op(A) X in the ORIGINAL ordering, op = 'N' or 'T'. X is n_cols x nrhs ('N') or
 * n_rows x nrhs ('T') with leading dimension ldx; Y is overwritten (ldy). */
void hmb_h2lib_matvec(const struct _hmatrix *A, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy);
void hmb_h2lib_h2_matvec(const struct _h2matrix *A, char trans, int nrhs, const double *X, int ldx, double *Y,
                         int ldy);

/* Sizes (rows, cols) of an hmatrix or h2matrix. */
void hmb_h2lib_size(const struct _hmatrix *A, int *rows, int *cols);
void hmb_h2lib_h2_size(const struct _h2matrix *A, int *rows, int *cols);

/* Bytes H2Lib allocated for A (getsize_hmatrix: borrowed leaf data is not counted). */
size_t hmb_h2lib_bytes(const struct _hmatrix *A);

/* ||A - B||_2 / ||A||_2 by H2Lib's power iteration (norm2diff_hmatrix_h2matrix). */
double hmb_h2lib_h2_reldiff(const struct _hmatrix *A, const struct _h2matrix *B);

#ifdef __cplusplus
}
#endif
#endif /* HMB_H2LIB_H */
