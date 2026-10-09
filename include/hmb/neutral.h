/* hmatrix-bridges: neutral in-memory view of an H-matrix.
 *
 * An exporter fills an hmb_hmatrix that points into the source library's storage
 * (or into copies it owns); an importer reads it and builds its own structures.
 * The view itself allocates nothing: whoever produced it owns every array it points
 * to and must keep that storage alive while the view is in use.
 *
 * Conventions (HMB_NEUTRAL_VERSION 1)
 *  - Real double precision only.
 *  - Indices are 0-based.
 *  - The view always describes the FULL matrix. Exporters of one-triangle
 *    (symmetric) storage must mirror the missing blocks.
 *  - Rows and columns are each numbered in a permuted ("tree") order:
 *    row_perm[k] is the ORIGINAL row index at permuted position k (same for col_perm).
 *    A NULL permutation means the identity.
 *  - A cluster covers the contiguous permuted range [offset, offset + size). Its sons
 *    are clusters son0 .. son0 + nsons - 1 and partition it in index order
 *    (nsons == 0 for a leaf). Row and column trees may share clusters.
 *  - bl[0] is the root block; it pairs row_root with col_root.
 *  - An HMB_SUPER block has rsons x csons children. Child (i, j) is
 *    bl[sons[son0 + i + j * rsons]]; its rc / cc fields name its clusters explicitly
 *    (normally the i-th son of rc and the j-th son of cc; a side that is not split has
 *    rsons == 1 or csons == 1 and keeps the parent's cluster).
 *  - An HMB_LOWRANK leaf of size rows x cols is
 *      U * V    (vtrans == 0: V is rank x cols, leading dimension ldV), or
 *      U * V^T  (vtrans == 1: V is cols x rank, leading dimension ldV),
 *    with U rows x rank, leading dimension ldU. rank == 0 is a valid zero block.
 *  - An HMB_DENSE leaf is D, rows x cols, leading dimension ldD.
 *  - All matrices are column-major.
 */
#ifndef HMB_NEUTRAL_H
#define HMB_NEUTRAL_H

#include <stddef.h>

#define HMB_NEUTRAL_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

enum { HMB_SUPER = 0, HMB_DENSE = 1, HMB_LOWRANK = 2 };

typedef struct {
  int offset, size;        /* permuted index range [offset, offset + size) */
  int nsons, son0;         /* sons: clusters son0 .. son0 + nsons - 1 */
  double bmin[3], bmax[3]; /* bounding box, if known (zeros otherwise) */
} hmb_cluster;

typedef struct {
  int rc, cc;              /* row / column cluster ids */
  int kind;                /* HMB_SUPER, HMB_DENSE or HMB_LOWRANK */
  int rsons, csons, son0;  /* HMB_SUPER: child (i,j) = bl[sons[son0 + i + j*rsons]] */
  int rank;                /* HMB_LOWRANK */
  const double *U; int ldU;
  const double *V; int ldV;
  int vtrans;              /* 0: block = U*V (V rank x cols); 1: block = U*V^T (V cols x rank) */
  const double *D; int ldD;/* HMB_DENSE */
} hmb_block;

typedef struct {
  int dim;                 /* spatial dimension of the bounding boxes (0 if none) */
  int n_rows, n_cols;
  const int *row_perm, *col_perm; /* perm[k] = original index at permuted position k */
  int row_root, col_root;  /* cluster ids of the two roots (equal if the tree is shared) */
  int ncl;
  const hmb_cluster *cl;
  int nbl;                 /* bl[0] is the root block */
  const hmb_block *bl;
  const int *sons;         /* child lists of HMB_SUPER blocks */
} hmb_hmatrix;

/* ---- Utilities (src/neutral.c). All are read-only on A and reentrant. ---- */

/* Number of scalars stored in the leaves. */
size_t hmb_stored_values(const hmb_hmatrix *A);

/* Structural checks: sons partition their parents, the block tree tiles the
 * matrix exactly once, leaf dimensions and leading dimensions are consistent.
 * Returns 0 if valid; otherwise writes a reason into msg (if msg != NULL). */
int hmb_validate(const hmb_hmatrix *A, char *msg, size_t msglen);

/* H2Lib "strictness" (needed by H2Lib's H -> H2 conversion):
 * v[0] = dense leaves whose row or column cluster is not a leaf cluster;
 * v[1] = super blocks that leave a non-leaf row or column cluster unsplit. */
void hmb_strict_violations(const hmb_hmatrix *A, long v[2]);

/* Y = op(A) X in the ORIGINAL ordering; op = 'N' or 'T'. X is n_cols x nrhs
 * ('N') or n_rows x nrhs ('T'), leading dimension ldx; Y is overwritten. */
void hmb_matvec(const hmb_hmatrix *A, char trans, int nrhs,
                const double *X, int ldx, double *Y, int ldy);

/* out(i, j) = A(rows[i], cols[j]) for ORIGINAL 0-based indices; out is m x n,
 * column-major, leading dimension ldo. Cost: one tree descent per entry plus a
 * dot product of length rank for entries in low-rank leaves. */
void hmb_entries(const hmb_hmatrix *A, int m, const int *rows, int n, const int *cols,
                 double *out, int ldo);

#ifdef __cplusplus
}
#endif
#endif /* HMB_NEUTRAL_H */
