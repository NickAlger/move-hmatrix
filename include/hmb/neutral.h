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
 *    bl[sons[son0 + i + j * rsons]]. A side that is split uses all sons of its cluster:
 *    rsons == cl[rc].nsons and child (i, j) has rc == cl[rc].son0 + i. A side that is not
 *    split has rsons == 1 and the children keep the parent's cluster. Same for columns.
 *    Over a cluster with a single son both have rsons == 1: the children's cluster (the
 *    son, or the parent itself) tells them apart (H2Lib's regular clustering makes such
 *    clusters). Every block is reached from bl[0] exactly once.
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
  int dim;                 /* spatial dimension of the bounding boxes, 0 to 3 (0 if none) */
  int n_rows, n_cols;
  const int *row_perm, *col_perm; /* perm[k] = original index at permuted position k */
  int row_root, col_root;  /* cluster ids of the two roots (equal if the tree is shared) */
  int ncl;
  const hmb_cluster *cl;
  int nbl;                 /* bl[0] is the root block */
  const hmb_block *bl;
  int sons_len;            /* length of sons[] */
  const int *sons;         /* child lists of HMB_SUPER blocks */
} hmb_hmatrix;

/* ---- Utilities (src/neutral.c). All are read-only on A and reentrant. ---- */

/* Number of scalars stored in the leaves. */
size_t hmb_stored_values(const hmb_hmatrix *A);

/* Structural checks: permutations are permutations, sons partition their parents, the
 * block tree follows the cluster trees and tiles the matrix exactly once, leaf pointers
 * and leading dimensions are consistent. Returns 0 if valid; otherwise a negative value,
 * with a reason in msg (if msg != NULL). Cost O(ncl + nbl + n_rows + n_cols). */
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
 * column-major, leading dimension ldo. The request is split down the block tree, so the
 * cost is about (m + n) * depth plus a dot product of length rank for each entry that
 * falls in a low-rank leaf. Computes the inverse permutations on each call (O(n_rows +
 * n_cols)); for many small requests, compute them once and use hmb_entries_ip. An index
 * outside the matrix is a caller bug: reported on stderr, then abort(). */
void hmb_entries(const hmb_hmatrix *A, int m, const int *rows, int n, const int *cols,
                 double *out, int ldo);

/* iperm[original index] = permuted position, for rows (irow, length n_rows) and columns
 * (icol, length n_cols). Either pointer may be NULL to skip it. */
void hmb_inverse_perms(const hmb_hmatrix *A, int *irow, int *icol);

/* hmb_entries with precomputed inverse permutations (from hmb_inverse_perms). irow (icol)
 * may be NULL only when row_perm (col_perm) is NULL; otherwise the call aborts. */
void hmb_entries_ip(const hmb_hmatrix *A, const int *irow, const int *icol, int m, const int *rows,
                    int n, const int *cols, double *out, int ldo);

#ifdef __cplusplus
}
#endif
#endif /* HMB_NEUTRAL_H */
