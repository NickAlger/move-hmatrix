/* hmatrix-bridges: growable owner behind an hmb_hmatrix view, for code written in C
 * (exporters and file parsers). C++ code can use hmb::NeutralStore instead.
 *
 * Usage:
 *   hmb_cstore s; hmb_cstore_init(&s);
 *   int root = hmb_cstore_add_clusters(&s, 1);          // ids are indices into s.cl
 *   s.cl[root].offset = 0; s.cl[root].size = n;
 *   int s0 = hmb_cstore_add_clusters(&s, 2);            // sons must have consecutive ids
 *   s.cl[root].son0 = s0; s.cl[root].nsons = 2; ...
 *   hmb_block b = {0}; ...; int id = hmb_cstore_add_block(&s, &b);   // first block = root
 *   double *U = hmb_cstore_alloc(&s, rows * rank);      // owned leaf data (zeroed)
 *   const hmb_hmatrix *A = hmb_cstore_finalize(&s);     // fills s.A's array pointers
 *   ...
 *   hmb_cstore_free(&s);
 *
 * Adding clusters, blocks or son slots may move s.cl / s.bl / s.sons: always index through
 * the store (s.cl[id]) rather than keeping element pointers, and call hmb_cstore_finalize()
 * after the last addition. Leaf data from hmb_cstore_alloc() never moves.
 */
#ifndef HMB_CSTORE_H
#define HMB_CSTORE_H

#include "hmb/neutral.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  hmb_hmatrix A;            /* the view (dim, n_rows, n_cols, roots: set these yourself) */
  hmb_cluster *cl;  int ncl,   cap_cl;
  hmb_block   *bl;  int nbl,   cap_bl;
  int         *sons; int nsons, cap_sons;
  int *row_perm, *col_perm; /* owned permutation arrays (NULL unless allocated) */
  double **chunks;  int nchunks, cap_chunks;
} hmb_cstore;

void hmb_cstore_init(hmb_cstore *s);
/* Frees everything the store owns; the view becomes invalid. */
void hmb_cstore_free(hmb_cstore *s);

/* Appends n zero-initialised clusters; returns the id of the first. */
int hmb_cstore_add_clusters(hmb_cstore *s, int n);
/* Appends a copy of *b; returns its id. The first block added is the root (bl[0]). */
int hmb_cstore_add_block(hmb_cstore *s, const hmb_block *b);
/* Appends n son slots (initialised to -1); returns the index of the first (a block's son0). */
int hmb_cstore_add_sons(hmb_cstore *s, int n);
/* Zero-initialised owned storage for n doubles (never moves; freed by hmb_cstore_free). */
double *hmb_cstore_alloc(hmb_cstore *s, size_t n);
/* Owned permutation array of length n for rows (which == 0) or columns (which == 1),
 * initialised to the identity; the view points at it after hmb_cstore_finalize(). */
int *hmb_cstore_perm(hmb_cstore *s, int which, int n);
/* Points s->A at the store's arrays and returns &s->A. Call after the last addition. */
const hmb_hmatrix *hmb_cstore_finalize(hmb_cstore *s);

#ifdef __cplusplus
}
#endif
#endif /* HMB_CSTORE_H */
