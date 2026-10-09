/* move-hmatrix: hmb_cstore implementation (see include/hmb/cstore.h). */
#include "hmb/cstore.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *what) {
  fprintf(stderr, "hmb_cstore: %s\n", what);
  abort();
}

/* Grows *cap (doubling, at least 16) until it holds need elements; new slots are zeroed. */
static void *grow(void *p, int *cap, int need, size_t elem) {
  if (need <= *cap) return p;
  size_t c = *cap ? (size_t)*cap : 16;
  while (c < (size_t)need) c *= 2;
  if (c > INT_MAX) c = INT_MAX;
  void *q = realloc(p, c * elem);
  if (!q) die("out of memory");
  memset((char *)q + (size_t)*cap * elem, 0, (c - (size_t)*cap) * elem);
  *cap = (int)c;
  return q;
}

/* Number of elements after appending n to count; aborts on a negative n or an overflow. */
static int grown(int count, int n) {
  if (n < 0) die("negative number of elements requested");
  if (n > INT_MAX - count) die("too many elements (int overflow)");
  return count + n;
}

void hmb_cstore_init(hmb_cstore *s) { memset(s, 0, sizeof *s); }

void hmb_cstore_free(hmb_cstore *s) {
  for (int i = 0; i < s->nchunks; i++) free(s->chunks[i]);
  free(s->chunks);
  free(s->cl);
  free(s->bl);
  free(s->sons);
  free(s->row_perm);
  if (s->col_perm != s->row_perm) free(s->col_perm);
  memset(s, 0, sizeof *s);
}

int hmb_cstore_add_clusters(hmb_cstore *s, int n) {
  int first = s->ncl, total = grown(s->ncl, n);
  if (n == 0) return first;
  s->cl = (hmb_cluster *)grow(s->cl, &s->cap_cl, total, sizeof(hmb_cluster));
  memset(s->cl + first, 0, (size_t)n * sizeof(hmb_cluster));
  s->ncl = total;
  return first;
}

int hmb_cstore_add_block(hmb_cstore *s, const hmb_block *b) {
  s->bl = (hmb_block *)grow(s->bl, &s->cap_bl, grown(s->nbl, 1), sizeof(hmb_block));
  s->bl[s->nbl] = *b;
  return s->nbl++;
}

int hmb_cstore_add_sons(hmb_cstore *s, int n) {
  int first = s->nsons, total = grown(s->nsons, n);
  if (n == 0) return first;
  s->sons = (int *)grow(s->sons, &s->cap_sons, total, sizeof(int));
  for (int i = 0; i < n; i++) s->sons[first + i] = -1;
  s->nsons = total;
  return first;
}

double *hmb_cstore_alloc(hmb_cstore *s, size_t n) {
  s->chunks = (double **)grow(s->chunks, &s->cap_chunks, grown(s->nchunks, 1), sizeof(double *));
  double *p = (double *)calloc(n ? n : 1, sizeof(double));
  if (!p) die("out of memory for leaf data");
  s->chunks[s->nchunks++] = p;
  return p;
}

int *hmb_cstore_perm(hmb_cstore *s, int which, int n) {
  if (n < 0) die("negative permutation length");
  int **slot = which == 0 ? &s->row_perm : &s->col_perm;
  if (*slot && (which == 0 ? s->col_perm : s->row_perm) != *slot) free(*slot);
  *slot = (int *)malloc((size_t)(n ? n : 1) * sizeof(int));
  if (!*slot) die("out of memory");
  for (int i = 0; i < n; i++) (*slot)[i] = i;
  return *slot;
}

const hmb_hmatrix *hmb_cstore_finalize(hmb_cstore *s) {
  s->A.ncl = s->ncl;
  s->A.cl = s->cl;
  s->A.nbl = s->nbl;
  s->A.bl = s->bl;
  s->A.sons = s->sons;
  s->A.sons_len = s->nsons;
  if (s->row_perm) s->A.row_perm = s->row_perm;
  if (s->col_perm) s->A.col_perm = s->col_perm;
  return &s->A;
}
