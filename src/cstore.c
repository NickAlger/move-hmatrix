/* hmatrix-bridges: hmb_cstore implementation (see include/hmb/cstore.h). */
#include "hmb/cstore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *grow(void *p, int *cap, int need, size_t elem) {
  if (need <= *cap) return p;
  int c = *cap ? *cap : 16;
  while (c < need) c *= 2;
  void *q = realloc(p, (size_t)c * elem);
  if (!q) {
    fprintf(stderr, "hmb_cstore: out of memory\n");
    abort();
  }
  memset((char *)q + (size_t)*cap * elem, 0, (size_t)(c - *cap) * elem);
  *cap = c;
  return q;
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
  int first = s->ncl;
  s->cl = (hmb_cluster *)grow(s->cl, &s->cap_cl, s->ncl + n, sizeof(hmb_cluster));
  memset(s->cl + first, 0, (size_t)n * sizeof(hmb_cluster));
  s->ncl += n;
  return first;
}

int hmb_cstore_add_block(hmb_cstore *s, const hmb_block *b) {
  s->bl = (hmb_block *)grow(s->bl, &s->cap_bl, s->nbl + 1, sizeof(hmb_block));
  s->bl[s->nbl] = *b;
  return s->nbl++;
}

int hmb_cstore_add_sons(hmb_cstore *s, int n) {
  int first = s->nsons;
  s->sons = (int *)grow(s->sons, &s->cap_sons, s->nsons + n, sizeof(int));
  for (int i = 0; i < n; i++) s->sons[first + i] = -1;
  s->nsons += n;
  return first;
}

double *hmb_cstore_alloc(hmb_cstore *s, size_t n) {
  s->chunks = (double **)grow(s->chunks, &s->cap_chunks, s->nchunks + 1, sizeof(double *));
  double *p = (double *)calloc(n ? n : 1, sizeof(double));
  if (!p) {
    fprintf(stderr, "hmb_cstore: out of memory (%zu doubles)\n", n);
    abort();
  }
  s->chunks[s->nchunks++] = p;
  return p;
}

int *hmb_cstore_perm(hmb_cstore *s, int which, int n) {
  int **slot = which == 0 ? &s->row_perm : &s->col_perm;
  if (*slot && (which == 0 ? s->col_perm : s->row_perm) != *slot) free(*slot);
  *slot = (int *)malloc((size_t)(n ? n : 1) * sizeof(int));
  if (!*slot) {
    fprintf(stderr, "hmb_cstore: out of memory\n");
    abort();
  }
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
