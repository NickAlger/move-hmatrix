/* Native STARS-H helpers for tests/test_starsh.cpp: matrices built directly with STARS-H's
 * API (kernel approximations with on-the-fly blocks, block lists in column order, broken
 * partitions) and probes of STARS-H's own routines. Kept in C so that the test itself
 * includes no STARS-H header. */
#include "test_starsh_native.h"

#include <starsh.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct ts_matrix {
  STARSH_problem *P;
  STARSH_cluster *C;
  STARSH_blrf *F;
  STARSH_blrm *M;
};

struct starsh_blrm *ts_blrm(ts_matrix *t) { return t->M; }

void ts_free(ts_matrix *t) {
  if (!t) return;
  if (t->M) starsh_blrm_free(t->M);
  if (t->F) starsh_blrf_free(t->F);
  if (t->C) starsh_cluster_free(t->C);
  if (t->P) starsh_problem_free(t->P);
  free(t);
}

/* K(x, y) = 1 / (0.1 + |x - y|) on points xyz[3 * i + d] (the Sym kernel of hmb_test.hpp) */
static void sphere_kernel(int nrows, int ncols, STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
                          void *result, int ld) {
  const double *x = (const double *)row_data, *y = (const double *)col_data;
  double *out = (double *)result;
  for (int j = 0; j < ncols; j++)
    for (int i = 0; i < nrows; i++) {
      const double *a = x + 3 * irow[i], *b = y + 3 * icol[j];
      double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      out[i + (size_t)j * ld] = 1.0 / (0.1 + sqrt(dx * dx + dy * dy + dz * dz));
    }
}

static void zero_kernel(int nrows, int ncols, STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
                        void *result, int ld) {
  (void)irow, (void)icol, (void)row_data, (void)col_data;
  for (int j = 0; j < ncols; j++) memset((double *)result + (size_t)j * ld, 0, sizeof(double) * (size_t)nrows);
}

/* data: the kernel's context (STARS-H passes the cluster's data to kernels in its
 * approximation and product routines, the problem's row_data in get_block) */
static STARSH_cluster *tile_cluster(int n, const int *perm, int tile, void *data) {
  int nb = (n + tile - 1) / tile;
  STARSH_int *pivot = malloc(sizeof(STARSH_int) * n), *start = malloc(sizeof(STARSH_int) * nb),
             *size = malloc(sizeof(STARSH_int) * nb);
  for (int k = 0; k < n; k++) pivot[k] = perm ? perm[k] : k;
  for (int k = 0; k < nb; k++) {
    start[k] = (STARSH_int)k * tile;
    size[k] = k == nb - 1 ? n - (STARSH_int)k * tile : tile;
  }
  STARSH_cluster *C = NULL;
  starsh_cluster_new(&C, data, n, pivot, nb, 0, NULL, start, size, NULL, NULL, NULL, STARSH_PLAIN);
  return C;
}

ts_matrix *ts_kernel_tlr(int n, const double *xyz, const int *perm, int tile, double tol, int maxrank, char symm,
                         int onfly) {
  if (!starsh_blrm_approximate) starsh_init();
  ts_matrix *t = calloc(1, sizeof(ts_matrix));
  STARSH_int shape[2] = {n, n};
  char name[] = "sphere kernel";
  t->C = tile_cluster(n, perm, tile, (void *)xyz);
  int info = starsh_problem_new(&t->P, 2, shape, symm, 'd', (void *)xyz, (void *)xyz, sphere_kernel, name);
  if (info == 0) info = starsh_blrf_new_tlr(&t->F, t->P, symm, t->C, t->C);
  if (info == 0) info = starsh_blrm_approximate(&t->M, t->F, maxrank, tol, onfly);
  if (info) {
    ts_free(t);
    return NULL;
  }
  return t;
}

/* random tile factors; the far-field list enumerates tiles column by column */
ts_matrix *ts_random_tlr_colmajor(int n, int tile, int rank, unsigned seed) {
  ts_matrix *t = calloc(1, sizeof(ts_matrix));
  STARSH_int shape[2] = {n, n};
  char name[] = "random tiles";
  t->C = tile_cluster(n, NULL, tile, NULL);
  starsh_problem_new(&t->P, 2, shape, 'N', 'd', NULL, NULL, zero_kernel, name);
  const int nb = (int)t->C->nblocks;
  STARSH_int nf = (STARSH_int)nb * nb, *bf = malloc(sizeof(STARSH_int) * 2 * nf), k = 0;
  for (int j = 0; j < nb; j++)
    for (int i = 0; i < nb; i++) {
      bf[2 * k] = i;
      bf[2 * k + 1] = j;
      k++;
    }
  starsh_blrf_new_from_coo(&t->F, t->P, 'N', t->C, t->C, nf, bf, 0, NULL, STARSH_TLR);
  int *far_rank = malloc(sizeof(int) * nf);
  Array **U = malloc(sizeof(Array *) * nf), **V = malloc(sizeof(Array *) * nf);
  srand(seed);
  for (k = 0; k < nf; k++) {
    int m = (int)t->C->size[bf[2 * k]], c = (int)t->C->size[bf[2 * k + 1]], su[2] = {m, rank}, sv[2] = {c, rank};
    array_new(U + k, 2, su, 'd', 'F');
    array_new(V + k, 2, sv, 'd', 'F');
    for (int q = 0; q < m * rank; q++) ((double *)U[k]->data)[q] = rand() / (double)RAND_MAX - 0.5;
    for (int q = 0; q < c * rank; q++) ((double *)V[k]->data)[q] = rand() / (double)RAND_MAX - 0.5;
    far_rank[k] = rank;
  }
  starsh_blrm_new(&t->M, t->F, far_rank, U, V, 0, NULL, NULL, NULL, NULL, '2');
  return t;
}

/* Partitions the export must refuse: 0 a tile listed twice, 1 a tile missing, 2 cluster
 * ranges that overlap without nesting ([0, 6) and [4, 10) on a 10 x 10 matrix). */
ts_matrix *ts_broken(int kind) {
  ts_matrix *t = calloc(1, sizeof(ts_matrix));
  const int n = 10;
  STARSH_int shape[2] = {n, n};
  char name[] = "broken";
  starsh_problem_new(&t->P, 2, shape, 'N', 'd', NULL, NULL, zero_kernel, name);
  STARSH_int *pivot = malloc(sizeof(STARSH_int) * n), *start = malloc(sizeof(STARSH_int) * 3),
             *size = malloc(sizeof(STARSH_int) * 3);
  for (int k = 0; k < n; k++) pivot[k] = k;
  /* clusters: 0 = [0, 5), 1 = [5, 10); kind 2 adds 2 = [4, 10) and uses it */
  start[0] = 0, size[0] = 5, start[1] = 5, size[1] = 5, start[2] = 4, size[2] = 6;
  starsh_cluster_new(&t->C, NULL, n, pivot, kind == 2 ? 3 : 2, 0, NULL, start, size, NULL, NULL, NULL, STARSH_PLAIN);
  STARSH_int nn = kind == 1 ? 3 : 4, *bn = malloc(sizeof(STARSH_int) * 2 * nn);
  STARSH_int list[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (int k = 0; k < nn; k++) {
    bn[2 * k] = list[k][0];
    bn[2 * k + 1] = list[k][1];
  }
  if (kind == 0) bn[2 * 3] = 0, bn[2 * 3 + 1] = 0; /* (0, 0) twice, (1, 1) missing */
  if (kind == 2) bn[2 * 3] = 2, bn[2 * 3 + 1] = 1; /* rows [4, 10) next to rows [0, 5) */
  starsh_blrf_new_from_coo(&t->F, t->P, 'N', t->C, t->C, 0, NULL, nn, bn, STARSH_TLR);
  Array **D = malloc(sizeof(Array *) * nn);
  for (int k = 0; k < nn; k++) {
    int s[2] = {(int)t->C->size[bn[2 * k]], (int)t->C->size[bn[2 * k + 1]]};
    array_new(D + k, 2, s, 'd', 'F');
    memset(D[k]->data, 0, D[k]->data_nbytes);
  }
  starsh_blrm_new(&t->M, t->F, NULL, NULL, NULL, 0, D, NULL, NULL, NULL, '2');
  return t;
}

/* Far-field blocks for which starsh_blrm_get_block returns factors other than the block's own
 * (GOTCHA(starsh-get-block)). */
int ts_get_block_mismatches(struct starsh_blrm *M) {
  int bad = 0;
  STARSH_blrf *F = M->format;
  for (STARSH_int bi = 0; bi < F->nblocks_far; bi++) {
    int shape[2], rank;
    void *U, *V, *D;
    starsh_blrm_get_block(M, F->block_far[2 * bi], F->block_far[2 * bi + 1], shape, &rank, &U, &V, &D);
    if (U != M->far_U[bi]->data || V != M->far_V[bi]->data || rank != M->far_rank[bi]) bad++;
  }
  return bad;
}

/* Largest |kernel - reference| over a few random submatrices: the problem's kernel of a
 * matrix owned by the adapter must return the stored matrix's entries (ORIGINAL indices). */
double ts_kernel_maxdiff(struct starsh_blrm *M, const hmb_hmatrix *ref, int trials, unsigned seed) {
  STARSH_problem *P = M->format->problem;
  const int nr = (int)P->shape[0], nc = (int)P->shape[1], m = nr < 7 ? nr : 7, n = nc < 5 ? nc : 5;
  STARSH_int ri[7], ci[5];
  int rr[7], cc[5];
  double a[35], b[35], worst = 0;
  srand(seed);
  for (int t = 0; t < trials; t++) {
    for (int i = 0; i < m; i++) rr[i] = rand() % nr, ri[i] = rr[i];
    for (int j = 0; j < n; j++) cc[j] = rand() % nc, ci[j] = cc[j];
    P->kernel(m, n, ri, ci, P->row_data, P->col_data, a, m);
    hmb_entries(ref, m, rr, n, cc, b, m);
    for (int q = 0; q < m * n; q++) worst = fmax(worst, fabs(a[q] - b[q]));
  }
  return worst;
}

double ts_dfe(struct starsh_blrm *M) { return starsh_blrm__dfe(M); }

int ts_onfly(struct starsh_blrm *M) { return M->onfly; }
