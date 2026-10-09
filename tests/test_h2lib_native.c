/* H2Lib-side helpers of tests/test_h2lib.cpp: H-matrices built by H2Lib itself, and a check
 * of which leaves an H2Lib matrix shares with a neutral view. H2Lib's headers stay in this C
 * translation unit (GOTCHA(header-clash)). */
#include "hmb_h2lib.h"

#include <math.h>
#include <stdlib.h>

#include "basic.h"
#include "aca.h"
#include "block.h"
#include "cluster.h"
#include "clustergeometry.h"
#include "hmatrix.h"

typedef struct {
  pcluster root;
  uint *idx;
  phmatrix hm;
} kernel_matrix;

/* K(x, y) = 1 / (0.1 + |x - y|), H2Lib's matrixentry_t convention (ntrans: transposed block) */
static void entry(const uint *ridx, const uint *cidx, void *data, const bool ntrans, pamatrix N) {
  const double *x = (const double *)data;
  for (uint j = 0; j < N->cols; j++)
    for (uint i = 0; i < N->rows; i++) {
      const uint r = ntrans ? ridx[j] : ridx[i], c = ntrans ? cidx[i] : cidx[j];
      double d2 = 0;
      for (int d = 0; d < 3; d++) d2 += (x[3 * r + d] - x[3 * c + d]) * (x[3 * r + d] - x[3 * c + d]);
      N->a[i + (size_t)j * N->ld] = 1.0 / (0.1 + sqrt(d2));
    }
}

static void fill(phmatrix hm, const double *xyz, double eps) {
  if (hm->son) {
    for (uint k = 0; k < hm->rsons * hm->csons; k++) fill(hm->son[k], xyz, eps);
  } else if (hm->r) {
    decomp_partialaca_rkmatrix(entry, (void *)xyz, hm->rc->idx, hm->rc->size, hm->cc->idx, hm->cc->size, eps, NULL,
                               NULL, hm->r);
  } else if (hm->f) {
    entry(hm->rc->idx, hm->cc->idx, (void *)xyz, false, hm->f);
  }
}

/* The kernel on n points (xyz point-major) as an H2Lib H-matrix: adaptive or regular
 * clustering (leaf size), strict or non-strict block tree (admissible_2_cluster, eta),
 * partial ACA at eps. Returns an owner for test_h2lib_kernel_hmatrix / _free. */
void *test_h2lib_kernel_new(int n, const double *xyz, int leaf, double eta, double eps, int regular, int strict) {
  kernel_matrix *K = (kernel_matrix *)calloc(1, sizeof *K);
  pclustergeometry cg = new_clustergeometry(3, (uint)n);
  for (int i = 0; i < n; i++)
    for (int d = 0; d < 3; d++) cg->x[i][d] = cg->smin[i][d] = cg->smax[i][d] = xyz[3 * i + d];
  K->idx = (uint *)malloc(sizeof(uint) * (size_t)n);
  for (int i = 0; i < n; i++) K->idx[i] = (uint)i;
  K->root = regular ? build_regular_cluster(cg, (uint)n, K->idx, (uint)leaf, 0)
                    : build_adaptive_cluster(cg, (uint)n, K->idx, (uint)leaf);
  del_clustergeometry(cg);
  real e = eta;
  pblock b = strict ? build_strict_block(K->root, K->root, &e, admissible_2_cluster)
                    : build_nonstrict_block(K->root, K->root, &e, admissible_2_cluster);
  K->hm = build_from_block_hmatrix(b, 0);
  del_block(b);
  fill(K->hm, xyz, eps);
  return K;
}

struct _hmatrix *test_h2lib_kernel_hmatrix(void *K) { return ((kernel_matrix *)K)->hm; }

void test_h2lib_kernel_free(void *owner) {
  kernel_matrix *K = (kernel_matrix *)owner;
  del_hmatrix(K->hm);
  del_cluster(K->root);
  free(K->idx);
  free(K);
}

/* Walks A and the view V (same block tree) together: *same counts the leaves whose data V
 * shares with A (dense block, U, and V when V's vtrans = 1; rank-0 and empty leaves have
 * nothing to share), *other the rest. */
static void alias_rec(pchmatrix hm, const hmb_hmatrix *V, int id, long *same, long *other) {
  const hmb_block *b = V->bl + id;
  if (!hm->son && (hm->rc->size == 0 || hm->cc->size == 0)) {
    (*same)++;
    return;
  }
  if (hm->son) {
    if (b->kind != HMB_SUPER || (uint)b->rsons != hm->rsons || (uint)b->csons != hm->csons) {
      (*other)++;
      return;
    }
    for (uint k = 0; k < hm->rsons * hm->csons; k++) alias_rec(hm->son[k], V, V->sons[b->son0 + (int)k], same, other);
  } else if (hm->f) {
    if (b->kind == HMB_DENSE && b->D == hm->f->a) (*same)++;
    else (*other)++;
  } else if (hm->r) {
    if (b->kind == HMB_LOWRANK && (b->rank == 0 || (b->U == hm->r->A.a && (!b->vtrans || b->V == hm->r->B.a)))) (*same)++;
    else (*other)++;
  }
}

void test_h2lib_aliasing(const struct _hmatrix *A, const hmb_hmatrix *V, long *same, long *other) {
  *same = *other = 0;
  alias_rec(A, V, 0, same, other);
}

/* Number of clusters with exactly one son in A's row tree. */
static long single_sons(pccluster t) {
  long c = t->sons == 1;
  for (uint i = 0; i < t->sons; i++) c += single_sons(t->son[i]);
  return c;
}

long test_h2lib_single_sons(const struct _hmatrix *A) { return single_sons(A->rc); }
