/* hmatrix-bridges: STARS-H adapter (see hmb_starsh.h). The only translation unit that
 * includes STARS-H's headers. */
#include "hmb_starsh.h"

#include <starsh.h>

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct hmb_starsh {
  STARSH_problem *P;
  STARSH_cluster *R, *C; /* C == R when rows and columns share one cluster object */
  STARSH_blrf *F;
  STARSH_blrm *M;
  hmb_cstore view;       /* zero-copy view of M: the kernel's entry source and hmb_starsh_view */
  int *irow, *icol;      /* inverse permutations of the view (NULL for the identity) */
};

/* ------------------------------------------------------------------------------------ */
static void clear(char *msg, size_t msglen) {
  if (msg && msglen) msg[0] = 0;
}

static int fail(char *msg, size_t msglen, const char *fmt, ...) {
  if (msg && msglen) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, msglen, fmt, ap);
    va_end(ap);
  }
  return -1;
}

static void *xmalloc(size_t bytes) {
  void *p = malloc(bytes ? bytes : 1);
  if (!p) {
    fprintf(stderr, "hmatrix-bridges (STARS-H adapter): out of memory (%zu bytes)\n", bytes);
    abort();
  }
  return p;
}

static int *to_int(int n, const STARSH_int *v) {
  int *r = (int *)xmalloc(sizeof(int) * (size_t)n);
  for (int i = 0; i < n; i++) r[i] = (int)v[i];
  return r;
}

/* A column-major rows x cols Array over a buffer of our own (array_free releases it with
 * free()); a zero-sized Array still gets a valid buffer. */
static Array *new_array(int rows, int cols) {
  void *buf = xmalloc(sizeof(double) * (size_t)rows * (size_t)cols);
  int shape[2] = {rows, cols};
  Array *a = NULL;
  if (array_from_buffer(&a, 2, shape, 'd', 'F', buf) != STARSH_SUCCESS) {
    free(buf);
    return NULL;
  }
  return a;
}

/* Kernel of every problem owned by an hmb_starsh: the stored matrix's own entries. STARS-H
 * passes ORIGINAL indices (pivot + start). Reentrant (STARS-H's OpenMP routines call kernels
 * concurrently). */
static void stored_kernel(int nrows, int ncols, STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
                          void *result, int ld) {
  (void)col_data;
  const hmb_starsh *h = (const hmb_starsh *)row_data;
  int *r = to_int(nrows, irow), *c = to_int(ncols, icol);
  hmb_entries_ip(&h->view.A, h->irow, h->icol, nrows, r, ncols, c, (double *)result, ld);
  free(r);
  free(c);
}

static int export_core(STARSH_blrm *M, hmb_cstore *s, hmb_starsh_export_stats *stats, const char *who, char *msg,
                       size_t msglen);

/* GOTCHA(starsh-kernel-data): STARS-H hands its kernels the problem's row_data/col_data in
 * starsh_problem_get_block and starsh_blrf_get_block, but the clusters' data fields in the
 * approximation routines and in dmml/dca/dfe (on-the-fly blocks, error norms): both must
 * point at the kernel's context. */
static void set_kernel(hmb_starsh *h, STARSH_kernel *kernel, void *ctx) {
  h->P->kernel = kernel;
  h->P->row_data = h->P->col_data = ctx;
  h->R->data = h->C->data = ctx;
}

/* Builds the handle's own view and makes the problem's kernel read the stored matrix. */
static int finish_handle(hmb_starsh *h, const char *who, char *msg, size_t msglen) {
  int rc = export_core(h->M, &h->view, NULL, who, msg, msglen);
  if (rc) return rc;
  const hmb_hmatrix *A = &h->view.A;
  if (A->row_perm) h->irow = (int *)xmalloc(sizeof(int) * (size_t)A->n_rows);
  if (A->col_perm) h->icol = (int *)xmalloc(sizeof(int) * (size_t)A->n_cols);
  hmb_inverse_perms(A, h->irow, h->icol);
  set_kernel(h, stored_kernel, h);
  return 0;
}

void hmb_starsh_free(hmb_starsh *h) {
  if (!h) return;
  /* STARS-H frees bottom-up: the matrix reads its format, the format its clusters */
  if (h->M) starsh_blrm_free(h->M);
  if (h->F) starsh_blrf_free(h->F);
  if (h->C && h->C != h->R) starsh_cluster_free(h->C);
  if (h->R) starsh_cluster_free(h->R);
  if (h->P) starsh_problem_free(h->P);
  hmb_cstore_free(&h->view);
  free(h->irow);
  free(h->icol);
  free(h);
}

struct starsh_blrm *hmb_starsh_blrm(const hmb_starsh *h) { return h->M; }
const hmb_hmatrix *hmb_starsh_view(const hmb_starsh *h) { return &h->view.A; }

/* ---- ranges ---------------------------------------------------------------------------- */
typedef struct {
  int off, size;
} range_t;

/* offset ascending, then size descending: a range comes before the ranges nested in it */
static int cmp_range(const void *a, const void *b) {
  const range_t *x = (const range_t *)a, *y = (const range_t *)b;
  if (x->off != y->off) return x->off < y->off ? -1 : 1;
  if (x->size != y->size) return x->size > y->size ? -1 : 1;
  return 0;
}

static int unique_ranges(range_t *r, int n) {
  if (n == 0) return 0;
  qsort(r, (size_t)n, sizeof(range_t), cmp_range);
  int m = 1;
  for (int k = 1; k < n; k++)
    if (cmp_range(r + k, r + m - 1) != 0) r[m++] = r[k];
  return m;
}

static int find_range(const range_t *r, int m, int off, int size) {
  range_t key = {off, size};
  const range_t *p = (const range_t *)bsearch(&key, r, (size_t)m, sizeof(range_t), cmp_range);
  return p ? (int)(p - r) : -1;
}

/* ======================================================================================
 * neutral view -> STARS-H
 * ====================================================================================== */
typedef struct {
  int ri, ci; /* STARS-H row / column cluster indices */
  int b;      /* neutral block id */
} coo_t;

static int cmp_coo(const void *a, const void *b) {
  const coo_t *x = (const coo_t *)a, *y = (const coo_t *)b;
  if (x->ri != y->ri) return x->ri < y->ri ? -1 : 1;
  if (x->ci != y->ci) return x->ci < y->ci ? -1 : 1;
  return 0;
}

static int perm_at(const int *p, int k) { return p ? p[k] : k; }

/* A STARS-H cluster over the given ranges (copied) and permutation (NULL = identity). */
static STARSH_cluster *make_cluster(int n, const int *perm, const range_t *rg, int m) {
  STARSH_int *pivot = (STARSH_int *)xmalloc(sizeof(STARSH_int) * (size_t)n);
  STARSH_int *start = (STARSH_int *)xmalloc(sizeof(STARSH_int) * (size_t)m);
  STARSH_int *size = (STARSH_int *)xmalloc(sizeof(STARSH_int) * (size_t)m);
  for (int k = 0; k < n; k++) pivot[k] = perm_at(perm, k);
  for (int k = 0; k < m; k++) {
    start[k] = rg[k].off;
    size[k] = rg[k].size;
  }
  STARSH_cluster *C = NULL;
  /* starsh_cluster_new takes ownership of pivot, start and size (freed by starsh_cluster_free) */
  if (starsh_cluster_new(&C, NULL, n, pivot, m, 0, NULL, start, size, NULL, NULL, NULL, STARSH_PLAIN) !=
      STARSH_SUCCESS) {
    free(pivot);
    free(start);
    free(size);
    return NULL;
  }
  return C;
}

int hmb_starsh_import(const hmb_hmatrix *A, hmb_starsh **out, char *msg, size_t msglen) {
  clear(msg, msglen);
  const char *who = "hmb_starsh_import";
  char vmsg[256];
  if (!out) return fail(msg, msglen, "%s: null output pointer", who);
  *out = NULL;
  if (hmb_validate(A, vmsg, sizeof vmsg)) return fail(msg, msglen, "%s: invalid view: %s", who, vmsg);
  const int nr = A->n_rows, nc = A->n_cols;
  if (nr == 0 || nc == 0) return fail(msg, msglen, "%s: empty matrix (%d x %d)", who, nr, nc);

  /* Leaves with a nonempty area: the format is the flat list of them. */
  int nleaf = 0, nfar = 0, nnear = 0;
  int *leaf = (int *)xmalloc(sizeof(int) * (size_t)A->nbl);
  for (int b = 0; b < A->nbl; b++) {
    const hmb_block *bk = A->bl + b;
    if (bk->kind == HMB_SUPER || A->cl[bk->rc].size == 0 || A->cl[bk->cc].size == 0) continue;
    leaf[nleaf++] = b;
    if (bk->kind == HMB_LOWRANK) nfar++;
    else nnear++;
  }

  /* One cluster object for both sides when the view's trees and orderings coincide. */
  int shared = A->row_root == A->col_root && nr == nc;
  for (int k = 0; shared && k < nr; k++)
    if (perm_at(A->row_perm, k) != perm_at(A->col_perm, k)) shared = 0;

  range_t *rr = (range_t *)xmalloc(sizeof(range_t) * (size_t)(2 * nleaf));
  range_t *cr = shared ? rr : (range_t *)xmalloc(sizeof(range_t) * (size_t)nleaf);
  int mr = 0, mc = 0;
  for (int k = 0; k < nleaf; k++) {
    const hmb_block *bk = A->bl + leaf[k];
    rr[mr].off = A->cl[bk->rc].offset;
    rr[mr++].size = A->cl[bk->rc].size;
    range_t *dst = shared ? rr + mr++ : cr + mc++;
    dst->off = A->cl[bk->cc].offset;
    dst->size = A->cl[bk->cc].size;
  }
  mr = unique_ranges(rr, mr);
  if (shared) mc = mr;
  else mc = unique_ranges(cr, mc);

  hmb_starsh *h = (hmb_starsh *)calloc(1, sizeof(hmb_starsh));
  if (!h) abort();
  hmb_cstore_init(&h->view);
  h->R = make_cluster(nr, A->row_perm, rr, mr);
  h->C = shared ? h->R : make_cluster(nc, A->col_perm, cr, mc);
  STARSH_int shape[2] = {nr, nc};
  char name[] = "hmatrix-bridges import";
  int info = (!h->R || !h->C) ? STARSH_MALLOC_ERROR
                              : starsh_problem_new(&h->P, 2, shape, 'N', 'd', h, h, stored_kernel, name);
  if (info != STARSH_SUCCESS) {
    free(leaf);
    free(rr);
    if (!shared) free(cr);
    hmb_starsh_free(h);
    return fail(msg, msglen, "%s: STARS-H cluster or problem creation failed (error %d)", who, info);
  }

  /* COO lists sorted by (row cluster, column cluster): starsh_blrm_get_block indexes the
   * factors by CSR position, which equals the COO index only in this order
   * (GOTCHA(starsh-get-block)). */
  coo_t *fc = (coo_t *)xmalloc(sizeof(coo_t) * (size_t)nfar), *nc_ = (coo_t *)xmalloc(sizeof(coo_t) * (size_t)nnear);
  int kf = 0, kn = 0;
  for (int k = 0; k < nleaf; k++) {
    const hmb_block *bk = A->bl + leaf[k];
    coo_t e = {find_range(rr, mr, A->cl[bk->rc].offset, A->cl[bk->rc].size),
               find_range(cr, mc, A->cl[bk->cc].offset, A->cl[bk->cc].size), leaf[k]};
    if (bk->kind == HMB_LOWRANK) fc[kf++] = e;
    else nc_[kn++] = e;
  }
  qsort(fc, (size_t)nfar, sizeof(coo_t), cmp_coo);
  qsort(nc_, (size_t)nnear, sizeof(coo_t), cmp_coo);
  free(leaf);
  free(rr);
  if (!shared) free(cr);

  STARSH_int *block_far = nfar ? (STARSH_int *)xmalloc(sizeof(STARSH_int) * 2 * (size_t)nfar) : NULL;
  STARSH_int *block_near = nnear ? (STARSH_int *)xmalloc(sizeof(STARSH_int) * 2 * (size_t)nnear) : NULL;
  for (int k = 0; k < nfar; k++) {
    block_far[2 * k] = fc[k].ri;
    block_far[2 * k + 1] = fc[k].ci;
  }
  for (int k = 0; k < nnear; k++) {
    block_near[2 * k] = nc_[k].ri;
    block_near[2 * k + 1] = nc_[k].ci;
  }
  /* takes ownership of block_far and block_near. It fails only when an allocation fails; its
   * partial format is then left alone (freeing it would free the lists twice). */
  STARSH_blrf *F = NULL;
  info = starsh_blrf_new_from_coo(&F, h->P, 'N', h->R, h->C, nfar, block_far, nnear, block_near, STARSH_TLR);
  if (info != STARSH_SUCCESS) {
    free(block_far);
    free(block_near);
    free(fc);
    free(nc_);
    hmb_starsh_free(h);
    return fail(msg, msglen, "%s: starsh_blrf_new_from_coo failed (error %d)", who, info);
  }
  h->F = F;

  /* Leaf data, copied into Arrays STARS-H takes over (starsh_blrm_free releases them). */
  int *far_rank = nfar ? (int *)xmalloc(sizeof(int) * (size_t)nfar) : NULL;
  Array **far_U = nfar ? (Array **)calloc((size_t)nfar, sizeof(Array *)) : NULL;
  Array **far_V = nfar ? (Array **)calloc((size_t)nfar, sizeof(Array *)) : NULL;
  Array **near_D = nnear ? (Array **)calloc((size_t)nnear, sizeof(Array *)) : NULL;
  int ok = !(nfar && (!far_U || !far_V)) && !(nnear && !near_D);
  for (int k = 0; ok && k < nfar; k++) {
    const hmb_block *bk = A->bl + fc[k].b;
    const int m = A->cl[bk->rc].size, n = A->cl[bk->cc].size, r = bk->rank;
    far_rank[k] = r;
    far_U[k] = new_array(m, r);
    far_V[k] = new_array(n, r);
    if (!far_U[k] || !far_V[k]) {
      ok = 0;
      break;
    }
    double *U = (double *)far_U[k]->data, *V = (double *)far_V[k]->data;
    for (int l = 0; l < r; l++) memcpy(U + (size_t)l * m, bk->U + (size_t)l * bk->ldU, sizeof(double) * (size_t)m);
    /* GOTCHA(lowrank-convention): STARS-H stores U * V^T with V cols x rank */
    if (bk->vtrans)
      for (int l = 0; l < r; l++) memcpy(V + (size_t)l * n, bk->V + (size_t)l * bk->ldV, sizeof(double) * (size_t)n);
    else
      for (int l = 0; l < r; l++)
        for (int j = 0; j < n; j++) V[j + (size_t)l * n] = bk->V[l + (size_t)j * bk->ldV];
  }
  for (int k = 0; ok && k < nnear; k++) {
    const hmb_block *bk = A->bl + nc_[k].b;
    const int m = A->cl[bk->rc].size, n = A->cl[bk->cc].size;
    near_D[k] = new_array(m, n);
    if (!near_D[k]) {
      ok = 0;
      break;
    }
    double *D = (double *)near_D[k]->data;
    for (int j = 0; j < n; j++) memcpy(D + (size_t)j * m, bk->D + (size_t)j * bk->ldD, sizeof(double) * (size_t)m);
  }
  free(fc);
  free(nc_);
  info = ok ? starsh_blrm_new(&h->M, h->F, far_rank, far_U, far_V, 0, near_D, NULL, NULL, NULL, '2')
            : STARSH_MALLOC_ERROR;
  if (info != STARSH_SUCCESS) {
    for (int k = 0; k < nfar && far_U; k++) {
      array_free(far_U[k]);
      array_free(far_V[k]);
    }
    for (int k = 0; k < nnear && near_D; k++) array_free(near_D[k]);
    free(far_rank);
    free(far_U);
    free(far_V);
    free(near_D);
    hmb_starsh_free(h);
    return fail(msg, msglen, "%s: creating the STARS-H matrix failed (error %d)", who, info);
  }
  int rc = finish_handle(h, who, msg, msglen);
  if (rc) {
    hmb_starsh_free(h);
    return rc;
  }
  *out = h;
  return 0;
}

/* ======================================================================================
 * entries -> STARS-H TLR approximation
 * ====================================================================================== */
typedef struct {
  hmb_starsh_entry_fn *fn;
  void *ctx;
  int thread_safe;
} entry_ctx;

static void entry_kernel(int nrows, int ncols, STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
                         void *result, int ld) {
  (void)col_data;
  const entry_ctx *e = (const entry_ctx *)row_data;
  int *r = to_int(nrows, irow), *c = to_int(ncols, icol);
  if (e->thread_safe) e->fn(nrows, ncols, r, c, (double *)result, ld, e->ctx);
  else {
#pragma omp critical(hmb_starsh_entries)
    e->fn(nrows, ncols, r, c, (double *)result, ld, e->ctx);
  }
  free(r);
  free(c);
}

int hmb_starsh_from_entries(int n, const int *perm, hmb_starsh_entry_fn *entries, void *ctx,
                            const hmb_starsh_tlr_opts *o, hmb_starsh **out, char *msg, size_t msglen) {
  clear(msg, msglen);
  const char *who = "hmb_starsh_from_entries";
  if (!out) return fail(msg, msglen, "%s: null output pointer", who);
  *out = NULL;
  if (n <= 0 || !entries || !o) return fail(msg, msglen, "%s: need n > 0, an entry callback and options", who);
  if (o->tile_size <= 0 || o->maxrank <= 0 || !(o->tol >= 0))
    return fail(msg, msglen, "%s: need tile_size > 0, maxrank > 0 and tol >= 0 (got %d, %d, %g)", who, o->tile_size,
                o->maxrank, o->tol);
  if (o->symm != 'N' && o->symm != 'S') return fail(msg, msglen, "%s: symm must be 'N' or 'S'", who);
  if (perm) {
    char *seen = (char *)calloc((size_t)n, 1);
    int bad = !seen;
    for (int k = 0; !bad && k < n; k++) {
      if (perm[k] < 0 || perm[k] >= n || seen[perm[k]]) bad = 1;
      else seen[perm[k]] = 1;
    }
    free(seen);
    if (bad) return fail(msg, msglen, "%s: perm is not a permutation of 0 .. %d", who, n - 1);
  }
  /* GOTCHA(starsh-init): starsh_blrm_approximate is a NULL function pointer and the RSVD
   * oversampling is -1 until starsh_init() has run (it reads STARSH_BACKEND, STARSH_LRENGINE
   * and STARSH_OVERSAMPLE, and reports its choice on stderr). */
  if (!starsh_blrm_approximate) starsh_init();
  if (!starsh_blrm_approximate) return fail(msg, msglen, "%s: STARS-H selected no approximation routine", who);
  const int ts = o->tile_size < n ? o->tile_size : n;
  const int maxrank = o->maxrank < ts ? o->maxrank : ts; /* no tile can need more */
  const int nb = (n + ts - 1) / ts;

  hmb_starsh *h = (hmb_starsh *)calloc(1, sizeof(hmb_starsh));
  if (!h) abort();
  hmb_cstore_init(&h->view);
  range_t *rg = (range_t *)xmalloc(sizeof(range_t) * (size_t)nb);
  for (int k = 0; k < nb; k++) {
    rg[k].off = k * ts;
    rg[k].size = (k == nb - 1) ? n - k * ts : ts;
  }
  h->R = h->C = make_cluster(n, perm, rg, nb);
  free(rg);
  entry_ctx ec = {entries, ctx, o->thread_safe};
  STARSH_int shape[2] = {n, n};
  char name[] = "hmatrix-bridges entries";
  int info = !h->R ? STARSH_MALLOC_ERROR
                   : starsh_problem_new(&h->P, 2, shape, o->symm, 'd', &ec, &ec, entry_kernel, name);
  if (info == STARSH_SUCCESS) {
    set_kernel(h, entry_kernel, &ec);
    info = starsh_blrf_new_tlr(&h->F, h->P, o->symm, h->R, h->C);
  }
  /* GOTCHA(starsh-false-far-leak): when tiles need more than maxrank, STARS-H's approximation
   * leaks ~104 bytes per far-field block (suppressed in tests/lsan.supp) */
  if (info == STARSH_SUCCESS) info = starsh_blrm_approximate(&h->M, h->F, maxrank, o->tol, 0);
  if (info != STARSH_SUCCESS) {
    hmb_starsh_free(h);
    return fail(msg, msglen, "%s: STARS-H approximation failed (error %d)", who, info);
  }
  /* ec goes out of scope: from here on the kernel reads the stored matrix */
  int rc = finish_handle(h, who, msg, msglen);
  if (rc) {
    hmb_starsh_free(h);
    return rc;
  }
  *out = h;
  return 0;
}

/* ======================================================================================
 * STARS-H -> neutral view
 * ====================================================================================== */
typedef struct {
  int rnode, cnode; /* cluster ids in the store */
  int kind;
  int rank, ldU, ldV;
  const double *U, *V;
  const double *D;
  int ldD;
} leaf_t;

typedef struct {
  hmb_cstore *s;
  const leaf_t *L;
  int *cpar;        /* parent cluster id (-1 for roots) */
  const char *who;
  char *msg;
  size_t msglen;
} bt_ctx;

/* Cluster tree of [0, n) over the given ranges (sorted, unique; the root is added when no
 * range covers [0, n)); ids in breadth-first order so that sons are consecutive. rid[k] is
 * the cluster id of range k. Returns the root id, or -1 with a message. */
static int build_tree(hmb_cstore *s, int n, const range_t *rg, int m, int *rid, const char *side,
                      const char *who, char *msg, size_t msglen) {
  const int has_root = m > 0 && rg[0].off == 0 && rg[0].size == n;
  const int t = m + !has_root; /* tree nodes: [root] + ranges */
  range_t *nd = (range_t *)xmalloc(sizeof(range_t) * (size_t)t);
  int *par = (int *)xmalloc(sizeof(int) * (size_t)t), *stack = (int *)xmalloc(sizeof(int) * (size_t)t);
  int *nch = (int *)calloc((size_t)t + 1, sizeof(int)), *cid = (int *)xmalloc(sizeof(int) * (size_t)t);
  int *queue = (int *)xmalloc(sizeof(int) * (size_t)t), *first = (int *)xmalloc(sizeof(int) * ((size_t)t + 1));
  int *kids = (int *)xmalloc(sizeof(int) * (size_t)t);
  if (!nch) abort();
  int rc = 0, top = 0;
  nd[0].off = 0;
  nd[0].size = n;
  memcpy(nd + !has_root, rg, sizeof(range_t) * (size_t)m);
  par[0] = -1;
  stack[top++] = 0;
  for (int k = 1; k < t && rc == 0; k++) {
    const int lo = nd[k].off, hi = nd[k].off + nd[k].size;
    if (lo < 0 || hi > n || nd[k].size <= 0) {
      rc = fail(msg, msglen, "%s: %s cluster range [%d, %d) is outside [0, %d)", who, side, lo, hi, n);
      break;
    }
    while (top > 1 && nd[stack[top - 1]].off + nd[stack[top - 1]].size <= lo) top--;
    const range_t *p = nd + stack[top - 1];
    if (hi > p->off + p->size)
      rc = fail(msg, msglen, "%s: %s cluster ranges [%d, %d) and [%d, %d) overlap without nesting (no cluster tree "
                "holds both)", who, side, p->off, p->off + p->size, lo, hi);
    par[k] = stack[top - 1];
    nch[par[k]]++;
    stack[top++] = k;
  }
  /* children in index (= offset) order; they must partition their parent */
  first[0] = 0;
  for (int k = 0; k < t; k++) first[k + 1] = first[k] + nch[k];
  memset(nch, 0, sizeof(int) * (size_t)t);
  for (int k = 1; k < t && rc == 0; k++) kids[first[par[k]] + nch[par[k]]++] = k;
  for (int k = 0; k < t && rc == 0; k++) {
    if (nch[k] == 0) continue;
    int next = nd[k].off;
    for (int q = first[k]; q < first[k + 1]; q++) {
      if (nd[kids[q]].off != next) break;
      next += nd[kids[q]].size;
    }
    if (next != nd[k].off + nd[k].size)
      rc = fail(msg, msglen, "%s: %s indices of cluster [%d, %d) are not all covered by its sub-ranges: the blocks "
                "do not tile the matrix", who, side, nd[k].off, nd[k].off + nd[k].size);
  }
  int root = -1;
  if (rc == 0) {
    int head = 0, tail = 0;
    root = cid[0] = hmb_cstore_add_clusters(s, 1);
    s->cl[root].offset = 0;
    s->cl[root].size = n;
    queue[tail++] = 0;
    while (head < tail) {
      const int k = queue[head++];
      if (nch[k] == 0) continue;
      const int s0 = hmb_cstore_add_clusters(s, nch[k]);
      s->cl[cid[k]].son0 = s0;
      s->cl[cid[k]].nsons = nch[k];
      for (int q = 0; q < nch[k]; q++) {
        const int c = kids[first[k] + q];
        cid[c] = s0 + q;
        s->cl[s0 + q].offset = nd[c].off;
        s->cl[s0 + q].size = nd[c].size;
        queue[tail++] = c;
      }
    }
    for (int k = 0; k < m; k++) rid[k] = cid[k + !has_root];
  }
  free(nd);
  free(par);
  free(stack);
  free(nch);
  free(cid);
  free(queue);
  free(first);
  free(kids);
  return rc ? -1 : root;
}

/* the son of cluster anc on the path to its strict descendant node (-1 if not a descendant) */
static int son_towards(const bt_ctx *t, int node, int anc) {
  while (node >= 0 && t->cpar[node] != anc) node = t->cpar[node];
  return node;
}

/* Block (r, c) holding the leaves ids[0 .. cnt): a leaf if one leaf matches it exactly,
 * otherwise split along every side that no leaf spans entirely. Returns the block id, or
 * -1 with a message. */
static int build_block(const bt_ctx *t, int r, int c, const int *ids, int cnt) {
  hmb_cstore *s = t->s;
  const int r0 = s->cl[r].offset, r1 = r0 + s->cl[r].size, c0 = s->cl[c].offset, c1 = c0 + s->cl[c].size;
  if (cnt == 0)
    return fail(t->msg, t->msglen, "%s: rows [%d, %d) x columns [%d, %d) are covered by no block", t->who, r0, r1,
                c0, c1);
  int split_r = 1, split_c = 1;
  for (int k = 0; k < cnt; k++) {
    const leaf_t *l = t->L + ids[k];
    if (l->rnode == r && l->cnode == c) {
      if (cnt > 1)
        return fail(t->msg, t->msglen, "%s: block rows [%d, %d) x columns [%d, %d) overlaps %d other block(s)", t->who,
                    r0, r1, c0, c1, cnt - 1);
      hmb_block b;
      memset(&b, 0, sizeof b);
      b.rc = r;
      b.cc = c;
      b.kind = l->kind;
      b.rank = l->rank;
      b.U = l->U;
      b.ldU = l->ldU;
      b.V = l->V;
      b.ldV = l->ldV;
      b.vtrans = 1;
      b.D = l->D;
      b.ldD = l->ldD;
      return hmb_cstore_add_block(s, &b);
    }
    if (l->rnode == r) split_r = 0;
    if (l->cnode == c) split_c = 0;
  }
  if (!split_r && !split_c)
    return fail(t->msg, t->msglen, "%s: blocks inside rows [%d, %d) x columns [%d, %d) overlap", t->who, r0, r1, c0,
                c1);
  const int rsons = split_r ? s->cl[r].nsons : 1, csons = split_c ? s->cl[c].nsons : 1;
  const int rs0 = s->cl[r].son0, cs0 = s->cl[c].son0, nb = rsons * csons;
  hmb_block b;
  memset(&b, 0, sizeof b);
  b.rc = r;
  b.cc = c;
  b.kind = HMB_SUPER;
  b.rsons = rsons;
  b.csons = csons;
  b.son0 = hmb_cstore_add_sons(s, nb);
  const int id = hmb_cstore_add_block(s, &b), son0 = b.son0;
  int *bucket = (int *)xmalloc(sizeof(int) * (size_t)cnt), *cnt_b = (int *)calloc((size_t)nb + 1, sizeof(int));
  int *sorted = (int *)xmalloc(sizeof(int) * (size_t)cnt);
  if (!cnt_b) abort();
  for (int k = 0; k < cnt; k++) {
    const leaf_t *l = t->L + ids[k];
    const int i = split_r ? son_towards(t, l->rnode, r) - rs0 : 0, j = split_c ? son_towards(t, l->cnode, c) - cs0 : 0;
    if (i < 0 || i >= rsons || j < 0 || j >= csons) { /* cannot happen: leaf ranges nest in the tree */
      free(bucket);
      free(cnt_b);
      free(sorted);
      return fail(t->msg, t->msglen, "%s: internal error placing a block in rows [%d, %d) x columns [%d, %d)", t->who,
                  r0, r1, c0, c1);
    }
    bucket[k] = i + j * rsons;
    cnt_b[bucket[k] + 1]++;
  }
  for (int q = 0; q < nb; q++) cnt_b[q + 1] += cnt_b[q];
  int *pos = (int *)xmalloc(sizeof(int) * (size_t)nb);
  memcpy(pos, cnt_b, sizeof(int) * (size_t)nb);
  for (int k = 0; k < cnt; k++) sorted[pos[bucket[k]]++] = ids[k];
  free(pos);
  free(bucket);
  int rc = id;
  for (int j = 0; j < csons && rc >= 0; j++)
    for (int i = 0; i < rsons && rc >= 0; i++) {
      const int q = i + j * rsons;
      const int ch = build_block(t, split_r ? rs0 + i : r, split_c ? cs0 + j : c, sorted + cnt_b[q], cnt_b[q + 1] - cnt_b[q]);
      if (ch < 0) rc = -1;
      else s->sons[son0 + q] = ch;
    }
  free(sorted);
  free(cnt_b);
  return rc;
}

/* permutation copied out of STARS-H's pivot; NULL when it is the identity */
static int *copy_pivot(hmb_cstore *s, int which, const STARSH_cluster *C, int n) {
  int ident = 1;
  for (int k = 0; k < n && ident; k++) ident = C->pivot[k] == k;
  if (ident) return NULL;
  int *p = hmb_cstore_perm(s, which, n);
  for (int k = 0; k < n; k++) p[k] = (int)C->pivot[k];
  return p;
}

static int check_array(const Array *a, int rows, int min_cols) {
  return a && a->ndim == 2 && a->dtype == 'd' && a->shape[0] == rows && a->shape[1] >= min_cols &&
         (a->data || (size_t)rows * (size_t)min_cols == 0);
}

static int export_core(STARSH_blrm *M, hmb_cstore *s, hmb_starsh_export_stats *stats, const char *who, char *msg,
                       size_t msglen) {
  if (!M || !M->format || !M->format->problem || !M->format->row_cluster || !M->format->col_cluster)
    return fail(msg, msglen, "%s: incomplete STARS-H matrix (missing format, problem or cluster)", who);
  if (s->ncl || s->nbl) return fail(msg, msglen, "%s: the store must be empty", who);
  STARSH_blrf *F = M->format;
  STARSH_problem *P = F->problem;
  STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
  if (P->ndim != 2 || P->dtype != 'd')
    return fail(msg, msglen, "%s: only real double scalar kernels (ndim %d, dtype '%c')", who, P->ndim, P->dtype);
  if (P->shape[0] > INT_MAX || P->shape[1] > INT_MAX) return fail(msg, msglen, "%s: matrix too large", who);
  const int nr = (int)P->shape[0], nc = (int)P->shape[1];
  const char symm = F->symm;
  if (R->ndata != nr || C->ndata != nc)
    return fail(msg, msglen, "%s: clusters hold %zd and %zd indices for a %d x %d matrix", who, (ssize_t)R->ndata,
                (ssize_t)C->ndata, nr, nc);
  if (symm == 'S' && (R != C || nr != nc))
    return fail(msg, msglen, "%s: symmetric format without a shared cluster object", who);
  hmb_starsh_export_stats st;
  memset(&st, 0, sizeof st);

  const long long nfar = F->nblocks_far, nnear = F->nblocks_near;
  const long long cap = (nfar + nnear) * (symm == 'S' ? 2 : 1);
  leaf_t *L = (leaf_t *)xmalloc(sizeof(leaf_t) * (size_t)cap);
  int *lr = (int *)xmalloc(sizeof(int) * (size_t)cap), *lc = (int *)xmalloc(sizeof(int) * (size_t)cap);
  int nl = 0, rc = 0;
  /* far-field (low-rank) blocks: U * V^T, ld = rows / cols (STARS-H's convention) */
  for (long long bi = 0; bi < nfar && rc == 0; bi++) {
    const STARSH_int i = F->block_far[2 * bi], j = F->block_far[2 * bi + 1];
    if (i < 0 || i >= R->nblocks || j < 0 || j >= C->nblocks) {
      rc = fail(msg, msglen, "%s: far block %lld refers to clusters (%zd, %zd)", who, bi, (ssize_t)i, (ssize_t)j);
      break;
    }
    const int m = (int)R->size[i], n = (int)C->size[j], r = M->far_rank[bi];
    if (r < 0 || (r > 0 && (!check_array(M->far_U[bi], m, r) || !check_array(M->far_V[bi], n, r)))) {
      rc = fail(msg, msglen, "%s: far block %lld (%d x %d) has rank %d and factors that do not match it", who, bi, m, n,
                r);
      break;
    }
    if (m == 0 || n == 0) continue;
    const double *U = r ? (const double *)M->far_U[bi]->data : NULL, *V = r ? (const double *)M->far_V[bi]->data : NULL;
    leaf_t l = {0, 0, HMB_LOWRANK, r, m, n, U, V, NULL, 0};
    L[nl] = l;
    lr[nl] = (int)i;
    lc[nl++] = (int)j;
    st.far_blocks++;
    st.aliased_values += (long long)r * (m + n);
    if (r > st.max_rank) st.max_rank = r;
    if (symm == 'S' && i != j) { /* GOTCHA(symmetric-storage): (j, i) = (U V^T)^T = V U^T */
      leaf_t t = {0, 0, HMB_LOWRANK, r, n, m, V, U, NULL, 0};
      L[nl] = t;
      lr[nl] = (int)j;
      lc[nl++] = (int)i;
      st.mirrored_blocks++;
    }
  }
  /* near-field (dense) blocks; evaluated through the kernel when not stored (onfly) */
  for (long long bi = 0; bi < nnear && rc == 0; bi++) {
    const STARSH_int i = F->block_near[2 * bi], j = F->block_near[2 * bi + 1];
    if (i < 0 || i >= R->nblocks || j < 0 || j >= C->nblocks) {
      rc = fail(msg, msglen, "%s: near block %lld refers to clusters (%zd, %zd)", who, bi, (ssize_t)i, (ssize_t)j);
      break;
    }
    const int m = (int)R->size[i], n = (int)C->size[j];
    if (m == 0 || n == 0) continue;
    const double *D;
    if (M->onfly == 0) {
      if (!check_array(M->near_D[bi], m, n) || M->near_D[bi]->shape[1] != n) {
        rc = fail(msg, msglen, "%s: near block %lld (%d x %d) has a dense array of another shape", who, bi, m, n);
        break;
      }
      D = (const double *)M->near_D[bi]->data;
      st.aliased_values += (long long)m * n;
    } else { /* the kernel arguments starsh_blrm__dmml / __dca use (GOTCHA(starsh-kernel-data)) */
      double *E = hmb_cstore_alloc(s, (size_t)m * n);
      P->kernel(m, n, R->pivot + R->start[i], C->pivot + C->start[j], R->data, C->data, E, m);
      D = E;
      st.copied_values += (long long)m * n;
    }
    leaf_t l = {0, 0, HMB_DENSE, 0, 0, 0, NULL, NULL, D, m};
    L[nl] = l;
    lr[nl] = (int)i;
    lc[nl++] = (int)j;
    st.near_blocks++;
    if (symm == 'S' && i != j) { /* GOTCHA(symmetric-storage): transposed copy */
      double *T = hmb_cstore_alloc(s, (size_t)m * n);
      for (int a = 0; a < m; a++)
        for (int b = 0; b < n; b++) T[b + (size_t)a * n] = D[a + (size_t)b * m];
      leaf_t t = {0, 0, HMB_DENSE, 0, 0, 0, NULL, NULL, T, n};
      L[nl] = t;
      lr[nl] = (int)j;
      lc[nl++] = (int)i;
      st.mirrored_blocks++;
      st.copied_values += (long long)m * n;
    }
  }

  /* cluster trees rebuilt from the ranges the blocks use */
  const int shared = R == C;
  range_t *rg = (range_t *)xmalloc(sizeof(range_t) * (size_t)(2 * nl + 1)), *cg = rg + nl;
  int *rid = (int *)xmalloc(sizeof(int) * (size_t)(2 * nl + 1)), *cid = rid + nl;
  int mr = 0, mc = 0, row_root = -1, col_root = -1;
  if (rc == 0) {
    for (int k = 0; k < nl; k++) {
      rg[mr].off = (int)R->start[lr[k]];
      rg[mr++].size = (int)R->size[lr[k]];
      range_t *d = shared ? rg + mr++ : cg + mc++;
      d->off = (int)C->start[lc[k]];
      d->size = (int)C->size[lc[k]];
    }
    if (shared) {
      mr = unique_ranges(rg, mr);
      row_root = col_root = build_tree(s, nr, rg, mr, rid, "row/column", who, msg, msglen);
      cg = rg;
      cid = rid;
      mc = mr;
    } else {
      mr = unique_ranges(rg, mr);
      mc = unique_ranges(cg, mc);
      row_root = build_tree(s, nr, rg, mr, rid, "row", who, msg, msglen);
      if (row_root >= 0) col_root = build_tree(s, nc, cg, mc, cid, "column", who, msg, msglen);
    }
    if (row_root < 0 || col_root < 0) rc = -1;
  }
  if (rc == 0) {
    for (int k = 0; k < nl; k++) {
      L[k].rnode = rid[find_range(rg, mr, (int)R->start[lr[k]], (int)R->size[lr[k]])];
      L[k].cnode = cid[find_range(cg, mc, (int)C->start[lc[k]], (int)C->size[lc[k]])];
    }
    int *cpar = (int *)xmalloc(sizeof(int) * (size_t)s->ncl);
    for (int k = 0; k < s->ncl; k++) cpar[k] = -1;
    for (int k = 0; k < s->ncl; k++)
      for (int q = 0; q < s->cl[k].nsons; q++) cpar[s->cl[k].son0 + q] = k;
    int *ids = (int *)xmalloc(sizeof(int) * (size_t)nl);
    for (int k = 0; k < nl; k++) ids[k] = k;
    bt_ctx t = {s, L, cpar, who, msg, msglen};
    if (build_block(&t, row_root, col_root, ids, nl) != 0) rc = -1; /* the root block is bl[0] */
    free(ids);
    free(cpar);
  }
  free(rg);
  free(rid);
  free(L);
  free(lr);
  free(lc);
  if (rc) return rc;

  s->A.dim = 0;
  s->A.n_rows = nr;
  s->A.n_cols = nc;
  s->A.row_root = row_root;
  s->A.col_root = col_root;
  int *rp = copy_pivot(s, 0, R, nr);
  if (shared) s->col_perm = rp;
  else copy_pivot(s, 1, C, nc);
  const hmb_hmatrix *A = hmb_cstore_finalize(s);
  char vmsg[256];
  if (hmb_validate(A, vmsg, sizeof vmsg)) return fail(msg, msglen, "%s: internal error, invalid view: %s", who, vmsg);
  if (stats) *stats = st;
  return 0;
}

int hmb_starsh_export(struct starsh_blrm *M, hmb_cstore *store, hmb_starsh_export_stats *stats, char *msg,
                      size_t msglen) {
  clear(msg, msglen);
  if (!store) return fail(msg, msglen, "hmb_starsh_export: null store");
  return export_core(M, store, stats, "hmb_starsh_export", msg, msglen);
}

/* ======================================================================================
 * operations in the ORIGINAL ordering
 * ====================================================================================== */
static int check_matrix(STARSH_blrm *M, const char *who, char *msg, size_t msglen) {
  clear(msg, msglen);
  if (!M || !M->format || !M->format->problem || !M->format->row_cluster || !M->format->col_cluster)
    return fail(msg, msglen, "%s: incomplete STARS-H matrix", who);
  const STARSH_problem *P = M->format->problem;
  if (P->ndim != 2 || P->dtype != 'd') return fail(msg, msglen, "%s: only real double scalar kernels", who);
  return 0;
}

char hmb_starsh_symm(const struct starsh_blrm *M) { return M && M->format ? M->format->symm : 0; }

int hmb_starsh_matvec(struct starsh_blrm *M, int nrhs, const double *X, int ldx, double *Y, int ldy, char *msg,
                      size_t msglen) {
  const char *who = "hmb_starsh_matvec";
  if (check_matrix(M, who, msg, msglen)) return -1;
  const STARSH_blrf *F = M->format;
  const int nr = (int)F->problem->shape[0], nc = (int)F->problem->shape[1];
  const STARSH_int *pr = F->row_cluster->pivot, *pc = F->col_cluster->pivot;
  if (nrhs < 0 || ldx < (nc > 1 ? nc : 1) || ldy < (nr > 1 ? nr : 1))
    return fail(msg, msglen, "%s: bad nrhs %d or leading dimensions (%d, %d) for a %d x %d matrix", who, nrhs, ldx, ldy,
                nr, nc);
  if (nrhs == 0) return 0;
  double *Xp = (double *)xmalloc(sizeof(double) * (size_t)nc * nrhs);
  double *Yp = (double *)xmalloc(sizeof(double) * (size_t)nr * nrhs);
  for (int c = 0; c < nrhs; c++)
    for (int k = 0; k < nc; k++) Xp[k + (size_t)c * nc] = X[pc[k] + (size_t)c * ldx];
  /* the sequential product: starsh_blrm__dmml_omp sizes its per-thread buffers for ranks
   * <= 100 only (GOTCHA(starsh-dmml-omp)) */
  int info = starsh_blrm__dmml(M, nrhs, 1.0, Xp, nc, 0.0, Yp, nr);
  if (info == 0)
    for (int c = 0; c < nrhs; c++)
      for (int k = 0; k < nr; k++) Y[pr[k] + (size_t)c * ldy] = Yp[k + (size_t)c * nr];
  free(Xp);
  free(Yp);
  return info ? fail(msg, msglen, "%s: starsh_blrm__dmml failed (error %d)", who, info) : 0;
}

int hmb_starsh_dense(struct starsh_blrm *M, double *D, int ldd, char *msg, size_t msglen) {
  const char *who = "hmb_starsh_dense";
  if (check_matrix(M, who, msg, msglen)) return -1;
  const STARSH_blrf *F = M->format;
  const int nr = (int)F->problem->shape[0], nc = (int)F->problem->shape[1];
  const STARSH_int *pr = F->row_cluster->pivot, *pc = F->col_cluster->pivot;
  if (ldd < (nr > 1 ? nr : 1)) return fail(msg, msglen, "%s: ldd = %d < %d", who, ldd, nr);
  Array *T = new_array(nr, nc);
  if (!T) return fail(msg, msglen, "%s: out of memory", who);
  memset(T->data, 0, sizeof(double) * (size_t)nr * nc); /* dca writes the blocks only */
  int info = starsh_blrm__dca(M, T);
  const double *t = (const double *)T->data;
  if (info == STARSH_SUCCESS)
    for (int j = 0; j < nc; j++)
      for (int i = 0; i < nr; i++) D[pr[i] + (size_t)pc[j] * ldd] = t[i + (size_t)j * nr];
  array_free(T);
  return info ? fail(msg, msglen, "%s: starsh_blrm__dca failed (error %d)", who, info) : 0;
}
