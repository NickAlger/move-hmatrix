/* move-hmatrix: H2Lib adapter (see hmb_h2lib.h). The only translation unit of the
 * package that includes H2Lib's headers (GOTCHA(header-clash)). */
#include "hmb_h2lib.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* H2Lib last: basic.h undefines UINT_MAX, settings.h defines I */
#include "basic.h"
#include "cluster.h"
#include "h2compression.h"
#include "h2matrix.h"
#include "hmatrix.h"
#include "matrixnorms.h"
#include "rkmatrix.h"
#include "truncation.h"

/* GOTCHA(scalar-type): H2Lib fixes its scalar type per build (USE_COMPLEX, USE_FLOAT); the
 * neutral form is real double. */
_Static_assert(sizeof(field) == sizeof(double) && sizeof(real) == sizeof(double),
               "move-hmatrix needs H2Lib built for real double precision (no USE_COMPLEX, no USE_FLOAT)");

static int fail(char *msg, size_t len, int code, const char *fmt, ...) {
  if (msg && len) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, len, fmt, ap);
    va_end(ap);
  }
  return code;
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) {
    fprintf(stderr, "hmb_h2lib: out of memory (%zu bytes)\n", n);
    abort();
  }
  return p;
}

static double wall(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------------------------ */
int hmb_h2lib_init(int pardepth) {
  static int initialised = 0;
  if (!initialised) {
    int argc = 0;
    char **argv = NULL;
#ifdef _OPENMP
    const int levels = omp_get_max_active_levels();
#endif
    init_h2lib(&argc, &argv);
#ifdef _OPENMP
    /* GOTCHA(h2lib-omp-asserts): init_h2lib() calls omp_set_nested(1) for the whole process */
    omp_set_max_active_levels(levels);
#endif
    initialised = 1;
  }
  if (pardepth >= 0) max_pardepth = pardepth;
  return 0;
}

/* ------------------------------------------------------------------------------------ */
/* Import                                                                                */
/* ------------------------------------------------------------------------------------ */
struct hmb_h2lib_hm {
  pcluster rroot, croot; /* croot == rroot when rows and columns share the tree */
  uint *ridx, *cidx;     /* index arrays the clusters point into (cidx == ridx when shared) */
  phmatrix hm;
};

typedef struct {
  const hmb_hmatrix *A;
  pcluster *rmap, *cmap; /* view cluster id -> H2Lib cluster of the row / column tree */
  int borrow;
  hmb_h2lib_import_info *info;
} import_ctx;

static pcluster make_tree(const hmb_hmatrix *A, int id, uint *idx, pcluster *map) {
  const hmb_cluster *c = A->cl + id;
  pcluster t = new_cluster((uint)c->size, idx + c->offset, (uint)c->nsons, (uint)A->dim);
  for (int d = 0; d < A->dim; d++) {
    t->bmin[d] = c->bmin[d];
    t->bmax[d] = c->bmax[d];
  }
  for (int s = 0; s < c->nsons; s++) t->son[s] = make_tree(A, c->son0 + s, idx, map);
  update_cluster(t);
  map[id] = t;
  return t;
}

static uint *index_array(const int *perm, int n) {
  uint *idx = (uint *)xmalloc(sizeof(uint) * (size_t)n);
  for (int k = 0; k < n; k++) idx[k] = (uint)(perm ? perm[k] : k);
  return idx;
}

/* An amatrix (allocated by H2Lib) aliasing rows x cols column-major storage with leading
 * dimension ld. The view's data is const; H2Lib only reads it on the documented paths. */
static pamatrix alias_amatrix(pamatrix a, const double *src, uint rows, uint cols, int ld) {
  init_pointer_amatrix(a, (pfield)src, rows, cols);
  /* GOTCHA(h2lib-pointer-amatrix): init_pointer_amatrix sets cols = rows and ld = rows */
  a->cols = cols;
  a->ld = (uint)ld;
  return a;
}

/* rows x a->cols block of src (leading dimension ld) into a */
static void copy_block(const double *src, int ld, pamatrix a) {
  for (uint j = 0; j < a->cols; j++) memcpy(a->a + (size_t)j * a->ld, src + (size_t)j * ld, sizeof(field) * a->rows);
}

/* B (cols x rank) = rows coff .. coff + B->rows of the view's column factor, which is V^T
 * (vtrans = 0, V rank x cols) or V (vtrans = 1, V cols x rank). H2Lib's rkmatrix is A*B^H
 * with B cols x rank (GOTCHA(lowrank-convention)). */
static void copy_colfactor(const hmb_block *b, int coff, pamatrix B) {
  if (b->vtrans) {
    for (uint l = 0; l < B->cols; l++)
      memcpy(B->a + (size_t)l * B->ld, b->V + coff + (size_t)l * b->ldV, sizeof(field) * B->rows);
    return;
  }
  for (uint j = 0; j < B->rows; j++) { /* read V once, column by column */
    const double *v = b->V + (size_t)(coff + j) * b->ldV;
    for (uint l = 0; l < B->cols; l++) B->a[j + (size_t)l * B->ld] = v[l];
  }
}

/* Dense leaf (rc, cc) = rows roff.. and columns coff.. of the view's dense block b. */
static phmatrix dense_leaf(import_ctx *x, pccluster rc, pccluster cc, const hmb_block *b, int roff, int coff) {
  const uint m = rc->size, n = cc->size;
  const double *src = b->D ? b->D + roff + (size_t)coff * b->ldD : NULL;
  phmatrix hm;
  if (x->borrow && m > 0 && n > 0) {
    hm = new_hmatrix(rc, cc);
    hm->f = alias_amatrix((pamatrix)allocmem(sizeof(amatrix)), src, m, n, b->ldD);
    x->info->borrowed += (size_t)m * n;
  } else {
    hm = new_full_hmatrix(rc, cc);
    if (m > 0 && n > 0) copy_block(src, b->ldD, hm->f);
    x->info->copied += (size_t)m * n;
  }
  x->info->dense_leaves++;
  update_hmatrix(hm);
  return hm;
}

/* Low-rank leaf (rc, cc): the view's low-rank block b restricted to rows roff.. and
 * columns coff... An empty block holds nothing (its factors may be NULL): rank 0. */
static phmatrix lowrank_leaf(import_ctx *x, pccluster rc, pccluster cc, const hmb_block *b, int roff, int coff) {
  const uint m = rc->size, n = cc->size, k = m > 0 && n > 0 ? (uint)b->rank : 0;
  phmatrix hm;
  if (x->borrow && k > 0) {
    hm = new_hmatrix(rc, cc);
    hm->r = (prkmatrix)allocmem(sizeof(rkmatrix));
    alias_amatrix(&hm->r->A, b->U + roff, m, k, b->ldU);
    if (b->vtrans) {
      alias_amatrix(&hm->r->B, b->V + coff, n, k, b->ldV);
      x->info->borrowed += (size_t)k * (m + n);
    } else {
      init_amatrix(&hm->r->B, n, k);
      copy_colfactor(b, coff, &hm->r->B);
      x->info->borrowed += (size_t)k * m;
      x->info->copied += (size_t)k * n;
    }
    hm->r->k = k;
  } else {
    hm = new_rk_hmatrix(rc, cc, k);
    for (uint l = 0; l < k; l++)
      memcpy(hm->r->A.a + (size_t)l * hm->r->A.ld, b->U + roff + (size_t)l * b->ldU, sizeof(field) * m);
    if (k > 0) copy_colfactor(b, coff, &hm->r->B);
    x->info->copied += (size_t)k * (m + n);
  }
  x->info->lowrank_leaves++;
  update_hmatrix(hm);
  return hm;
}

/* The view's block tree as it is. */
static phmatrix mirror(import_ctx *x, int id) {
  const hmb_hmatrix *A = x->A;
  const hmb_block *b = A->bl + id;
  pccluster rc = x->rmap[b->rc], cc = x->cmap[b->cc];
  if (b->kind == HMB_SUPER) {
    phmatrix hm = new_super_hmatrix(rc, cc, (uint)b->rsons, (uint)b->csons);
    for (int k = 0; k < b->rsons * b->csons; k++) ref_hmatrix(hm->son + k, mirror(x, A->sons[b->son0 + k]));
    update_hmatrix(hm);
    return hm;
  }
  return b->kind == HMB_DENSE ? dense_leaf(x, rc, cc, b, 0, 0) : lowrank_leaf(x, rc, cc, b, 0, 0);
}

static int contains(const hmb_hmatrix *A, int outer, int inner) {
  const hmb_cluster *o = A->cl + outer, *i = A->cl + inner;
  return o->offset <= i->offset && i->offset + i->size <= o->offset + o->size;
}

/* Block (t, s) of a strict block tree, built from the smallest block of the view that holds
 * it (searched from block id down): a low-rank block gives a low-rank leaf on (t, s) at any
 * level (strictness allows it), a dense block on two leaf clusters a dense leaf; a dense
 * block above the leaf level, or a super block, is split on every non-leaf side of (t, s).
 * GOTCHA(strict-tree): this is the lossless refinement H2Lib's H -> H2 conversion needs. */
static phmatrix refine(import_ctx *x, int t, int s, int id) {
  const hmb_hmatrix *A = x->A;
  const hmb_block *b = A->bl + id;
  while (b->kind == HMB_SUPER) {
    int next = -1;
    for (int k = 0; k < b->rsons * b->csons && next < 0; k++) {
      const int c = A->sons[b->son0 + k];
      if (contains(A, A->bl[c].rc, t) && contains(A, A->bl[c].cc, s)) next = c;
    }
    if (next < 0) break;
    id = next;
    b = A->bl + id;
  }
  const hmb_cluster *tc = A->cl + t, *sc = A->cl + s;
  const int roff = tc->offset - A->cl[b->rc].offset, coff = sc->offset - A->cl[b->cc].offset;
  pccluster rc = x->rmap[t], cc = x->cmap[s];
  if (b->kind == HMB_LOWRANK) return lowrank_leaf(x, rc, cc, b, roff, coff);
  if (b->kind == HMB_DENSE && tc->nsons == 0 && sc->nsons == 0) return dense_leaf(x, rc, cc, b, roff, coff);
  if (tc->nsons == 0 && sc->nsons == 0) { /* a super block always has a son holding (t, s) */
    fprintf(stderr, "hmb_h2lib_import: internal error: no block of the view holds a leaf pair\n");
    abort();
  }
  const int rs = tc->nsons ? tc->nsons : 1, cs = sc->nsons ? sc->nsons : 1;
  phmatrix hm = new_super_hmatrix(rc, cc, (uint)rs, (uint)cs);
  for (int j = 0; j < cs; j++)
    for (int i = 0; i < rs; i++)
      ref_hmatrix(hm->son + i + j * rs, refine(x, tc->nsons ? tc->son0 + i : t, sc->nsons ? sc->son0 + j : s, id));
  update_hmatrix(hm);
  return hm;
}

static int same_perm(const hmb_hmatrix *A) {
  if (A->row_perm == A->col_perm) return 1;
  if (A->n_rows != A->n_cols) return 0;
  for (int k = 0; k < A->n_rows; k++)
    if ((A->row_perm ? A->row_perm[k] : k) != (A->col_perm ? A->col_perm[k] : k)) return 0;
  return 1;
}

int hmb_h2lib_import(const hmb_hmatrix *A, int mode, hmb_h2lib_hm **out, hmb_h2lib_import_info *info, char *msg,
                     size_t msglen) {
  char vmsg[256];
  hmb_h2lib_import_info local;
  memset(&local, 0, sizeof local);
  if (!out) return fail(msg, msglen, HMB_H2LIB_EINVAL, "out = NULL");
  *out = NULL;
  if (mode & ~(HMB_H2LIB_BORROW | HMB_H2LIB_STRICT)) return fail(msg, msglen, HMB_H2LIB_EINVAL, "unknown mode %d", mode);
  if (hmb_validate(A, vmsg, sizeof vmsg)) return fail(msg, msglen, HMB_H2LIB_EINVAL, "invalid view: %s", vmsg);

  hmb_h2lib_hm *M = (hmb_h2lib_hm *)calloc(1, sizeof *M);
  pcluster *rmap = (pcluster *)calloc((size_t)A->ncl, sizeof(pcluster));
  pcluster *cmap = (pcluster *)calloc((size_t)A->ncl, sizeof(pcluster));
  if (!M || !rmap || !cmap) {
    fprintf(stderr, "hmb_h2lib: out of memory\n");
    abort();
  }
  const int shared = A->row_root == A->col_root && same_perm(A);
  M->ridx = index_array(A->row_perm, A->n_rows);
  M->rroot = make_tree(A, A->row_root, M->ridx, rmap);
  local.index_ints = (size_t)A->n_rows;
  if (shared) {
    M->cidx = M->ridx;
    M->croot = M->rroot;
    memcpy(cmap, rmap, sizeof(pcluster) * (size_t)A->ncl);
  } else {
    M->cidx = index_array(A->col_perm, A->n_cols);
    M->croot = make_tree(A, A->col_root, M->cidx, cmap);
    local.index_ints += (size_t)A->n_cols;
  }
  import_ctx x = {A, rmap, cmap, (mode & HMB_H2LIB_BORROW) != 0, &local};
  M->hm = (mode & HMB_H2LIB_STRICT) ? refine(&x, A->row_root, A->col_root, 0) : mirror(&x, 0);
  free(rmap);
  free(cmap);
  *out = M;
  if (info) *info = local;
  return 0;
}

struct _hmatrix *hmb_h2lib_hmatrix(const hmb_h2lib_hm *M) { return M ? M->hm : NULL; }

void hmb_h2lib_free(hmb_h2lib_hm *M) {
  if (!M) return;
  if (M->hm) del_hmatrix(M->hm);
  if (M->croot && M->croot != M->rroot) del_cluster(M->croot);
  if (M->rroot) del_cluster(M->rroot);
  if (M->cidx != M->ridx) free(M->cidx);
  free(M->ridx);
  free(M);
}

/* ------------------------------------------------------------------------------------ */
/* Export                                                                                */
/* ------------------------------------------------------------------------------------ */
typedef struct {
  hmb_cstore *s;
  pccluster *node; /* view cluster id -> H2Lib cluster */
  int cap, dim;
  char *msg;
  size_t len;
} export_ctx;

static void remember(export_ctx *x, int id, pccluster t) {
  if (id >= x->cap) {
    x->cap = 2 * id + 16;
    x->node = (pccluster *)realloc(x->node, sizeof(pccluster) * (size_t)x->cap);
    if (!x->node) {
      fprintf(stderr, "hmb_h2lib: out of memory\n");
      abort();
    }
  }
  x->node[id] = t;
}

/* Numbers the cluster tree of root breadth-first (sons get consecutive ids) and derives each
 * cluster's offset from its index array's position inside its father's. */
static int export_tree(export_ctx *x, pccluster root, int *rid) {
  hmb_cstore *s = x->s;
  const int r = hmb_cstore_add_clusters(s, 1);
  s->cl[r].offset = 0;
  s->cl[r].size = (int)root->size;
  remember(x, r, root);
  for (int q = r; q < s->ncl; q++) {
    pccluster t = x->node[q];
    for (int d = 0; d < x->dim; d++) {
      s->cl[q].bmin[d] = t->bmin[d];
      s->cl[q].bmax[d] = t->bmax[d];
    }
    if (t->sons == 0) continue;
    const int s0 = hmb_cstore_add_clusters(s, (int)t->sons);
    const uint *next = t->idx;
    int offset = s->cl[q].offset;
    s->cl[q].nsons = (int)t->sons;
    s->cl[q].son0 = s0;
    for (uint i = 0; i < t->sons; i++) {
      pccluster u = t->son[i];
      if (u->idx != next)
        return fail(x->msg, x->len, HMB_H2LIB_EUNSUPPORTED,
                    "the index arrays of the sons of cluster %d (size %u) are not consecutive inside its own", q,
                    t->size);
      s->cl[s0 + i].offset = offset;
      s->cl[s0 + i].size = (int)u->size;
      remember(x, s0 + (int)i, u);
      next += u->size;
      offset += (int)u->size;
    }
    if (next != t->idx + t->size)
      return fail(x->msg, x->len, HMB_H2LIB_EUNSUPPORTED, "the sons of cluster %d (size %u) do not cover it", q, t->size);
  }
  *rid = r;
  return 0;
}

/* Id of a son block's cluster c: the father block's cluster p (side not split) or p's i-th
 * son (side split into all sons of p). */
static int son_cluster(const export_ctx *x, int p, uint nsplit, uint i, pccluster c) {
  const hmb_cluster *pc = x->s->cl + p;
  if (nsplit == (uint)pc->nsons && x->node[pc->son0 + (int)i] == c) return pc->son0 + (int)i;
  if (nsplit == 1 && c == x->node[p]) return p;
  return -1;
}

static int export_block(export_ctx *x, pchmatrix hm, int rcid, int ccid) {
  hmb_block b;
  memset(&b, 0, sizeof b);
  b.rc = rcid;
  b.cc = ccid;
  const uint m = hm->rc->size, n = hm->cc->size;
  if (hm->son) {
    b.kind = HMB_SUPER;
    b.rsons = (int)hm->rsons;
    b.csons = (int)hm->csons;
    b.son0 = hmb_cstore_add_sons(x->s, b.rsons * b.csons);
    const int id = hmb_cstore_add_block(x->s, &b);
    for (uint j = 0; j < hm->csons; j++)
      for (uint i = 0; i < hm->rsons; i++) {
        pchmatrix c = hm->son[i + j * hm->rsons];
        const int crc = son_cluster(x, rcid, hm->rsons, i, c->rc), ccc = son_cluster(x, ccid, hm->csons, j, c->cc);
        if (crc < 0 || ccc < 0)
          return fail(x->msg, x->len, HMB_H2LIB_EUNSUPPORTED,
                      "son (%u, %u) of a %u x %u block lies on a cluster that is neither its father's nor one of "
                      "its sons",
                      i, j, m, n);
        const int cid = export_block(x, c, crc, ccc);
        if (cid < 0) return cid;
        x->s->sons[b.son0 + (int)(i + j * hm->rsons)] = cid;
      }
    return id;
  }
  if (hm->f) {
    if (hm->f->rows != m || hm->f->cols != n)
      return fail(x->msg, x->len, HMB_H2LIB_EUNSUPPORTED, "a %u x %u dense leaf on %u x %u clusters", hm->f->rows,
                  hm->f->cols, m, n);
    b.kind = HMB_DENSE;
    b.D = hm->f->a;
    b.ldD = hm->f->ld > 0 ? (int)hm->f->ld : 1;
  } else if (hm->r) {
    const prkmatrix r = hm->r;
    if (r->A.rows != m || r->B.rows != n || r->A.cols < r->k || r->B.cols < r->k)
      return fail(x->msg, x->len, HMB_H2LIB_EUNSUPPORTED, "a low-rank leaf whose factors do not match its clusters");
    /* GOTCHA(lowrank-convention): A*B^H with B cols x rank is U*V^T, vtrans = 1 */
    b.kind = HMB_LOWRANK;
    b.rank = (int)r->k;
    b.U = r->A.a;
    b.ldU = r->A.ld > 0 ? (int)r->A.ld : 1;
    b.V = r->B.a;
    b.ldV = r->B.ld > 0 ? (int)r->B.ld : 1;
    b.vtrans = 1;
  } else {
    b.kind = HMB_LOWRANK; /* H2Lib treats a leaf without data as a zero block */
  }
  return hmb_cstore_add_block(x->s, &b);
}

int hmb_h2lib_export(const struct _hmatrix *A, hmb_cstore *out, char *msg, size_t msglen) {
  if (!A || !out) return fail(msg, msglen, HMB_H2LIB_EINVAL, "null argument");
  if (out->ncl || out->nbl || out->nsons || out->nchunks)
    return fail(msg, msglen, HMB_H2LIB_EINVAL, "the output store must be empty (hmb_cstore_init)");
  export_ctx x = {out, NULL, 0, 0, msg, msglen};
  x.dim = A->rc->dim <= 3 && A->rc->dim == A->cc->dim ? (int)A->rc->dim : 0;
  int rroot = -1, croot = -1, rc = export_tree(&x, A->rc, &rroot);
  if (rc == 0) {
    if (A->cc == A->rc) croot = rroot;
    else rc = export_tree(&x, A->cc, &croot);
  }
  if (rc == 0) {
    const int root = export_block(&x, A, rroot, croot);
    rc = root < 0 ? root : 0;
  }
  free(x.node);
  if (rc == 0) {
    hmb_cstore_finalize(out);
    out->A.dim = x.dim;
    out->A.n_rows = (int)A->rc->size;
    out->A.n_cols = (int)A->cc->size;
    out->A.row_root = rroot;
    out->A.col_root = croot;
    /* zero-copy permutations: the root clusters' index arrays (unsigned and signed int may
       alias each other) */
    out->A.row_perm = (const int *)A->rc->idx;
    out->A.col_perm = (const int *)A->cc->idx;
    char vmsg[256];
    if (hmb_validate(&out->A, vmsg, sizeof vmsg))
      rc = fail(msg, msglen, HMB_H2LIB_EUNSUPPORTED, "the exported view is invalid: %s", vmsg);
  }
  if (rc) {
    hmb_cstore_free(out);
    hmb_cstore_init(out);
  }
  return rc;
}

/* ------------------------------------------------------------------------------------ */
/* Strictness, H -> H2, products, sizes                                                  */
/* ------------------------------------------------------------------------------------ */
static void strict_rec(pchmatrix hm, long v[2]) {
  if (hm->son) {
    if ((hm->rc->sons > 0 && hm->son[0]->rc == hm->rc) || (hm->cc->sons > 0 && hm->son[0]->cc == hm->cc)) v[1]++;
    for (uint k = 0; k < hm->rsons * hm->csons; k++) strict_rec(hm->son[k], v);
  } else if (hm->f && (hm->rc->sons > 0 || hm->cc->sons > 0)) {
    v[0]++;
  }
}

void hmb_h2lib_strict_violations(const struct _hmatrix *A, long v[2]) {
  v[0] = v[1] = 0;
  strict_rec(A, v);
}

static unsigned kmax_basis(pcclusterbasis cb) {
  unsigned k = cb->k;
  for (uint i = 0; i < cb->sons; i++) {
    unsigned ks = kmax_basis(cb->son[i]);
    if (ks > k) k = ks;
  }
  return k;
}

int hmb_h2lib_to_h2(const struct _hmatrix *A, double eps, struct _h2matrix **out, hmb_h2lib_h2_info *info, char *msg,
                    size_t msglen) {
  if (!A || !out) return fail(msg, msglen, HMB_H2LIB_EINVAL, "null argument");
  *out = NULL;
  if (!(eps > 0)) return fail(msg, msglen, HMB_H2LIB_EINVAL, "eps = %g must be positive", eps);
  long v[2];
  hmb_h2lib_strict_violations(A, v);
  if (v[0] || v[1])
    /* GOTCHA(strict-tree): H2Lib would return a wrong H2 matrix or crash */
    return fail(msg, msglen, HMB_H2LIB_ENOTSTRICT,
                "H -> H2 needs a strict block tree, this one has %ld dense leaves on non-leaf clusters and %ld "
                "super blocks that keep a non-leaf cluster; import with HMB_H2LIB_STRICT to refine it losslessly",
                v[0], v[1]);
  ptruncmode tm = new_blockreleucl_truncmode();
  const double t0 = wall();
  pclusterbasis rb = buildrowbasis_hmatrix(A, tm, eps);
  const double t1 = wall();
  pclusterbasis cb = buildcolbasis_hmatrix(A, tm, eps);
  const double t2 = wall();
  ph2matrix H2 = build_projected_hmatrix_h2matrix(A, rb, cb);
  const double t3 = wall();
  del_truncmode(tm);
  if (info) {
    info->rowbasis_s = t1 - t0;
    info->colbasis_s = t2 - t1;
    info->projection_s = t3 - t2;
    info->bytes = getsize_h2matrix(H2);
    info->near_bytes = getnearsize_h2matrix(H2);
    info->far_bytes = getfarsize_h2matrix(H2);
    info->rank_sum_row = H2->rb->ktree;
    info->rank_sum_col = H2->cb->ktree;
    info->rank_max_row = kmax_basis(H2->rb);
    info->rank_max_col = kmax_basis(H2->cb);
  }
  *out = H2;
  return 0;
}

void hmb_h2lib_free_h2(struct _h2matrix *A) {
  if (A) del_h2matrix(A);
}

void hmb_h2lib_matvec(const struct _hmatrix *A, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy) {
  const int t = trans == 'T' || trans == 't';
  const uint nin = t ? A->rc->size : A->cc->size, nout = t ? A->cc->size : A->rc->size;
  for (int c = 0; c < nrhs; c++) {
    avector xv, yv;
    init_pointer_avector(&xv, (pfield)(X + (size_t)c * ldx), nin);
    init_pointer_avector(&yv, Y + (size_t)c * ldy, nout);
    clear_avector(&yv);
    if (t) addevaltrans_hmatrix_avector(1.0, A, &xv, &yv); /* both permute through the roots' idx */
    else addeval_hmatrix_avector(1.0, A, &xv, &yv);
    uninit_avector(&yv);
    uninit_avector(&xv);
  }
}

void hmb_h2lib_h2_matvec(const struct _h2matrix *A, char trans, int nrhs, const double *X, int ldx, double *Y,
                         int ldy) {
  const int t = trans == 'T' || trans == 't';
  const uint nin = t ? A->rb->t->size : A->cb->t->size, nout = t ? A->cb->t->size : A->rb->t->size;
  for (int c = 0; c < nrhs; c++) {
    avector xv, yv;
    init_pointer_avector(&xv, (pfield)(X + (size_t)c * ldx), nin);
    init_pointer_avector(&yv, Y + (size_t)c * ldy, nout);
    clear_avector(&yv);
    if (t) addevaltrans_h2matrix_avector(1.0, A, &xv, &yv);
    else addeval_h2matrix_avector(1.0, A, &xv, &yv);
    uninit_avector(&yv);
    uninit_avector(&xv);
  }
}

void hmb_h2lib_size(const struct _hmatrix *A, int *rows, int *cols) {
  *rows = (int)A->rc->size;
  *cols = (int)A->cc->size;
}

void hmb_h2lib_h2_size(const struct _h2matrix *A, int *rows, int *cols) {
  *rows = (int)A->rb->t->size;
  *cols = (int)A->cb->t->size;
}

size_t hmb_h2lib_bytes(const struct _hmatrix *A) { return getsize_hmatrix(A); }

double hmb_h2lib_h2_reldiff(const struct _hmatrix *A, const struct _h2matrix *B) {
  const real nA = norm2_hmatrix(A), d = norm2diff_hmatrix_h2matrix(B, A);
  return nA > 0 ? d / nA : d;
}
