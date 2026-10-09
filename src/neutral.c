/* hmatrix-bridges: reference implementations of the neutral-view utilities (neutral.h).
 * Simple and checked rather than tuned; products go through the Fortran BLAS dgemm. */
#include "hmb/neutral.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
            const double *beta, double *c, const int *ldc);

static void gemm(char ta, char tb, int m, int n, int k, double alpha, const double *a, int lda,
                 const double *b, int ldb, double beta, double *c, int ldc) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) { /* C = beta * C */
    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++) c[i + (size_t)j * ldc] = beta == 0.0 ? 0.0 : beta * c[i + (size_t)j * ldc];
    return;
  }
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) {
    fprintf(stderr, "hmatrix-bridges: out of memory (%zu bytes)\n", n);
    abort();
  }
  return p;
}

static int imax(int a, int b) { return a > b ? a : b; }

/* ------------------------------------------------------------------------------------ */
size_t hmb_stored_values(const hmb_hmatrix *A) {
  size_t s = 0;
  for (int k = 0; k < A->nbl; k++) {
    const hmb_block *b = A->bl + k;
    size_t m = (size_t)A->cl[b->rc].size, n = (size_t)A->cl[b->cc].size;
    if (b->kind == HMB_DENSE) s += m * n;
    else if (b->kind == HMB_LOWRANK) s += (size_t)b->rank * (m + n);
  }
  return s;
}

/* ------------------------------------------------------------------------------------ */
typedef struct {
  char *msg;
  size_t len;
} errbuf;

static int fail(errbuf *e, const char *fmt, ...) {
  if (e->msg && e->len) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->msg, e->len, fmt, ap);
    va_end(ap);
  }
  return -1;
}

static int check_perm(errbuf *e, const char *which, const int *p, int n) {
  if (!p) return 0;
  char *seen = (char *)calloc((size_t)(n ? n : 1), 1);
  int rc = 0;
  for (int k = 0; k < n && rc == 0; k++) {
    if (p[k] < 0 || p[k] >= n) rc = fail(e, "%s_perm[%d] = %d is out of range [0, %d)", which, k, p[k], n);
    else if (seen[p[k]]) rc = fail(e, "%s_perm is not a permutation: %d appears twice", which, p[k]);
    else seen[p[k]] = 1;
  }
  free(seen);
  return rc;
}

/* Walks one cluster tree from root; mark[] counts visits within this tree. */
static int check_tree(errbuf *e, const hmb_hmatrix *A, const char *which, int root, int n, char *mark) {
  if (root < 0 || root >= A->ncl) return fail(e, "%s_root = %d is not a cluster id (ncl = %d)", which, root, A->ncl);
  if (A->cl[root].offset != 0 || A->cl[root].size != n)
    return fail(e, "%s root cluster %d covers [%d, %d), expected [0, %d)", which, root, A->cl[root].offset,
                A->cl[root].offset + A->cl[root].size, n);
  int *stack = (int *)xmalloc(sizeof(int) * (size_t)A->ncl + sizeof(int));
  int top = 0, rc = 0;
  stack[top++] = root;
  while (top > 0 && rc == 0) {
    int c = stack[--top];
    const hmb_cluster *p = A->cl + c;
    if (mark[c]++) { rc = fail(e, "%s tree reaches cluster %d twice", which, c); break; }
    if (p->size < 0 || p->offset < 0 || p->offset + p->size > n) {
      rc = fail(e, "%s cluster %d range [%d, %d) is outside [0, %d)", which, c, p->offset, p->offset + p->size, n);
      break;
    }
    if (p->nsons < 0 || (p->nsons > 0 && (p->son0 < 0 || p->son0 + p->nsons > A->ncl))) {
      rc = fail(e, "%s cluster %d has sons %d .. %d, outside [0, %d)", which, c, p->son0, p->son0 + p->nsons - 1, A->ncl);
      break;
    }
    int next = p->offset;
    for (int s = 0; s < p->nsons; s++) {
      const hmb_cluster *q = A->cl + p->son0 + s;
      if (q->offset != next) {
        rc = fail(e, "%s cluster %d: son %d starts at %d, expected %d (sons must partition the parent in order)",
                  which, c, p->son0 + s, q->offset, next);
        break;
      }
      next += q->size;
      stack[top++] = p->son0 + s;
    }
    if (rc == 0 && p->nsons > 0 && next != p->offset + p->size)
      rc = fail(e, "%s cluster %d: sons cover [%d, %d), parent covers [%d, %d)", which, c, p->offset, next, p->offset,
                p->offset + p->size);
  }
  free(stack);
  return rc;
}

int hmb_validate(const hmb_hmatrix *A, char *msg, size_t msglen) {
  errbuf e = {msg, msglen};
  if (msg && msglen) msg[0] = 0;
  if (!A) return fail(&e, "null view");
  if (A->n_rows < 0 || A->n_cols < 0) return fail(&e, "negative size %d x %d", A->n_rows, A->n_cols);
  if (A->ncl < 1 || !A->cl) return fail(&e, "no clusters");
  if (A->nbl < 1 || !A->bl) return fail(&e, "no blocks");
  if (A->sons_len < 0 || (A->sons_len > 0 && !A->sons)) return fail(&e, "sons_len = %d but sons = NULL", A->sons_len);
  if (check_perm(&e, "row", A->row_perm, A->n_rows) || check_perm(&e, "col", A->col_perm, A->n_cols)) return -1;

  int rc = 0, top = 0;
  char *rmark = (char *)calloc((size_t)A->ncl, 1), *cmark = (char *)calloc((size_t)A->ncl, 1);
  char *bmark = (char *)calloc((size_t)A->nbl, 1);
  int *stack = (int *)xmalloc(sizeof(int) * (size_t)A->nbl + sizeof(int));
  if (check_tree(&e, A, "row", A->row_root, A->n_rows, rmark) ||
      check_tree(&e, A, "col", A->col_root, A->n_cols, cmark)) {
    rc = -1;
    goto done;
  }
  if (A->bl[0].rc != A->row_root || A->bl[0].cc != A->col_root) {
    rc = fail(&e, "root block pairs clusters (%d, %d), expected (row_root %d, col_root %d)", A->bl[0].rc, A->bl[0].cc,
              A->row_root, A->col_root);
    goto done;
  }
  stack[top++] = 0;
  bmark[0] = 1;
  while (top > 0) {
    int id = stack[--top];
    const hmb_block *b = A->bl + id;
    if (b->rc < 0 || b->rc >= A->ncl || b->cc < 0 || b->cc >= A->ncl) {
      rc = fail(&e, "block %d: cluster ids (%d, %d) out of range", id, b->rc, b->cc);
      goto done;
    }
    const hmb_cluster *r = A->cl + b->rc, *c = A->cl + b->cc;
    const int m = r->size, n = c->size;
    if (b->kind == HMB_SUPER) {
      if (b->rsons < 1 || b->csons < 1 || b->son0 < 0 || b->son0 + b->rsons * b->csons > A->sons_len) {
        rc = fail(&e, "block %d: %d x %d children at sons[%d..] exceed sons_len %d", id, b->rsons, b->csons, b->son0,
                  A->sons_len);
        goto done;
      }
      if (b->rsons > 1 && b->rsons != r->nsons) {
        rc = fail(&e, "block %d: split into %d row parts but row cluster %d has %d sons", id, b->rsons, b->rc, r->nsons);
        goto done;
      }
      if (b->csons > 1 && b->csons != c->nsons) {
        rc = fail(&e, "block %d: split into %d column parts but column cluster %d has %d sons", id, b->csons, b->cc,
                  c->nsons);
        goto done;
      }
      for (int j = 0; j < b->csons; j++)
        for (int i = 0; i < b->rsons; i++) {
          int ch = A->sons[b->son0 + i + j * b->rsons];
          if (ch < 0 || ch >= A->nbl) {
            rc = fail(&e, "block %d: child (%d, %d) = %d is not a block id (nbl = %d)", id, i, j, ch, A->nbl);
            goto done;
          }
          if (ch == 0 || bmark[ch]) {
            rc = fail(&e, "block %d reached twice (child (%d, %d) of block %d)", ch, i, j, id);
            goto done;
          }
          int want_r = b->rsons > 1 ? r->son0 + i : b->rc, want_c = b->csons > 1 ? c->son0 + j : b->cc;
          if (A->bl[ch].rc != want_r || A->bl[ch].cc != want_c) {
            rc = fail(&e, "block %d: child (%d, %d) = block %d pairs clusters (%d, %d), expected (%d, %d)", id, i, j, ch,
                      A->bl[ch].rc, A->bl[ch].cc, want_r, want_c);
            goto done;
          }
          bmark[ch] = 1;
          stack[top++] = ch;
        }
    } else if (b->kind == HMB_DENSE) {
      if ((size_t)m * (size_t)n > 0 && !b->D) {
        rc = fail(&e, "dense block %d (%d x %d) has D = NULL", id, m, n);
        goto done;
      }
      if (b->ldD < imax(1, m)) {
        rc = fail(&e, "dense block %d: ldD = %d < rows = %d", id, b->ldD, m);
        goto done;
      }
    } else if (b->kind == HMB_LOWRANK) {
      if (b->rank < 0) {
        rc = fail(&e, "low-rank block %d: negative rank %d", id, b->rank);
        goto done;
      }
      if (b->vtrans != 0 && b->vtrans != 1) {
        rc = fail(&e, "low-rank block %d: vtrans = %d (must be 0 or 1)", id, b->vtrans);
        goto done;
      }
      if (b->rank > 0 && m > 0 && n > 0) {
        if (!b->U || !b->V) {
          rc = fail(&e, "low-rank block %d (rank %d) has a NULL factor", id, b->rank);
          goto done;
        }
        if (b->ldU < imax(1, m)) {
          rc = fail(&e, "low-rank block %d: ldU = %d < rows = %d", id, b->ldU, m);
          goto done;
        }
        int need = b->vtrans ? n : b->rank;
        if (b->ldV < imax(1, need)) {
          rc = fail(&e, "low-rank block %d: ldV = %d < %d (%s)", id, b->ldV, need,
                    b->vtrans ? "V is cols x rank" : "V is rank x cols");
          goto done;
        }
      }
    } else {
      rc = fail(&e, "block %d: unknown kind %d", id, b->kind);
      goto done;
    }
  }
  for (int k = 0; k < A->nbl; k++)
    if (!bmark[k]) {
      rc = fail(&e, "block %d is not reachable from the root block", k);
      goto done;
    }
done:
  free(rmark);
  free(cmark);
  free(bmark);
  free(stack);
  return rc;
}

/* ------------------------------------------------------------------------------------ */
void hmb_strict_violations(const hmb_hmatrix *A, long v[2]) {
  v[0] = v[1] = 0;
  for (int k = 0; k < A->nbl; k++) {
    const hmb_block *b = A->bl + k;
    int rnonleaf = A->cl[b->rc].nsons > 0, cnonleaf = A->cl[b->cc].nsons > 0;
    if (b->kind == HMB_DENSE && (rnonleaf || cnonleaf)) v[0]++;
    if (b->kind == HMB_SUPER && ((rnonleaf && b->rsons == 1) || (cnonleaf && b->csons == 1))) v[1]++;
  }
}

/* ------------------------------------------------------------------------------------ */
typedef struct {
  const hmb_hmatrix *A;
  int t, nrhs;
  const double *X; /* permuted input, leading dimension ldx */
  int ldx;
  double *Y;       /* permuted output, leading dimension ldy */
  int ldy;
  double *T;       /* rank x nrhs workspace */
} mv_ctx;

static void mv_rec(mv_ctx *c, int id) {
  const hmb_hmatrix *A = c->A;
  const hmb_block *b = A->bl + id;
  if (b->kind == HMB_SUPER) {
    for (int k = 0; k < b->rsons * b->csons; k++) mv_rec(c, A->sons[b->son0 + k]);
    return;
  }
  const int m = A->cl[b->rc].size, n = A->cl[b->cc].size;
  const int ro = A->cl[b->rc].offset, co = A->cl[b->cc].offset;
  if (m == 0 || n == 0) return;
  if (b->kind == HMB_DENSE) {
    if (!c->t) gemm('N', 'N', m, c->nrhs, n, 1.0, b->D, b->ldD, c->X + co, c->ldx, 1.0, c->Y + ro, c->ldy);
    else gemm('T', 'N', n, c->nrhs, m, 1.0, b->D, b->ldD, c->X + ro, c->ldx, 1.0, c->Y + co, c->ldy);
    return;
  }
  const int k = b->rank;
  if (k == 0) return;
  if (!c->t) { /* y_r += U (Vop x_c), Vop = V (k x n) or V^T (V n x k) */
    gemm(b->vtrans ? 'T' : 'N', 'N', k, c->nrhs, n, 1.0, b->V, b->ldV, c->X + co, c->ldx, 0.0, c->T, k);
    gemm('N', 'N', m, c->nrhs, k, 1.0, b->U, b->ldU, c->T, k, 1.0, c->Y + ro, c->ldy);
  } else { /* y_c += Vop^T (U^T x_r) */
    gemm('T', 'N', k, c->nrhs, m, 1.0, b->U, b->ldU, c->X + ro, c->ldx, 0.0, c->T, k);
    gemm(b->vtrans ? 'N' : 'T', 'N', n, c->nrhs, k, 1.0, b->V, b->ldV, c->T, k, 1.0, c->Y + co, c->ldy);
  }
}

void hmb_matvec(const hmb_hmatrix *A, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy) {
  const int t = (trans == 'T' || trans == 't');
  const int nin = t ? A->n_rows : A->n_cols, nout = t ? A->n_cols : A->n_rows;
  const int *pin = t ? A->row_perm : A->col_perm, *pout = t ? A->col_perm : A->row_perm;
  if (nrhs <= 0) return;
  int kmax = 0;
  for (int b = 0; b < A->nbl; b++)
    if (A->bl[b].kind == HMB_LOWRANK) kmax = imax(kmax, A->bl[b].rank);
  double *Xp = (double *)xmalloc(sizeof(double) * (size_t)nin * nrhs);
  double *Yp = (double *)calloc((size_t)(nout ? nout : 1) * nrhs, sizeof(double));
  double *T = (double *)xmalloc(sizeof(double) * (size_t)(kmax ? kmax : 1) * nrhs);
  if (!Yp) abort();
  for (int c = 0; c < nrhs; c++)
    for (int k = 0; k < nin; k++) Xp[k + (size_t)c * nin] = X[(pin ? pin[k] : k) + (size_t)c * ldx];
  mv_ctx ctx = {A, t, nrhs, Xp, imax(1, nin), Yp, imax(1, nout), T};
  mv_rec(&ctx, 0);
  for (int c = 0; c < nrhs; c++)
    for (int k = 0; k < nout; k++) Y[(pout ? pout[k] : k) + (size_t)c * ldy] = Yp[k + (size_t)c * nout];
  free(Xp);
  free(Yp);
  free(T);
}

/* ------------------------------------------------------------------------------------ */
typedef struct {
  int pos; /* permuted position */
  int idx; /* index in the request */
} req;

static int cmp_req(const void *a, const void *b) {
  int pa = ((const req *)a)->pos, pb = ((const req *)b)->pos;
  return (pa > pb) - (pa < pb);
}

typedef struct {
  const hmb_hmatrix *A;
  const req *R, *C;
  double *out;
  int ldo;
} ent_ctx;

/* first index in [lo, hi) whose pos >= bound (positions sorted ascending) */
static int lower(const req *v, int lo, int hi, int bound) {
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (v[mid].pos < bound) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

static void ent_rec(const ent_ctx *e, int id, int r0, int r1, int c0, int c1) {
  if (r0 >= r1 || c0 >= c1) return;
  const hmb_hmatrix *A = e->A;
  const hmb_block *b = A->bl + id;
  if (b->kind == HMB_SUPER) {
    int rb[b->rsons + 1], cb[b->csons + 1];
    rb[0] = r0;
    for (int i = 0; i < b->rsons; i++) {
      const hmb_cluster *ci = A->cl + A->bl[A->sons[b->son0 + i]].rc;
      rb[i + 1] = i == b->rsons - 1 ? r1 : lower(e->R, rb[i], r1, ci->offset + ci->size);
    }
    cb[0] = c0;
    for (int j = 0; j < b->csons; j++) {
      const hmb_cluster *cj = A->cl + A->bl[A->sons[b->son0 + j * b->rsons]].cc;
      cb[j + 1] = j == b->csons - 1 ? c1 : lower(e->C, cb[j], c1, cj->offset + cj->size);
    }
    for (int j = 0; j < b->csons; j++)
      for (int i = 0; i < b->rsons; i++)
        ent_rec(e, A->sons[b->son0 + i + j * b->rsons], rb[i], rb[i + 1], cb[j], cb[j + 1]);
    return;
  }
  const int ro = A->cl[b->rc].offset, co = A->cl[b->cc].offset;
  if (b->kind == HMB_DENSE) {
    for (int c = c0; c < c1; c++) {
      const double *Dc = b->D + (size_t)(e->C[c].pos - co) * b->ldD - ro;
      double *oc = e->out + (size_t)e->C[c].idx * e->ldo;
      for (int r = r0; r < r1; r++) oc[e->R[r].idx] = Dc[e->R[r].pos];
    }
    return;
  }
  const int k = b->rank;
  for (int c = c0; c < c1; c++) {
    const int jc = e->C[c].pos - co;
    double *oc = e->out + (size_t)e->C[c].idx * e->ldo;
    for (int r = r0; r < r1; r++) {
      const int ir = e->R[r].pos - ro;
      double s = 0;
      if (b->vtrans)
        for (int l = 0; l < k; l++) s += b->U[ir + (size_t)l * b->ldU] * b->V[jc + (size_t)l * b->ldV];
      else
        for (int l = 0; l < k; l++) s += b->U[ir + (size_t)l * b->ldU] * b->V[l + (size_t)jc * b->ldV];
      oc[e->R[r].idx] = s;
    }
  }
}

void hmb_inverse_perms(const hmb_hmatrix *A, int *irow, int *icol) {
  if (irow)
    for (int k = 0; k < A->n_rows; k++) irow[A->row_perm ? A->row_perm[k] : k] = k;
  if (icol)
    for (int k = 0; k < A->n_cols; k++) icol[A->col_perm ? A->col_perm[k] : k] = k;
}

void hmb_entries_ip(const hmb_hmatrix *A, const int *irow, const int *icol, int m, const int *rows, int n,
                    const int *cols, double *out, int ldo) {
  if (m <= 0 || n <= 0) return;
  req *R = (req *)xmalloc(sizeof(req) * (size_t)m), *C = (req *)xmalloc(sizeof(req) * (size_t)n);
  for (int i = 0; i < m; i++) {
    R[i].pos = irow ? irow[rows[i]] : (A->row_perm ? -1 : rows[i]);
    R[i].idx = i;
  }
  for (int j = 0; j < n; j++) {
    C[j].pos = icol ? icol[cols[j]] : (A->col_perm ? -1 : cols[j]);
    C[j].idx = j;
  }
  qsort(R, (size_t)m, sizeof(req), cmp_req);
  qsort(C, (size_t)n, sizeof(req), cmp_req);
  if ((m && R[0].pos < 0) || (n && C[0].pos < 0)) {
    fprintf(stderr, "hmb_entries_ip: a permuted view needs inverse permutations (hmb_inverse_perms)\n");
    abort();
  }
  ent_ctx e = {A, R, C, out, ldo};
  ent_rec(&e, 0, 0, m, 0, n);
  free(R);
  free(C);
}

void hmb_entries(const hmb_hmatrix *A, int m, const int *rows, int n, const int *cols, double *out, int ldo) {
  int *irow = A->row_perm ? (int *)xmalloc(sizeof(int) * (size_t)A->n_rows) : NULL;
  int *icol = A->col_perm ? (int *)xmalloc(sizeof(int) * (size_t)A->n_cols) : NULL;
  hmb_inverse_perms(A, irow, icol);
  hmb_entries_ip(A, irow, icol, m, rows, n, cols, out, ldo);
  free(irow);
  free(icol);
}
