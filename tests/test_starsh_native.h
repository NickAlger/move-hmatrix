/* Native STARS-H helpers for tests/test_starsh.cpp (implemented in test_starsh_native.c,
 * the only test file that includes starsh.h). Test support only. */
#ifndef HMB_TEST_STARSH_NATIVE_H
#define HMB_TEST_STARSH_NATIVE_H

#include "hmb/neutral.h"

#ifdef __cplusplus
extern "C" {
#endif

struct starsh_blrm;
typedef struct ts_matrix ts_matrix; /* problem + cluster + format + matrix, built with STARS-H's API */

/* TLR approximation of the Sym kernel of hmb_test.hpp at points xyz (3 x n, point-major,
 * referenced, not copied) by starsh_blrm_approximate; onfly = 1 keeps dense blocks as kernel
 * calls. */
ts_matrix *ts_kernel_tlr(int n, const double *xyz, const int *perm, int tile, double tol, int maxrank, char symm,
                         int onfly);
/* Random rank-r tiles with the far-field list in column order (not sorted by block row). */
ts_matrix *ts_random_tlr_colmajor(int n, int tile, int rank, unsigned seed);
/* 10 x 10 dense-block partitions the export must refuse: kind 0 a block listed twice,
 * 1 a block missing, 2 cluster ranges overlapping without nesting. */
ts_matrix *ts_broken(int kind);
struct starsh_blrm *ts_blrm(ts_matrix *t);
void ts_free(ts_matrix *t);

int ts_get_block_mismatches(struct starsh_blrm *M);
double ts_kernel_maxdiff(struct starsh_blrm *M, const hmb_hmatrix *ref, int trials, unsigned seed);
double ts_dfe(struct starsh_blrm *M);
int ts_onfly(struct starsh_blrm *M);

#ifdef __cplusplus
}
#endif
#endif
