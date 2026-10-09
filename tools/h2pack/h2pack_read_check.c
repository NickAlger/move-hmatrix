/* hmatrix-bridges: read an H2Pack file set with H2P_read_from_file, multiply, compare.
 *
 * usage: h2pack_read_check --meta F_meta.json --aux F_aux.json --bin F.bin
 *            --x X.bin --y Y.bin [--nrhs K] [--tol T]
 *            --converter <H2Pack>/examples/meta_json_to_txt.py [--python python3]
 *            [--allow-unpatched]
 *   X.bin, Y.bin: n x K raw little-endian float64, column-major, ORIGINAL ordering
 *   (h2pack_write.py write_vectors). The program computes A X with H2P_matmul (all K
 *   columns) and H2P_matvec (first column), prints both relative Frobenius differences to Y
 *   with the threshold T (default 1e-12) and exits with status 0 if both are <= T, 1 if not,
 *   2 on a usage or I/O error, 3 if H2Pack could not read the files.
 *
 * H2Pack's reader, and what this program does about it:
 *   - GOTCHA(h2pack-cwd-scripts): H2P_read_from_file runs "./meta_json_to_txt.py" in the
 *     current directory through system() (python3 from PATH, via the script's shebang) and
 *     writes its intermediate tmp_metadata.txt there under that fixed name; without the script
 *     it prints an error and returns, leaving the output pointer untouched. The read here runs
 *     in a private temporary directory holding a small wrapper of that name, which runs
 *     H2Pack's converter (--converter) with the chosen interpreter (--python); file names are
 *     passed as absolute paths. The directory is removed afterwards.
 *   - GOTCHA(h2pack-reader-bugs): the program is linked with h2pack_reader_fix.o and
 *     -Wl,--wrap=H2P_calc_node_inadm_lists (see h2pack_reader_fix.h); it refuses to report
 *     results when the fix did not run (a shared libH2Pack, another linker) unless
 *     --allow-unpatched is given.
 *   - GOTCHA(h2pack-blas-threads): H2Pack's construction routines set the BLAS thread count
 *     (BLAS_SET_NUM_THREADS); the count is saved before H2Pack runs and restored afterwards,
 *     and a change is reported.
 *   - GOTCHA(scalar-type): H2Pack's DTYPE must be double (DTYPE_SIZE 8, its default); checked at
 *     compile time.
 */
#include "h2pack_reader_fix.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* before H2Pack.h, whose H2Pack_aux_structs.h uses memcpy without it */
#include <sys/stat.h>
#include <unistd.h>

#include "H2Pack.h"

_Static_assert(sizeof(DTYPE) == 8, "h2pack_read_check needs H2Pack built with double precision (DTYPE_SIZE 8)");

#if defined(USE_MKL)
#define BLAS_GET_THREADS() mkl_get_max_threads()
#define BLAS_SET_THREADS(k) mkl_set_num_threads(k)
#else
#define BLAS_GET_THREADS() openblas_get_num_threads()
#define BLAS_SET_THREADS(k) openblas_set_num_threads(k)
#endif

static void usage(void) {
  fprintf(stderr, "usage: h2pack_read_check --meta F_meta.json --aux F_aux.json --bin F.bin --x X.bin --y Y.bin\n"
                  "           [--nrhs K] [--tol T] --converter meta_json_to_txt.py [--python python3]\n"
                  "           [--allow-unpatched]\n");
}

/* whole file of float64; *count = number of values */
static double *read_doubles(const char *path, size_t *count) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  double *v = NULL;
  if (fseek(f, 0, SEEK_END) == 0) {
    long bytes = ftell(f);
    rewind(f);
    if (bytes > 0 && bytes % 8 == 0) {
      *count = (size_t)bytes / 8;
      v = (double *)malloc((size_t)bytes);
      if (v && fread(v, 8, *count, f) != *count) {
        free(v);
        v = NULL;
      }
    }
  }
  fclose(f);
  return v;
}

static double relerr(size_t m, const double *a, const double *ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < m; i++) {
    num += (a[i] - ref[i]) * (a[i] - ref[i]);
    den += ref[i] * ref[i];
  }
  return den > 0 ? sqrt(num / den) : sqrt(num);
}

/* Reads the file set inside a private directory with a converter wrapper (see above).
 * Returns NULL if H2Pack could not read it. */
static H2Pack_p read_h2pack(const char *meta, const char *aux, const char *bin, const char *converter,
                            const char *python) {
  char cwd[PATH_MAX], dir[PATH_MAX], script[PATH_MAX + 32], tmp[PATH_MAX + 32];
  char am[PATH_MAX], aa[PATH_MAX], ab[PATH_MAX], ac[PATH_MAX];
  if (!realpath(meta, am) || !realpath(aux, aa) || !realpath(bin, ab) || !realpath(converter, ac)) {
    fprintf(stderr, "h2pack_read_check: cannot resolve %s, %s, %s or %s: %s\n", meta, aux, bin, converter,
            strerror(errno));
    return NULL;
  }
  if (strchr(ac, '\'') || strchr(python, '\'')) {
    fprintf(stderr, "h2pack_read_check: quotes in the converter or interpreter path are not supported\n");
    return NULL;
  }
  const char *tdir = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
  snprintf(dir, sizeof dir, "%s/h2pack_read_check.XXXXXX", tdir);
  if (!getcwd(cwd, sizeof cwd) || !mkdtemp(dir)) {
    fprintf(stderr, "h2pack_read_check: cannot create a temporary directory in %s: %s\n", tdir, strerror(errno));
    return NULL;
  }
  snprintf(script, sizeof script, "%s/meta_json_to_txt.py", dir);
  snprintf(tmp, sizeof tmp, "%s/tmp_metadata.txt", dir);
  H2Pack_p h = NULL;
  FILE *f = fopen(script, "w");
  if (f) {
    fprintf(f, "#!/bin/sh\nexec '%s' '%s' \"$@\"\n", python, ac);
    fclose(f);
    if (chmod(script, 0700) == 0 && chdir(dir) == 0) {
      H2P_read_from_file(&h, am, aa, ab, 0, NULL, NULL, NULL, 0); /* AOT: B and D from the file */
      if (chdir(cwd) != 0) fprintf(stderr, "h2pack_read_check: cannot return to %s\n", cwd);
    }
  }
  unlink(script);
  unlink(tmp);
  rmdir(dir);
  return h;
}

int main(int argc, char **argv) {
  const char *meta = NULL, *aux = NULL, *bin = NULL, *xf = NULL, *yf = NULL, *conv = NULL, *py = "python3";
  int nrhs = 1, allow_unpatched = 0;
  double tol = 1e-12;
  for (int a = 1; a < argc; a++) {
    const char *k = argv[a], *v = a + 1 < argc ? argv[a + 1] : NULL;
    if (!strcmp(k, "--allow-unpatched")) {
      allow_unpatched = 1;
      continue;
    }
    if (!v) {
      usage();
      return 2;
    }
    a++;
    if (!strcmp(k, "--meta")) meta = v;
    else if (!strcmp(k, "--aux")) aux = v;
    else if (!strcmp(k, "--bin")) bin = v;
    else if (!strcmp(k, "--x")) xf = v;
    else if (!strcmp(k, "--y")) yf = v;
    else if (!strcmp(k, "--converter")) conv = v;
    else if (!strcmp(k, "--python")) py = v;
    else if (!strcmp(k, "--nrhs")) nrhs = atoi(v);
    else if (!strcmp(k, "--tol")) tol = atof(v);
    else {
      usage();
      return 2;
    }
  }
  if (!meta || !aux || !bin || !xf || !yf || !conv || nrhs < 1) {
    usage();
    return 2;
  }
  size_t nx = 0, ny = 0;
  double *X = read_doubles(xf, &nx), *Yref = read_doubles(yf, &ny);
  if (!X || !Yref || nx != ny || nx % (size_t)nrhs) {
    fprintf(stderr, "h2pack_read_check: cannot read %s and %s as two n x %d float64 arrays\n", xf, yf, nrhs);
    return 2;
  }
  const int n = (int)(nx / (size_t)nrhs);

  const int blas_threads = BLAS_GET_THREADS(); /* GOTCHA(h2pack-blas-threads) */
  H2Pack_p h = read_h2pack(meta, aux, bin, conv, py);
  if (!h) {
    fprintf(stderr, "h2pack_read_check: H2Pack could not read %s / %s / %s (see its messages above)\n", meta, aux,
            bin);
    return 3;
  }
  const int fixed = hmb_h2pack_reader_fix_calls();
  if (!fixed && !allow_unpatched) {
    fprintf(stderr, "h2pack_read_check: the reader fix did not run (libH2Pack linked dynamically, or no "
                    "-Wl,--wrap=H2P_calc_node_inadm_lists): H2Pack's reader may have corrupted the heap\n");
    return 3;
  }
  if (h->krnl_mat_size != n) {
    fprintf(stderr, "h2pack_read_check: the matrix has order %d, the vectors %d rows\n", h->krnl_mat_size, n);
    H2P_destroy(&h);
    return 2;
  }
  double *Y = (double *)malloc(sizeof(double) * nx), *y = (double *)malloc(sizeof(double) * (size_t)n);
  if (!Y || !y) return 2;
  H2P_matmul(h, CblasColMajor, nrhs, X, n, Y, n);
  H2P_matvec(h, X, y);
  const int blas_after = BLAS_GET_THREADS();
  if (blas_after != blas_threads) BLAS_SET_THREADS(blas_threads);

  const double e_mm = relerr(nx, Y, Yref), e_mv = relerr((size_t)n, y, Yref);
  printf("H2Pack read: n=%d nodes=%d levels=%d leaves=%d min_adm_level=%d admissible=%d inadmissible=%d %s; "
         "reader fix %s; BLAS threads %d%s\n",
         n, h->n_node, h->max_level + 1, h->n_leaf_node, h->is_HSS ? h->HSS_min_adm_level : h->min_adm_level,
         h->is_HSS ? h->HSS_n_r_adm_pair : h->n_r_adm_pair, h->is_HSS ? h->HSS_n_r_inadm_pair : h->n_r_inadm_pair,
         h->is_HSS ? "HSS" : "H2", fixed ? "applied" : "NOT applied", blas_threads,
         blas_after != blas_threads ? " (changed by H2Pack, restored)" : " (unchanged)");
  printf("H2P_matmul %d columns vs reference: %.3e <= %.3e %s\n", nrhs, e_mm, tol, e_mm <= tol ? "ok" : "FAILED");
  printf("H2P_matvec first column vs reference: %.3e <= %.3e %s\n", e_mv, tol, e_mv <= tol ? "ok" : "FAILED");
  H2P_destroy(&h);
  free(X);
  free(Yref);
  free(Y);
  free(y);
  return e_mm <= tol && e_mv <= tol ? 0 : 1;
}
