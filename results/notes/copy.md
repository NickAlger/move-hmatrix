# Agent B (copy): neutral-core tests, Htool-DDM block copy, H2Lib adapter, copy example

(Report of the agent that wrote the block-copy route; saved by the integrator. Line numbers
are as of the report and may have moved since.)

**Status.** All four deliverables are done, plus the optional N = 20,000 run (with `nice`).
- From a clean `build-copy`, the 5 checks pass (526 individual checks after the second round: test_htool_copy 123, test_h2lib 214, with lifetime checks added).
- Everything compiles with -Wall and no warnings.
- Each check also ran under ASan/LSan with no report, after the last change to the code it covers.
- Section 7 asked for six changes to files the agent did not own, as one patch (applied by the integrator).

## 1. Files

Owned files modified:
- `src/neutral.c`: fixes for bugs the new tests found.
  - The cluster-tree walk overflowed its stack array on a cluster DAG (heap-buffer-overflow under ASan). Clusters are now marked when pushed.
  - hmb_validate rejected a split over a cluster with a single son, and hmb_strict_violations miscounted it. H2Lib's regular clustering makes such clusters, so H2Lib's own export failed validation.
  - `dim` outside 0..3 was accepted, but bmin/bmax hold only 3 values.
  - hmb_entries_ip checked neither the request indices nor missing inverse permutations. It now prints a message and calls abort(), because the API has no error return.
  - No pointer arithmetic before the start of a dense block.
  - An empty request allocates nothing.
- `src/cstore.c`: growth is computed in size_t and capped at INT_MAX. Negative counts are refused. n = 0 never passes NULL to memset/memcpy. Running out of memory is reported.
- `mk/10-core.mk`: `CORE_TESTS := $(BIN)/test_neutral`.

Created:
- `tests/test_neutral.cpp`: tests the neutral core against a dense reference, on random views with NaN padding.
- `adapters/htool/hmb_htool_copy.hpp`: neutral-facing API for the Htool-DDM block copy (export and import), with its conventions and refusals.
- `adapters/htool/htool_export.hpp`: native export of an unwrapped `htool::HMatrix<double>`.
- `adapters/htool/htool_import.hpp`: native import of one side's cluster tree as an `htool::Cluster<double>`.
- `adapters/htool/hmb_htool_copy.cpp`: the exporter (walks Htool's block tree, mirroring where needed) and the importer (builds Htool trees and the HMatrix, injects the factors).
- `mk/21-htool-copy.mk`: `HTOOL_COPY_OBJS` and test_htool_copy.
- `tests/test_htool_copy.cpp`: export and import checked against Htool's own products.
- `adapters/h2lib/hmb_h2lib.h`: neutral-facing C API. It forward-declares H2Lib's structs instead of including H2Lib.
- `adapters/h2lib/hmb_h2lib.c`: the package's only H2Lib translation unit: init, import (copy, borrow, strict), export, strictness, H -> H2, products, sizes and norms.
- `adapters/h2lib/hmb_h2lib.hpp`: C++ conveniences (shared ownership, exceptions, BlackBox).
- `mk/20-h2lib.mk`: `H2LIB_OBJS` and test_h2lib.
- `tests/test_h2lib.cpp`, plus `tests/test_h2lib_native.c` (H2Lib-side helpers, in C).
- `tests/test_h2lib_htool.cpp`: the Htool -> H2Lib route, registered in mk/50-copy.mk.
- `examples/copy_htool_h2lib.cpp`: port of E1's driver.
- `mk/50-copy.mk`: registers the example and test_h2lib_htool.

## 2. Public API

**Htool-DDM block copy**, `hmb_htool_copy.hpp` (namespace hmb; errors throw `std::invalid_argument("hmb::htool_...: why")`):
- `std::shared_ptr<const hmb_hmatrix> htool_export(const HtoolHandle &A, HtoolExportInfo *info = nullptr)`
  - Returns the full matrix as a view (the shared_ptr's control block owns the tree arrays, the copies and the handle); aliases Htool's dense blocks, U, V (vtrans = 0) and permutations, so symmetry 'N' is zero-copy.
  - Mirrors symmetric storage, copying only what must be transposed. The store owns the copies and keeps A alive.
- `HtoolHandle htool_import(const hmb_hmatrix &A, const HtoolImportOptions &opt = {}, HtoolImportInfo *info = nullptr)`
  - Returns a new Htool matrix (symmetry 'N') with exactly the view's block tree and copies of all leaves. No recompression.
  - Accepts any tree shape, with one shared cluster tree or two.
- `struct HtoolExportInfo { long long aliased_values, copied_values; int mirrored_leaves, added_blocks; }`
- `struct HtoolImportOptions { double epsilon = 1e-6; double eta = 10; }`: only recorded, for Htool's later arithmetic.
- `struct HtoolImportInfo { long long copied_values, transposed_values; }`
- Native, `htool_export.hpp`: `void htool_export(const htool::HMatrix<double> &A, NeutralStore &out, HtoolExportInfo *info = nullptr)`. A and its trees must outlive `out`.
- Native, `htool_import.hpp`: `std::unique_ptr<htool::Cluster<double>> htool_import_cluster_tree(const hmb_hmatrix &A, int side)`.

**H2Lib**, `hmb_h2lib.h` (C). Functions return 0, or HMB_H2LIB_EINVAL / ENOTSTRICT / EUNSUPPORTED with a message in `msg`.
- `int hmb_h2lib_init(int pardepth)`: calls init_h2lib() once, then restores the OpenMP max-active-levels setting that init_h2lib() changed; sets H2Lib's max_pardepth (negative keeps H2Lib's choice).
- `int hmb_h2lib_import(const hmb_hmatrix *A, int mode, hmb_h2lib_hm **out, hmb_h2lib_import_info *info, char *msg, size_t msglen)`
  - HMB_H2LIB_COPY (0) copies every leaf; HMB_H2LIB_BORROW (1) aliases dense blocks and U, and V when vtrans = 1; HMB_H2LIB_STRICT (2) refines to a strict tree, losslessly; BORROW | STRICT is allowed.
  - `info` = {copied, borrowed, index_ints, dense_leaves, lowrank_leaves}.
- `struct _hmatrix *hmb_h2lib_hmatrix(const hmb_h2lib_hm *M)` and `void hmb_h2lib_free(hmb_h2lib_hm *M)`.
- `int hmb_h2lib_export(const struct _hmatrix *A, hmb_cstore *out, char *msg, size_t msglen)`: zero-copy view; U = A, V = B, vtrans = 1; permutations are the root clusters' idx.
- `void hmb_h2lib_strict_violations(const struct _hmatrix *A, long v[2])`.
- `int hmb_h2lib_to_h2(const struct _hmatrix *A, double eps, struct _h2matrix **out, hmb_h2lib_h2_info *info, char *msg, size_t msglen)`: H -> H2 (row basis, column basis, projection; blockwise relative Euclidean truncation); refuses a non-strict tree with ENOTSTRICT.
- `void hmb_h2lib_free_h2(struct _h2matrix *A)`.
- `hmb_h2lib_matvec` / `hmb_h2lib_h2_matvec(..., char trans, int nrhs, const double *X, int ldx, double *Y, int ldy)`: Y = op(A) X in the ORIGINAL ordering.
- `hmb_h2lib_size`, `hmb_h2lib_h2_size`, `size_t hmb_h2lib_bytes(const struct _hmatrix *A)`.
- `double hmb_h2lib_h2_reldiff(const struct _hmatrix *A, const struct _h2matrix *B)`: ||A - B||_2 / ||A||_2 by power iteration.

**H2Lib, C++**, `hmb_h2lib.hpp`: `H2libHandle h2lib_import(const hmb_hmatrix &A, int mode = HMB_H2LIB_COPY, hmb_h2lib_import_info *info = nullptr, std::shared_ptr<const void> keepalive = nullptr)`, `std::shared_ptr<const hmb_hmatrix> h2lib_export(const H2libHandle &M)`, `H2libH2Handle h2lib_to_h2(const H2libHandle &M, double eps, hmb_h2lib_h2_info *info = nullptr)`, `BlackBox h2lib_blackbox(const H2libHandle &M)`, `BlackBox h2lib_h2_blackbox(const H2libH2Handle &H)`; `H2libHandle = std::shared_ptr<hmb_h2lib_hm>`, `H2libH2Handle = std::shared_ptr<struct _h2matrix>` (keeps its H-matrix alive). Each deleter releases its sources right after freeing its H2Lib object (a later fix: they used to live as long as any weak_ptr).

## 3. Checks

`rm -rf build-copy && make B=build-copy -j2 && make B=build-copy check HMB_TEST_FILTER='^(test_neutral|test_htool_copy|test_h2lib|test_h2lib_htool|copy_htool_h2lib)$'`
(run_tests.sh sets OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=1)

```
check                                        result  seconds  detail
test_neutral                                 PASS        0.0  test_neutral: 136 checks; n=240
test_h2lib                                   PASS        1.8  test_h2lib: 211 checks; n=2000
test_htool_copy                              PASS        0.6  test_htool_copy: 121 checks; n=2000
test_h2lib_htool                             PASS        0.8  test_h2lib_htool: 41 checks; n=2000
copy_htool_h2lib                             PASS        1.6  copy_htool_h2lib: 12 checks; n=4000: copy 1.6-3.7, H->H2 72 matvecs
```

Thresholds: lossless transfers 512 eps = 1.137e-13 (relative 2-norm difference of op(A) X over
1-3 random right-hand sides); accuracy against the kernel and of the H2 conversion 10 x tol.
Worst measured: rounding level 3.3e-16 (test_neutral) to 1.05e-15 (example); tolerance level 8.3e-7.

- **test_neutral** (136 checks): hmb_validate rejects 36 separately injected defects, each with
  a reason (permutations, sons, splits inconsistent with the trees, children reached twice,
  unreachable blocks, leading dimensions, NULL factors, vtrans, rank, kind, roots, dim, a
  cluster DAG); accepts rank-0 leaves with NULL factors, one-sided and single-son splits, 0 x 0
  and 1 x 1 matrices, nrhs = 0. On four random views (strict; 3-way with one-sided splits;
  rectangular 280 x 190 with separate trees and permutations; single sons and empty clusters),
  all with random vtrans, ranks 0-4 and NaN in every padding: exact tiling, strict counts
  matching an independent count, hmb_matvec N/T (nrhs 1 and 3, padded ldx/ldy) within 3.3e-16
  of dense with Y's padding untouched, hmb_entries exact with duplicated unsorted requests,
  cstore/NeutralStore copies identical, the neutral_bridge wrappers exact from 2 threads.
- **test_htool_copy** (121 checks, N = 2,000): Htool matrices [N], [N, inconsistent], [N, leaf 10],
  [S], [S, leaf 10], [S, upper], [skipped level], [S, inconsistent]. Exports validate; full storage
  is zero-copy (2,267,798 values aliased, 0 copied); symmetric storage mirrored ([S]: 454 leaves,
  936,052 of 2,267,798 values copied) and the view is symmetric; view products match Htool within
  4.5e-16; unread triangles of symmetric diagonal leaves set to NaN are ignored.
  import(export(A)) multiplies like A within 4e-16 and re-exports the same tree. [S, inconsistent]
  is refused (Htool's own product is 6.73e-1 off the kernel). Hand-made views with unusual shapes
  import exactly.
- **test_h2lib** (211 checks, N = 2,000): H2Lib-built matrices (adaptive, regular, planar with 43
  single-son clusters, non-strict with 352 dense leaves on non-leaf clusters) export zero-copy and
  validate; re-import in copy, borrow and strict modes reproduces the products at rounding level,
  with exact copy counts and pointer-checked aliasing. H -> H2 at 1e-6: 4.7-5.9e-7 (spectral
  4.8-5.6e-8); refused on the non-strict tree; accurate after a strict re-import. Hand-made views
  in all four modes: products within 7.3e-16; conversion refused before the strict import
  (violations (48, 20), (16, 6), (18, 18)) and successful after it. Init keeps the OpenMP nesting.
- **test_h2lib_htool** (41 checks): Htool [strict], [leaf 10] (strict violations (1,218, 0)) and
  [inconsistent] ((0, 3)) into H2Lib in copy, borrow and strict modes; products within 7.8e-16;
  borrow copies only Htool's V (520,616 of 2,267,798 values); H -> H2 refused on both non-strict
  trees; after strict import, H2 is 2.5-3.1e-7 off Htool and 4.9-5.5e-7 off the kernel.
- **copy_htool_h2lib** (12 checks, N = 4,000): export zero-copy; imports 1.05e-15 off Htool and
  7.35e-7 off the kernel; H2 3.79e-7 off Htool, 8.34e-7 off the kernel; with --leaf 10 or
  --inconsistent the conversion of the copy import is refused.

## 4. Measured costs

Setup as in E1: kernel 1/(0.1 + |x - y|) + 10 [x = y] on N random points of the unit sphere;
Htool-DDM 0ee4cac (ACA 1e-6, eta 10, leaf 64); H2Lib 920662c; H -> H2 truncation 1e-6. One
thread for everything timed (Htool sequential build and product, H2Lib pardepth 0,
OPENBLAS_NUM_THREADS=1). Unit: one sequential Htool matvec (median of 7, averaged over the start
and end of the run). Machine: 2-core VM shared with four other agents, load 2.7-5.

| Run | source matvec | export | copy import | borrow | strict | H -> H2 | lossless error | H2 vs Htool / kernel |
|---|---|---|---|---|---|---|---|---|
| N = 4,000, final run (load 2.7) | 3.28 ms | 0.06 | 3.70 | 1.62 | 3.20 | 72.3 | 1.05e-15 | 3.79e-7 / 8.34e-7 |
| N = 4,000, range over 7 runs | 3.2-7.7 ms | 0.04-0.13 | 2.5-4.4 | 1.1-2.4 | 2.5-6.5 | 46-99 | same | same |
| E1 main_4000 | 3.27 ms | 0.04 | 3.6 | 1.25 | 3.9 | 73.5 | 1.01e-15 | 4.21e-7 / - |
| N = 4,000, --leaf 10 | 5.02 ms | 0.71 | 4.05 | 3.35 | 6.77 | 116 (strict import) | 1.09e-15 | 4.44e-7 / 9.13e-7 |
| E1 leaf10_4000 | 5.12 ms | 0.64 | 4.2 | 2.3 | 5.8 | 108 (strict import) | 1.01e-15 | 4.94e-7 / - |
| N = 4,000, --inconsistent | 3.02 ms | 0.06 | 4.26 | 1.63 | 3.32 | 74.2 (strict import) | 1.05e-15 | 3.79e-7 / 8.34e-7 |
| E1 inconsistent_strict_4000 | 3.20 ms | 0.05 | 3.6 | 1.34 | 3.33 | 77.0 | 1.01e-15 | 4.21e-7 / - |
| N = 20,000, one run with nice (*) | 59.9 ms | 0.11 | 2.62 | 2.43 | 4.82 | 63.0 | 1.06e-15 | 3.34e-7 / 8.10e-7 |
| E1 main_20000 | 40.6 ms | 0.09 | 2.30 | 1.16 | 2.52 | 64.4 | 1.41e-15 | 5.35e-7 / - |

Costs except the "source matvec" column are in units of one source matvec. Minimum over the 7
N = 4,000 runs: copy 3.8, borrow 1.6, strict 3.3, H -> H2 73 (E1: 3.6, 1.25, 3.9, 73.5).
(*) The N = 20,000 run predates the last borrow-import optimization. With --leaf 10, H -> H2 on the
copy import is refused (2,356 dense leaves on non-leaf clusters); E1 converted that tree and got a
5.74e-1 error.

Side by side with E1's own code (same process, same matrix, medians of 5-7):

| Step | N = 4,000, package | N = 4,000, prototype | N = 20,000, package | N = 20,000, prototype |
|---|---|---|---|---|
| export | 0.2 ms | 0.2 ms | 6.7 ms | 7.1 ms |
| import, copy | 14.8-15.1 ms | 14.2-15.0 ms | 148 ms | 242 ms |
| import, borrow | 2.2-4.1 ms | 1.9-6.6 ms | 68 ms | 76 ms |
| import, strict | 13.9-17.5 ms | 13.2-17.0 ms | 166 ms | 201 ms |

Memory and copies (deterministic, identical to E1's logs, including H2 bytes and rank sums):

| N | Htool stored values | H2Lib copy import | H2Lib borrow import | H2 | process peak RSS |
|---|---|---|---|---|---|
| 4,000 | 5,597,044 (0.35 N^2) | 45.0 MB = 1.006 x Htool's leaf bytes | 13.8 MB = 0.308 x; 1,688,770 values copied (Htool's V transposed) | 26.2 MB = 0.584 x (near 17.8, far 8.2 MB); rank sums 6,697, max rank 31 | 139 MB |
| 20,000 | 39,336,068 (0.10 N^2) | 316.7 MB = 1.006 x | 131.8 MB = 0.419 x; 16,225,170 values copied | 114.7 MB = 0.364 x; rank sums 45,808, max rank 34 | 923 MB |

## 5. Gotchas

- **lowrank-convention**: Htool U V (V rank x cols) vs H2Lib A B^H (B cols x rank). Htool export
  aliases V with vtrans 0; a mirrored leaf uses U as its V with vtrans 1; Htool import transposes a
  vtrans 1 V; H2Lib import `copy_colfactor`; H2Lib export vtrans 1. Covered by random-vtrans views,
  a nonsymmetric kernel and 'T' checks.
- **symmetric-storage**: Htool 'S' stores one triangle of blocks and multiplies symmetric diagonal
  leaves with symv/symm, never reading their other triangle. Export mirrors and rebuilds diagonal
  leaves from the UPLO triangle. Tests [S], [S, leaf 10], [S, upper], NaN in unread triangles.
- **strict-tree**: refusal in `hmb_h2lib_to_h2`; lossless refinement in the strict import.
- **header-clash**: only `hmb_h2lib.c` and `tests/test_h2lib_native.c` include H2Lib;
  `hmb_h2lib.h` forward-declares `struct _hmatrix` and `struct _h2matrix`.
- **scalar-type**: `_Static_assert` for real double in `hmb_h2lib.c`.
- **h2lib-omp-asserts**: `hmb_h2lib_init` restores max-active-levels; inputs are validated before
  any H2Lib call so no H2Lib assert can fire on bad input.

New quirks (ids marked in the code):
- **htool-sym-inconsistent**: with 'S' and block_tree_consistency off, Htool logs an error and
  builds anyway; its products then omit the missing triangle (6.73e-1 off at N = 2,000). The export
  refuses such matrices, and `htool_build` refuses the combination.
- **htool-user-transpose**: Htool's user-numbering products permute the input with the source tree
  and the output with the target tree also for 'T'; wrong when the trees differ (imported two-tree
  or rectangular matrices). `blackbox_from_htool` now permutes itself.
- **htool-partition**: Htool's user_to_cluster / cluster_to_user read the first partition cluster
  without checking; a root without sons crashes. The importer adds a partition cluster under a leaf root.
- **htool-empty-cluster**: for a block with no rows, Htool's products pass lda = 0 to dgemm
  (reference BLAS prints an error and skips; OpenBLAS accepts). The importer drops empty clusters
  and their blocks.
- **h2lib-pointer-amatrix**: H2Lib's init_pointer_amatrix sets cols = rows and ld = rows;
  `alias_amatrix` fixes cols and ld after the call.

Handled without an id: Htool children that skip a cluster level (the export adds intermediate
blocks); H2Lib leaves without data (exported as rank 0); single-son clusters from H2Lib's regular
clustering (neutral.c accepts single-son splits).

## 6. Limitations

- Htool export refuses MPI-local block rows, symmetric storage over two trees, 'S' built with
  block_tree_consistency off, blocks straddling the diagonal under 'S', and leaves without data.
- Htool import always produces full storage ('N'); empty clusters and their blocks are dropped;
  cluster balls come from the view's bounding boxes; only Htool's sequential build was tested.
- H2Lib: real double builds only; borrowed imports are read-only (never use H2Lib's in-place
  routines on them); no H2 export (the neutral form has no nested bases); only E1's blockreleucl
  truncation is exposed; `hmb_h2lib_matvec` loops over right-hand sides; `hmb_h2lib_init` is
  process-global; export needs sons' index arrays consecutive inside their father's (H2Lib's own
  clustering guarantees it; otherwise EUNSUPPORTED).
- hmb_entries / hmb_entries_ip abort on out-of-range indices (no error return in the API).

## 7. Requested changes (applied by the integrator)

1. `include/hmb/neutral.h`: document single-son splits, the dim range and the abort behaviour.
2. `include/hmb/neutral_bridge.hpp`: `entries_from_neutral` computes the inverse permutations once
   (33 us vs 0.8 us per single-entry request at N = 20,000).
3. `include/hmb/neutral_store.hpp`: delete NeutralStore's copy operations (a copy's view would point
   into the original); keep moves.
4. `adapters/htool/htool_native.hpp`: `blackbox_from_htool` permutes itself (htool-user-transpose).
5. `adapters/htool/hmb_htool.cpp`: `htool_build` refuses 'S' with block_tree_consistency off.
6. `adapters/htool/hmb_htool.hpp`: correct the block_tree_consistency comment.
