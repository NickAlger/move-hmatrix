# Gotchas

Library quirks that break transfers, found while building and testing these bridges. Sixteen
of them return wrong numbers with a success code; others crash, end the program with exit
status 0, fail without a message, or leak. Each one has a
short id; the code that deals with it carries a `GOTCHA(<id>)` comment, so
`grep -rn "GOTCHA(<id>)"` finds the handling, and each entry below names the tests that cover
it. Figures quoted from the authors' reports cite `results/notes/`.

Versions: Htool-DDM 0ee4cac, H2Lib 920662c, hmat-oss 45a2fdb, ButterflyPACK a8cf81b,
H2Opus cbc7985, STARS-H c115e75, H2Pack 2aa58a6 (see `deps/versions.txt`). A quirk may be
fixed in later versions; the tests will show it.

## Summary

| Id | Library | If ignored | Package |
|---|---|---|---|
| [index-base](#index-base) | hmat-oss, ButterflyPACK | wrong values with success code (91%) | handled, tested |
| [perm-direction](#perm-direction) | ButterflyPACK, H2Opus, STARS-H, hmat-oss | wrong ordering | handled, tested |
| [lowrank-convention](#lowrank-convention) | all | missing transpose passes symmetric tests | handled, tested |
| [bpack-transpose](#bpack-transpose) | ButterflyPACK | silently wrong unless A symmetric | handled, tested |
| [strict-tree](#strict-tree) | H2Lib | wrong H² (34-57%) or crash | refused / refined, tested |
| [symmetric-storage](#symmetric-storage) | Htool, hmat-oss, STARS-H, H2Opus, H2Pack | half the matrix missing; 144% error | mirrored / refused, tested |
| [scalar-type](#scalar-type) | hmat-oss, H2Lib, H2Opus, H2Pack, STARS-H | garbage with success code | refused, tested |
| [tolerance-stacking](#tolerance-stacking) | entry and matvec routes | missed tolerance (up to 56x) | measured, documented |
| [inverse-asymmetry](#inverse-asymmetry) | any approximate solve fed to H2Opus | 4.3-4.7x larger residual | symmetrized in the chain example |
| [htool-user-transpose](#htool-user-transpose) | Htool | 'T' product wrong (142%) | handled, tested |
| [htool-sym-inconsistent](#htool-sym-inconsistent) | Htool | product wrong (67%) | refused, tested |
| [hmat-epsilon](#hmat-epsilon) | hmat-oss | accuracy capped at 1e-4 | handled, tested |
| [factorized-state](#factorized-state) | hmat-oss | factors returned as entries | refused, tested |
| [starsh-get-block](#starsh-get-block) | STARS-H | another block's factors | handled, tested |
| [h2opus-tlr-diag-rank](#h2opus-tlr-diag-rank) | H2Opus | product wrong (2.8e-2) | handled, tested |
| [h2opus-tlr-gemv-multi](#h2opus-tlr-gemv-multi) | H2Opus | multi-vector product wrong (43-74%) | handled, tested |
| [fortran-stop](#fortran-stop) | ButterflyPACK | process ends with status 0 | diagnosed, status 70, tested |
| [bpack-errsol](#bpack-errsol) | ButterflyPACK | segfault in later matvecs | handled, tested |
| [h2pack-reader-bugs](#h2pack-reader-bugs) | H2Pack | heap over-read and corruption | link-time fix, tested |
| [h2pack-zero-rank](#h2pack-zero-rank) | H2Pack | BLAS illegal-value errors | avoided by the writer, tested |
| [h2pack-cwd-scripts](#h2pack-cwd-scripts) | H2Pack | reader silently fails outside its directory | handled, tested |
| [htool-partition](#htool-partition) | Htool | crash on a one-cluster tree | handled, tested |
| [htool-empty-cluster](#htool-empty-cluster) | Htool | dgemm with lda = 0 | handled, tested |
| [h2lib-pointer-amatrix](#h2lib-pointer-amatrix) | H2Lib | aliased blocks get wrong shape | handled, tested |
| [starsh-kernel-data](#starsh-kernel-data) | STARS-H | segfault in dfe | handled, tested |
| [starsh-init](#starsh-init) | STARS-H | NULL function pointer | handled |
| [starsh-dmml-omp](#starsh-dmml-omp) | STARS-H | buffer overflow beyond rank 100 | avoided |
| [gemv-mutates-input](#gemv-mutates-input) | hmat-oss | input vector permuted in place | handled, tested |
| [h2lib-omp-asserts](#h2lib-omp-asserts) | H2Lib | nested OpenMP switched on; abort on bad input | handled, tested |
| [bpack-global-state](#bpack-global-state) | ButterflyPACK | thread count and seed reset | restored, tested |
| [h2pack-blas-threads](#h2pack-blas-threads) | H2Pack | BLAS thread count reset | restored |
| [h2opus-workspace](#h2opus-workspace) | H2Opus | 203 MB kept for a 27 MB matrix | compacted, tested |
| [header-clash](#header-clash) | H2Lib, Htool, H2Opus | compile errors or redefinitions | separate translation units |
| [mpi-stub](#mpi-stub) | ButterflyPACK | clash with a real MPI | documented |
| [bpack-openblas-batch](#bpack-openblas-batch) | ButterflyPACK build | undefined `cblas_dgemm_batch` | build script |
| [starsh-codegen-cwd](#starsh-codegen-cwd) | STARS-H build | library silently without kernels | build script |
| [hmat-blas-threads](#hmat-blas-threads) | hmat-oss build | OpenBLAS thread count races | build script |
| [h2opus-thrust-backend](#h2opus-thrust-backend) | H2Opus build | wrong Thrust backend | documented |
| [hmat-node-leak](#hmat-node-leak) | hmat-oss | 16 bytes leaked per node | LSan suppression |
| [starsh-false-far-leak](#starsh-false-far-leak) | STARS-H | 104 bytes leaked per far block | LSan suppression |

## Silent wrong answers

### index-base
- **What happens.** hmat-oss's `get_values` takes 1-based indices while its assembly
  callbacks are 0-based. Without the +1 it returns rc 0 and the neighbouring entries (91%
  error); index 0 reads out of bounds. ButterflyPACK's permutation (new->old) and its entry
  callbacks are 1-based.
- **Package.** `hmatoss_entries` adds 1 and checks bounds; the ButterflyPACK adapter converts
  its permutation and entry indices and range-checks them.
- **Tests.** `test_hmatoss_entries` (raw call without +1: 91% error with rc 0; adapter
  9e-17), `test_bpack_entries` ("indices reach the source 0-based").

### perm-direction
- **What happens.** Libraries multiply and sample in their own permuted order, and their
  permutation arrays point different ways (new->old or old->new). ButterflyPACK and H2Opus
  sample in tree order; STARS-H's pivot and H2Opus's `index_map` are new->old, like the
  neutral form. A consistent permutation mistake still produces small errors in some
  tests; it shows up as inflated ranks.
- **Package.** Every BlackBox and EntrySource works in the original ordering; adapters
  permute on the way in and out.
- **Tests.** Accuracy with random and shuffled numberings; rank bounds calibrated to catch a
  consistent mistake (`test_bpack_matvec`: scrambled points give ranks 924 instead of
  172-210; `test_h2opus_hara`: 250 instead of 31, with a silent 6e-2 error).

### lowrank-convention
- **What happens.** Low-rank blocks are stored as U·V with V rank x cols (Htool-DDM), A·Bᴴ
  with B cols x rank (H2Lib), A·Bᵀ (hmat-oss, ButterflyPACK, STARS-H, H2Opus TLR). A missing
  transpose passes every test whose blocks are symmetric.
- **Package.** The neutral form takes both layouts (`vtrans` 0 or 1), so exporters alias
  instead of transposing; importers transpose only when the destination needs it.
- **Tests.** Random `vtrans` in hand-made views, non-symmetric kernels, and 'T' products in
  every adapter test.

### bpack-transpose
- **What happens.** ButterflyPACK's construction from matvecs asks for AᵀX in a third
  (HODLR) to a half (H) of its calls. Answering with AX is correct only for symmetric A, and
  the error is silent on symmetric test kernels.
- **Package.** 'T' requests go to `apply('T')`, or to `apply('N')` when the BlackBox is
  flagged symmetric.
- **Tests.** `test_bpack_matvec`, `test_bpack_htool`: count the 'T' calls reaching the source
  and check accuracy against a 32%-non-symmetric source.

### strict-tree
- **What happens.** H2Lib's H -> H² conversion needs a strict block tree: no dense leaf on a
  non-leaf cluster, and no super block that leaves a non-leaf cluster unsplit. On another
  tree it returns a wrong H² (34-57% error; 57% at N = 4,000 with leaf size 10) or crashes.
  Htool-DDM's trees break it where ACA fell back to a dense block above the leaf level
  (always with small leaves) and, with block-tree consistency off, through one-sided splits.
- **Package.** `hmb_h2lib_to_h2` refuses a non-strict tree (`HMB_H2LIB_ENOTSTRICT`) and
  points at the strict import mode, which refines the tree losslessly while importing.
  `hmb_strict_violations` counts the two kinds of violation on any neutral view.
- **Tests.** `test_h2lib`, `test_h2lib_htool`, `copy_htool_h2lib --leaf 10` and
  `--inconsistent`: refused first; after a strict import, zero violations, the same product,
  and H² within the tolerance.

### symmetric-storage
- **What happens.** Several libraries store one triangle: Htool-DDM ('S', and it never reads
  the other triangle of symmetric diagonal leaves), hmat-oss (lower or, after `transpose`,
  upper; `get_values` leaves the unstored triangle unwritten, 48.4% of the entries, with rc 0),
  STARS-H ('S'), H2Opus TLR (symmetric type), H2Pack (one block per pair). H2Opus's
  construction from matvecs builds a symmetric H² from AX samples only: a non-symmetric
  operator gives a silent 144% error (results/notes/matvec.md).
- **Package.** The neutral form always holds the full matrix: exporters mirror (aliasing
  low-rank factors with swapped roles, copying transposed dense blocks).
  `hmatoss_entries` serves the unstored triangle from the stored one.
  `h2opus_from_matvec` refuses a BlackBox not flagged symmetric unless the caller asserts
  `assume_symmetric`.
- **Tests.** `test_htool_copy` ([S], [S, upper], NaN in the unread triangles),
  `test_hmatoss_entries`, `test_hmatoss_copy`, `test_starsh`, `test_h2opus_tlr`,
  `test_h2opus_hara` (refusals).

### scalar-type
- **What happens.** hmat-oss picks the scalar type at run time and writes through `void*`:
  a float matrix read into a double buffer gives garbage with rc 0. H2Lib fixes one type per
  build (struct layouts depend on the `-D` flags). H2Opus, H2Pack and STARS-H's paths used
  here are real only.
- **Package.** Double precision only, enforced: hmat-oss interface and stream type checked,
  `_Static_assert` / `static_assert` for H2Lib and H2Opus builds, complex input refused by
  the H2Pack writer.
- **Tests.** `test_hmatoss_entries` ("scalar-type"), `test_h2pack_writer.py`.

### tolerance-stacking
- **What happens.** A rebuilt matrix carries both libraries' errors. On the entry route with
  equal tolerances the result is 1.5x less accurate than its source; making the source 100x
  tighter brings it to the accuracy of a direct build. On the matvec route, ButterflyPACK's
  construction stops early: HODLR lands at 0.6-7x the requested tolerance, H at a median 4x
  with a heavy tail (4 of 164 runs above 15x, worst 56x; results/notes/matvec.md) and 5-13x
  at N = 4,000-16,000. HARA reached 2-5x.
- **Package.** Measured, not hidden: the matvec adapters report `check_error` (the error
  against the source on a few random vectors, one extra source call) after every
  construction. Advice: entries route, destination tolerance 10-100x looser than the
  source's, or the source 10-100x tighter than the target; matvec route, ask 10x tighter
  than the target and check `check_error`.
- **Tests.** `entries_hmatoss_htool` (three self-checks), `matvec_htool_bpack` (tol vs
  tol/10), `check_error` agreement checks in the ButterflyPACK and H2Opus tests.

### inverse-asymmetry
- **What happens.** An approximate solve of a symmetric matrix is not symmetric. A HODLR
  rebuilt from matvecs is symmetric only up to its errors (6e-6 at tolerance 1e-6,
  N = 4,000), and inverting amplifies them: ‖(S − Sᵀ)x‖/‖Sx‖ = 1.9-2.1e-5 for
  ButterflyPACK's solve. HARA (symmetric only), told to assume symmetry, cannot fit the skew
  part: its H² ends up off by about the asymmetry, with a 4.3-4.7x larger residual (5.3e-5
  vs 1.1-1.2e-5) and 30% higher ranks (results/notes/matvec.md).
- **Package.** `bpack_solve_blackbox` is flagged non-symmetric.
  `examples/chain_four_libraries.cpp` samples (S + Sᵀ)/2 through the transposed HODLR
  solve; `--assume-symmetric` shows the difference.
- **Tests.** The chain example's checks (asymmetry within S's own error; HARA within 20 tol
  of (S + Sᵀ)/2).

### htool-user-transpose
- **What happens.** Htool-DDM's user-numbering products (`add_hmatrix_vector_product`,
  `add_hmatrix_matrix_product`) permute the input with the source tree and the output with
  the target tree whatever the operation, which is wrong for 'T' when the trees differ
  (rectangular or imported two-tree matrices): 1.42 relative error.
- **Package.** `blackbox_from_htool` (behind `htool_blackbox`) permutes itself and calls
  Htool's internal product.
- **Tests.** `test_htool_copy` (two-tree imports: the adapter's 'T' product at rounding
  level; Htool's own user-numbering product printed for comparison, 1.42 relative error).

### htool-sym-inconsistent
- **What happens.** With symmetric storage ('S') and block-tree consistency off, Htool-DDM's
  builder logs an error and builds anyway; its own products then omit the missing triangle
  (0.67 relative error at N = 2,000).
- **Package.** `htool_build` refuses the combination; `htool_export` refuses such a matrix.
- **Tests.** `test_htool_copy` ([S, inconsistent], built with Htool's own builder).

### hmat-epsilon
- **What happens.** hmat-oss truncates every compressed block at the matrix's own low-rank
  epsilon (default 1e-4, set by `set_low_rank_epsilon`), not at the compressor's: ACA+ at
  1e-10 gives 1.1e-4 error unless both are set.
- **Package.** `hmatoss_build` sets both.
- **Tests.** `test_hmatoss_entries` ("build: ||A - K||", 3.6e-7 at 1e-6).

### factorized-state
- **What happens.** After an LU, hmat-oss's `get_values` silently returns the factors (rc 0);
  after LLᵀ, `gemv` applies the factor. The state is visible only in the serialization
  stream, and serializing a hodlrsym-factorized matrix throws after printing a backtrace.
- **Package.** Entries, matvec and export refuse a factorized matrix.
- **Tests.** `test_hmatoss_entries` ("factorized").

### starsh-get-block
- **What happens.** STARS-H's `starsh_blrm_get_block` indexes factors by their position in
  a row-sorted list, not by the block's index, so it returns another block's factors unless
  the block list is sorted by block row (20 of 25 blocks wrong in a column-ordered list).
- **Package.** `hmb_starsh_import` sorts the block lists; `hmb_starsh_export` reads the lists
  directly and never calls `get_block`.
- **Tests.** `test_starsh` (every import; the column-ordered native list).

### h2opus-tlr-diag-rank
- **What happens.** For a non-symmetric TLR matrix, H2Opus's `construct_tlr_matrix` leaves a
  stale rank (and factors) on diagonal tiles, and `tlr_gemv` adds them: 2.8e-2 error.
- **Package.** `h2opus_tlr_from_entries` and `h2opus_tlr_wrap` clear the diagonal ranks; the
  import writes rank 0 there.
- **Tests.** `test_h2opus_tlr_native` (bug reproduced, then fixed), `test_h2opus_tlr`.

### h2opus-tlr-gemv-multi
- **What happens.** `tlr_gemv` with several vectors and more than one parallel column
  overlaps its per-column buffers: 0.43-0.74 relative error. Its reduction also assumes the
  leading dimension of Y is the padded size.
- **Package.** One parallel column when nrhs > 1; padded leading dimensions always.
- **Tests.** `test_h2opus_tlr_native`, and every multi-vector product in `test_h2opus_tlr`.

## Crashes, exits and memory errors

### fortran-stop
- **What happens.** When its randomized sampling does not converge, ButterflyPACK ends the
  whole program with a Fortran STOP, exit status 0, from inside the construction call.
- **Package.** While the adapter is inside ButterflyPACK (construction, factorization,
  product, solve), an `atexit` handler prints a diagnostic naming the options to change
  (rank0, rankrate, itermax), flushes Fortran output (`bpack_flush.f90`) and exits with
  status 70 (`BpackOptions::stop_exit_status`; 0 keeps ButterflyPACK's). A transposed solve
  with an H matrix also STOPs, so `bpack_solve_blackbox` refuses it. Direct ButterflyPACK
  calls outside the adapter keep status 0, so `tests/run_tests.sh` requires both status 0
  and the `HMB_TEST_PASSED` line.
- **Tests.** `test_bpack_stop` (three child processes), `test_bpack_solve`.

### bpack-errsol
- **What happens.** ButterflyPACK's H and BLR formats keep no copy of the forward operator
  unless `ErrSol = 1` (or `precon > 1`), and HODLR deletes it during factorization; a matvec
  afterwards segfaults.
- **Package.** `BpackOptions::keep_forward` (default true) sets ErrSol; it doubles the memory
  of H (21.4 vs 10.8 MB at N = 1,500, test_bpack_safety). `bpack_blackbox` throws instead of crashing on an H
  matrix built without it. `bpack_factor` records the factorization; with
  `keep_forward = false`, products of a factored HODLR throw `std::logic_error`.
- **Tests.** `test_bpack_safety`, `test_bpack_solve`; every ButterflyPACK test multiplies
  after construction.

### h2pack-reader-bugs
- **What happens.** `H2P_read_from_file` allocates one bounding box, then
  `H2P_calc_node_inadm_lists` reads one per node (heap over-read on every file with more
  than one node) and writes past `node_inadm_lists` from the garbage: heap corruption that
  aborts or not depending on thread count and output. The HSS matvec of a file-read matrix
  uses the corrupted counts.
- **Package.** `tools/h2pack/h2pack_reader_fix.c`, linked with
  `-Wl,--wrap=H2P_calc_node_inadm_lists`, replaces that call with zeroed lists and the
  counts the reader computed itself. Needs a static `libH2Pack.a` and a GNU linker. The
  writer emits H² only (no HSS).
- **Tests.** `test_h2pack_file` ("reader fix ran"); an ASan run against an instrumented H2Pack
  showed the overflow without the fix and none with it.

### h2pack-zero-rank
- **What happens.** A basis with zero columns makes H2Pack call DGEMV with lda = 0 (OpenBLAS
  prints "parameter number 6 had an illegal value"; reference BLAS stops).
- **Package.** The writer pads such bases to one zero column.
- **Tests.** `test_h2pack_file` ("no BLAS parameter errors"), `test_h2pack_writer.py`.

### h2pack-cwd-scripts
- **What happens.** `H2P_read_from_file` runs `./meta_json_to_txt.py` from the current
  directory through `system()`, writes a fixed `tmp_metadata.txt` there, and returns silently
  without a matrix if the script is missing.
- **Package.** `h2pack_read_check` reads from a private temporary directory with a wrapper
  that runs H2Pack's converter with `$(PYTHON)`.
- **Tests.** Every read in `test_h2pack_file`.

### htool-partition
- **What happens.** Htool-DDM's `user_to_cluster` / `cluster_to_user` read the first
  partition cluster without checking; a root without sons has none and crashes.
- **Package.** The importer adds a partition cluster under a leaf root.
- **Tests.** `test_htool_copy` ([leaf root]).

### htool-empty-cluster
- **What happens.** For a block with no rows, Htool-DDM's products call dgemm with lda = 0,
  an illegal argument (reference BLAS prints an error and skips the call; OpenBLAS accepts).
- **Package.** The importer drops empty clusters and their blocks (the matrix is unchanged).
- **Tests.** `test_htool_copy` ([1-son, empty]).

### h2lib-pointer-amatrix
- **What happens.** H2Lib's `init_pointer_amatrix` (and `new_pointer_amatrix`) sets
  cols = rows and ld = rows, so an aliased rectangular or padded block gets the wrong shape.
- **Package.** The borrow import fixes cols and ld after the call.
- **Tests.** `test_h2lib` (borrow imports of rectangular, padded leaves; pointer checks).

### starsh-kernel-data
- **What happens.** STARS-H passes kernels the problem's `row_data` in some routines
  (`problem_get_block`, `blrf_get_block`) and the cluster's `data` in others (approximation,
  `dmml`, `dca`, `dfe`); a kernel that trusts one of them segfaults in the other.
- **Package.** The adapter sets both.
- **Tests.** `test_starsh` (`dfe` and kernel checks on five views).

### starsh-init
- **What happens.** `starsh_blrm_approximate` is a NULL function pointer and the RSVD
  oversampling is -1 until `starsh_init()` runs (which also prints three lines to stderr).
- **Package.** `hmb_starsh_from_entries` calls it once.

### starsh-dmml-omp
- **What happens.** STARS-H's OpenMP matvec `starsh_blrm__dmml_omp` sizes per-thread buffers
  for rank 100 and `nrows/nbrows` near-field rows: buffer overflow beyond those (found by
  reading the code).
- **Package.** Always uses the sequential `starsh_blrm__dmml`.

### gemv-mutates-input
- **What happens.** hmat-oss's `gemv` permutes the input vector in place and restores it
  afterwards: read-only input crashes, and other threads see it permuted during the call.
- **Package.** `hmatoss_blackbox` copies X first.
- **Tests.** `test_hmatoss_entries` ("blackbox: input X left untouched").

### h2lib-omp-asserts
- **What happens.** `init_h2lib()` turns on nested OpenMP for the whole process; H2Lib aborts
  on failed asserts, including on bad input.
- **Package.** `hmb_h2lib_init` restores the max-active-levels setting; every adapter entry
  point validates its input before calling H2Lib.
- **Tests.** `test_h2lib` ("init keeps the OpenMP nesting"; error cases never reach an assert).

## Global state

### bpack-global-state
- **What happens.** ButterflyPACK's `construct_init` sets the OpenMP thread count to
  `$OMP_NUM_THREADS` (1 when unset) and reseeds its random generator from the clock and the
  process id, so results vary between runs and cannot be made reproducible.
- **Package.** A guard saves and restores the thread count around every ButterflyPACK call.
- **Tests.** `test_bpack_safety` (raw call changes 3 threads to 2; the adapter keeps 3).

### h2pack-blas-threads
- **What happens.** H2Pack sets the OpenBLAS thread count in its construction routines.
- **Package.** `h2pack_read_check` saves and restores it and reports a change; the build
  needs OpenBLAS.

### h2opus-workspace
- **What happens.** After HARA, the H2Opus handle keeps HARA's workspace and the H² arrays keep
  spare capacity: 203 MB retained for a 27 MB matrix at N = 4,000.
- **Package.** `h2opus_from_matvec` copies the result into a fresh matrix and handle (26.7 MB).
- **Tests.** `test_h2opus_hara` ("retained memory close to H2Opus's count").

Not a library bug, but measured: HARA runs many short OpenMP regions and slowed 6-17x on a
busy machine with spinning threads. `tests/run_tests.sh` sets `OMP_WAIT_POLICY=passive`.

## Build and link

### header-clash
- **What happens.** H2Lib's headers define the macros `I` and `UINT_MAX` and a type named
  `real`; Htool-DDM and H2Opus declare the Fortran BLAS prototypes with different `const`;
  H2Lib compiles without `-fPIC`.
- **Package.** Each library lives in its own translation units; neutral-facing headers
  (`hmb_*.h*`) never include a library header, so any combination compiles together
  (`LAYOUT.md`). libh2.a is linked statically into executables.

### mpi-stub
- **What happens.** The sequential ButterflyPACK build links its own MPI stub, which defines
  the Fortran MPI symbols; linking a real MPI into the same program conflicts.
- **Package.** Sequential use only; documented in `hmb_bpack.hpp`. The build script refuses
  MPI compiler wrappers and keeps MPI out of every build.

### bpack-openblas-batch
- **What happens.** ButterflyPACK's CMake adds `-DHAVE_OPENBLAS_BATCH` whenever the BLAS name
  contains "openblas", but OpenBLAS 0.3.26 has no `cblas_dgemm_batch`: every program linking
  ButterflyPACK fails with an undefined reference.
- **Package.** `deps/build_deps.sh` probes for the symbol and adds `-UHAVE_OPENBLAS_BATCH`.

### starsh-codegen-cwd
- **What happens.** STARS-H's configure step generates its kernel sources with
  `python ../misc_scripts/...`, relative to where cmake runs. Run elsewhere, configure
  succeeds and `libstarsh.a` silently lacks the kernels. It also needs a `python` command.
- **Package.** `deps/build_deps.sh` runs cmake from `<source>/src` with `python` pointing to
  python3, and checks with `nm` that the kernels are present.

### hmat-blas-threads
- **What happens.** Configured against OpenBLAS, hmat-oss changes OpenBLAS's global thread
  count inside most of its calls (and restores it to the processor count, not the previous
  value), which races when several threads call it.
- **Package.** `deps/build_deps.sh` compiles it out (`HAVE_OPENBLAS_SET_NUM_THREADS=OFF`,
  `HAVE_GOTO_GET_NUM_PROCS=OFF`), as the study's build had it. The thread-safety of
  `hmatoss_entries` also relies on hmat-oss being built without `HMAT_SCALAR_ARRAY_ORTHO`
  (its CMake never sets it).

### h2opus-thrust-backend
- **What happens.** H2Opus's headers choose Thrust's backend from `_OPENMP`, so every file
  that includes them must be compiled with `-fopenmp`, or objects disagree.
- **Package.** The package always compiles with `$(OPENMP)`; documented in the generated
  `config.mk`.

## Leaks inside the libraries

Suppressed for LeakSanitizer in `tests/lsan.supp`, which `tests/run_tests.sh` uses by default.

### hmat-node-leak
hmat-oss 45a2fdb leaks 16 bytes per block-tree node (an `FPCompressionSettings` allocated in
`LocalSettings`, `h_matrix.hpp:85`, never freed by `~HMatrix`), for built and for deserialized
matrices: about 31 KB per N = 2,000 matrix.

### starsh-false-far-leak
STARS-H's `starsh_blrm__drsdd` leaks an `Array` header (104 bytes) per far-field block
whenever a tile needs more than maxrank. Needs `ASAN_OPTIONS=fast_unwind_on_malloc=0`
(libstarsh.a has no frame pointers).
