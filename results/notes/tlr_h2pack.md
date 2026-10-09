# Agent E: STARS-H and H2Opus TLR adapters, H2Pack file bridge

All six checks pass from a clean `build-tlr`, with one thread, and under AddressSanitizer. The
STARS-H to H2Opus TLR block copy is lossless: the two libraries' products differ by 7.7e-17.
On a quiet machine it costs 8-11 STARS-H matvecs at N = 4,000 and under 3 at N = 16,000,
against 360-810 for H2Opus rebuilding the matrix from the kernel's entries.

I found five library defects, beyond the eight quirks in §5:

- **H2Pack's file reader corrupts the heap** on every file with more than one node. A program
  that reads H2Pack files must link `tools/h2pack/h2pack_reader_fix.o` with
  `-Wl,--wrap=H2P_calc_node_inadm_lists`, and needs a static `libH2Pack.a` for it.
- **H2Opus `tlr_gemv` is wrong for more than one vector** when it uses more than one parallel
  column. The adapter calls it with one parallel column in that case.
- **H2Opus `construct_tlr_matrix` leaves stale ranks on the diagonal tiles** of a
  non-symmetric matrix, and `tlr_gemv` adds them to the product. The adapter clears them.
- **STARS-H `starsh_blrm_get_block` returns another block's factors** unless the block list is
  sorted by block row. The import sorts it; the export does not call `get_block`.
- **STARS-H's approximation leaks** about 104 bytes per far-field block. It cannot be fixed from
  outside, so it is suppressed for LeakSanitizer.

## 0. Changes in this revision (the integrator's follow-up)

- **API harmonization.**
  - The H2Opus TLR types are now `H2opusTlrHandle`, `H2opusTlrMatrix`, `H2opusTlrStats`,
    `H2opusTlrOptions` and `H2opusTlrImportOptions`, and the native alias is `H2opusTlr`.
    H2Opus's own names (`H2OpusTLRTile`, `H2OpusTLR_Symmetric`, ...) are unchanged.
  - `h2opus_tlr_export` returns `std::shared_ptr<const hmb_hmatrix>`, and
    `h2opus_tlr_import` takes `const hmb_hmatrix &`.
  - New C++ header `adapters/starsh/hmb_starsh.hpp`: `StarshHandle`, `starsh_import`,
    `starsh_from_entries`, `starsh_export`, `starsh_blackbox`. It needed one new C function,
    `hmb_starsh_symm`.
- **Exceptions from an EntrySource.** `starsh_from_entries` and `h2opus_tlr_from_entries` call
  the source inside the library's OpenMP loops, where an escaping exception calls
  `std::terminate` (and would cross STARS-H's C code). Both now keep the first exception, read
  the remaining entries as 0 without calling the source again, and rethrow it as
  `std::runtime_error` once the construction returns. Both behaviors are tested.
- **Example.** `copy_tlr_starsh_h2opus` uses `hmb_starsh.hpp`. Its A^T check now goes through
  `starsh_blackbox`, and it has 11 checks (the two "build succeeded" checks are gone).
- **Tests.** New `tests/test_starsh_cpp.cpp` (25 checks), registered in `mk/20-starsh.mk`.
  `test_h2opus_tlr` has two new checks (34).
- **Costs re-measured** on the now quiet machine (§4). The earlier figures were taken at a load
  of 2-6 on 2 cores and were up to twice as high.
- **Done by the integrator** from my earlier requests (§7): the `GOTCHAS.md` entries,
  `bisection_perm` in `tests/common/hmb_test.hpp`, `tests/lsan.supp` (my
  `tests/test_starsh_lsan.supp` merged into it, applied by `tests/run_tests.sh`), and the
  Makefile's `TOOLS` (now used by `mk/20-h2pack.mk`).

## 1. Files (under /root/hmatrix-bridges)

| File | Purpose |
|---|---|
| `adapters/starsh/hmb_starsh.h` | Neutral-facing C API of the STARS-H adapter |
| `adapters/starsh/hmb_starsh.c` | Its implementation; the only file that includes `starsh.h` |
| `adapters/starsh/hmb_starsh.hpp` | C++ conveniences over the C API (handle, exceptions, EntrySource, BlackBox); includes no STARS-H header |
| `adapters/h2opus/hmb_h2opus_tlr.hpp` | Neutral-facing C++ API of the H2Opus TLR adapter |
| `adapters/h2opus/h2opus_tlr_native.hpp` | Native side: `TTLR_Matrix` types and `h2opus_tlr_wrap` |
| `adapters/h2opus/h2opus_tlr.cpp` | Its implementation |
| `tools/h2pack/h2pack_write.py` | Python module and CLI: writes H² arrays in H2Pack's format, with checks, reference products, a demo builder and an `.npz` input format |
| `tools/h2pack/h2pack_read_check.c` | Reads a file set with `H2P_read_from_file`, multiplies with `H2P_matmul` and `H2P_matvec`, compares |
| `tools/h2pack/h2pack_reader_fix.{c,h}` | Link-time fix for the memory bug in H2Pack's reader |
| `tools/h2pack/README.md` | Why the bridge is file-based, the input format, the reader's rules and bugs |
| `mk/20-starsh.mk`, `mk/22-h2opus-tlr.mk`, `mk/20-h2pack.mk`, `mk/50-tlr.mk` | Make fragments |
| `tests/test_starsh.cpp` | STARS-H adapter test (C API) |
| `tests/test_starsh_cpp.cpp` | STARS-H adapter test (C++ API, `hmb_starsh.hpp`) |
| `tests/test_starsh_native.{c,h}` | Helpers that call STARS-H's API in C |
| `tests/test_h2opus_tlr.cpp`, `tests/test_h2opus_tlr_native.cpp` | H2Opus TLR tests |
| `tests/test_h2pack_file.sh` | Registered as a SCRIPT_CHECK |
| `tests/test_h2pack_writer.py` | Writer checks, run by the script above |
| `examples/copy_tlr_starsh_h2opus.cpp` | The STARS-H to H2Opus TLR example |
| `results/notes/tlr_h2pack.md` | This report |

My brief lists only `tests/test_starsh*.cpp`. I also created two non-`.cpp` files with that
prefix, `test_starsh_native.{c,h}`, because LAYOUT.md asks for library-header code in a
separate C file. My suppression file is now part of the integrator's `tests/lsan.supp`.

## 2. Public API

**STARS-H, C** (`hmb_starsh.h`). Every function returns 0 or a negative value, with the reason
in `msg`.

- `int hmb_starsh_import(const hmb_hmatrix *A, hmb_starsh **M, char *msg, size_t msglen)`:
  copies every leaf of any valid view into a new `STARSH_blrm` (format 'N').
  - STARS-H's format is a flat list of leaves with one permutation per side, so it holds TLR,
    HODLR and general H partitions alike (`starsh_hodlr.c` showed this).
  - Block lists are sorted by (row cluster, column cluster), as `get_block` requires.
  - For a TLR view, the cluster list has exactly the layout of `starsh_cluster_new_plain`.
- `int hmb_starsh_from_entries(int n, const int *perm, hmb_starsh_entry_fn *f, void *ctx, const hmb_starsh_tlr_opts *o, hmb_starsh **M, char *msg, size_t msglen)`:
  STARS-H's own TLR approximation (`starsh_blrf_new_tlr` + `starsh_blrm_approximate`) from an
  entry callback. Options: `tile_size`, `tol`, `maxrank`, `symm` ('N' or 'S'), `thread_safe`.
- `struct starsh_blrm *hmb_starsh_blrm(const hmb_starsh *)`, `const hmb_hmatrix *hmb_starsh_view(const hmb_starsh *)` and `void hmb_starsh_free(hmb_starsh *)`.
  The problem's kernel of an adapter-owned matrix evaluates the stored matrix. So `get_block`,
  `dfe` and a later re-approximation see the represented matrix.
- `int hmb_starsh_export(struct starsh_blrm *M, hmb_cstore *store, hmb_starsh_export_stats *st, char *msg, size_t msglen)`:
  any `STARSH_blrm` to a neutral view, zero-copy.
  - It copies only the mirrored dense blocks of 'S' formats and the near blocks of on-the-fly
    matrices.
  - It rebuilds the cluster and block trees from the block ranges.
  - It refuses overlapping blocks, missing blocks, and ranges that overlap without nesting.
- `char hmb_starsh_symm(const struct starsh_blrm *M)` (new): 'S' for a symmetric format (one
  triangle stored), 'N' otherwise, 0 for an incomplete matrix.
- `int hmb_starsh_matvec(struct starsh_blrm *M, int nrhs, const double *X, int ldx, double *Y, int ldy, char *msg, size_t msglen)`:
  `starsh_blrm__dmml` in the original ordering. STARS-H has no transposed product.
- `int hmb_starsh_dense(struct starsh_blrm *M, double *D, int ldd, char *msg, size_t msglen)`:
  `starsh_blrm__dca` in the original ordering.

**STARS-H, C++** (`hmb_starsh.hpp`, new, namespace `hmb`). Errors throw
`std::invalid_argument("hmb::starsh_<function>: <reason>")`.

- `using StarshHandle = std::shared_ptr<hmb_starsh>;`
- `StarshHandle starsh_import(const hmb_hmatrix &A);`
- `StarshHandle starsh_from_entries(const EntrySource &A, const int *perm, const hmb_starsh_tlr_opts &opt);`
  - STARS-H calls `A.get` concurrently only when both `A.thread_safe` and `opt.thread_safe` are
    set; otherwise one call at a time.
  - An exception thrown by `A.get` comes back as `std::runtime_error` (see §0).
- `std::shared_ptr<const hmb_hmatrix> starsh_export(const StarshHandle &M, hmb_starsh_export_stats *stats = nullptr);`
  - An aliasing pointer: its control block owns the view's trees and copied blocks, and keeps
    `M` alive.
  - It exports afresh on each call (0.05 ms at N = 4,000). `hmb_starsh_view` returns the
    handle's own view at no cost.
- `BlackBox starsh_blackbox(const StarshHandle &M);` for square matrices. 'N' is STARS-H's
  `starsh_blrm__dmml`. STARS-H has no transposed product, so 'T' works as follows, as the
  header comment says:
  - a symmetric format ('S', `symmetric == true`) is its own transpose: the same STARS-H product;
  - any other matrix: `hmb_matvec` on the handle's own zero-copy view of the same leaves
    (exact, but the package's reference kernel, not tuned).

**H2Opus TLR** (C++, `hmb_h2opus_tlr.hpp`, namespace `hmb`). Errors throw
`std::invalid_argument` or `std::runtime_error`, message starting with the function name.

- `struct H2opusTlrMatrix; using H2opusTlrHandle = std::shared_ptr<H2opusTlrMatrix>;`
- `H2opusTlrHandle h2opus_tlr_import(const hmb_hmatrix &A, const H2opusTlrImportOptions &opt = {}, H2opusTlrStats *stats = nullptr);`
  a uniform tile view (with padding) to `TTLR_Matrix<double, CPU>`.
  - `opt.symmetric` selects `H2OpusTLR_Symmetric` and reads the lower tiles only.
  - A low-rank diagonal tile is expanded to dense.
  - A dense off-diagonal tile, or one of rank > block size, is stored exactly as D·I.
  - Other shapes are refused with the reason.
- `H2opusTlrHandle h2opus_tlr_from_entries(const EntrySource &entries, const int *perm, const H2opusTlrOptions &opt = {}, H2opusTlrStats *stats = nullptr);`
  H2Opus's own `construct_tlr_matrix`. Its `eps` is an absolute per-tile threshold. It clears
  the stale diagonal ranks, and returns the source's exceptions as `std::runtime_error`.
- `std::shared_ptr<const hmb_hmatrix> h2opus_tlr_export(const H2opusTlrHandle &A);` a
  zero-copy view with `ld` = block size, as an aliasing pointer that keeps the matrix alive.
- `void h2opus_tlr_matvec(const H2opusTlrHandle &A, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy);`,
  `BlackBox h2opus_tlr_blackbox(const H2opusTlrHandle &A);` and
  `H2opusTlrStats h2opus_tlr_stats(const H2opusTlrHandle &A);`: `tlr_gemv` for A and A^T in
  the original ordering, serialized per handle.
- Native header `h2opus_tlr_native.hpp`: `using H2opusTlr = TTLR_Matrix<double, H2OPUS_HWTYPE_CPU>;`,
  `struct H2opusTlrMatrix { std::unique_ptr<H2opusTlr> A; h2opusHandle_t handle; std::mutex lock; H2opusTlrStats stats; };`
  and `H2opusTlrHandle h2opus_tlr_wrap(std::unique_ptr<H2opusTlr> A);`.

**H2Pack bridge** (`tools/h2pack/`, unchanged):

- Python:
  - `H2Matrix(n, children, start, size, U, E, B, D, perm=None, coords=None)`: the input format,
    documented precisely in the module docstring.
  - `write_h2pack(h, prefix, dim_point=None, qr_stop_tol=0.0) -> dict`: raises
    `H2PackFormatError` with the reason.
  - `h2_matvec(h, X)` and `h2_dense(h)`: reference products.
  - `kernel_h2(points, kernel, leaf_size, eta, tol)`: demo builder.
  - `save_npz` / `load_npz` and `write_vectors` / `read_vectors`.
- CLI: `h2pack_write.py demo PREFIX [...]` and `h2pack_write.py write IN.npz PREFIX`.
- C: `h2pack_read_check --meta --aux --bin --x --y [--nrhs K] [--tol T] --converter <H2Pack>/examples/meta_json_to_txt.py [--python PY] [--allow-unpatched]`,
  and `int hmb_h2pack_reader_fix_calls(void)`.

## 3. Checks

Final run from a clean build: `rm -rf build-tlr && make B=build-tlr -j2 && make B=build-tlr check HMB_TEST_FILTER='starsh|h2opus_tlr|h2pack|copy_tlr'`,
with 2 OpenMP threads, at a load below 1:

```
check                                        result  seconds  detail
test_starsh                                  PASS        0.5  test_starsh: 82 checks; n=2000 tile=192 eps=1e-06
test_starsh_cpp                              PASS        0.3  test_starsh_cpp: 25 checks; n=800 tile=100 eps=1e-06
test_h2opus_tlr                              PASS        0.3  test_h2opus_tlr: 34 checks; n=2001 bs=256 eps=1e-06
test_h2opus_tlr_native                       PASS        0.1  test_h2opus_tlr_native: 6 checks; n=1000 bs=128
copy_tlr_starsh_h2opus                       PASS        2.1  copy_tlr_starsh_h2opus: 11 checks; N=4000 tile=300 eps=1e-06
test_h2pack_file                             PASS        1.5  test_h2pack_file: 16 checks; sphere n=2000 eta=2, line n=1024, tol 1e-
passed 6, failed 0, filtered out 18
```

The clean build of the whole package took 31 s. My files compile warning-free with `-Wall`.
The only warnings in the build are six linker "executable stack" warnings from ButterflyPACK's
`dBplus_randomized.f90.o` in another agent's binaries.

All six also pass with `OMP_NUM_THREADS=1`. They pass under ASan, from a clean
`make B=build-tlr-asan OPT='-O1 -g -fsanitize=address -fno-omit-frame-pointer' HMB_LDFLAGS='-fopenmp -fsanitize=address'`
followed by `make check` with the same variables. `tests/run_tests.sh` now applies
`tests/lsan.supp` and `fast_unwind_on_malloc=0` by default. The only suppressed leaks in my
checks are inside STARS-H's `starsh_blrm__drsdd` (see starsh-false-far-leak). The libraries
themselves are not instrumented, except for one extra H2Pack run noted under
h2pack-reader-bugs.

**test_starsh** (C API)

- Views tested:
  - TLR 200 (shared tree and permutation);
  - TLR 203×171 (rectangular, different row and column permutations);
  - TLR 150 (identity);
  - HODLR 97 (depth 3, uneven);
  - a 30×25 view with row-only and column-only splits.
- All views have padded leading dimensions (NaN in the padding), rank-0 tiles and both `vtrans`
  values.
- On every view:
  - STARS-H `dmml` vs `hmb_matvec`: 2e-17 to 1.5e-16 (threshold 1e-13);
  - `dca` vs the dense reference: ≤ 8.8e-17 (1e-14);
  - `get_block` finds every block;
  - the kernel equals the stored entries exactly;
  - `dfe` ≤ 8.8e-17;
  - export(import(view)) is bit-identical, with the same leaves and values, and zero-copy.
- An invalid view is refused with the validator's reason.
- STARS-H's own TLR, from the kernel (n = 2000, tile 192, tol 1e-6, maxrank 96):
  - 'N': 111 far + 10 dense tiles;
  - 'S': 56 + 10, 55 mirrored;
  - export vs `dmml`: 1.4e-16;
  - vs the exact kernel: 7.9e-7 (≤ 10 eps);
  - the exported 'S' view is symmetric to 5e-16;
  - the kernel reads the stored matrix after the build.
- Native STARS-H matrices:
  - on-the-fly near blocks: 159,744 values evaluated, 1.4e-16;
  - 'S' with dense off-diagonal tiles: 45 mirrored, 161,280 values copied, 1.6e-16;
  - block list in column order: `get_block` wrong on 20 of 25 blocks (the bug reproduced),
    export vs `dmml` = 0.
- Three broken partitions are refused, with "overlaps", "covered by no block" and
  "without nesting".

**test_starsh_cpp** (C++ API, new; n = 800, tile 100, tol 1e-6, maxrank 50). It includes
`hmb_starsh.hpp` first and is compiled without STARS-H's include flags, so the header is
self-contained.

- `starsh_from_entries` from `Problem::entries()`, then `starsh_export`:
  - 'N': 56 far + 8 near blocks;
  - 'S': 28 + 8, 28 mirrored.
- `starsh_blackbox`, for both formats:
  - `symmetric` is false for 'N' and true for 'S';
  - 'N' vs `hmb_matvec` on the exported view: 1.2e-16 (threshold 1e-13);
  - 'T' vs `hmb_matvec(view, 'T')`: 0 for 'N' (same leaves), 4.2e-16 for 'S' (STARS-H's own
    product);
  - vs the exact kernel: 7.4e-7 (≤ 10 eps).
- `starsh_import(*starsh_export(S))` round trip: entries bit-identical, the re-import is 'N',
  and its product matches the source's to 0.
- The exported view keeps the matrix alive once handle and BlackBox are gone, reads it
  correctly (9.6e-17), and frees it with the view.
- The entry callback with a leading dimension above m (5 × 4 entries, ld 8) places every entry
  and leaves the padding alone.
- A source with `thread_safe = false` (and `opt.thread_safe = 1`) gets its 72 calls one at a
  time with 2 OpenMP threads.
- A throwing source gives `std::runtime_error("hmb::starsh_from_entries: the EntrySource threw: no row 123")`.
- Refused, with the message checked: an invalid view, `tile_size` 0, a rectangular EntrySource,
  a BlackBox of a 3 × 2 matrix, `trans` 'C'.

**test_h2opus_tlr**

- Views tested, all with permutations or identity, padded `ld`, rank 0..6, both `vtrans`, and
  one tile of rank bs + 3:
  - 200/32 (padded to 224);
  - 192/32 (no padding);
  - symmetric 150/40 (padded).
- On each: `tlr_gemv` vs `hmb_matvec` for N and T, 1.6e-16 to 2.8e-16 (1e-13); round-trip
  entries ≤ 3.2e-16 (1e-14). The symmetric type stores 10,640 values against 14,880.
- Refused:
  - different row and column orderings;
  - unequal tiles;
  - a rectangular view;
  - a row-only split;
  - a nested (HODLR) partition.
- H2Opus's own construction (n = 2001, bs = 256, 8 tiles padded to 2048):
  - non-symmetric: 7 of 8 stale diagonal ranks cleared;
  - export vs `tlr_gemv` for N and T: 1.2e-16 to 5.7e-16;
  - vs the kernel: 1.7e-9, against a bound of 10·n_block·eps/‖A‖_F = 3.8e-8.
- New: the exported view keeps the matrix alive after the handle is gone (product 1.1e-16 vs
  `tlr_gemv`) and frees it with the view.
- New: a throwing source gives
  `std::runtime_error("hmb::h2opus_tlr_from_entries: the EntrySource threw: no entry (123, 45)")`.

**test_h2opus_tlr_native** (raw H2Opus objects, n = 1000, bs = 128)

- The diagonal-rank bug is present: 7 of 8 tiles, `tlr_gemv` error 2.8e-2.
- The multi-vector bug is present: 3 vectors with npc = 2 are off by 6.9e-1; with npc = 1 they
  match single vectors to 7e-16.
- After `h2opus_tlr_wrap`: 1.2e-11 against the bound 7.5e-10; export vs `tlr_gemv` 1.1e-16.

**test_h2pack_file**

- Cases:
  - 3-D sphere, n = 2000: nested bases on levels 4-5, 8 padded bases;
  - 1-D, n = 1024: bases on levels 2-5;
  - a rank-0 leaf basis.
- H2Pack vs the numpy H² product: `H2P_matmul` 0.0, `H2P_matvec` ≤ 5.2e-16 (threshold 1e-12).
- H2Pack vs the dense kernel: 7.0e-10 and 1.1e-9 (≤ 10·tol, tol = 1e-9).
- Every read prints no BLAS "illegal value" message and runs the reader fix.
- The `.npz` route writes byte-identical files.
- `test_h2pack_writer.py`, 16 checks:
  - `h2_matvec` equals the dense expansion;
  - the `.npz` round trip;
  - padding of bases;
  - refusals with their reason: complex data, a non-symmetric diagonal block, both blocks of a
    pair, a missing diagonal block, a missing block, a dense block on a non-leaf, an
    admissible pair across two levels, 3 children with `dim_point` 1, 9 children, a bad
    permutation.

**copy_tlr_starsh_h2opus** (11 checks): the transfer, A and A^T lossless, both libraries
against the kernel, and the rebuild route; numbers in §4.

## 4. Measured costs (for RESULTS.md)

Setup for `examples/copy_tlr_starsh_h2opus`:

- Kernel 1/(0.1+r) on N = 4000 random points of the sphere.
- Recursive-bisection order; 14 tiles of 300, the last of 100, padded in H2Opus.
- STARS-H tolerance eps = 1e-6, maxrank 100.
- Timings are the best of 5 runs.
- Quiet machine: 2 cores, load 0.1-1.1 during the runs.

| Storage, threads | Transfer (export + import) | One STARS-H matvec | Values copied | Lossless (H2Opus vs STARS-H) | Accuracy vs kernel |
|---|---|---|---|---|---|
| 'N', 1 | 0.06 ms + 16.8-18.1 ms = **9.7-10.7 source matvecs** (4 runs) | 1.6-1.8 ms | 4,572,000 (export copies 0) | 7.7e-17 | 7.09e-7 (both libraries) |
| 'S', 1 | 0.05-0.06 ms + 11.5-12.6 ms = **8.8-10.0 source matvecs** (4 runs) | 1.3 ms | 2,881,000 (91 tiles mirrored, zero-copy) | 7.7e-17 | 7.10e-7 |
| 'N', 2 | **9.5-11.1 source matvecs** (5 runs) | 1.6-1.8 ms | as above | 7.7e-17 | 7.09e-7 |
| 'S', 2 | **7.9-10.5 source matvecs** (5 runs) | 1.2-1.3 ms | as above | 7.7e-17 | 7.10e-7 |

- **A second thread shortens neither the transfer nor STARS-H's product.** The import's
  copies are sequential, and the adapter uses STARS-H's sequential product (starsh-dmml-omp).
  Only the two constructions speed up.
- **Where the import time goes.** I measured 8,600-8,900 minor page faults (about 36 MB of
  newly allocated H2Opus storage) per import, against 2 per matvec. The copy into fresh memory,
  not the copying itself, sets the cost.
- **Alternative route.** H2Opus rebuilding its TLR matrix from the kernel's entries
  (`construct_tlr_matrix`, 16M entries, absolute eps = eps·‖A‖_F/n_block) took 1.11-1.17 s =
  632-706 source matvecs on 1 thread, and 0.62-0.69 s = 361-407 on 2 threads, with accuracy
  3.1e-7.
- **STARS-H build.** 1.30-1.38 s ('N', 183 far + 13 near tiles) and 0.72-0.73 s ('S', 92 + 13)
  on 1 thread; 0.61-0.69 s and 0.37-0.42 s on 2. Max rank 100. H2Opus stores 4.74M values ('N')
  or 3.0M ('S'), which adds about 35 MB or 22 MB resident.
- **The integrator's reference runs** (`results/logs/package/copy_tlr_starsh_h2opus_n*.log`,
  1 thread, made with the previous binary) agree:
  - N = 4,000: 8.9 ('N') and 8.2 ('S') matvecs, rebuild 560;
  - N = 16,000: 2.6 and 2.7 matvecs, rebuild 805 (16.5 s), STARS-H builds of 19.2 s and 9.9 s.
- **Under load** (2-6 on 2 cores, the previous revision's runs) the same transfer cost 11-23
  matvecs, and up to 331 with 2 threads: H2Opus's many short OpenMP regions stall when cores
  are busy.
- **H2Pack.** `write_h2pack` takes 0.02 s for the n = 2000 sphere (1.3M values, best of 3);
  the demo's numpy H² construction takes 0.5 s. The transfer is exact (0 for `H2P_matmul`,
  5e-16 for `H2P_matvec`).

## 5. Gotchas

| Id | What happens | Handled at | Test |
|---|---|---|---|
| **starsh-get-block** (new) | `starsh_blrm_get_block` indexes the factors by CSR position, not COO index (`blrm.c:262-289`). It returns another block's factors unless the lists are sorted by block row. | `hmb_starsh.c:245-258` sorts on import; the export reads the COO lists directly. | test_starsh: 0 wrong on every import; the column-ordered list gives 20/25 wrong and its export is exact |
| **starsh-kernel-data** (new) | STARS-H passes kernels the problem's `row_data` in `problem_get_block` / `blrf_get_block`, but the cluster's `data` in the approximation, `dmml`, `dca` and `dfe`. A first version segfaulted in `dfe`. | `hmb_starsh.c:81-89` sets both; `:720` uses the cluster data like `dca`. | test_starsh: `dfe` and kernel checks on 5 views; STARS-H builds |
| **starsh-init** (new) | `starsh_blrm_approximate` is a NULL pointer and the RSVD oversampling is -1 until `starsh_init()` runs, which also prints 3 lines to stderr. | `hmb_starsh.c:393-397` | every `from_entries` call |
| **starsh-dmml-omp** (new) | `starsh_blrm__dmml_omp` hard-codes `maxrank = 100` for its per-thread buffers, and `maxnb = nrows/nbrows` for on-the-fly blocks: buffer overflow beyond those (`openmp/blrm/dmml.c`). | Always the sequential `dmml`: `hmb_starsh.c:845-847` | code reading only, not run |
| **starsh-false-far-leak** (new) | `drsdd` leaks an `Array` header, about 104 bytes, per far block whenever tiles exceed maxrank. | Documented at `hmb_starsh.c:421`; suppressed in `tests/lsan.supp` | LSan saw 2,052 objects (71 kB) in test_starsh, 2,136 (74 kB) in test_starsh_cpp and 1,806 in the example |
| **h2opus-tlr-diag-rank** (new) | For the non-symmetric type, `construct_tlr_matrix` compresses a stale buffer into each diagonal tile (the generator skips `block == block_column`), and `tlr_gemv` adds it: error 2.8e-2. | `h2opus_tlr.cpp:127-142`, in `from_entries` and `wrap`; the import writes rank 0 | test_h2opus_tlr_native (bug shown, then fixed); test_h2opus_tlr (7/8 cleared) |
| **h2opus-tlr-gemv-multi** (new) | `tlr_gemv` with several vectors and npc > 1 overlaps its per-column buffers (`tlr_gemv_marshal.h`: offset `(j-j_start)*n`), giving relative errors of 0.43-0.74. Its reduction also assumes `ldy` = padded n. | `h2opus_tlr.cpp:386-390`: npc = 1 when nrhs > 1; padded `ld` always | test_h2opus_tlr_native; every 3- and 4-vector product in the tests |
| **h2pack-cwd-scripts** (new) | `H2P_read_from_file` runs `./meta_json_to_txt.py` from the current directory via `system()`, writes a fixed `tmp_metadata.txt` there, and returns silently without the script. | `h2pack_read_check.c:92-127`: a private temporary directory plus a wrapper that runs H2Pack's converter with `$(PYTHON)` | every read in test_h2pack_file |
| **h2pack-zero-rank** (new) | A basis with 0 columns makes H2Pack call DGEMV with lda = 0. OpenBLAS prints "parameter number 6 had an illegal value"; reference BLAS stops. The study's 1-D prototype printed 4 of them. | The writer pads to one zero column: `h2pack_write.py:323-341` | test_h2pack_file "no BLAS parameter errors" (8 padded, rank-0 case); writer test |
| **h2pack-reader-bugs** | See the note below this table. | `h2pack_reader_fix.c:15-27`, `mk/20-h2pack.mk:16-19`; the writer stays clear of the rest (`h2pack_write.py:345`, `is_HSS = 0`) | test_h2pack_file "reader fix ran"; the ASan run described below |
| **h2pack-blas-threads** | H2Pack sets BLAS threads in its construction routines (`build.c:82,354` and others), not in read or matvec. | Saved and restored, change reported: `h2pack_read_check.c:170,192-193` | prints "unchanged" in every read |
| **lowrank-convention** | STARS-H and H2Opus store U·Vᵀ with V cols × rank. | `hmb_starsh.c:305`, `h2opus_tlr.cpp:220` | both `vtrans` in all views |
| **perm-direction** | STARS-H's pivot and H2Opus's `index_map` are new→old, like the neutral perm; both libraries multiply in permuted order. | matvec wrappers in both adapters | separate row and column permutations (STARS-H), H2Opus permuted views |
| **symmetric-storage** | STARS-H 'S' and H2Opus `Symmetric` store one triangle; H2Pack stores one block per pair. | `hmb_starsh.c:695,731` mirror; `starsh_blackbox` answers 'T' with STARS-H's product for 'S' (`hmb_starsh.hpp:181`); H2Opus import option; writer rules | 'S' tests in all three |
| **scalar-type** | All three are real-only. | dtype check (STARS-H); double-only instantiation (H2Opus); complex refused (`h2pack_write.py:186`); `_Static_assert` in the checker | writer refusal test |
| **header-clash** | H2Opus headers appear only in `h2opus_tlr.cpp` and `h2opus_tlr_native.hpp`; `hmb_starsh.hpp` includes no STARS-H header. | layout | the example links STARS-H and H2Opus together; test_starsh_cpp builds without STARS-H's include flags |

**h2pack-reader-bugs, exact trigger.** `H2P_read_from_file` (`H2Pack_file_IO.c:744-745`)
allocates one bounding box. `H2P_calc_node_inadm_lists` then reads one box per node from it, a
heap over-read on every file with more than one node. From the garbage it writes past
`node_inadm_lists` (3^d entries per node). In the study's 1-D file, 2 to 29 nodes went over
that limit in every run. The heap corruption then aborted with "double free or corruption" or
not, depending on the thread count and on stdout. The HSS matvec of a file-read matrix uses
the counts this function computes.

**What the fix does, and how I checked it.** A link-time wrapper replaces only the reader's call
with zeroed lists and the admissible-pair counts the reader computed itself. In a scratch run
against the study's ASan-instrumented H2Pack objects, the unfixed checker hit
`heap-buffer-overflow in H2P_calc_node_inadm_lists`; the fixed one ran clean and exact. The
writer cannot avoid this bug: every file with more than one node triggers it.

## 6. Limitations and unfinished work

- **STARS-H**
  - The import always builds format 'N': there is no one-triangle import.
  - STARS-H has no transposed product. `starsh_blackbox` answers A^T through `hmb_matvec` on
    the handle's view (the package's untuned reference kernel), except for 'S' formats.
  - `starsh_blackbox` is for square matrices only (a BlackBox has one size). A rectangular
    matrix imports, exports and multiplies through the C API.
  - The export rebuilds the block tree from the ranges it uses. Leaves are identical to the
    source, but the tree can be flatter than the original.
  - Only the OpenMP backend was tested; MPI and StarPU were not.
- **H2Opus TLR**
  - CPU only, `H2OpusTLRTile` allocation only, no triangular types.
  - Imported U factors are not orthonormalized (`tlr_gemv` does not need it).
  - `from_entries` takes H2Opus's absolute eps.
  - `h2opus_tlr_wrap` is covered only for a non-symmetric matrix (test_h2opus_tlr_native).
- **H2Pack**
  - Writes H2 only, not HSS, and no partially admissible pairs.
  - AOT only: files carry no skeleton points, so H2Pack's JIT mode is unavailable.
  - Readers need the link-time fix, a static `libH2Pack.a` and a GNU-compatible linker.
  - Reading HSS files with the fix is untested here; the evidence is the prototype's `hss.c`.
  - SPDHSS on file-read matrices is unsupported: the reader leaves `node_height` NULL.

## 7. Requests to the owners of other files

All five requests of the previous revision are done:

1. `GOTCHAS.md` has every id from §5.
2. `config.mk.example` and `deps/build_deps.sh` name the static `libH2Pack.a`. The STARS-H
   adapter and its test helper compile against `deps/install/config.mk` (checked with
   `make CONFIG=deps/install/config.mk`): its CBLAS and LAPACKE headers are on the default
   include path, so the earlier note on `STARSH_CPPFLAGS` is not needed here.
3. The Makefile has `TOOLS`; `mk/20-h2pack.mk` uses it.
4. `tests/run_tests.sh` applies `tests/lsan.supp` by default.
5. `bisection_perm` is in `tests/common/hmb_test.hpp`; the three local copies are gone.

New, for information only: the reference logs
`results/logs/package/copy_tlr_starsh_h2opus_n*.log` come from the previous binary and say
"13 checks". A rerun of `results/run_reference.sh` prints 11, with the same options and the
same `RESULT` line.
