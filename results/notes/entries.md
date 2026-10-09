# Agent C (entries): hmat-oss adapter and the entry route

All three of my checks pass. That holds for a clean build (`rm -rf build-entries && make B=build-entries -j2 && make B=build-entries check HMB_TEST_FILTER='hmatoss|entries_hmatoss'`), under AddressSanitizer, and against agent A's fresh pinned builds (`CONFIG=deps/install/config.mk`). Everything compiles with -Wall and no warnings.

Revision after the integrator's API harmonization:
- `hmatoss_export` now returns `std::shared_ptr<const hmb_hmatrix>`, an aliasing pointer that owns the copied data.
- `hmatoss_import` now takes `const hmb_hmatrix &`.
- The LeakSanitizer suppressions moved to `tests/lsan.supp`, which `tests/run_tests.sh` applies by default.
- The neutral form now lets a block with `rsons == 1` descend into a cluster's single son (`neutral.h`, `hmb_validate`). My importer would have crashed hmat-oss on such views; that is fixed and tested (section 2).

## 1. Files created

- `adapters/hmatoss/hmb_hmatoss.hpp`: the neutral-facing API (build, entries, blackbox, stats, export, import) with its conventions documented. It includes no hmat-oss header.
- `adapters/hmatoss/hmatoss_native.hpp`: the native side. It holds `struct HmatossMatrix` (interface, `hmat_matrix_t*`, cluster tree) and `hmatoss_wrap`.
- `adapters/hmatoss/hmatoss_stream.hpp`: internal to the adapter. It documents hmat-oss's serialization stream format (undocumented upstream) and declares the structure parser and the shared checks.
- `adapters/hmatoss/hmb_hmatoss.cpp`: build (block-assembly callbacks), wrap, entries over `get_values`, matvec over `gemv`, stats, and the structure-stream parser.
- `adapters/hmatoss/hmb_hmatoss_copy.cpp`: export and import through `write_struct`/`write_data` and `read_struct`/`read_data`. Export uses a streaming parser that copies each leaf array straight into the store. Import uses a lazy writer whose staging buffer holds one leaf at most.
- `mk/20-hmatoss.mk`: adapter objects and the two tests.
- `mk/50-entries.mk`: the example (needs hmat-oss and Htool).
- `tests/test_hmatoss_entries.cpp`: build, entries, the +1 regression, concurrency, symmetric storage, matvec, and the refusals (33 checks).
- `tests/test_hmatoss_copy.cpp`: export, import and round trips, plus import of hand-made views (35 checks).
- The hmat-oss LeakSanitizer suppressions I wrote now live in the shared `tests/lsan.supp` (merged by the integrator).
- `examples/entries_hmatoss_htool.cpp`: port of E3's bench.cpp through the package API, self-checking (11 checks).
- `results/notes/entries.md`: this report.

## 2. Public API (namespace `hmb`)

| Signature | Meaning |
|---|---|
| `HmatossHandle = std::shared_ptr<HmatossMatrix>` | Opaque handle (defined in `hmatoss_native.hpp`). It owns the hmat-oss matrix and the cluster tree it was built on. |
| `HmatossOptions {epsilon=1e-6, eta=2, leaf_size=64, lower_symmetric=false}` | ACA+ tolerance, which is also the matrix's low-rank epsilon. Standard admissibility; median clustering. |
| `HmatossHandle hmatoss_build(const EntrySource&, int dim, const double *coords, const HmatossOptions& = {}, HmatossStats* = nullptr)` | Assembles an n x n hmat-oss matrix from entries. Square only. The source is called sequentially with 0-based indices, for whole blocks, single rows and single columns. |
| `EntrySource hmatoss_entries(const HmatossHandle&)` | Entries over `get_values`. It adds the +1, bounds-checks indices, and serves the unstored triangle of symmetric storage by symmetry. `thread_safe = true`. Rectangular matrices are allowed. |
| `BlackBox hmatoss_blackbox(const HmatossHandle&)` | Matvec ('N' or 'T') in the original ordering; square only. `symmetric` is true for symmetric storage. X is copied before the call. Not meant for concurrent calls. |
| `HmatossStats hmatoss_stats(const HmatossHandle&)` | Sizes, storage ('N', 'L' or 'U'), stored values, dense values, leaf counts and build time. |
| `std::shared_ptr<const hmb_hmatrix> hmatoss_export(const HmatossHandle&, HmatossExportStats* = nullptr)` | Block copy hmat-oss -> neutral view, validated. The result is an aliasing shared_ptr whose control block owns the copied leaves, trees and permutations, so the view outlives the source. Every stored value is copied once. Symmetric storage is mirrored: dense blocks are transposed copies, low-rank blocks alias the stored factors. Rows and columns share one tree in the view when hmat-oss's two trees are identical. Bounding boxes come from hmat-oss's coordinates. |
| `HmatossHandle hmatoss_import(const hmb_hmatrix&, const HmatossImportOptions& = {})` | Block copy neutral -> hmat-oss, lossless (no recompression), full storage. Any valid view is accepted; an invalid one is refused with `hmb_validate`'s message. Options: `epsilon` (stored low-rank epsilon, used by hmat-oss's later LU, gemm and truncate) and optional coordinates. |
| `HmatossHandle hmatoss_wrap(const hmat_interface_t&, hmat_matrix_t*, bool take_ownership=false)` (native) | Wraps a matrix the user already holds. Refuses any interface that is not `HMAT_DOUBLE_PRECISION`. |

On the "every pairing" requirement for import: hmat-oss's child `(i, j)` is number `i + j * nrChildRow`, with `nrChildRow = 1` when the rows are kept and otherwise all row sons. That is exactly the neutral layout, which `hmb_validate` already enforces. So every valid view is representable: 1 x 1 splits are collapsed, empty blocks are written as hmat-oss void leaves, and invalid views are refused with a clear message (tested).

Single sons:
- Under the revised neutral rule, a block with `rsons == 1` over a cluster with a single son may keep the cluster or descend into the son.
- hmat-oss can only "keep" (`keepSameRows`). With a descended child, hmat-oss assigned the grandchildren no cluster and crashed (segfault, verified with the fix removed).
- The importer now writes a chain of single sons as one hmat-oss cluster (`put_cluster` in `hmb_hmatoss_copy.cpp`), so both forms are the same hmat-oss block. The export already collapsed such chains.
- The rectangular hand-made view in `test_hmatoss_copy` covers both forms: a 1 x 1 block that descends into a single son and then splits it, and a block that keeps the single-son cluster.

## 3. Checks

Final clean build after the API revision, `make B=build-entries check HMB_TEST_FILTER='hmatoss|entries_hmatoss'`. The machine was shared with four agents, so the cost in source matvecs varies from run to run: at N = 4,000 it ranged from 58 to 95 in optimized builds and reached 114 under ASan.

```
check                                        result  seconds  detail
test_hmatoss_entries                         PASS        0.7  test_hmatoss_entries: 33 checks; n=1500 eps=1e-06
test_hmatoss_copy                            PASS        1.2  test_hmatoss_copy: 35 checks; n=2000 eps=1e-06
entries_hmatoss_htool                        PASS        3.7  entries_hmatoss_htool: 11 checks; N=4000 transfer 57.6 source matvecs,
passed 3, failed 0, filtered out 20
```

### test_hmatoss_entries (n = 1,500, eps = 1e-6)

| Area | What is verified | Measured value |
|---|---|---|
| Build | Assembled matrix vs kernel | 3.6e-7 <= 10 eps |
| Entries | Whole matrix vs hmat-oss's own `get_block` dense extraction (an independent code path) | 1.3e-16 relative max |
| Entries | 300 random requests (rows, columns, blocks; unsorted, with duplicates) | 1.3e-16 |
| index-base | Raw 0-based `get_values` | rc 0 and 91% error |
| index-base | Adapter on the same indices | 9.3e-17 |
| Concurrency | 768 requests from 4 OpenMP threads vs sequential results | Bitwise identical; up to 4 calls in flight |
| Bounds | Out-of-range index | Refused |
| Matvec | Blackbox 'N' and 'T' vs kernel | 4.9e-7 and 5.2e-7 |
| Matvec | Blackbox vs dense product of hmat's own entries | 1.2e-15 |
| Matvec | Input X | Untouched |
| symmetric-storage | Lower storage reported as 'L' | ok |
| symmetric-storage | Raw `get_values` on the whole matrix | Leaves 48.4% unwritten, rc 0 |
| symmetric-storage | Adapter output | Writes every entry; equals `get_block` (8.9e-17); exactly symmetric; random requests ok |
| symmetric-storage | Blackbox flag and 'N'/'T' vs kernel | 5.5e-7 |
| scalar-type | A float interface reading into a double buffer, through hmat-oss directly | rc 0; 8 of 16 values garbage, 8 unwritten |
| scalar-type | `hmatoss_wrap` on the float interface | Refused |
| factorized-state | After LU of a wrapped matrix, `get_values` | Returns the factors (rel. change 0.80, rc 0) |
| factorized-state | `hmatoss_entries`, `hmatoss_blackbox`, `hmatoss_export` on the LU-factorized matrix | All refuse |
| factorized-state | hodlrsym-factorized HODLR matrix | Refused with a clear message (write_struct throws for that state) |

### test_hmatoss_copy (n = 2,000, eps = 1e-6)

- **Export, non-symmetric kernel**
  - The view is valid.
  - 2,165,064 values are copied, equal to hmat's stored count.
  - The tree is shared and the dimension is 3.
  - All low-rank leaves have vtrans = 1.
  - Bounding boxes contain their points.
  - `hmb_matvec` 'N' and 'T' equal hmat-oss's `gemv`: difference 0.0.
  - `hmb_entries` equals `get_values` to 1.1e-16.
- **Round trip import(export(A))**
  - Same leaf counts.
  - `gemv` 'N' and 'T' bitwise identical.
  - All entries bitwise identical.
  - export(import(view)) equals the view field by field and bit for bit.
- **Lower-symmetric export**
  - The view is valid, with 460 leaves mirrored and 601,246 dense values transposed.
  - `hmb_matvec` equals `gemv` to 2.3e-16 ('N') and 2.0e-16 ('T').
  - Entries equal the adapter's to 1.1e-16.
  - Importing it gives full storage; its entries are bitwise equal and its `gemv` agrees to 2.6e-16.
- **Upper storage (hmat-oss transpose of the lower-stored matrix)**
  - Export is valid.
  - Entries equal the lower-stored matrix's, bitwise.
  - Exported entries agree to 1.1e-16.
- **Hand-made rectangular view (37 x 29, separate row and column trees, random permutations)**
  - Shapes covered: a 3-way split, a one-sided split, a single-son cluster (one block keeps it, another descends into the son and splits it), an empty cluster, a 1 x 1 split, rank 9 on an 8 x 13 block, rank 0, empty blocks, padded ldD/ldU/ldV (NaN in the padding), vtrans 0 and 1.
  - Entries equal the view to rounding (<= 1e-14).
  - hmat `gemv` equals the view to rounding (<= 1e-14).
  - export(import(view)) is valid and matches exactly: fewer blocks (1 x 1 splits collapsed), and the empty 7 x 0 leaf is dropped.
- **Hand-made square view (64, one shared tree)**
  - Entries agree to 3.0e-16, `gemv` 'N' to 1.5e-17 and 'T' to 8.4e-17.
  - The export of the import matches exactly.
  - Blackbox 'T' agrees to 1.2e-16.
- **Invalid view** (a 2-way split on a cluster with 3 sons): refused with `hmb::hmatoss_import: invalid view: block 0: split into 2 row parts but row cluster 0 has 3 sons`.

### entries_hmatoss_htool (N = 4,000)

- Transfer requests fewer than N² entries.
- The source served concurrent requests.
- Transfer vs source, transfer vs kernel, source vs kernel and direct vs kernel are each within 10 eps.
- The 1-thread and 2-thread transfers are identical.
- The three tolerance checks pass (section 4).
- The rebuilt matrix is independent of the source: the hmat-oss matrix was destroyed, verified with a weak_ptr.

### AddressSanitizer

- Ran all three with `make B=build-entries-asan OPT='-O1 -g -fsanitize=address -fno-omit-frame-pointer' HMB_LDFLAGS='-fopenmp -fsanitize=address'`.
- `tests/run_tests.sh` now applies `tests/lsan.supp` by default, which covers an hmat-oss leak (GOTCHA(hmat-node-leak)).
- Result after the revision: all pass, with no ASan or LSan report beyond the suppressed library allocations.

## 4. Measured costs (for RESULTS.md)

Problem: E3's. Non-symmetric kernel on a shuffled Fibonacci sphere, source hmat-oss (ACA+, eta 2, leaf 64), destination Htool (eta 10, leaf 64, OpenMP build), eps_src = eps_dst = 1e-6, OPENBLAS_NUM_THREADS = 1.

How the runs were made:
- Each larger run was done once, with `nice`, started after waiting for the machine to look idle (as E3 did). Other agents' compiles then resumed during the runs.
- In the N = 20,000 run the second core was busy during the 2-thread timing, so its 2-thread numbers are not usable. E3's idle figure was 1.29 s, i.e. 35 matvecs.

Definitions:
- "Matvecs" means time divided by one hmat-oss matvec timed next to the builds.
- Errors are ‖Y_a − Y_b‖_F / ‖Y_b‖_F over 3 random vectors jointly. E3 took the maximum of the per-vector errors, so its numbers are slightly higher.
- Our "direct" build goes through `Problem::entries()` (a `std::function`) and includes Htool's cluster tree. It is therefore 10-25% slower than E3's direct build, which makes the transfer/direct ratio lower (3.2-4.0x here, against 4.3-4.8x in E3).

| | N = 4,000 (default) | N = 5,000 | N = 20,000 (`--no-tolerance`) |
|---|---|---|---|
| Source build; stored values | 0.44 s; 5.14M (32% of N²) | 0.89 s; 6.35M (25%) | 4.29 s; 34.7M (8.7%, 278 MB) |
| Source matvec | 3.8 ms | 5.8 ms | 33.7 ms |
| Direct Htool build, 1 thread | 0.082 s = 21.7 matvecs | 0.118 s = 20.5 | 0.657 s = 19.5 |
| Transfer, 1 thread | 0.259 s = 68.4 matvecs = 3.15x direct | 0.446 s = 77.2 = 3.77x | 2.638 s = 78.3 = 4.02x (E3: 70.5, 4.3x) |
| Transfer, 2 threads | 0.131 s = 36.2 matvecs | 0.229 s = 38.9 | not usable (busy core) |
| Transfer / source build | 0.59x | 0.50x | 0.62x |
| Calls: row/col + block | 41,754 + 562 | 80,000 + 1,016 (= E3) | 316,144 + 4,436 (= E3) |
| Entries requested | 6.09M = 38% of N² = 1.19x src stored | 7.97M = 31.9% = 1.26x (= E3) | 46.18M = 11.5% = 1.33x (= E3) |
| Destination stored; max rank (direct) | 1.19x src; 31 (23) | 1.25x; 26 (23) (= E3) | 1.33x; 30 (25) (= E3) |
| Extra RSS of the transfer | +48 MB | +54 MB | +364.5 MB (E3: +364.5 MB) |
| Error: transfer vs source | 1.35e-6 | 0.94e-6 | 1.38e-6 |
| Error: transfer / source / direct vs kernel | 1.48e-6 / 0.76e-6 / 1.79e-6 | 1.09e-6 / 0.70e-6 / 0.56e-6 | 1.53e-6 / 0.86e-6 / 0.85e-6 |

Every deterministic quantity at N = 5,000 and 20,000 matches E3's logs exactly: call counts, entry counts, the source's and the destination's stored values, leaf counts and ranks. So `hmatoss_build` through the block-assembly callbacks produces the same matrix as E3's per-entry `simple_compute`.

### Tolerance stacking (GOTCHA(tolerance-stacking))

Error is against the kernel; entries are those requested by the destination; time is the build time.

| Setting (eps_src -> eps_dst) | N = 5,000 | N = 4,000 |
|---|---|---|
| Same tolerance, 1e-6 -> 1e-6 | 1.09e-6; 7.97M entries; 0.51 s | 1.48e-6 |
| Source 100x tighter, 1e-8 -> 1e-6 | 5.53e-7; 7,581,912 entries | 1.78e-6 |
| Direct build from the kernel | 5.57e-7; 7,578,628 entries | 1.79e-6 |
| Destination 100x tighter, 1e-6 -> 1e-8 | 7.03e-7 (the source's own error); 21.4M entries = 2.7x; 1.18 s = 2.3x | 7.59e-7; 2.3x the entries |

What this shows:
- **Same tolerance.** The rebuild is 1.5-2x less accurate than its own source.
- **Source 100x tighter (destination 10-100x looser than the source, the advice).** The rebuild is the direct build: the same error within 1% and the same entry count within 0.04%.
- **Destination 100x tighter.** It costs 2.3-2.7x the entries and 2.3-3x the time, and gains nothing beyond the source.
- **Caveat at N = 4,000.** Htool's own direct build at 1e-6 is less accurate (1.79e-6) than the same-tolerance rebuild from hmat-oss (1.48e-6). So "the errors add" holds against the source, not always against a direct build. The example checks only statements that hold at every size measured.

### Block copy (scratch measurement, once, `nice`, loaded machine, 1 thread)

| | Export | Import |
|---|---|---|
| N = 4,000, full storage | 1.0e-2 s = 3.3 matvecs | 7.5 matvecs |
| N = 20,000, full storage | 0.098 s = 1.8 matvecs (34.7M values copied) | 0.198 s = 3.7 matvecs |
| N = 20,000, lower-symmetric | 1.8 matvecs (3.75M dense values transposed for the mirror) | 3.0 matvecs |

## 5. Gotchas

### index-base

- **What happens.** `get_values` takes 1-based original indices, while the assembly callbacks are 0-based. Without the +1 it returns rc 0 and the neighbouring entries (91% error). With index 0 it also reads `indices_rev()[-1]`, out of bounds.
- **Where handled.** `hmb_hmatoss.cpp:369` adds the +1 and checks bounds. The 0-based assembly side is at `hmb_hmatoss.cpp:214`.
- **Test.** `test_hmatoss_entries.cpp:151` calls raw `get_values` without the +1 (91% error, rc 0) and the adapter on the same indices (9e-17). Indices of 0 are avoided so the test runs under ASan.

### symmetric-storage

- **What happens.** On lower-symmetric storage, `get_values` leaves the upper triangle unwritten (48.4%) with rc 0. After hmat-oss's `transpose` the storage is upper instead. `get_values` also skips NULL ("inert") children; none of the C API's admissibility conditions produce them.
- **Where handled.**
  - Entries zero-fill the output, then fill the missing triangle from a transposed request, using hmat positions from the stream (`hmb_hmatoss.cpp:380`).
  - Export mirrors the unstored blocks (`hmb_hmatoss_copy.cpp:275`). Low-rank mirrors alias the stored factors; dense mirrors are transposed copies.
- **Tests.**
  - In `test_hmatoss_entries`: the raw call leaves 48.4% unwritten; the adapter writes every entry, equals `get_block`, and is exactly symmetric.
  - In `test_hmatoss_copy`: "symmetric export", the symmetric round trip, and "upper storage".

### scalar-type

- **What happens.** hmat-oss picks the type at run time and writes through `void*`. A float matrix read into a double buffer gives 8 garbage values and 8 untouched ones out of 16, with rc 0.
- **Where handled.**
  - The interface type is checked in `checked()` (`hmb_hmatoss.cpp:50`) and in `hmatoss_wrap` (`:310`).
  - The stream's type field is checked at `:180`.
  - Import writes type 1, double (`hmb_hmatoss_copy.cpp:586`).
- **Test.** The "scalar-type" checks in `test_hmatoss_entries`.

### lowrank-convention

- **What happens.** hmat-oss stores A·Bᵀ with B cols x rank.
- **Where handled.** Export writes vtrans = 1 and aliases the factors (`hmb_hmatoss_copy.cpp:327`). Import transposes a vtrans = 0 V (`:523`).
- **Tests.**
  - On the non-symmetric kernel, export `matvec` 'N' and 'T' equal `gemv` exactly.
  - A check that all low-rank leaves have vtrans = 1.
  - The hand-made views mix vtrans 0 and 1 with padding.

### tolerance-stacking

- **What happens.** See section 4.
- **Where handled.** `entries_hmatoss_htool.cpp:176`.
- **Test.** Three self-checks in the example.

### hmat-epsilon (new)

- **What happens.** hmat-oss truncates every compressed block at the matrix's own low-rank epsilon. That defaults to 1e-4 and is set by `set_low_rank_epsilon`; the compressor's epsilon does not apply. My probe: ACA+ at 1e-10 without `set_low_rank_epsilon` gives a matvec error of 1.1e-4; with it, 1.4e-10. This also explains the ~3e-6 errors of the study's survey `test1.c` at SVD 1e-12.
- **Where handled.** `hmb_hmatoss.cpp:283` sets both epsilons.
- **Test.** The "build: ||A - K||" check (3.6e-7 at 1e-6).

### gemv-mutates-input (new)

- **What happens.** hmat-oss's `gemv` permutes the input vector in place, then restores it (`c_wrapping.hpp` gemv). Read-only input would crash. A shared input is seen permuted by other threads during the call.
- **Where handled.** `hmb_hmatoss.cpp:415` copies X.
- **Test.** "blackbox: input X left untouched".

### factorized-state (new)

- **What happens.**
  - After LU, `get_values` returns the factors with rc 0 (relative change 0.80). The pivot check is an assert, compiled out.
  - After LLt or hodlrsym, `gemv` applies the factor, by design.
  - The factorization state is visible only in the serialization stream.
  - hodlrsym is not even serializable: `write_struct` throws `std::runtime_error` after printing a backtrace to stderr, because `convert_factorization_to_int` has no HODLRSYM case.
- **Where handled.** `require_unfactorized` (`hmb_hmatoss.cpp:198`) runs when an EntrySource, BlackBox or export is created. The catch in `read_structure` turns the hodlrsym exception into a clear error.
- **Tests.** The "factorized" checks: LU then entries, blackbox and export are refused; a hodlrsym-factorized matrix is refused.

### hmat-node-leak (new)

- **What happens.** hmat-oss 45a2fdb leaks 16 bytes per block-tree node, for built and for deserialized matrices: `LocalSettings` allocates an `FPCompressionSettings` (`h_matrix.hpp:85`) that `~HMatrix` never frees. That is about 31 KB per N = 2,000 matrix. LeakSanitizer then fails every test that uses hmat-oss.
- **Where handled.** `tests/lsan.supp` (my suppressions, merged there by the integrator), scoped to the two hmat-oss constructors. `tests/run_tests.sh` applies it by default.
- **Coverage.** The ASan runs report only those suppressed allocations.

Watch items (not active, no id):
- If hmat-oss is built with `HAVE_OPENBLAS_SET_NUM_THREADS` and `HAVE_GOTO_GET_NUM_PROCS`, its `DisableThreadingInBlock` restores OpenBLAS to `goto_get_num_procs()` threads after every call, not to the previous count. Both the study build and agent A's build have these off.
- `hmatoss_entries`' `thread_safe` relies on hmat-oss being built without `HMAT_SCALAR_ARRAY_ORTHO`, which its CMake never sets. With it, reading a dense leaf resets an orthogonality flag, a benign write race.

## 6. Limitations and unfinished items

- `hmatoss_build` is square only. It offers ACA+, median clustering and standard admissibility only; hmat-oss's other compressors and admissibility conditions are reachable through `hmatoss_wrap`.
- hmat-oss's global settings (`hmat_set_parameters`) are process-wide.
- Export copies all leaf data. The public C API gives no stable pointer into hmat-oss's leaves (the stream callback's buffer is valid only during the call), so a zero-copy export would need hmat-oss's uninstalled C++ headers.
- Import is lossless but, without the optional coordinates, its cluster trees carry zero coordinates. Those trees must not be reused to build new matrices by admissibility.
- Import does not set orthogonality flags. Writing 0 means "unknown" and is always safe.
- The BlackBox is not verified for concurrent `apply` calls.
- Symmetric export requires identical row and column trees, which is always the case for hmat-oss's symmetric storage.
- Timings are noisy on the shared machine. Use the idle-gated N = 5,000 and 20,000 rows, single-threaded.
- `hmatoss_entries` makes two `get_values` calls for requests that cross the diagonal of symmetric storage. That is about the same total work, since each call computes only stored entries.
- Nothing from the brief is unfinished.

## 7. Requested changes to files I do not own

1. **`tests/run_tests.sh`**: done by the integrator (`tests/lsan.supp` applied by default).
2. **`GOTCHAS.md`**: done by the integrator. It has `hmat-epsilon`, `gemv-mutates-input`, `factorized-state` and `hmat-node-leak`, and both refinements: index 0 as an out-of-bounds read under `index-base`, and upper storage after `transpose` under `symmetric-storage`.
3. **`include/hmb/neutral_store.hpp`** (optional). `NeutralStore::alloc` zero-initializes, which is wasted for exporters that immediately overwrite the data (here, every value copied from the stream). An uninitialized variant would save one pass over memory. It is not needed for correctness.
