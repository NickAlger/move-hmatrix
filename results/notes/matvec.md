# Agent D (matvec): ButterflyPACK and H2Opus HARA adapters, the matvec route, the four-library chain

All checks pass: 11 programs (8 tests, 3 examples), from a clean build, under AddressSanitizer,
and against the fresh pinned builds (`deps/install/config.mk`). The adapters and the two
matvec examples are ported from experiment E2 (`/root/interop/matvec-rebuild`); the examples
reproduce its counts, errors and memory figures.

Second round (integrator's follow-up):
- ButterflyPACK **factorization and solves**: `bpack_factor`, `bpack_solve_blackbox`,
  `bpack_factor_stats`, tested by the new `test_bpack_solve` (45 checks).
- The **four-library chain** `examples/chain_four_libraries.cpp` (`mk/50-chain.mk`): Htool ->
  block copy -> H2Lib (H -> H2) -> matvecs -> ButterflyPACK HODLR, factorization -> solves ->
  H2Opus H² of the inverse. Final residual ‖K (H2Opus x) − x‖/‖x‖ = **1.1e-5** at N = 4,000
  (tolerances 1e-6), in 65 s on one thread (44-46 s on two).
- **Deviation from the request, by measurement:** the request was to feed HARA the solve with
  `assume_symmetric`. The solve's asymmetry is 1.9-2.0e-5, 19x the tolerance (the HODLR's own
  asymmetry, 6e-6, amplified by the inversion). With `assume_symmetric` the final residual is
  5.3e-5 and the ranks 225. The example therefore samples (S + Sᵀ)/2 through ButterflyPACK's
  transposed HODLR solve: residual 1.1e-5, ranks 174, less memory and time.
  `--assume-symmetric` runs the requested variant. New gotcha `inverse-asymmetry`.

New findings:
- **Round 1:**
  - ButterflyPACK's H matvec construction has a **heavy error tail**: 4 of 164 runs above 15x
    the tolerance, worst 56x. Both adapters measure `check_error` after every matvec
    construction.
  - HARA leaves **200 MB retained for a 27 MB matrix**. The adapter compacts the result.
  - HARA slows **6-17x when other jobs load the cores**. This is now handled by
    `OMP_WAIT_POLICY=passive` in `run_tests.sh`.
  - A ButterflyPACK **STOP inside the adapter ends the process with status 70**, not 0, after
    a diagnostic.
- **Round 2:**
  - **A solve of a symmetric matrix is not symmetric** (above). Symmetrize it before a
    symmetric-only consumer.
  - **ButterflyPACK's HODLR factorization is exact for the stored HODLR**: with low-rank
    blocks its updates are direct, and the residual against the rebuilt matrix is 1e-15. H-LU
    truncates at the tolerance (7-9e-7).
  - **ButterflyPACK has no transposed H-LU solve.** `inv_mult('T')` on H stops the process; the
    adapter refuses it.
  - **The inverse of K + 10 I is much less compressible than K**: HARA ranks 174 vs 31-33.
    Building the H² of the inverse took 56 s at N = 4,000, 85% of the chain, mostly in HARA's
    recompression. Applying it costs 3.2 source matvecs per vector, against 1.1 for
    ButterflyPACK's own solve: at this size, the last hop does not pay for itself.

## 1. Files created

| File | Contents |
|---|---|
| `adapters/butterflypack/hmb_bpack.hpp` | Neutral-facing API. Options and statistics; `bpack_from_matvec` / `bpack_from_entries` / `bpack_blackbox`; `bpack_factor` / `bpack_solve_blackbox` / `bpack_factor_stats`. Documents the conventions, measured costs and every gotcha. |
| `adapters/butterflypack/bpack_native.hpp` | `BpackMatrix` (ButterflyPACK handles, 0-based `new2old`, factorization state) and `bpack_mutex()`, with the rules for native ButterflyPACK calls. |
| `adapters/butterflypack/hmb_bpack.cpp` | Options and clustering; matvec and block-entry callbacks (ordering, 1-based indices, transposes, exception capture); thread-count guard; STOP diagnostics per activity (build, factor, multiply, solve); `check_error`; factor and solve. |
| `adapters/butterflypack/bpack_flush.f90` | 13-line Fortran helper that flushes Fortran output before the STOP handler's `_Exit`. |
| `adapters/h2opus/hmb_h2opus.hpp` | Neutral-facing API: `h2opus_from_matvec` (HARA) and `h2opus_blackbox`. |
| `adapters/h2opus/h2opus_native.hpp` | `H2opusMatrix` (`HMatrix`, H2Opus handle, `new2old`, product lock). |
| `adapters/h2opus/h2opus_hara.cpp` | Structure from the points; sampler in H2Opus's order; norm estimate; HARA; compaction; `hgemv` BlackBox; `check_error`. |
| `tests/test_bpack_h2opus_common.hpp` / `.cpp` | Test helpers: an exact dense kernel BlackBox (own translation unit: its `dgemm_` prototype clashes with H2Opus's) and a call-counting wrapper. |
| `tests/test_bpack_matvec.cpp` | Matvec route from dense sources (symmetric and nonsymmetric): accuracy, transposes, statistics, ordering, threads. |
| `tests/test_bpack_entries.cpp` | Entry route: accuracy, 0-based indices, entry counts, serialized calls, ordering. |
| `tests/test_bpack_solve.cpp` | **New.** Factorization and solves: residuals against the kernel and against the rebuilt matrix, forward product unchanged, `keep_forward = false`, transposed solves, statistics, lifetime, refusals. |
| `tests/test_bpack_safety.cpp` | Thread count (raw vs adapter), `keep_forward`, native solve after `bpack_factor`, source exceptions, nested ButterflyPACK use, validation. |
| `tests/test_bpack_stop.cpp` | Fortran STOP diagnostic, in child processes. |
| `tests/test_bpack_htool.cpp` | Htool sources (`'S'` and nonsymmetric `'N'`) into HODLR and H. |
| `tests/test_h2opus_hara.cpp` | HARA from dense sources: accuracy, statistics, norm, permutation, ranks, exact symmetry, retained memory, refusals, exceptions, concurrent products. |
| `tests/test_h2opus_hara_htool.cpp` | Htool sources into HARA. |
| `examples/matvec_htool_bpack.cpp` | Port of `drv_bp`: Htool -> HODLR and H (at tol and tol/10), with entry-route baselines. |
| `examples/matvec_htool_h2opus.cpp` | Port of `drv_h2` (HARA part). |
| `examples/chain_four_libraries.cpp` | **New.** The four-library chain (section 4); options `--n --eps --tol --assume-symmetric`; neutral-facing headers only. |
| `mk/20-butterflypack.mk` | `BPACK_OBJS`, the Fortran object rule, 5 single-library tests. |
| `mk/20-h2opus.mk` | `H2OPUS_HARA_OBJS`, 1 test. |
| `mk/50-matvec.mk` | 2 Htool-source tests and 2 examples. |
| `mk/50-chain.mk` | **New.** The chain example, built when all four libraries are configured. |
| `results/notes/matvec.md` | This report. |

## 2. Public API (namespace `hmb`)

```cpp
enum class BpackFormat { HODLR, H };
struct BpackOptions { BpackFormat format = HODLR; double tolerance = 1e-6; int leaf_size = 64;
  double near_para = 2.0; int rank0 = 64; double rankrate = 2.0; int itermax = 10;
  bool keep_forward = true; int check_vectors = 4; int verbosity = -1; int stop_exit_status = 70; };
struct BpackStats { long calls, calls_transposed; long long vectors, vectors_transposed;
  double check_error; long entry_calls; long long entries; double seconds_init, seconds_build,
  seconds_callback, seconds_in_source; int rank_max; double bpack_mem_fill_mb; };
BpackHandle bpack_from_matvec(const BlackBox &A, int dim, const double *coords, const BpackOptions & = {}, BpackStats * = nullptr);
BpackHandle bpack_from_entries(const EntrySource &A, int dim, const double *coords, const BpackOptions & = {}, BpackStats * = nullptr);
BlackBox bpack_blackbox(const BpackHandle &);
int bpack_size(const BpackHandle &);
BpackStats bpack_stats(const BpackHandle &);
const char *bpack_format_name(BpackFormat);
// new in round 2
struct BpackFactorStats { bool factored; double seconds; int rank_max; double bpack_mem_factor_mb; };
void bpack_factor(const BpackHandle &A, BpackFactorStats *stats = nullptr);
BlackBox bpack_solve_blackbox(const BpackHandle &A);
BpackFactorStats bpack_factor_stats(const BpackHandle &A);
// native (bpack_native.hpp)
struct BpackMatrix { F2Cptr bmat, option, stats, msh, ker, ptree; int n, nloc; std::vector<int> new2old;
  BpackOptions options; BpackStats build_stats; BpackFactorStats factor_stats; bool forward_deleted; };
std::recursive_mutex &bpack_mutex();

struct H2opusOptions { double tolerance = 1e-6; int leaf_size = 64; double eta = 1.0; int max_samples = 512;
  int block_size = 32; int converged_samples = 10; int norm_iterations = 10; bool assume_symmetric = false;
  int check_vectors = 4; bool verbose = false; };
struct H2opusStats { long calls; long long vectors; long norm_calls; double norm_estimate, abs_tolerance,
  seconds_structure, seconds_norm, seconds_hara, seconds_callback, seconds_in_source, memory_mb;
  int depth, max_rank; double check_error; };
H2opusHandle h2opus_from_matvec(const BlackBox &A, int dim, const double *coords, const H2opusOptions & = {}, H2opusStats * = nullptr);
BlackBox h2opus_blackbox(const H2opusHandle &);
int h2opus_size(const H2opusHandle &);
H2opusStats h2opus_stats(const H2opusHandle &);
// native (h2opus_native.hpp)
struct H2opusMatrix { HMatrix H; h2opusHandle_t handle; std::vector<int> new2old; std::mutex lock;
  H2opusOptions options; H2opusStats build_stats; };
```

What each function does:
- **`bpack_from_matvec`** rebuilds a BlackBox operator (original ordering, `coords` dim x n,
  point-major) as a ButterflyPACK HODLR or H by randomized sampling. A^T requests go to
  `apply('T')`, or to `apply('N')` when the BlackBox is symmetric. `check_error` is the relative
  error against the source on `check_vectors` Gaussian vectors (one extra source call).
- **`bpack_from_entries`** rebuilds from an EntrySource with ButterflyPACK's block callback
  (`elem_extract = 2`), translating its 1-based indices.
- **`bpack_blackbox`** multiplies in the original ordering, `'N'` and exact `'T'`; `symmetric`
  is false. It throws `std::logic_error` when there is no forward operator: an H built with
  `keep_forward = false`, or a HODLR built so and then factored. In the second case, BlackBoxes
  made earlier throw too.
- **`bpack_factor`** (new) factors in place: HODLR by ButterflyPACK's recursive inversion, H by
  H-LU, on matrices built either way. A second call does nothing. It runs under the mutex,
  the thread-count guard and the STOP diagnostic ("factors a HODLR matrix"). With the default
  `keep_forward`, `bpack_blackbox` gives bit-identical products before and after.
- **`bpack_solve_blackbox`** (new) applies A⁻¹ in the original ordering through
  ButterflyPACK's `inv_mult`.
  - It factors first if needed. I chose that over throwing, because the factorization is
    idempotent and cheap next to the construction.
  - `'T'` applies A⁻ᵀ for HODLR. For H, `apply('T')` throws `std::logic_error`: ButterflyPACK
    has no transposed H-LU solve and would stop the process.
  - `symmetric` is false even for a symmetric A (`inverse-asymmetry`). The BlackBox keeps the
    matrix alive.
- **`bpack_factor_stats`** (new) returns `factored`, time, ButterflyPACK's `Rank_max` (matrix
  and factors) and `Mem_Factor`.
- **`h2opus_from_matvec`** builds a symmetric H² with HARA on a KD-tree of the points.
  - The relative tolerance becomes HARA's absolute one through a sampled ‖A‖₂ estimate.
  - A BlackBox not flagged symmetric is refused unless `assume_symmetric` is set. Its doc now
    says to symmetrize an operator that is symmetric only up to errors above the tolerance.
  - `block_size` must be 16 or 32.
- **`h2opus_blackbox`** multiplies with `hgemv`. `symmetric` is true (exactly symmetric storage).
- **Errors:** `std::invalid_argument`, `std::runtime_error` or `std::logic_error`, with the
  message prefixed by the function name.
- **Source exceptions** never cross library frames. They are caught, the remaining samples get
  zeros, and the exception is rethrown as `runtime_error` once the library returns.

## 3. Checks

Final run from a clean build (`rm -rf build-matvec && make B=build-matvec -j2 && make
B=build-matvec check HMB_TEST_FILTER='^(test_bpack|test_h2opus_hara|matvec_htool|chain_four)'`).
The full `make all` built warning-free; the only linker notes are ButterflyPACK's
executable-stack warnings. Runner defaults: 2 threads, passive wait. The machine was quiet
(load 0.0-0.5):

```
check                                        result  seconds  detail
test_bpack_matvec                            PASS        6.0  test_bpack_matvec: 44 checks; n=2000 tol=1e-06
test_bpack_entries                           PASS        2.9  test_bpack_entries: 33 checks; n=2000 tol=1e-06
test_bpack_solve                             PASS        7.6  test_bpack_solve: 45 checks; n=2000 tol=1e-06
test_bpack_safety                            PASS        5.4  test_bpack_safety: 23 checks; n=1500
test_bpack_stop                              PASS        0.4  test_bpack_stop: 10 checks
test_h2opus_hara                             PASS        5.1  test_h2opus_hara: 29 checks; n=2000 tol=1e-06
test_bpack_htool                             PASS        5.8  test_bpack_htool: 18 checks; n=2000 tol=1e-06
test_h2opus_hara_htool                       PASS        3.5  test_h2opus_hara_htool: 7 checks; n=2000 tol=1e-06
chain_four_libraries                         PASS       44.9  chain_four_libraries: 11 checks; n=4000 eps=1e-06 tol=1e-06: final res
matvec_htool_bpack                           PASS       19.3  matvec_htool_bpack: 8 checks; n=4000 tol=1e-06
matvec_htool_h2opus                          PASS        6.9  matvec_htool_h2opus: 4 checks; n=4000 tol=1e-06
passed 11, failed 0, filtered out 15
```

- **AddressSanitizer** (`build-matvec-asan`, `-O1 -g -fsanitize=address`, leak checks on): all
  8 tests PASS, and the chain at `--n 2000` PASS (final residual 3.0-3.4e-6, two runs), with no reports.
  ButterflyPACK and H2Opus themselves are prebuilt, not instrumented.
- **Fresh pinned builds** (`CONFIG=deps/install/config.mk`, `build-matvec-deps`): all 11 PASS
  (chain 44.7-45 s, final residual 1.12-1.21e-5).

What each check verifies (n = 2,000, tolerance 1e-6 unless stated):

**`test_bpack_solve`** (new; kernel K + 10 I, dense exact source D; HODLR and H, from matvecs
and from entries):

| | factor s (build s) | rank / Mem_Factor | forward e vs K | residual vs K (bound 20 (e + tol)) | vs B (bound 10 tol) | 'T' |
|---|---|---|---|---|---|---|
| HODLR matvec | 0.055 (1.15) | 170 / 18.7 MB | 1.18e-6 | 3.07e-6 (2.6 e) | 1.4e-15 | 3.12e-6 |
| H matvec | 0.248 (1.29) | 24 / 15.8 MB | 4.56e-6 | 9.22e-6 (2.0 e) | 7.1e-7 | refused |
| HODLR entries | 0.053 (0.68) | 146 / 17.1 MB | 1.78e-6 | 4.50e-6 (2.5 e) | 1.6e-15 | 4.50e-6 |
| H entries | 0.235 (0.66) | 22 / 15.5 MB | 3.2e-7 | 1.06e-6 (3.3 e) | 8.5e-7 | refused |

- **Residuals and statistics.** Residuals use the source's product, not ButterflyPACK's. Each
  configuration also checks: not factored before; statistics after; a second `bpack_factor`
  does nothing.
- **Forward product unchanged** by the factorization, `'N'` and `'T'`: change 0.0 (bound 1e-14).
- **Thread count** kept through factorization and solves.
- **`keep_forward = false`:**
  - HODLR: products refused after the factorization, through the old BlackBox and a new one;
    the solve still works (4.5e-6).
  - H: products refused, yet it factors and solves (1.07e-6, bound 20 x 2 tol).
- **On-demand factorization:** `bpack_solve_blackbox` factors an unfactored matrix. The solve
  keeps working after the handle is released.
- **Refusals:** null handles, and `trans` other than N/T.

**`test_bpack_safety`** (n = 1,500): the raw `construct_init` resets the thread count (3 -> 2);
through the adapter it survives constructions, products and destruction.
- **`keep_forward = false`:** H's `bpack_blackbox` throws instead of segfaulting; HODLR still
  multiplies.
- **Default `keep_forward`:** HODLR multiplies after `bpack_factor` (1.6e-6). Native
  `d_c_bpack_solve` through `new2old` recovers x (9.9e-16, bound 10 tol).
- **Source exceptions** come back as `runtime_error`, with no further source calls.
- **Nested use** (an H built from a HODLR's BlackBox), and 6 invalid inputs refused.

**`chain_four_libraries`** (11 checks, all relative to the tolerances):
- **Lossless and compression hops:**
  - Htool vs kernel ≤ 10 eps;
  - export zero-copy;
  - H2Lib copy vs Htool ≤ 512 rounding units;
  - H2 vs H ≤ 10 eps;
  - HODLR vs H2 ≤ 20 tol;
  - every A^T request answered by H2Lib's `'T'` (17 of 17).
- **Solve:** residual vs HODLR ≤ 10 tol; residual vs K ≤ 10 (a e4 + r_B).
  - a = (‖KX‖/‖X‖)(‖SX‖/‖X‖) is measured on random vectors: 2.75 at N = 2,000, 3.96 at 4,000.
  - The asymmetry stays ≤ 5 (a e4 + tol): the skew part is S's error, so removing it removes
    no signal.
- **H2Opus:** error vs (S + Sᵀ)/2 ≤ 20 tol (with `--assume-symmetric`: vs S, ≤ 20 tol + 2 x the
  asymmetry); final residual ≤ 10 (r_T + a e6).

**Unchanged since round 1** (values in the first report, all still PASS):
- `test_bpack_matvec`: HODLR 1.1-1.3e-6 vs bound 20 tol, H 3.2-5.5e-6 vs bound 100 tol; checks
  `'T'` routing, counts, ranks and threads.
- `test_bpack_entries`: 1.4-2.0e-6 (HODLR), 3.1-3.5e-7 (H) vs bound 10 tol.
- `test_bpack_stop`: status 70 and the diagnostic; status 0 when asked; a normal child prints
  no diagnostic.
- `test_bpack_htool`, `test_h2opus_hara` (1.36e-6, exact symmetry, retained memory, refusals,
  exceptions, concurrency), `test_h2opus_hara_htool`.
- The two matvec examples.

## 4. Measured costs (for RESULTS.md)

### The four-library chain, N = 4,000, `OMP_NUM_THREADS=1`

Setup:
- **Problem and tolerances:** K = 1/(0.1 + r) + 10 I on random sphere points; eps = tol =
  1e-6.
- **Libraries:** Htool full storage, eta 10, leaf 64; H2Lib serial; ButterflyPACK HODLR,
  leaf 64, rank0 64, rankrate 2; H2Opus leaf 64, eta 1.
- **Cost unit:** one single-vector Htool matvec, 2.73 ms (mean of the start and end timings).
- **Log:** `.../scratchpad/chainx/chain_n4000_omp1.log`.

| hop | route | s | x 1 source matvec | into the hop | MB | vs previous hop | vs kernel |
|---|---|---|---|---|---|---|---|
| 1 | Htool-DDM: ACA from the kernel's entries | 0.076 | 28 | 5.60 M entries (0.35 N²) | 44.8 | - | 7.9e-7 |
| 2 | block copy Htool -> H2Lib (strict import) | 0.022 | 8 | 5.60 M values copied (1.00x; export aliases all) | 45.0 | 9.5e-16 | 7.9e-7 |
| 3 | H -> H2 inside H2Lib | 0.220 | 81 | basis ranks ≤ 31 | 26.2 | 4.0e-7 | 8.8e-7 |
| 4 | matvecs H2Lib H2 -> ButterflyPACK HODLR | 8.26 | 3,028 | 4,076 vectors (1,998 A^T), 52 calls | 35.8 | 4.5e-6 | 4.6e-6 |
| 5 | ButterflyPACK factorization | 0.197 | 72 | - | 52.9 | residual vs HODLR 2.2e-15 | residual 1.55e-5 |
| 6 | solves -> H2Opus H² (HARA on (S + Sᵀ)/2) | 56.1 | 20,569 | 3,025 vectors (20 for the norm), 115 calls | 108.6 | 9.7e-6 vs S (6.2e-8 vs (S + Sᵀ)/2) | **final residual 1.13e-5** |

- **Total:** 64.9 s = 23,785 source matvecs.
- **Hop 4:** 6.5 of its 8.3 s are spent in H2Lib's products, which H2Lib applies one vector at
  a time (1.6 ms each). `check_error` was 3.6e-6.
- **Hop 5:** the solve's asymmetry is 1.93e-5. Amplification a = 3.96.
- **Hop 6:** HARA ranks reach 174 (K's H² bases: 31). ButterflyPACK's solves take 6.0 of the
  56 s (two solves per sample); the rest is HARA's recompression. Symmetrizing alone lowers the
  residual from 1.55e-5 (S) to 1.13e-5 ((S + Sᵀ)/2).
- **Applying A⁻¹ to one vector:** H2Opus 8.8 ms (3.2 source matvecs); ButterflyPACK's solve
  3.05 ms (1.1).
- **Two threads** (the `make check` run): 43.7-46 s; hop 6 takes 35 s; final residual
  1.15-1.21e-5.
- **`--assume-symmetric`** (one thread): hop 6 62.7 s, rank 225, 129.4 MB, error vs S 1.85e-5,
  **final residual 5.3e-5**, total 72.1 s.
- **Same comparison at N = 2,000** (one thread, experiment `scratchpad/chainx/symx.cpp`):
  - symmetrized: residual 2.95e-6, rank 100, 8.1 s;
  - `assume_symmetric`: residual 8.0e-6, rank 114, 8.2 s.

### ButterflyPACK factorization (n = 2,000, 2 threads; `test_bpack_solve`)

Factorization takes 5-36% of the construction time: HODLR 0.05 s, H 0.24 s, against
construction times of 0.7-1.3 s. In the chain at N = 4,000 the HODLR factorization took
0.20 s, against 8.3 s for its construction from matvecs. Residuals are in section 3.

### Round 1 (unchanged; one thread, E2's setup)

**N = 4,000**: source matvec 3.32 ms single vector, 0.47 ms per vector in blocks of 64.

| Construction | calls (A^T) | vectors / entries | s | x 1 src matvec | err vs src | err vs kernel | rank | retained MB | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| HODLR matvec 1e-6 | 52 (17) | 4,072 | 3.37 | 1,014 | 4.4e-6 | 4.5e-6 | 233 | 37.8 | 119 |
| H matvec 1e-6 | 260 (116) | 7,254 | 4.60 | 1,385 | 6.7e-6 | 6.7e-6 | 17 | 71.3 | 77 |
| H matvec 1e-7 | 199 (85) | 6,145 | 4.44 | 1,337 | 1.05e-6 | 1.3e-6 | 24 | 83.1 | 88 |
| HODLR entries 1e-6 | - | 15.6 M (0.97 N²) | 1.89 | 567 | 2.5e-6 | 2.4e-6 | 198 | 34.3 | 83 |
| H entries 1e-6 | - | 40.5 M (2.53 N²) | 1.73 | 521 | 9.1e-7 | 4.3e-7 | 18 | 73.6 | 73 |
| H² HARA 1e-6 | 75 (0; 8 norm) | 2,065 | 8.78 | 2,750 | 2.1e-6 | 2.2e-6 | 33 | 26.7 | 241 |

**N = 16,000** (one run each, loaded machine: times 20-40% above E2's):

| Construction | calls (A^T) | vectors / entries | s | err vs src | err vs kernel | rank | retained MB | peak MB | E2 (s, err, peak) |
|---|---|---|---|---|---|---|---|---|---|
| HODLR matvec 1e-6 | 70 (23) | 8,358 | 41.6 | 5.9e-6 | 6.0e-6 | 447 | 280 | 845 | 40.6, 7.4e-6, 849 |
| H matvec 1e-6 | 429 (201) | 10,837 | 51.7 | 7.6e-6 | 7.7e-6 | 19 | 374 | 390 | 42.0, 7.4e-6, 387 |
| H matvec 1e-7 | 364 (168) | 10,976 | 66.6 | 1.4e-6 | 1.8e-6 | 25 | 457 | 472 | 50.5, 3.2e-6, 466 |
| HODLR entries 1e-6 | - | 107.6 M (0.42 N²) | 42.8 | 3.7e-6 | 3.6e-6 | 349 | 240 | 420 | 24.2, 3.7e-6, 382 |
| H entries 1e-6 | - | 259.9 M (1.02 N²) | 20.9 | 1.2e-6 | 5.0e-7 | 21 | 391 | 391 | 13.7, 1.2e-6, 391 |
| H² HARA 1e-6 | 141 (0; 10 norm) | 4,173 | 143.9 | 4.6e-6 | 4.7e-6 | 54 | 131 | 978 | 105.2, 4.6e-6, 977 |

The integrator's quiet-machine reruns are in `results/logs/package/`.

ButterflyPACK's matvec-route accuracy (tolerance 1e-6, n = 1,500-2,000):
- HODLR: 88 runs at 0.6-1.6x the tolerance.
- H: 164 runs; median 4x, 90th percentile 8x, worst four 15.0, 16.9, 35.5 and 56.3x, all on
  the shifted matrix.

## 5. Gotchas

| id | What happens | Where handled | Test |
|---|---|---|---|
| `index-base` | ButterflyPACK's new2old and its entry callbacks are 1-based. | `hmb_bpack.cpp:395` (conversion, validation), `:273` (block entries, range checks) | `test_bpack_entries` (0-based indices); every accuracy check; native solve through `new2old` |
| `perm-direction` | ButterflyPACK and H2Opus sample, multiply and solve in their own order (new2old). | `hmb_bpack.cpp:192` (callback), `apply()` `:426` (products and solves); `h2opus_hara.cpp:39` | accuracy with random and shuffled numberings; rank bounds; residuals of solves |
| `bpack-transpose` | ButterflyPACK's matvec construction asks for AᵀX in a third to half of its calls. | `hmb_bpack.cpp:200` | `test_bpack_matvec`, `test_bpack_htool`; the chain counts H2Lib's `'T'` products (17 of 17) |
| `tolerance-stacking` | The matvec route misses its tolerance (HODLR 1-7x, H 4-8x, tail to 56x). | `hmb_bpack.hpp:42`; `check_error` at `hmb_bpack.cpp:462/:528` and `h2opus_hara.cpp:118/:209` | `check_error` agreement; bounds 20 tol (HODLR, HARA) and 100 tol (H) |
| `fortran-stop` | ButterflyPACK ends the process with STOP, status 0: on non-convergence, internal errors, and a transposed H-LU solve. | `hmb_bpack.cpp:46-122`; StopGuards around constructions (`:510`, `:542`), factorization (`:583`), products and solves (`:446`); H `'T'` solve refused beforehand (`apply()`) | `test_bpack_stop`; `test_bpack_solve` (H `'T'` refused) |
| `bpack-global-state` | `construct_init` sets the thread count to $OMP_NUM_THREADS and reseeds Fortran's RNG. | `ThreadCountGuard` (`hmb_bpack.cpp:32`) around every call: `:358, :405, :447, :493, :520, :551, :586, :593` | `test_bpack_safety`; thread checks in `test_bpack_matvec` / `_entries` / `_solve` |
| `bpack-errsol` | H keeps no forward operator unless ErrSol = 1; HODLR's factorization deletes it. Products then segfault. | `keep_forward` -> ErrSol (`hmb_bpack.cpp:380`); `forward_missing()` (`:414`) refuses H without it and a factored HODLR without it (`:439`, `:566`); `forward_deleted` set in `bpack_factor` (`:591`) | `test_bpack_safety`; `test_bpack_solve` (products refused after factorization, solves still work, forward product unchanged with `keep_forward`) |
| `mpi-stub` | The sequential build's stub defines Fortran MPI symbols; `dBPACK_wrapper.h` #defines `MPI_Fint`. | `hmb_bpack.hpp:72`, `bpack_native.hpp`; `comm = 0` (`hmb_bpack.cpp:360`); `nloc != n` and non-1x1 grids are errors | not testable without MPI |
| `symmetric-storage` | HARA samples only AX and builds a symmetric H²; a nonsymmetric operator gives a silent wrong matrix (144%). | refused unless `assume_symmetric` (`h2opus_hara.cpp:147`) | `test_h2opus_hara`, `_htool` |
| `scalar-type` | H2Opus's precision is fixed at build time. | `static_assert` (`h2opus_hara.cpp:17`) | compile time |
| `header-clash` | Htool's and H2Opus's BLAS prototypes clash, and H2Lib's macros break other headers. | separate translation units; the chain includes neutral-facing headers only (`chain_four_libraries.cpp:57`) | the chain and the Htool-source tests compile |
| `h2opus-workspace` | After HARA, the handle keeps HARA's workspace: 203 MB retained for a 27 MB H². | compaction (`h2opus_hara.cpp:199`) | `test_h2opus_hara` (retained memory) |
| **`inverse-asymmetry` (new)** | An approximate solve of a symmetric A is not symmetric. The HODLR is symmetric only up to its errors (6e-6), and the inversion amplifies them: 1.9-2.0e-5 at N = 4,000, tolerance 1e-6. HARA with `assume_symmetric` cannot fit the skew part: its result is off by about the asymmetry, giving a 4.7x larger residual (5.3e-5 vs 1.13e-5) and 30% higher ranks. | documented at `hmb_bpack.hpp:95` and on `bpack_solve_blackbox`, and in `hmb_h2opus.hpp:25-30`; the chain samples (S + Sᵀ)/2 through the transposed HODLR solve (`chain_four_libraries.cpp:102`, `:258`) | chain checks: asymmetry ≤ 5 (a e4 + tol); H2Opus vs (S + Sᵀ)/2 ≤ 20 tol (6.2e-8); `--assume-symmetric` for the comparison |

Other quirks (no code id):
- **HARA under CPU contention.** Many short OpenMP regions; up to 17x slower on a loaded machine
  (`hmb_h2opus.hpp`). The integrator applied passive waiting in the runner.
- **`Mem_Fill`** leaves out the low-rank blocks of H built from matvecs.
- **Entry callbacks.** The block callback gives the same results with about 1,000x fewer calls
  than the single-entry one, which ButterflyPACK calls concurrently.
- **H2Opus asserts** `block_size ∈ {16, 32}`.
- **The HARA-built H² is exactly symmetric.**
- **HODLR factorization with low-rank blocks is direct.** `LR_Sblock` and `LR_minusBC` do no
  sampling, so it cannot fail to converge, and its residual against the stored matrix is 1e-15.
- **ButterflyPACK's factor runs a test solve** (`BPACK_Test_Solve_error`, one random vector)
  when ErrSol = 1, i.e. with `keep_forward`. The cost is included in
  `BpackFactorStats::seconds`.
- **H2Lib's products are per vector** (`hmb_h2lib_h2_matvec` loops over columns). For the
  chain's 4,000 sampling vectors, that is 79% of hop 4. A blocked H² product would speed up the
  matvec route from H2Lib; that is agent B's file, so this is only an observation.

## 6. Limitations and anything unfinished

- **ButterflyPACK formats:** HODLR and H with low-rank blocks only. BLR from matvecs exists in
  ButterflyPACK but is not exposed (untested); butterfly, HSS-BF and H² cannot be built from
  matvecs.
- **Factorization:**
  - No transposed solve for H (refused).
  - The log-determinant, which ButterflyPACK computes during factorization, is not exposed; call
    `d_c_bpack_logdet` through `bpack_native.hpp`.
  - ButterflyPACK's symmetric HODLR factorization (`option%sym`) is not used.
- **No MPI:** sequential ButterflyPACK only.
- **H2Opus:** CPU, real double, symmetric operators only, one process.
- **Run-to-run variation:** ButterflyPACK reseeds from the clock, so counts, ranks and errors
  vary between runs. The chain's final residual varied 1.12-1.21e-5 over 5 runs.
- **STOP:** the diagnostic covers every adapter call into ButterflyPACK, not native calls.
- **Chain time:** dominated by HARA rebuilding the inverse (ranks 174). At N = 4,000 it is about
  1 minute on one thread; it grows quickly with N. It was not run at 16,000: HARA alone took
  144 s on K there, and at N = 4,000 the inverse's ranks were 5x K's.

## 7. Requested changes to files I do not own

Round 1's requests 1 (runner) and 2 (LAYOUT.md) were applied by the integrator. Round 1's
request 3 (GOTCHAS.md) was applied too.

1. **`GOTCHAS.md`: add `inverse-asymmetry`** (silent wrong answers), and a summary row:
   `| [inverse-asymmetry](#inverse-asymmetry) | ButterflyPACK -> H2Opus (any approximate solve) | 4.7x larger residual | symmetrized in the chain example |`

   ```markdown
   ### inverse-asymmetry
   - **What happens.** An approximate solve of a symmetric matrix is not symmetric. A HODLR
     rebuilt from matvecs is symmetric only up to its errors (6e-6 at tolerance 1e-6,
     N = 4,000), and inverting amplifies them: ‖(S − Sᵀ)x‖/‖Sx‖ = 1.9-2.0e-5 for
     ButterflyPACK's solve. HARA (symmetric only), told to assume symmetry, cannot fit the
     skew part: its H² is off by about the asymmetry, with a 4.7x larger residual
     (5.3e-5 vs 1.1e-5) and 30% higher ranks.
   - **Package.** `bpack_solve_blackbox` is flagged nonsymmetric. `examples/chain_four_libraries.cpp`
     samples (S + Sᵀ)/2 through the transposed HODLR solve; `--assume-symmetric` shows the
     difference. Documented in `hmb_bpack.hpp` and `hmb_h2opus.hpp`.
   - **Tests.** The chain's checks (asymmetry within S's own error; HARA within 20 tol of
     (S + Sᵀ)/2: 6.2e-8).
   ```

2. **`GOTCHAS.md`: update two entries.**
   - `fortran-stop`, Package: "While a ButterflyPACK construction runs" -> "While the adapter
     is inside ButterflyPACK (construction, factorization, product, solve)". Append: "A
     transposed solve with an H matrix also STOPs; `bpack_solve_blackbox` refuses it."
     Tests: add `test_bpack_solve`.
   - `bpack-errsol`, Package: append "`bpack_factor` records the factorization; with
     `keep_forward = false`, products of a factored HODLR throw `std::logic_error`." Tests:
     add `test_bpack_solve`.

3. **`RESULTS.md`:** the chain table and the factorization numbers in section 4.
