# HiCMA and HiCMA-PaRSEC deep dive

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects HiCMA 1.0.0, HiCMA-PaRSEC at 366f7e0, STARS-H at c115e75 and TLR-MVM at 9c86f91; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

The KAUST stack provides distributed, task-based tile low-rank (TLR) arithmetic: a uniform grid of tiles, dense on the diagonal or a band and low-rank elsewhere, with no cluster tree and no nesting. It targets SPD kernel and covariance matrices whose tiles come from a kernel callback through STARS-H. Its best-supported operation is TLR Cholesky at very large scale.

HiCMA 1.0.0 (StarPU, CPU only, January 2023) also has TLR×TLR multiplication, left-lower triangular solves and a complex TLR LU that discards its pivots. HiCMA-PaRSEC 0.1.0 (PaRSEC and DPLASMA, November 2025) offers only Cholesky, in FP64, mixed-precision band-dense and sparse (zero-rank-pruned) variants. Its CUDA tasks run only where the output tile is dense.

Neither has a matvec: STARS-H and TLR-MVM provide one, but only on their own objects. There is no TLR addition, inversion, matrix function, LDLᵀ, QR or eigensolver, no construction from sparse matrices or matvecs, and no conversion to hierarchical formats. The largest TLR runs are CPU-only; the exascale Gordon Bell runs used dense mixed-precision Cholesky.

## Snapshot

Only HiCMA-PaRSEC is still developed, mostly in unreleased branches; HiCMA, STARS-H and HCORE have been quiet since 2023, as have KAUST’s C++ rewrites. [HiCMA++](https://github.com/ecrc/hicmapp) 1.0.0 (Nov 2023) runs TLR GEMM and Cholesky on StarPU and MPI. [HCore++](https://github.com/ecrc/hcorepp) 3.0.0 (Nov 2023) has TLR tile GEMM and POTRF with CUDA and SYCL backends.

| Component | Role | Latest release | Last commit | Commits since 2024 |
| --- | --- | --- | --- | --- |
| [HiCMA](https://github.com/ecrc/hicma) | TLR linear algebra on StarPU | 1.0.0 (17 Jan 2023) | 17 Jan 2023 | 0 |
| [HiCMA-PaRSEC](https://github.com/ecrc/hicma-x) | TLR and mixed-precision Cholesky on PaRSEC | 0.1.0 (12 Nov 2025) | 18 Mar 2026 | 13 (history squashed at 0.1.0) |
| [STARS-H](https://github.com/ecrc/stars-h) | Tile generation and compression | 0.3.1 (11 Dec 2022) | 20 Nov 2023 | 0 |
| [HCORE](https://github.com/ecrc/hcore) | Low-rank tile kernels | 0.1.3 (19 Dec 2022) | 19 Dec 2022 | 0 |
| [TLR-MVM](https://github.com/ecrc/tlrmvm) | TLR matvec on CPUs and GPUs | 0.0.2 (27 Nov 2022) | 10 Dec 2024 | 2 |
| [ExaGeoStat](https://github.com/ecrc/exageostat) | Main application | 1.2.1 | 25 Mar 2025 | 4 |

| Item | Value |
| --- | --- |
| Licence | BSD-3-Clause for every component |
| Languages | C; HiCMA-PaRSEC adds PaRSEC task-graph files and CUDA/HIP kernels; STARS-H has some Fortran; TLR-MVM is C++ with CUDA/HIP and Python bindings. HiCMA and HiCMA-PaRSEC have no Python or Julia bindings |
| Dependencies | HiCMA: StarPU 1.2 or later with hwloc, STARS-H, HCORE, CBLAS/LAPACKE; optional MPI and GSL. Its Chameleon dependency was removed in 1.0.0 and the code vendored. HiCMA-PaRSEC: MPI, BLAS/LAPACKE, Fortran, and submodules pinned to personal forks of DPLASMA with PaRSEC, STARS-H and HCORE; optional CUDA 12 or ROCm, CUTLASS, GSL |
| Maintainers | KAUST’s Extreme Computing Research Center (Ltaief, Keyes, Alomairy, Abdulah; formerly Akbudak, Mikhalev) with UTK ICL (Bosilca, Dongarra). HiCMA-PaRSEC is led by Qinglei Cao, now at Saint Louis University |
| Documentation | Fair to poor. HiCMA’s Doxygen dates from April 2019. HiCMA-PaRSEC documents only test-driver parameters, and its TODO list includes “Better API” and “Fix hicma\_parsec\_init for external users” |
| Users | ExaGeoStat (TLR maximum-likelihood estimation through `HICMA_dpotrf_Tile` and `HICMA_dtrsmd_Tile`, with its own log-det and kriging) and ExaGeoStatCPP; Gordon Bell finalist and prize papers at SC22, SC23 and SC24 |

## Construction

Every TLR matrix is generated tile by tile from a kernel callback and compressed per tile by randomized SVD. There is no cluster tree, sparse input or matvec-based construction.

| Access | Formats | Algorithm | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | ✗ | — | — | “Sparse” in HiCMA-PaRSEC means a TLR matrix whose zero-rank tiles are pruned from the task graph |
| Block callback | HiCMA TLR (real and complex double); STARS-H BLR; HiCMA-PaRSEC ◐ | Each tile is formed as a dense block, then compressed by randomized SVD (one Gaussian sketch of max rank + 10, no power iterations) at a fixed accuracy with a rank cap. STARS-H standalone also offers SVD and RRQR; its “CROSS” engine is an alias of randomized SVD | HiCMA: a `STARSH_kernel` callback → `starsh_problem_new` → `starsh_blrf_new_tlr` → `HICMA_set_starsh_format` → `HICMA_dgytlr_Tile` (or `zgytlr`). HiCMA-PaRSEC: `hicma_parsec_matrix_generation` | O(N²) kernel evaluations, since every tile is formed in full. A tile above the rank cap aborts the run in HiCMA (via exit(0), which reports success) and, off the band, in HiCMA-PaRSEC. Only standalone STARS-H stores it as a dense tile. HiCMA-PaRSEC has no documented way to add a user kernel |
| Kernel + coordinates | Same | Same | STARS-H built-ins: spatial statistics (exponential, squared exponential, Matérn in 1–4D and nD, space-time, bivariate), Coulomb, electrodynamics, Cauchy, random TLR, 3D acoustic BEM (complex), RBF mesh deformation. HiCMA-PaRSEC: 16 problem kinds, 14 built on these (the space-time ones from its STARS-H fork) and two genomics kernel-ridge-regression (KRR) kinds | Points are generated (random, uniform or quasi-uniform) and Morton-sorted, or read from files |
| Matvecs only | ✗ | — | — | Randomized SVD always sketches an explicit dense tile |
| Dense matrix | STARS-H ✓; HiCMA ◐; TLR-MVM ✓; HiCMA-PaRSEC ◐ | STARS-H wraps an array as a callback (`starsh_problem_from_array`), which HiCMA can consume. TLR-MVM’s Python tool runs a per-tile SVD truncated relative to ‖A‖\_F and writes factor files | — | HiCMA has no routine to compress an existing descriptor; HiCMA-PaRSEC does so only in its genomics compile mode |
| Other (HiCMA-PaRSEC) | Dense kernel-ridge-regression and Hamming matrices | Distributed INT8/FP16 SYRK, then a Gaussian transform | — | Band-size auto-tuning regenerates near-diagonal tiles densely from the kernel |

- **No cluster tree.** Indices are split into contiguous blocks; the H and HODLR options in STARS-H are commented out. Ordering the points is the user’s job, and built-in generators use Morton order.
- **Implicit admissibility.** Every off-diagonal tile (HiCMA) or off-band tile (HiCMA-PaRSEC) is low-rank. STARS-H alone accepts arbitrary near and far tile lists.
- **Uniform tiles.** One tile size throughout; N should be a multiple of it, and partial tiles are marked FIXME in the low-rank kernels.

## Conversions

Tiles can be expanded to dense and their precision lowered, but the components share no format and nothing converts to hierarchical formats.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| TLR → dense | HiCMA `HICMA_duncompress`, `HICMA_zuncompress`; HiCMA-PaRSEC `hicma_parsec_matrix_uncompress` (internal); STARS-H `starsh_blrm__dca` | Exact (U·Vᵀ expansion) | Distributed in HiCMA and HiCMA-PaRSEC; the STARS-H version was verified by a run |
| Dense → TLR | STARS-H `starsh_problem_from_array` plus approximation (usable by HiCMA); TLR-MVM Python tool | Recompressing (randomized SVD, SVD or RRQR) | See Construction |
| FP64 ↔ FP32 or FP16, tile by tile | HiCMA-PaRSEC `hicma_parsec_convert_d2s` and `_s2d` | Lossy rounding; ranks unchanged | A per-tile decision map lowers a tile’s precision when its norm is small relative to ‖A‖\_F. Diagonal tiles stay FP64; low-rank tiles are FP64 or FP32. FP8 is compiled out |
| FP32 → FP16 or INT8 | TLR-MVM CUDA FP16 and INT8 classes | Lossy | Feeds the CUDA mixed-precision matvec |
| Between components; TLR ↔ H, H², HODLR, HSS | — | ✗ | HiCMA regenerates from the kernel rather than reading STARS-H matrices, and TLR-MVM reads only its own files. Other formats are listed as “current research” |

## Operations

Cholesky is the shared core; HiCMA adds multiplication, triangular solves and a pivot-free complex LU, while matvecs live only in the two companion libraries. No component provides matrix functions, inversion, LDLᵀ, QR, eigendecomposition or SVD.

| Operation | HiCMA 1.0.0 | HiCMA-PaRSEC 0.1.0 | STARS-H (own BLR object) | TLR-MVM |
| --- | --- | --- | --- | --- |
| Matvec | ✗ \[1\] | ✗ \[1\] | ✓ \[2\] | ✓ \[3\] |
| Transpose or adjoint matvec | ✗ | ✗ | ◐ symmetric only \[2\] | ✓ \[3\] |
| Block of vectors | ✗ | ✗ | ✓ \[2\] | ✗ one vector \[3\] |
| Direct solve | ✓ SPD \[4\] | ✗ \[5\] | ✗ | ✗ |
| Package-provided iterative solve | ✗ | ✗ | ✓ unpreconditioned CG \[6\] | ✗ |
| Triangular solve | ◐ left-lower, N and T \[4\] | ✗ | ✗ | ✗ |
| A + B, A + σI | ✗ \[7\] | ✗ \[7\] | ✗ | ✗ |
| Scaling | ◐ through α and β | ✗ | ◐ through α and β | ✗ |
| A · B | ✓ TLR×TLR \[8\] | ✗ dense kernels only \[9\] | ✗ | ✗ |
| LU | ◐ complex, pivots discarded \[10\] | ✗ | ✗ | ✗ |
| Cholesky | ✓ \[11\] | ✓ nine variants, two FP8 ones compiled out \[12\] | ✗ | ✗ |
| Log-determinant | ✗ (ExaGeoStat computes it) | ◐ internal check only \[13\] | ✗ | ✗ |
| Recompression | ◐ inside updates \[14\] | ◐ inside Cholesky \[14\] | ✗ | ✗ |

1. **No matvec in HiCMA or HiCMA-PaRSEC.** HiCMA’s public header has no TLR × vector or TLR × dense routine; `HICMA_dgemm_Tile` takes all three operands in TLR form. HiCMA-PaRSEC’s only gemv code is in the climate emulator and tests.
2. **STARS-H.** `starsh_blrm__dmml(M, nrhs, α, X, ldx, β, Y, ldy)` computes Y = αAX + βY with OpenMP or MPI, without a transpose flag. `dfe` measures the error against the kernel and `dca` expands to dense. These act on STARS-H’s own object, not on HiCMA descriptors; verified by a run.
3. **TLR-MVM.** `Tlrmvm{CPU,cuda,hip}::MVM()` applies one vector, optionally transposed or conjugated, in real or complex single or double precision. Its FP16 and INT8 variants on CUDA are complex single only and convert the factors on load. Batched classes handle many independent matrices with CUDA graphs; input comes from factor files. Read in code only.
4. **HiCMA solves.** `HICMA_dtrsm_Tile` (TLR right-hand side) and `HICMA_dtrsmd_Tile` (dense right-hand side) implement only the left-lower N and T cases; other branches assert “Not implemented yet”. There is no `posv` wrapper: a solve is `HICMA_dpotrf_Tile` plus two `HICMA_dtrsmd_Tile` calls.
5. **HiCMA-PaRSEC has no TLR solve.** Two KRR test drivers call DPLASMA’s dense `spotrs` only when the header macro CHECKSOLVE or PREDICTION is set.
6. **STARS-H CG.** `starsh_itersolvers__dcg_omp` and `_mpi`, unpreconditioned, for several right-hand sides; its header says iterative solvers are “not intended to be STARS-H business”. Verified: 438 iterations to 9e-9 at N = 4,000.
7. **No addition or shift.** A diagonal shift exists only at generation time (HiCMA-PaRSEC’s `--adddiag`, STARS-H’s nugget parameter). HiCMA-PaRSEC’s `hicma_parsec_matrix_add` builds a dense kernel matrix; it is not A + B.
8. **TLR multiplication.** `HICMA_dgemm_Tile` supports all four transpose combinations, with every tile (diagonal included) low-rank and each update recompressed by QR + SVD at fixed rank or accuracy. Real double only.
9. **HiCMA-PaRSEC products** are dense FP64 to INT8 GEMM and SYRK kernels for kernel ridge regression and Hamming distances. `perform_gemm` is declared in the public header but defined nowhere.
10. **Complex TLR LU.** `HICMA_zgetrf_Tile` is compiled into the 1.0.0 library. Its diagonal-tile kernel calls `zgetrf` and frees the pivots without applying them, so results are valid only when no row interchange occurs; there is no LU solve.
11. **HiCMA Cholesky.** `HICMA_dpotrf_Tile(Lower, …)` with dense diagonal and low-rank off-diagonal tiles, real double, lower only. Generated single and complex headers declare routines that have no implementation.
12. **HiCMA-PaRSEC Cholesky.** `hicma_parsec_potrf`, lower and real only. Variants: band-dense mixed precision with TLR (dense tiles FP64/32/16, low-rank FP64/32), band-dense FP64 with TLR, dense mixed-precision CPU and GPU variants, and sparse TLR with zero-rank tiles pruned. A performance model tunes the band size.
13. **HiCMA-PaRSEC log-det.** `hicma_parsec_matrix_det` sums log Lᵢᵢ over FP64 diagonal tiles, but is internal and called only on the dense reference factor in the check path.
14. **Recompression** happens only inside updates (HCORE: QR of stacked factors, then SVD at an absolute threshold or fixed rank). The process aborts if a truncated rank exceeds the cap.

## HPC suitability

Both implementations run every operation as an MPI-distributed task graph, which is the stack’s main strength. GPU use is limited to tasks with dense outputs in HiCMA-PaRSEC.

- **Threading.** HiCMA uses StarPU workers and HiCMA-PaRSEC uses PaRSEC threads; STARS-H has an OpenMP backend, and TLR-MVM uses OpenMP with single-threaded BLAS.
- **Distributed memory.** HiCMA uses StarPU-MPI with 2D block-cyclic tile layouts. HiCMA-PaRSEC requires MPI and adds a band distribution on a separate grid, sending low-rank tiles as variable-size messages. STARS-H has MPI backends for compression, matvec and CG.
- **GPU.** HiCMA has none. HiCMA-PaRSEC’s TLR Cholesky has CUDA bodies only for POTRF, TRSM, SYRK and GEMM tasks whose output tile is dense (verified in `potrf_L_dense_tlr_dp.jdf`); low-rank updates fall back to the CPU, and HIP exists only for the dense variants. TLR-MVM runs on CUDA and HIP, and STARS-H can compress tiles on GPUs, but HiCMA does not use that.
- **Runtimes.** StarPU for HiCMA; PaRSEC with DPLASMA for HiCMA-PaRSEC, both pinned to personal forks.

| Operation | HiCMA threads | HiCMA MPI | HiCMA GPU | PaRSEC threads | PaRSEC MPI | PaRSEC GPU |
| --- | --- | --- | --- | --- | --- | --- |
| Generation and compression | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| TLR Cholesky (FP64; mixed in HiCMA-PaRSEC only) | ✓ | ✓ | ✗ | ✓ | ✓ | ◐ dense-output tasks |
| Sparse TLR Cholesky | — | — | — | ✓ | ✓ | ✗ |
| Dense mixed-precision Cholesky | — | — | — | ✓ | ✓ | ✓ CUDA and HIP |
| TLR multiplication, triangular solves, complex LU | ✓ | ✓ | ✗ | — | — | — |
| Uncompression | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |

Largest documented runs, all CPU-only for TLR:

- **Sparse TLR Cholesky**, 3D RBF mesh deformation: 52.57M points at 1e-4 on 2,048 Shaheen II nodes (65K cores) in 36 min ([Cao et al., IPDPS 2022](https://netlib3.cs.utk.edu/files/publications/2022/icl-utk-1556-2022.pdf)).
- **TLR Cholesky**, 3D squared-exponential kernel: N = 42M at 1e-8 on 4,096 Shaheen II nodes (about 131K cores) in nearly 24 hours, against an estimated 77 days dense ([Cao et al., PASC 2020](https://www.netlib.org/utk/people/JackDongarra/PAPERS/extreme-cholesky-pasc-2020.pdf)). This used Lorapo, HiCMA-PaRSEC’s predecessor.
- **Band-dense TLR** (IPDPS 2021): 11.88M on 2,048 Shaheen II nodes at 1e-8, up to 7× faster with 44× less memory than the previous version.
- **Mixed precision with TLR on Fugaku** (SC22 Gordon Bell finalist): up to 12× faster than dense Cholesky; node counts and N were not found.
- **For context, dense:** the SC24 Gordon Bell climate prize ran dense FP64/FP16 Cholesky on 9,025 Frontier nodes at N ≈ 27.24M and 0.976 EFlop/s ([arXiv 2408.04440](https://arxiv.org/html/2408.04440v2)). TLR-MVM’s SC23 finalist run used 48 Cerebras CS-2 systems with code not in the public repository.

Practical limits: the maximum rank is tuned by hand and a tile that exceeds it aborts the run. Tiles have one size; HiCMA-PaRSEC handles real symmetric lower-stored matrices only, and its task priorities can overflow beyond 1,200 tile rows.

## Caveats and surprises

- **The headline features overstate the TLR stack.** HiCMA-PaRSEC’s “matrix multiplication and rank-k updates” are dense kernels, its TLR paths use CUDA only for dense-output tasks, and FP8 is compiled out with no build switch. The most-cited exascale runs were dense, not TLR.
- **Accuracy means different things in different places.** Initial compression uses a per-tile relative Frobenius criterion; recompression during factorization uses an absolute singular-value threshold. Precision decisions are relative to ‖A‖\_F / NT, and TLR-MVM is relative to ‖A‖\_F.
- **HiCMA-PaRSEC is a benchmark harness, not a library API.** `hicma_parsec_init` parses command-line arguments into a parameter struct, and features are toggled by compile-time macros in a header. Its release notes claim Windows support, unit tests and API documentation that the repository does not contain. CI, on pull requests only, runs seven small CPU-only TLR Cholesky cases and one climate-emulator case.
- **Headers over-promise.** HiCMA’s generated single and complex headers declare routines that have no implementation, and every triangular-solve branch except left-lower asserts at run time.
- **Construction always costs O(N²) kernel evaluations.** There is no ACA, interpolation or proxy method; STARS-H’s “CROSS” engine is randomized SVD, and its `dlrna` routine is a stub returning rank −1.
- **Two matvecs, neither for HiCMA.** STARS-H’s and TLR-MVM’s matvecs work only on their own objects.
- **Small corrections.** HiCMA 1.0.0 is not built over Chameleon, despite its README; the complex TLR LU ships in the 1.0.0 library, not only on a branch. Version strings disagree with the tags (HiCMA’s CMake says 0.1.1; HiCMA-PaRSEC’s headers say 1.0.0).
- **Maintenance.** HiCMA, STARS-H and HCORE have had no commits since 2023. HiCMA-PaRSEC depends on personal forks of DPLASMA, PaRSEC, HCORE and STARS-H, and active work happens in unreleased branches.

## Verification

The review read all components and built STARS-H; HiCMA, HiCMA-PaRSEC and TLR-MVM were not built, so their MPI, GPU and solver behaviour comes from code reading.

- **Code read:** HiCMA at v1.0.0 (fa8596b) and its `zgetrf` branch; HiCMA-PaRSEC at 366f7e0 and the branches of Qinglei Cao’s fork; STARS-H (c115e75), TLR-MVM (9c86f91), HCORE (ff81276), ExaGeoStat (ee4a82b), ExaGeoStatCPP, and the DPLASMA, HCORE and STARS-H forks pinned by HiCMA-PaRSEC.
- **STARS-H build:** OpenMP backend with OpenBLAS. The 2D squared-exponential test (N = 2,500, tile 500, tolerance 1e-9, max rank 90; 17 of 25 tiles stored dense) reached about 4.7e-11 relative error with every compression engine. “CROSS” gave exactly the randomized-SVD error, confirming the alias.
- **STARS-H check program:** a user kernel (1D exponential plus a 1e-2 shift, N = 4,000, tile 400, tolerance 1e-9, max rank 150) compressed every off-diagonal tile to rank 1, which is exact for this kernel (relative error 6.2e-16). This tests the API, not compression accuracy. The matvec with 3 right-hand sides matched the dense copy to 3.6e-16, and CG converged in 438 iterations to 9.05e-9.
- **Spot check:** the CUDA task bodies in HiCMA-PaRSEC’s TLR Cholesky were re-read independently and return to the CPU for every tile that is not dense.
- **Not accessible:** full texts of the SC22 paper (ACM), Euro-Par 2018 (Springer) and ISC 2020, so their node counts and sizes are not reported. The full STARS-H test suite did not finish in 10 minutes.

## Sources

- [HiCMA](https://github.com/ecrc/hicma) at v1.0.0: `README.md`, `RELEASE.md`, `CMakeLists.txt`, `include/hicma_z.h`, `include/hicma.h`, `compute/` (`dpotrf.c`, `pdpotrf.c`, `pdgemm.c`, `pdtrsm.c`, `pzgetrf.c`), `runtime/starpu/codelets/`, `timing/`, `testing/`
- [HiCMA-PaRSEC](https://github.com/ecrc/hicma-x) at 366f7e0: `README.md`, `RELEASE.md`, `TESTS.md`, `BUILD.md`, `TODOs`, `src/core/hicma_parsec.{h,c}`, `hicma_parsec_internal.{h,c}`, `src/kernels/` (`matrix_compress.jdf`, `potrf_L_dense_tlr_dp.jdf`, `potrf_L_dense_tlr_mp.jdf`, `potrf_L_dense_mp_gpu.jdf`, `decision_make_comp.jdf`, `matrix_det.jdf`, `matrix_add.jdf`), `tests/`, `scripts/test_CI.sh`
- [STARS-H](https://github.com/ecrc/stars-h), [HCORE](https://github.com/ecrc/hcore), [TLR-MVM](https://github.com/ecrc/tlrmvm), [ExaGeoStat](https://github.com/ecrc/exageostat)
- [Cao et al., PASC 2020](https://www.netlib.org/utk/people/JackDongarra/PAPERS/extreme-cholesky-pasc-2020.pdf)
- [Cao et al., IPDPS 2021](https://icl.utk.edu/files/publications/2021/icl-utk-1463-2021.pdf)
- [Cao et al., IPDPS 2022](https://netlib3.cs.utk.edu/files/publications/2022/icl-utk-1556-2022.pdf)
- [Abdulah et al., TPDS 2022](https://netlib.org/utk/people/JackDongarra/PAPERS/accer-geo-ipds-2022.pdf)
- [SC22 Gordon Bell finalist abstract](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/gb102.html)
- [SC24 climate emulator](https://arxiv.org/html/2408.04440v2) and [SC24 genomics kernel ridge regression](https://arxiv.org/html/2409.01712v1)
- [SC23 TLR-MVM on Cerebras](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/gb109.html)
- [ISC 2018 HiCMA poster](https://www.isc-hpc.com/agenda2018/conferences/isc_hpc/assets/2018/posters/apost108.pdf)
