# MUMPS deep dive

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects the MUMPS 5.7.3 source and the 5.9.1 changelog and user guide; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

MUMPS 5.9.1 (July 2026) is a mature MPI + OpenMP multifrontal sparse direct solver for LU and LDLᵀ in four arithmetics. Its only compressed format is an internal block low-rank (BLR) representation of frontal factors, optionally with compressed contribution blocks, switched on with ICNTL(35) and an absolute threshold CNTL(7). It is widely used in production; published runs reach 3.1 × 10⁸ complex unknowns on 19,200 cores.

Almost every solver feature works on BLR factors: multiple, sparse and distributed right-hand sides, transposed solves, a dense Schur complement, the determinant, selected entries of A⁻¹, null-space bases, iterative refinement and save/restore.

For an H-matrix user the gaps are structural. There is no user-visible compressed object, no input other than an assembled sparse matrix, and no matvec, arithmetic or conversion; multilevel BLR exists only in papers. The only iteration is refinement, and the new GPU option’s coverage of BLR is unknown.

## Snapshot

MUMPS ships one feature release each spring, with tarballs from its website, via a request form or at mumps-solver.org/MUMPS\_\<version>.tar.gz.

| Item | Value |
| --- | --- |
| Release | 5.9.1, July 2026 (5.9.0 April 2026; 5.8.2 January 2026; 5.8.0 May 2025; 5.7.3 July 2024), per the [download page](https://mumps-solver.org/index.php?page=dwnld) |
| Cadence | One feature release per spring since 5.3.0 (April 2020), then one to four bug-fix releases. No public version control or issue tracker; tarballs come from a web form or a direct URL, with SHA256 sums published |
| Licence | CeCILL-C (LGPL-like, under French law). The truncated-RRQR routines derived from LAPACK carry a BSD-3 header |
| Languages and bindings | Fortran 95/2003 core with a C interface; one driver per arithmetic (S, D, C, Z). Sequential MATLAB/Octave and Scilab interfaces ship in the tarball. Third-party: python-mumps and PyMUMPS on PyPI, MUMPS.jl, R `rmumps`, PETSc, Trilinos Amesos/Amesos2 |
| Dependencies | BLAS and LAPACK (LAPACK is required for BLR); MPI and ScaLAPACK for parallel builds (ScaLAPACK optional since 5.6/5.7). Orderings: built-in AMD variants, bundled PORD, optional METIS/ParMETIS and SCOTCH/PT-SCOTCH (strongly advised for BLR). Optional: OpenMP, CUDA/cuBLAS or XKBlas for GPUs, AVX-512 VBMI for adaptive-precision BLR |
| Maintainers | The MUMPS team and Mumps Technologies; copyright holders are CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria, Mumps Technologies and Univ. Bordeaux. BLR and mixed-precision research comes from IRIT/INP Toulouse and LIP6 Sorbonne |
| Documentation | A thorough user guide of about 140 pages, with per-parameter tables, defaults, incompatibilities and examples; no separate API reference |
| Users | PETSc (`MATSOLVERMUMPS`, which exposes the BLR controls), Elmer (GPU option, September 2026), COIN-OR/IPOPT, and 3D geophysics, electromagnetics and helioseismology codes. Amestoy, L’Excellent and Mary received the 2024 ECMI-EMS Lanczos Prize for MUMPS |

## Construction

The only input is an assembled sparse matrix; BLR compression happens inside its factorization, using groups found at analysis time.

| Access | Formats | Algorithm | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Assembled sparse matrix | Internal BLR panels of LU (SYM = 0) or LDLᵀ (SYM = 1, 2) fronts; optional BLR contribution blocks; small fronts stay full-rank | Each panel is factored, then its off-diagonal blocks are compressed by truncated QR with column pivoting, stopping at \|r\_kk\| < CNTL(7). A block stays full-rank if its rank exceeds half of mn/(m + n); low-rank updates are accumulated, without recompression by default | JOB = 1 analysis with ICNTL(35) ≠ 0 (clustering happens here), then JOB = 2 factorization; centralized or distributed triplets | Verified in code and by run on 5.7.3. Switching BLR on only at factorization fails with INFO(1) = −54 |
| Elemental sparse matrix | — | — | ICNTL(5) = 1 | BLR is refused with INFO(1) = −800 (verified on 5.7.3); the 5.9 changelog does not mention a change |
| Entry oracle, kernel + coordinates, matvecs only, dense matrix | ✗ | — | — | No callbacks, no geometry and no dense input; the Schur complement is a dense output |
| Inputs that shape BLR | — | — | User permutation (ICNTL(7)), block format with analysis by blocks (ICNTL(15)), user scaling and a Schur variable list | No low-rank data can be imported |

- **Groups.** Each front’s fully summed variables are grouped by METIS (preferred) or SCOTCH k-way partitioning of the separator graph plus a halo. Without either, the code falls back to contiguous chunks in elimination order. With distributed input, the graph is gathered on the host for clustering.
- **Block size.** 128, 256, 384 or 512, set by the number of fully summed variables (up to 1,000, 5,000, 10,000, or more).
- **What is compressed.** Only fronts with at least 1,000 variables and 128 fully summed ones. The Schur front and the contribution blocks of the root’s children are never compressed, and a ScaLAPACK root stays dense unless ICNTL(13) > 0.

| BLR control | Meaning | Default |
| --- | --- | --- |
| ICNTL(35) | 0 off; 1 automatic; 2 BLR factorization and BLR solve (memory gain); 3 BLR factorization with full-rank factors kept for the solve | 0 |
| CNTL(7) | Absolute dropping threshold ε, not relative to any norm, so scaling is advised; 0.0 means full precision | 0.0 |
| ICNTL(36) | 0 compresses after the triangular solve; 1 compresses before it, saving more flops while keeping pivoting | 1 since 5.9.0 (0 before) |
| ICNTL(37) | Compress contribution blocks: less memory and MPI traffic, more flops | 0 |
| ICNTL(38), ICNTL(39) | Expected compression of factors and contribution blocks (per mille), used only for memory estimates at analysis | 600, 500 |
| ICNTL(40) | Adaptive mixed-precision storage of BLR blocks (5.9.0) | Not read; that guide section was inaccessible |
| Related | ICNTL(13) > 0 lets the root be BLR; ICNTL(47) = 1 factors in single precision inside a double instance (5.9.0); INFOG(35) and RINFOG(14) report effective factor entries and flops | — |

## Conversions

No conversion is user-callable: the factors are never exposed, and nothing can be exported to dense, H, H² or HODLR. The transformations below happen inside the solver.

| From → To | Control | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| Full-rank front panel → BLR blocks | ICNTL(35) ≠ 0, CNTL(7) | Recompressing (lossy at ε) | During factorization only; cannot be applied to an existing full-rank factorization |
| BLR factorization → full-rank factor storage | ICNTL(35) = 3 | Approximate (ε-level) factors stored uncompressed | No memory gain, but all factors can go out of core |
| BLR blocks → adaptive precision (up to 7 formats of 2–8 bytes) | ICNTL(40), 5.9.0 | Lossy: singular vectors of each block are grouped into precisions by singular-value decay; full-rank blocks of small norm may be quantized | Storage only: blocks are converted back to fp64 before BLAS |
| Double instance → single-precision factors | ICNTL(47) = 1, 5.9.0 | Precision conversion | Refinement in double on request; compatibility with BLR not stated in the parts of the guide that were readable |
| Full-rank contribution block → BLR | ICNTL(37) = 1 | Recompressing | Saves memory and communication |
| Instance ↔ disk | JOB = 7 and 8 (save and restore) | Exact | Includes the BLR structures |

## Operations

Nearly every solver service works with BLR on; what is missing is everything that would treat the factors as a matrix. No mode provides a matvec, matrix functions, A + B, A + σI, A·B, an explicit inverse, QR, eigendecomposition, SVD or user-callable recompression.

| Operation | Full-rank | BLR factors, BLR solve (ICNTL(35) = 1, 2) | BLR factors, full-rank solve (ICNTL(35) = 3) |
| --- | --- | --- | --- |
| Direct solve: multiple, sparse and distributed right-hand sides; Aᵀ | ✓ \[1\] | ✓ | ✓ |
| Package-provided iterative solve | ◐ refinement \[2\] | ◐ | ◐ |
| Partial (triangular) solves | ◐ \[3\] | ◐ | ◐ |
| LU | ✓ \[4\] | ✓ | ✓ |
| Cholesky | ◐ LDLᵀ form \[5\] | ◐ | ◐ |
| LDLᵀ | ✓ \[5\] | ✓ | ✓ |
| Determinant | ✓ \[6\] | ✓ | ✓ |
| Schur complement (dense) | ✓ \[7\] | ✓ | ✓ |
| Selected entries of A⁻¹ | ✓ \[8\] | ✓ | ✓ |
| Null pivots and null-space basis | ✓ \[9\] | ✓ | ✓ |
| Inertia | ◐ real symmetric; exact with one process or ICNTL(13) > 0 | ◐ | ◐ |
| Error analysis, save and restore | ✓ | ✓ | ✓ |
| Forward elimination during factorization | ✓ | ✗ \[10\] | ✗ |
| Discard factors, out-of-core | ✓ | ◐ \[11\] | ✓ |
| Adaptive-precision factor storage | ✗ | ✓ 5.9.0 \[12\] | ? |

1. **Direct solve** (JOB = 3): dense multiple right-hand sides, sparse (ICNTL(20) = 1–3) and distributed (10, 11) right-hand sides, centralized or distributed solutions, and Aᵀx = b (ICNTL(9)). That is the transpose, not the conjugate transpose. The BLR solve has used compressed factors since 5.2.0 and handles A and Aᵀ; verified.
2. **Iterative refinement** (ICNTL(10)) is the only package-provided iteration; with ICNTL(47) = 1 it runs in double on single-precision factors. With BLR at ε = 1e-4 on the 30³ grid (LU), three fixed steps took ‖b−Ax‖₂/‖b‖₂ from 1.1e-2 to 4.7e-10 (verified). It is incompatible with several right-hand sides, distributed solutions, Schur complements and some other options.
3. **Partial solves.** ICNTL(26) condenses a right-hand side onto the Schur variables and expands the solution back; ICNTL(31) = 2 keeps only U for backward substitution. There is no general “apply L⁻¹” call.
4. **LU** (SYM = 0) uses threshold partial pivoting, static pivoting and delayed pivots, kept by both BLR variants.
5. **Symmetric matrices.** SYM = 1 is LDLᵀ without pivoting, not Cholesky, and SYM = 2 is LDLᵀ with 1×1 and 2×2 pivots. Complex SYM = 1 is treated as SYM = 2, and there is no Hermitian factorization.
6. **Determinant** (ICNTL(33)): mantissa and exponent, with the sign tracked through permutations and scaling. With BLR it is the determinant of the approximate factors: on a 40³ Laplacian at ε = 1e-6 the mantissa changed in the sixth digit.
7. **Schur complement** (ICNTL(19)): returned as a full matrix, centralized or 2D block-cyclic. The Schur front itself is never compressed. With BLR elsewhere at ε = 1e-8 on the 30³ grid, it matched full rank to 9.7e-11 in relative Frobenius norm (order 900).
8. **Selected entries of A⁻¹** (ICNTL(30)) exploit sparsity and parallelism. With BLR at ε = 1e-8 on the 30³ grid, 200 diagonal entries matched full rank to a maximum relative difference of 8.5e-12 (verified).
9. **Null space.** ICNTL(24) detects null pivots and ICNTL(25) returns a null-space basis. On a 30³ Neumann Laplacian (SYM = 2) with BLR at ε = 1e-8, it found the one null pivot, with ‖Az‖₂/‖z‖₂ = 1.2e-8 against 1.1e-14 at full rank.
10. **Forward elimination with BLR** fails with INFO(1) = −43 (verified).
11. **Discard and out-of-core.** Discarding all factors works with BLR since 5.6.0, but ICNTL(31) = 2 keeps the L factors of BLR fronts. With ICNTL(35) = 2 only full-rank factors go out of core.
12. **Adaptive precision** (ICNTL(40)) stores BLR blocks in up to seven custom formats chosen from their singular-value decay. It mainly speeds up the memory-bound solve.

## HPC suitability

MUMPS distributes factorization and solve over MPI and threads them with OpenMP, with BLR active in both. GPU offload arrived in 5.9.0 through XKBlas, with unknown coverage of BLR.

- **Threading.** OpenMP plus multithreaded BLAS. BLR clustering, panel compression and low-rank updates run in OpenMP loops, and tree parallelism (ICNTL(48)) applies to factorization and solve. Since 5.9.0, ICNTL(17) can turn some MPI ranks into OpenMP resources.
- **Distributed memory.** Static mapping with asynchronous dynamic scheduling: tree parallelism across ranks, large fronts split into 1D row blocks, and a 2D block-cyclic ScaLAPACK root. BLR applies to all fronts except that root, and compressing contribution blocks cuts communication. Ordering can run in parallel, but BLR clustering is centralized on the host.
- **GPU.** ICNTL(51) (5.9.0) offloads work to NVIDIA or AMD GPUs through XKBlas, a multi-GPU BLAS-3 library. Which operations it covers, and whether BLR kernels run on GPUs, was not documented in the readable parts of the guide. Most likely only large dense BLAS-3 calls are offloaded: the [2025 MUMPS workshop slides](https://mumps-solver.org/doc/workshop2025/Gautier_Polet_Slides.pdf) show only full-rank BLAS-3 offload and do not mention BLR.
- **Runtimes.** None beyond MUMPS’s own MPI scheduler; the GPU path uses the XKaapi runtime inside XKBlas.

| Operation | Threads | MPI | GPU |
| --- | --- | --- | --- |
| Ordering and symbolic analysis | ◐ threaded SCOTCH | ✓ PT-SCOTCH or ParMETIS | ✗ |
| BLR clustering | ✓ | ◐ centralized on the host | ✗ |
| Full-rank factorization | ✓ | ✓ | ✓ ICNTL(51), 5.9.0 |
| BLR factorization | ✓ | ✓ (root dense unless ICNTL(13) > 0) | ? |
| Solve, full-rank or BLR | ✓ | ✓ distributed right-hand sides and solutions | ? |
| Schur complement | ✓ | ✓ 2D block-cyclic output | ? |
| Selected entries of A⁻¹ | ✓ | ✓ | ? |
| Refinement, error analysis | ? | ◐ centralized solution only | ? |

Largest documented runs:

- **Helioseismology, N = 309,859,823** (complex unsymmetric LU, BLR ε = 1e-7 with adaptive-precision storage): factorization in 54 min with 28.6 TB on 100 Adastra nodes, 19,200 cores. At N = 132.2M on 50 nodes, BLR took 12 min and 6.70 TB against 63 min and 12.43 TB at full rank; BLR fits on 20 nodes where full rank needs 50 ([Faucher et al., 2025](https://arxiv.org/pdf/2511.22530)).
- **Offshore full-waveform inversion, 73.9M unknowns** (complex single, full rank): LU in 2,218 s with 8.3 TB on 128 Jean-Zay nodes. Mixed-precision BLR further cuts LU memory by up to about 26% and time by up to about 40% ([Operto et al., 2023](https://arxiv.org/pdf/2210.16767)).
- **3D controlled-source electromagnetics, 20.6M unknowns** at ε = 1e-7 on 900 cores: BLR factorization in 1,113 s against 2,803 s, with factors at 30.5% and flops at 7.9% of full rank ([Shantsev et al., 2017](https://perso.lip6.fr/Theo.Mary/doc/GJI17.pdf)).
- **North Sea full-waveform inversion, 17.4M unknowns** at ε = 1e-3 on 680 cores: LU in 424 s against 1,153 s, with factors of 209 GB against 722 GB. That version did not yet exploit the smaller factors to reduce memory use. The residual rose to 4.3e-4, yet the inverted models were indistinguishable ([Amestoy et al., 2016](https://www-pequan.lip6.fr/~tmary/doc/GEO16.pdf)).
- **Adaptive precision on one node** (Queen\_4147, n ≈ 4.1M, ε = 3e-9): factor storage 112 GB full rank, 57 GB BLR in fp64, 37 GB adaptive. The solve took 479 ms adaptive, 505 ms BLR fp64 and 871 ms full rank ([Amestoy et al., ACM TOMS 2026](https://www-pequan.lip6.fr/~tmary/doc/BlockMemAcc.pdf)).

Practical limits: scale the matrix, since the threshold is absolute; install METIS or SCOTCH, since clustering quality depends on them (without them, groups fall back to contiguous chunks). Small fronts, the Schur front and the ScaLAPACK root stay dense, and N and indices are 32-bit (entry counts 64-bit) unless a 64-bit-integer build is chosen.

## Caveats and surprises

- **Compression is skipped exactly where an H-matrix user might want it:** the Schur-complement front and the ScaLAPACK root stay dense.
- **SYM = 1 is not Cholesky**, and there is no Hermitian solver.
- **Multilevel BLR and shared-basis BLR are not released.** They are absent from the 5.7.3 source and the 5.9 changelog, existing only in papers.
- **Adaptive precision is storage-only.** Arithmetic stays in fp64 through an accessor, and its speed relies on AVX-512 VBMI.
- **The 5.9.1 guide contradicts itself.** Section 3.20.1 still calls the compress-after variant the default, while section 2.4 and the changelog say ICNTL(36) = 1 is now the default.
- **History worth knowing.** The first public BLR release was 5.1.1 (March 2017), still experimental. Before 5.2.0 the solve used full-rank factors, which is why 2017 papers report full-rank solve times.
- **Access limits.** MUMPS has no public version control, and the author’s sandbox could read its pages but not download from it. Only sections 1–5.3 of the 5.9.1 guide were readable, so its BLR API (5.20), GPU (5.27) and parameter-table (6) sections were not read; those details rely on the 5.7.3 guide and code plus the 5.9.1 changelog.

## Verification

The author read and ran MUMPS 5.7.3, the newest source available to them, and read 5.9.x through its changelog, the readable part of the 5.9.1 guide and third-party build files.

- **Source.** MUMPS 5.7.3 from branch topic-v25stage4 (commit 3b41f12) of the [deepin-community/mumps](https://github.com/deepin-community/mumps) GitHub repository (Debian packaging whose patches touch only makefiles), including the 5.7.3 user guide.
- **Code read.** BLR controls, the RRQR kernel and rank rule, clustering, front eligibility and its exclusions, the BLR solve, save and restore, the GPU stub and OpenMP coverage; no multilevel BLR code exists.
- **Build.** Sequential double precision with gfortran 13.3, OpenBLAS 0.3.26 (pthreads), PORD only and OpenMP, in about 3 minutes.
- **Runs on 3D Laplacians (30³, 40³).** BLR worked for SYM = 0, 1 and 2 with transposed solves and determinants, with Schur complements (9.7e-11), A⁻¹ entries (8.5e-12), null-space bases and refinement. These matrices are symmetric, so the transposed solves did not test transposition. A reviewer’s rerun with an unsymmetric 30³ operator at ε = 1e-4 confirmed transposed BLR solves, with relative residual 5.4e-3 for Aᵀ against 4.9e-3 for A. ICNTL(36) = 1 cut flops slightly further and ICNTL(37) = 1 added flops; elemental input and forward elimination failed with BLR, as documented.
- **5.9.x evidence.** The download-page changelog; sections 1–5.3 of the 5.9.1 guide; the [mumps-superbuild](https://github.com/scivision/mumps-superbuild) CMake files for 5.9.1 (GPU, XKBlas and AVX-512 flags); PETSc’s MUMPS package file and manual page; Elmer pull request 930.
- **Not tested.** No MPI, METIS or SCOTCH runs. HAL, the SEG library, the ACM Digital Library and ResearchGate were not accessible.

## Sources

- [MUMPS download page and changelog](https://mumps-solver.org/index.php?page=dwnld) and [documentation page](https://mumps-solver.org/index.php?page=doc)
- [MUMPS 5.9.1 user guide](https://mumps-solver.org/doc/userguide_5.9.1.pdf), sections 1–5.3
- MUMPS 5.7.3 source and user guide from [deepin-community/mumps](https://github.com/deepin-community/mumps): `src/dana_driver.F`, `dfac_driver.F`, `dini_defaults.F`, `dlr_core.F`, `dfac_lr.F`, `dana_lr.F`, `lr_common.F`, `dsol_driver.F`, `dsol_lr.F`, `dmumps_save_restore.F`, `dmumps_gpu.c`
- [mumps-superbuild](https://github.com/scivision/mumps-superbuild) and the [XKBlas](https://github.com/scivision/xkblas) README
- [Gautier and Polet, MUMPS workshop 2025](https://mumps-solver.org/doc/workshop2025/Gautier_Polet_Slides.pdf), on MUMPS with XKBlas
- [PETSc MATSOLVERMUMPS](https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/) and [Elmer pull request 930](https://github.com/ElmerCSC/elmerfem/pull/930)
- [Faucher et al., 2025](https://arxiv.org/pdf/2511.22530)
- [Operto et al., The Leading Edge 2023](https://arxiv.org/pdf/2210.16767)
- [Shantsev et al., GJI 2017](https://perso.lip6.fr/Theo.Mary/doc/GJI17.pdf)
- [Amestoy et al., Geophysics 2016](https://www-pequan.lip6.fr/~tmary/doc/GEO16.pdf)
- [Amestoy, Jego, L’Excellent, Mary, Pichon, ACM TOMS 2026](https://www-pequan.lip6.fr/~tmary/doc/BlockMemAcc.pdf)
- [Théo Mary’s publication list](https://perso.lip6.fr/Theo.Mary/publications.php), for the multilevel BLR and mixed-precision BLR papers
- [2024 Lanczos Prize announcement](https://ecmiindmath.org/2024/07/15/lanczos-ecmi-ems-prize-awarded-to-mumps/)
