# Survey of further H-matrix packages

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects the eight packages' repositories, release notes and papers as read on 8 Oct 2026; later versions may differ. [All background docs](README.md) · [Overview of the twelve packages](overview.md)

Eight further packages clear the bar: [ButterflyPACK](packages/butterflypack.md), [Htool-DDM](packages/htool-ddm.md), [hmat-oss](packages/hmat-oss.md), [H2Pack](packages/h2pack.md), [HMatrices.jl](packages/hmatrices-jl.md), [HiCMA (with HiCMA-PaRSEC)](packages/hicma.md), [MUMPS](packages/mumps.md) and [PaStiX](packages/pastix.md). They add butterfly and tensor formats, tile low-rank on runtime systems, and two sparse direct solvers with low-rank compression to the four libraries already covered ([STRUMPACK](packages/strumpack.md), [H2Opus](packages/h2opus.md), [HLIBpro](packages/hlibpro.md), [H2Lib](packages/h2lib.md); all twelve are compared in the [overview](overview.md)).

The bar was a maintained, documented library that people use on real applications. One-off research codes and toolboxes built for teaching or experimentation were left out. The search covered GitHub, project sites, the awesome-hierarchical-matrices list and recent papers. The deep dives on all eight follow the package summaries; each has its own document, linked above.

## Shortlist at a glance

The eight split into five general-purpose dense libraries, one tile low-rank stack, and two sparse direct solvers that compress their factors.

| Library | Formats | Construction | Factorizations | Parallelism | Latest release · last commit |
| --- | --- | --- | --- | --- | --- |
| [ButterflyPACK](https://github.com/liuyangzhuan/ButterflyPACK) | H, HODLR, butterfly (HODBF, H-BF, SHNBF), BLR, H², H-tensor | Entry or block evaluation; randomized matvecs | H-LU, HODLR/HODBF inversion, BLR LU, symmetric H² solver; log-det | MPI + OpenMP; PaRSEC for H and BLR | 5.0.0 (22 Aug 2026) · 17 Sep 2026 |
| [Htool-DDM](https://github.com/htool-ddm/htool) | H | Entry generator; ACA variants, SVD | H-LU, H-Cholesky; LDLᵀ unreleased | MPI + OpenMP | 1.0.2 (Feb 2026) · 6 Oct 2026 |
| [hmat-oss](https://github.com/jeromerobert/hmat-oss) | H, HODLR | Entry or block assembly; ACA, SVD, RRQR | LU, LLᵀ, LDLᵀ, HODLR | Sequential | 1.11.1 (Nov 2025) · 25 Sep 2026 |
| [H2Pack](https://github.com/scalable-matrix/H2Pack) | H², HSS | Kernel + points; proxy points | HSS ULV | OpenMP | No tags · no commits after Jul 2023 until 5 in Sep 2026 |
| [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl) | H | Entry access; ACA | LU, Cholesky | Julia threads (assembly, matvec) | 0.2.13 (Aug 2025) · 27 Feb 2026 |
| [HiCMA](https://github.com/ecrc/hicma) / [HiCMA-PaRSEC](https://github.com/ecrc/hicma-x) | TLR | STARS-H kernel generators; per-tile compression | TLR Cholesky | StarPU or PaRSEC over MPI; CUDA for dense-output tasks | 1.0.0 (Jan 2023) · 17 Jan 2023 / 0.1.0 (Nov 2025) · 18 Mar 2026 |
| [MUMPS](https://mumps-solver.org) | Sparse factors with BLR fronts | Sparse matrix | LU, LDLᵀ | MPI + OpenMP | 5.9.1 (Jul 2026) |
| [PaStiX](https://solverstack.gitlabpages.inria.fr/pastix/) | Sparse supernodal factors with low-rank blocks | Sparse matrix | LLᵀ, LDLᵀ, LU (static pivoting) | Threads, MPI, StarPU/PaRSEC; GPU for uncompressed blocks | 6.4.0 (Jul 2024) |

Release and commit data were read from the repositories on 8 Oct 2026 and re-checked in the deep dives.

## General-purpose hierarchical-matrix libraries

Five libraries build and operate on dense hierarchical matrices directly; only ButterflyPACK and Htool-DDM distribute that work over MPI.

**[ButterflyPACK](https://github.com/liuyangzhuan/ButterflyPACK)** is a Fortran 2008 library from Lawrence Berkeley National Laboratory with C++ and Python interfaces (BSD licence). It compresses blocks as low-rank or butterfly factors in HODLR, H, BLR and strongly admissible nested formats, built from entry evaluation or randomized matvecs. HODBF, H-BF and B-BF are HODLR, H and BLR with butterfly blocks; HSS-BF and SHNBF are nested butterfly formats with weak and strong admissibility. Version 5.0.0 added a symmetric parallel H² solver and PaRSEC for the H and BLR formats, and extended the H-tensor format added in 4.1.0. STRUMPACK uses ButterflyPACK for its HODLR and butterfly fronts.

**[Htool-DDM](https://github.com/htool-ddm/htool)** is a header-only C++17 library (MIT) for H-matrices compressed by ACA variants or SVD. It provides an MPI-distributed matvec and, through [HPDDM](https://github.com/hpddm/hpddm), Krylov solvers with domain-decomposition preconditioners. H-LU and H-Cholesky date from 0.9.0 and became task-parallel in 1.0; a sequential LDLᵀ is on the main branch but unreleased.

**[hmat-oss](https://github.com/jeromerobert/hmat-oss)** is a C++ library with a C API (GPL v2), first developed by Airbus. It assembles H-matrices with ACA, SVD or RRQR compression and provides LU, LLᵀ, LDLᵀ and HODLR factorizations. The open-source release is sequential and pins BLAS to one thread; Airbus keeps its parallel version proprietary.

**[H2Pack](https://github.com/scalable-matrix/H2Pack)** is a C library (MIT) from Edmond Chow’s group at Georgia Tech. It builds H² and HSS representations of kernel matrices in linear time with the proxy-point method, and provides matvec, matmul and an HSS ULV factorization and solve on shared memory. It is limited to square kernel matrices on one point set in up to three dimensions, with symmetric, translation-invariant, non-oscillatory kernels. It had no commits between July 2023 and September 2026 but is included for its documentation and published use.

**[HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl)** is a Julia package (MIT) aimed at boundary integral equations. It assembles H-matrices by ACA from any matrix-like object that returns entries. Assembly and matvec are threaded; H-LU and H-Cholesky are serial, and the experimental Distributed.jl path appears unable to run.

## Tile low-rank: HiCMA and HiCMA-PaRSEC

The KAUST tile low-rank stack targets runtime-scheduled, multi-node Cholesky of dense covariance and kernel matrices.

**[HiCMA](https://github.com/ecrc/hicma)** (BSD) stores a matrix as a grid of tiles, dense on the diagonal and low-rank elsewhere. It runs TLR multiplication, triangular solves and Cholesky as StarPU tasks in shared or distributed memory; its complex TLR LU discards pivots. Release 1.0.0 dates from January 2023, with no commits since. KAUST’s C++ rewrites, HiCMA++ 1.0.0 and HCore++ 3.0.0, both date from November 2023 and have been quiet since.

**[HiCMA-PaRSEC](https://github.com/ecrc/hicma-x)** (BSD-3-Clause) is the active successor on PaRSEC and DPLASMA, released as 0.1.0 in November 2025. Its one TLR operation is Cholesky, with a dense band, mixed precision (FP16 in dense tiles only) and pruning of zero-rank tiles. CUDA runs only tasks with dense outputs, and the multiplication and rank-k features in its release notes are dense kernels.

Two companions do the rest of the work. [STARS-H](https://github.com/ecrc/stars-h) generates the test and application matrices tile by tile from kernels and compresses each tile; it has had no commits since November 2023. [TLR-MVM](https://github.com/ecrc/tlrmvm) is a standalone TLR matvec for CPUs and NVIDIA and AMD GPUs, last changed in December 2024.

## Sparse direct solvers with low-rank compression

MUMPS and PaStiX take a sparse matrix and compress the dense fronts or supernodes inside its factorization; neither hands the user a compressed matrix object.

**[MUMPS](https://mumps-solver.org/index.php?page=dwnld)** (CeCILL-C) is the multifrontal MPI + OpenMP solver from the MUMPS consortium. Block low-rank factorization arrived publicly in 5.1.1, low-rank solve in 5.2.0, and adaptive mixed-precision BLR in 5.9.0 (April 2026), which also added an option to expose work to GPUs. Version 5.9.1 shipped in July 2026.

**[PaStiX](https://solverstack.gitlabpages.inria.fr/pastix/)** (LGPL-3.0) is Inria’s supernodal solver for LLᵀ, LDLᵀ and LU with static pivoting, with low-rank compression of supernodes to cut memory or time to solution. It runs on its own threads, StarPU or PaRSEC, with MPI between processes. The 6.4.0 release (July 2024), still the latest, documents two limits: GPU kernels skip compressed supernodes, and the Schur complement is unavailable under MPI.

## Deep dives

Among the general-purpose libraries, only ButterflyPACK builds and factors compressed matrices across MPI ranks. Htool-DDM distributes assembly and matvec but factors on one node; HiCMA distributes all its TLR operations; MUMPS and PaStiX distribute their internal BLR factorizations. None of the eight provides square roots or other matrix functions (the excluded hm-toolbox does), and only hmat-oss offers all four of addition, shift, multiplication and inversion, sequentially.

Findings come from each package’s source code, except MUMPS 5.9 features (changelog and user guide) and published run sizes (papers). ButterflyPACK, Htool-DDM, hmat-oss, H2Pack, STARS-H and MUMPS 5.7.3 were also built and tested; PaStiX was read through its published source listings, and HMatrices.jl without a Julia install.

### Construction

Entry or block callbacks are the common route. Only ButterflyPACK can also build from matvecs alone, and only the two sparse solvers start from a sparse matrix.

| Package | Formats | Sparse matrix | Entry or block callback | Matvecs only | Geometry |
| --- | --- | --- | --- | --- | --- |
| ButterflyPACK | HODLR/HODBF, H/H-BF, BLR/B-BF, HSS-BF/SHNBF, H², tensor variants | ✗ | ✓ all formats: ACA family, SVD or RRQR for low-rank blocks; ID with proxy sampling for butterflies | ✓ HODLR/HODBF; H and BLR with low-rank blocks only | Optional (Gram distance or user clustering); H² needs 1–3D points |
| Htool-DDM | H | ✗ | ✓ partial or full ACA, SVD, optional recompression | ✗ | Required |
| hmat-oss | H, including HODLR-shaped and all-low-rank tilings | ✗ | ✓ ACA variants, SVD; RRQR fails at tight tolerances | ✗ | Required |
| H2Pack | H², HSS | ✗ | ◐ kernel callback on points only; proxy-point or sample-point ID | ✗ | Required, up to 3D |
| HMatrices.jl | H, Hermitian half storage | ✗ | ✓ partial ACA, truncated SVD | ✗ | Required |
| HiCMA / HiCMA-PaRSEC | TLR; band-dense + TLR | ✗ | ◐ each tile formed densely, then randomized SVD (O(N²) evaluations); no documented way to add a kernel to HiCMA-PaRSEC beyond its 16 built-in problems | ✗ | None: flat tiles, user orders the points |
| MUMPS | BLR fronts, internal | ✓ assembled input; elemental input cannot use BLR | ✗ | ✗ | Graph-based clustering |
| PaStiX | BLR supernodes, internal | ✓ | ✗ | ✗ | Graph-based clustering |

Dense matrices otherwise enter through the same callbacks; only STARS-H (feeding HiCMA) and the TLR-MVM Python tool take a dense array.

### Conversions

Only H2Pack (H² to HSS) and ButterflyPACK (rebuilds through callbacks) cross formats; the rest offer dense extraction, recompression and precision changes at most.

| Package | Conversions provided |
| --- | --- |
| ButterflyPACK | To dense (Fortran API only). A new format or tolerance is rebuilt from an existing compressed matrix through its entry-extraction or matvec callbacks. Double-complex to single-complex copy for HODLR, H and BLR. |
| Htool-DDM | To dense, diagonal and sub-blocks; per-leaf recompression on the same tree |
| hmat-oss | To dense blocks, entries and diagonal; in-place transpose; truncation to a looser tolerance; leaf coarsening at assembly |
| H2Pack | H² to an SPD HSS approximation of A + σI by sketching and recompression; save and load to files |
| HMatrices.jl | To dense; per-leaf recompression (internal function) |
| HiCMA / HiCMA-PaRSEC | TLR to dense; dense to TLR through STARS-H; per-tile precision conversion in HiCMA-PaRSEC |
| MUMPS | None callable; options store BLR factors in adaptive precision or factor in single precision |
| PaStiX | Dense Schur complement extracted from compressed factors; single-precision factors |

### Operations

✓ provided, ◐ limited (see notes), ✗ absent. For packages with several formats, the mark is for the best-supported format.

| Operation | ButterflyPACK | Htool-DDM | hmat-oss | H2Pack | HMatrices.jl | HiCMA | MUMPS | PaStiX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matvec | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Transpose or adjoint matvec | ✓ (not H²) | ◐ | ✓ | Symmetric only | ◐ adjoint only | ✗ | ✗ | ✗ |
| Block of vectors | ✓ | ✓ | ✓ | ✓ | ◐ column loop | ✗ | ✗ | ✗ |
| Direct solve | ✓ | ✓ one node | ✓ | ✓ HSS, one RHS | ✓ one vector | ✓ HiCMA only | ✓ | ✓ |
| Own iterative solver | ✓ GMRES, TFQMR, CG, BiCGSTAB | ✓ HPDDM with GenEO | ✗ | ◐ example PCG | ✗ | ✗ | ◐ refinement | ✓ GMRES, CG, BiCGSTAB |
| Square root, inverse root, other functions | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| A + B | ✗ | ◐ low-rank + H | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| A + σI | ✗ | ✗ | ✓ | ◐ inside factorization | ✓ | ✗ | ✗ | ✗ |
| A · B | ✗ | ◐ internal | ✓ | ✗ | ◐ internal | ✓ HiCMA only | ✗ | ✗ |
| Inversion | ✓ HODLR, HSS-BF | ✗ | ✓ | ✗ | ✗ | ✗ | ◐ selected entries | ✗ |
| LU | ✓ H, BLR | ✓ | ✓ | ✓ HSS | ✓ no pivoting | ◐ complex, pivots discarded, no solve | ✓ | ✓ static pivoting |
| Cholesky | ✗ | ✓ | ✓ | ✓ HSS | ✓ | ✓ | ◐ LDLᵀ form | ✓ |
| LDLᵀ | ◐ HODLR, H² | ◐ unreleased | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| QR, eigen, SVD | ◐ eigen | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Log-determinant | ✓ | ✗ | ◐ wrong values | ◐ overflows | ✗ | ✗ | ✓ | ✗ |
| Recompression | ✗ | ✓ | ✓ | ◐ via HSS | ◐ internal | ✗ | ✗ | ✗ |
| Schur complement | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ dense | ◐ dense, one MPI process |

1. **ButterflyPACK:** H and BLR LU solve only untransposed systems; the H² solver needs a transpose-symmetric matrix and an MPI build. SHNBF and tensor formats solve only iteratively, and eigenvalues come from PARPACK (double complex, Fortran API) or a dense ScaLAPACK route. With default options the forward operator is dropped, so H matvecs crash unless ErrSol=1 is set.
2. **Htool-DDM:** a transposed matvec on a rectangular matrix overflows the heap in user numbering; the distributed operator handles it correctly. H×H and triangular H-solves are undocumented routines in cluster numbering.
3. **hmat-oss:** log-det returns ½·log det A after LLᵀ and 0 after LDLᵀ. LU on symmetric storage, adjoint matvec on complex-symmetric storage, and adding symmetric into general storage all return wrong results silently. After LLᵀ, matvec applies the factor L, which OpenTURNS uses to sample Gaussian processes.
4. **H2Pack:** only symmetric real kernels on one point set; a nonsymmetric kernel is accepted silently and gives wrong products. Only HSS is factored, by ULV LU or Cholesky of A + σI. Its log-det multiplies pivots per node and returned ±inf at n = 3,000–10,000 in tests.
5. **HMatrices.jl:** LU and Cholesky have no pivoting and solve one vector at a time; adding a sparse matrix ignores its scalar in admissible blocks.
6. **HiCMA:** HiCMA has Cholesky, left-lower triangular solves and TLR×TLR multiplication in real double; HiCMA-PaRSEC has only Cholesky. Matvecs exist only in STARS-H and TLR-MVM, on their own objects, and log-dets only in ExaGeoStat. In HiCMA, a tile whose rank exceeds the hand-set maximum ends the run through exit(0).
7. **MUMPS and PaStiX:** solves use the compressed factors, and both compression thresholds are absolute by default, so A must be scaled. MUMPS also returns the determinant, selected entries of A⁻¹ and null-space bases with BLR on. It never compresses fronts under 1,000 variables, the Schur front or the ScaLAPACK root.
8. **Hermitian matrices:** Htool-DDM, HMatrices.jl and PaStiX factor them. hmat-oss, MUMPS and the ButterflyPACK H² solver handle complex symmetric only, and HiCMA’s Cholesky and H2Pack are real only.

### HPC

Only ButterflyPACK, HiCMA and the two sparse solvers factor across nodes. Apart from MUMPS 5.9, whose GPU coverage of BLR blocks is unknown, no release updates or recompresses low-rank blocks on a GPU. GPU paths touching compressed data: ButterflyPACK’s butterfly matvec, HiCMA-PaRSEC tasks with dense outputs, STARS-H compression and the standalone TLR-MVM.

| Package | Threads | MPI | GPU | Largest documented run |
| --- | --- | --- | --- | --- |
| ButterflyPACK | OpenMP in compression and butterfly kernels | Construction, matvec, factorization, solve; PaRSEC for H and BLR LU | Release: MAGMA batched butterfly matvec only. CUDA H² and HODLR on unreleased branches | 3D Helmholtz, 21.2M unknowns via STRUMPACK on 32 Cori nodes: 0.6–0.8 TB of factors at ε = 1e-3 vs 1.97 TB exact, but no faster (674–1,037 s vs 660 s). A 100M-unknown 2D EFIE (Liu and Yang, arXiv:1901.11371) is reported without timings or code version |
| Htool-DDM | OpenMP: assembly, matvec, LU, Cholesky | Assembly, matvec, Krylov and Schwarz; factorization on one node | ✗ | 524k-point assembly in 10 s on 16 threads (ε = 1e-10); no MPI timings published |
| hmat-oss | ✗; pins BLAS to one thread | ✗ | ✗ | No published timings. Survey author’s unpublished test: N = 100,000 covariance on one core: assembly 2.8 s, LLᵀ 24 s, solve 0.2 s |
| H2Pack | OpenMP and SIMD, all operations | ✗ | ✗ | 15.4M unknowns (periodic RPY) on 24 cores: build ≈40 s, matvec ≈8.6 s at ε = 1e-4 |
| HMatrices.jl | Assembly and matvec only | ✗ (Distributed.jl path appears unable to run) | ✗ | 100k points in the README; no timings |
| HiCMA / HiCMA-PaRSEC | StarPU or PaRSEC tasks | All operations, 2D block-cyclic | HiCMA ✗; HiCMA-PaRSEC CUDA for tasks with dense outputs | Sparse TLR Cholesky, 52.6M at 1e-4 on 2,048 Shaheen II nodes in 36 min; TLR 42M at 1e-8 on 4,096 nodes in ≈24 h (Lorapo, HiCMA-PaRSEC’s predecessor). The Gordon Bell exascale runs were dense, not TLR |
| MUMPS | OpenMP and BLAS, tree parallelism | Factorization and solve; BLR clustering on one rank | 5.9 offload through XKBlas; BLR coverage unknown | 310M complex unknowns at BLR ε = 1e-7: 54 min, 28.6 TB on 100 Adastra nodes |
| PaStiX | Internal threads, StarPU, PaRSEC | BLR only with the internal schedulers (not StarPU or PaRSEC); analysis centralized | CUDA for uncompressed blocks only | Published BLR runs are single-node: up to 4.4× less memory, 3.3× faster |

## Adjacent packages

These are maintained and useful, but each is a wrapper, an application, a matvec-only method or a building block rather than a new hierarchical-matrix library.

| Package | What it is | Why it is not on the shortlist | Activity |
| --- | --- | --- | --- |
| PETSc [MATHTOOL](https://petsc.org/release/manualpages/Mat/MATHTOOL/) and [MATH2OPUS](https://petsc.org/release/manualpages/Mat/MATH2OPUS/) | PETSc matrix types wrapping Htool and H2Opus, created from a kernel or, for H2Opus, from another matrix | An access route to libraries already covered | Part of PETSc releases |
| [HierBEM](https://github.com/jihuan-tian/hierbem) | 3D Galerkin BEM on deal.II with its own H-matrix algebra and CUDA quadrature | Application library; its H-matrix code serves its BEM | 1.0.0; 268 commits since 2024 |
| [Bembel](https://github.com/temf/bembel) | Isogeometric Galerkin BEM (GPL-3) for Laplace, Helmholtz and Maxwell with an embedded FMM and OpenMP assembly | Compression tied to its discretization; no standalone matrix algebra | 1.2; last commit Jul 2026 |
| [FMM3D](https://github.com/flatironinstitute/FMM3D), [PVFMM](https://github.com/dmalhotra/pvfmm) | Production fast multipole libraries for Laplace, Helmholtz and Stokes kernels | Kernel matvec only: no stored compressed matrix, arithmetic or factorization | FMM3D 2.1.0 (tagged Nov 2024), last commit Sep 2026; PVFMM 1.3.0 (Nov 2021), last commit Oct 2026 |
| [ExaGeoStat](https://github.com/ecrc/exageostat) | Gaussian-process likelihood and prediction on Chameleon and HiCMA | An application of the TLR stack | 1.2.1; last commit Mar 2025 |
| [KBLAS](https://github.com/ecrc/kblas-gpu) | Batched GPU BLAS and LAPACK kernels that H2Opus builds on | A building block | 4.0.0; last commit Mar 2023 |

## Excluded

Most candidates fell short on maintenance: 13 of the 19 below have had no commits after 2021. The rest are research testbeds or prototyping toolboxes.

| Package | Reason | Last commit |
| --- | --- | --- |
| [HLR](https://github.com/rkriemann/hlr) | Active research testbed for HLIBpro algorithms; builds on HLIBpro, no releases | Jul 2026 |
| [hm-toolbox](https://github.com/numpi/hm-toolbox) | Sequential MATLAB HODLR/HSS prototyping toolbox; the only maintained package found with sqrtm, expm, QR and HODLR↔HSS conversion | Nov 2025 |
| [HierarchicalMatrices.jl](https://github.com/JuliaMatrices/HierarchicalMatrices.jl) | Experimental; 3 commits since 2024 | Oct 2024 |
| [Hatrix](https://github.com/rioyokotalab/Hatrix) | Research code | Mar 2024 |
| [HssMatrices.jl](https://github.com/bonevbs/HssMatrices.jl) | Inactive | Oct 2023 |
| [FLAM](https://github.com/klho/FLAM) | MATLAB recursive-skeletonization research code | May 2023 |
| [exafmm-t](https://github.com/exafmm/exafmm-t) | FMM, matvec only, inactive | Sep 2021 |
| [RLCM](https://github.com/jiechenjiechen/RLCM) | Research code | Apr 2021 |
| [ButterflyLab](https://github.com/ButterflyLab/ButterflyLab) | MATLAB butterfly research code | Oct 2020 |
| [DMHM](https://github.com/YingzhouLi/dmhm) | Distributed H-matrix research code | Sep 2020 |
| [HODLRlib](https://github.com/sivaramambikasaran/HODLR) | C++ HODLR library, inactive | Jul 2020 |
| [spaND](https://github.com/leopoldcambier/spaND_public) | Sparse low-rank research code | Oct 2019 |
| [GOFMM / HMLP](https://github.com/ChenhanYu/hmlp) | Research code | Apr 2019 |
| [BBFMM3D](https://github.com/ruoxi-wang/BBFMM3D) | FMM research code | Sep 2018 |
| [hmglib](https://github.com/zaspel/hmglib) | CUDA H-matrix research code | Jul 2018 |
| [HACApK GPU](https://github.com/hoshino-UTokyo/hacapk-gpu) | Inactive GPU fork of HACApK | Jul 2018 |
| [HMat (Yang, Li)](https://github.com/YingzhouLi/HMat) | MATLAB research code | Jan 2018 |
| [Dense\_HODLR](https://github.com/amiraa127/Dense_HODLR) | Inactive | May 2015 |
| [AHMED](https://github.com/xantares/ahmed) | Unmaintained mirror | Sep 2013 |

The original [HLib](http://hlib.org/) is also out: H2Lib and HLIBpro superseded it.

## Sources

- Repository READMEs, changelogs and git history, read 8 Oct 2026: [ButterflyPACK](https://github.com/liuyangzhuan/ButterflyPACK), [Htool-DDM](https://github.com/htool-ddm/htool), [hmat-oss](https://github.com/jeromerobert/hmat-oss), [H2Pack](https://github.com/scalable-matrix/H2Pack), [HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl), [HiCMA](https://github.com/ecrc/hicma), [HiCMA-PaRSEC](https://github.com/ecrc/hicma-x), [STARS-H](https://github.com/ecrc/stars-h), [TLR-MVM](https://github.com/ecrc/tlrmvm), [HierBEM](https://github.com/jihuan-tian/hierbem), [Bembel](https://github.com/temf/bembel). Last-commit dates in all tables come from each repository’s default branch.
- [MUMPS download page and release notes](https://mumps-solver.org/index.php?page=dwnld)
- [PaStiX 6.4.0 handbook](https://solverstack.gitlabpages.inria.fr/pastix/)
- [Htool-DDM documentation](http://htool-ddm.pages.math.cnrs.fr/)
- PETSc manual pages for [MATHTOOL](https://petsc.org/release/manualpages/Mat/MATHTOOL/) and [MATH2OPUS](https://petsc.org/release/manualpages/Mat/MATH2OPUS/)
- [Awesome hierarchical matrices](https://github.com/gchavez2/awesome_hierarchical_matrices), the curated list used as a starting point

Deep-dive sources beyond the repositories above:

- ButterflyPACK: [Liu, Ghysels, Claus, Li, SISC 2021](https://arxiv.org/pdf/2007.00202); [Liu et al., SISC 2021](https://arxiv.org/pdf/2002.03400); [hosted documentation, still v3.2.0](https://portal.nersc.gov/project/sparse/butterflypack/)
- Htool-DDM: [JOSS paper](https://doi.org/10.21105/joss.09279); [benchmark data](https://github.com/PierreMarchand20/htool_benchmarks)
- hmat-oss: [OpenTURNS HMatrix](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HMatrix.html); [Carratalá-Sáez thesis, 2021](https://www.tdx.cat/bitstream/handle/10803/671577/2021_Tesis_Carratala%20Saez_Rocio.pdf)
- H2Pack: [Huang, Xing, Chow, TOMS 2020](https://faculty.cc.gatech.edu/~echow/pubs/h2pack.pdf); [Xing, Huang, Chow, JCP 2022](https://faculty.cc.gatech.edu/~echow/pubs/xing-huang-chow-jcp-2021.pdf); [project wiki](https://github.com/scalable-matrix/H2Pack/wiki)
- HMatrices.jl: [documentation](https://integralequations.github.io/HMatrices.jl/stable/); [Inti.jl](https://github.com/IntegralEquations/Inti.jl)
- HiCMA: [Cao et al., PASC 2020](https://www.netlib.org/utk/people/JackDongarra/PAPERS/extreme-cholesky-pasc-2020.pdf); [Cao et al., IPDPS 2022](https://netlib3.cs.utk.edu/files/publications/2022/icl-utk-1556-2022.pdf); [SC24 climate emulator](https://arxiv.org/html/2408.04440v2); [ExaGeoStat](https://github.com/ecrc/exageostat)
- MUMPS: [user guide 5.9.1](https://mumps-solver.org/doc/userguide_5.9.1.pdf); [Faucher et al., 2025](https://arxiv.org/pdf/2511.22530); [Amestoy et al., Geophysics 2016](https://www-pequan.lip6.fr/~tmary/doc/GEO16.pdf)
- PaStiX: [GPU guard for compressed blocks](https://solverstack.gitlabpages.inria.fr/pastix/codelet__blok__dgemmsp_8c_source.html); [Pichon et al., J. Comput. Sci. 2018](https://doi.org/10.1016/j.jocs.2018.06.007); [Nies and Hoelzl, 2019](https://arxiv.org/abs/1907.13442)
