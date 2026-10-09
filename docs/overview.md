# H-matrix packages: overview and comparison

> Written in October 2026 by Nick Alger. Reflects the twelve package docs, each of which names the versions it read; later versions may differ. [All background docs](README.md)

The twelve packages fall into four groups. Single-node H-matrix libraries: HLIBpro, H2Lib, hmat-oss and HMatrices.jl. Distributed or GPU structured-matrix packages: STRUMPACK, ButterflyPACK, H2Opus, HiCMA and Htool-DDM. A kernel-only H² package: H2Pack. Sparse direct solvers that compress their own fronts or supernodes: MUMPS and PaStiX.

None of the twelve provides square roots, inverse square roots or other matrix functions. Only HLIBpro (threaded), H2Lib (sequential, with experimental H² arithmetic) and hmat-oss (sequential) offer addition, multiplication and inversion together, all on one node. Under MPI only single operations exist, such as HiCMA’s TLR multiplication and ButterflyPACK’s HODLR and HSS-BF inversion. Distributed factorization of dense compressed matrices comes from STRUMPACK (HSS, BLR), ButterflyPACK (HODLR, H, BLR, H²) and HiCMA (tile low-rank Cholesky).

Only STRUMPACK, ButterflyPACK and H2Opus build from matvecs alone. GPU work on compressed data is rare: H2Opus runs H² and tile low-rank on NVIDIA GPUs but has been dormant since 2022, and STRUMPACK compresses BLR fronts on NVIDIA GPUs inside its sparse solver. Each package has its own doc with the full findings; the table below links them.

## At a glance

Each package name links to its full doc. “Parallel reach” describes the operations on compressed data, not only assembly.

| Package | Formats | Builds from | Factorizations | Parallel reach | Release · activity |
| --- | --- | --- | --- | --- | --- |
| [STRUMPACK](packages/strumpack.md) | HSS, BLR, HODLR/HODBF and butterfly (through ButterflyPACK); sparse multifrontal | Entries, dense, kernel + points; matvecs for the ButterflyPACK formats; sparse matrices for its solver only | HSS ULV, BLR LU, HODLR inversion; sparse LU with compressed fronts | MPI + OpenMP; GPUs for sparse fronts; BLR fronts on NVIDIA only | 8.1.0, Oct 2026 |
| [ButterflyPACK](packages/butterflypack.md) | HODLR/HODBF, H, BLR, HSS-BF/SHNBF, H², tensor variants | Entries, matvecs | HODLR/HODBF inversion, H and BLR LU, symmetric H² solver; log-det | MPI + OpenMP; PaRSEC for H and BLR LU | 5.0.0, Aug 2026 |
| [H2Opus](packages/h2opus.md) | H², tile low-rank | Kernel + points (H²), matvecs (symmetric H²), entries (TLR) | TLR Cholesky and LDLᵀ only | MPI + NVIDIA GPUs for H² matvec and recompression | No releases; dormant since Oct 2022 |
| [HiCMA / HiCMA-PaRSEC](packages/hicma.md) | Tile low-rank | Kernel tiles through STARS-H | TLR Cholesky; pivot-free complex LU (HiCMA) | StarPU or PaRSEC over MPI; CUDA for dense tasks | 1.0.0 (Jan 2023) / 0.1.0 (Nov 2025) |
| [Htool-DDM](packages/htool-ddm.md) | H | Entries | H-LU, H-Cholesky; LDLᵀ unreleased | MPI for assembly, matvec and DDM solves; factorization on one node | 1.0.2, Feb 2026 |
| [HLIBpro](packages/hlibpro.md) | H (any admissibility, HODLR, MBLR); H² for matvec and iterative solves only | Entries, sparse, dense | LU, LDU, Cholesky, LDLᴴ, inversion | TBB threads; MPI claimed but undocumented | 3.2 (2025 or later); closed source |
| [H2Lib](packages/h2lib.md) | H, H², directional H² | BEM kernels, sparse (FEM), dense; entries one block at a time | H and H² LU, Cholesky and inversion | Sequential arithmetic; OpenMP assembly, H² recompression and DH² matvec | 3.0.1 (2016); master updated Jun 2026 |
| [hmat-oss](packages/hmat-oss.md) | H, including HODLR layouts | Entries | LU, LLᵀ, LDLᵀ, HODLR, inversion | Sequential | 1.11.1, Nov 2025 |
| [HMatrices.jl](packages/hmatrices-jl.md) | H | Entries | LU and Cholesky without pivoting | Julia threads for assembly and matvec | 0.2.13, Aug 2025 |
| [H2Pack](packages/h2pack.md) | H², HSS | Kernel + points only | HSS ULV LU and Cholesky | OpenMP + SIMD | No tags; updated Sep 2026 |
| [MUMPS](packages/mumps.md) | BLR fronts, internal | Sparse matrices | LU, LDLᵀ | MPI + OpenMP; GPU offload since 5.9 | 5.9.1, Jul 2026 |
| [PaStiX](packages/pastix.md) | BLR supernodes, internal | Sparse matrices | LLᴴ, LDLᵀ/LDLᴴ, LU with static pivoting | Threads, StarPU/PaRSEC; BLR under MPI only with internal schedulers; GPUs for full-rank blocks | 6.4.0, Jul 2024 |

HODBF: HODLR with butterfly blocks; HSS-BF, SHNBF: ButterflyPACK’s nested butterfly formats; MBLR: multilevel BLR.

## Landscape

![Grid placing the twelve packages by the operations they offer on compressed matrices (rows) and the parallel reach of those operations (columns)](figures/overview-landscape.svg)

*Twelve packages by operations on compressed matrices and parallel reach · 3 rows, 4 columns*

<details>
<summary>The same placement as a table</summary>

| Operations | One core | Threads | MPI | MPI + GPUs |
| --- | --- | --- | --- | --- |
| **Full arithmetic**: A + B, A·B, inversion and factorizations | H2Lib, hmat-oss | HLIBpro | None | None |
| **Apply and factor**: factor, solve or apply; little or no arithmetic |  | H2Pack, HMatrices.jl | STRUMPACK, ButterflyPACK, Htool-DDM, HiCMA | H2Opus |
| **Sparse solvers**: compression stays inside the factors |  |  | MUMPS, PaStiX | STRUMPACK sparse |

</details>

Placement reflects released code and each package’s best-supported format. ButterflyPACK, HiCMA, MUMPS and PaStiX sit under MPI because their GPU paths cover only a matvec kernel, dense blocks or an offload of unknown reach. H2Opus reaches GPUs for H² matvec and recompression but factors only tile low-rank matrices.

## Choosing by need

Most needs are met by two or three packages, and one, matrix functions, by none.

| Need | Packages that provide it | Watch for |
| --- | --- | --- |
| H-arithmetic: A + B, A·B, inversion | HLIBpro, H2Lib, hmat-oss | One node only. HLIBpro is closed source, H2Lib’s H² arithmetic is experimental, and hmat-oss is sequential with silent wrong-result cases |
| Square roots or other matrix functions | None | Outside the twelve, the sequential MATLAB hm-toolbox has sqrtm and expm for HODLR and HSS; see the survey |
| Construction from matvecs alone | ButterflyPACK, STRUMPACK (its ButterflyPACK formats), H2Opus | ButterflyPACK and STRUMPACK need products with A and Aᵀ; H2Opus builds only symmetric H², on one node |
| A sparse matrix turned into an H-matrix | HLIBpro, H2Lib | MUMPS, PaStiX and STRUMPACK compress only inside their own factorizations |
| Distributed factorization of a dense compressed matrix | ButterflyPACK, STRUMPACK (HSS, BLR), HiCMA (TLR Cholesky) | ButterflyPACK’s H² solver needs transpose-symmetric matrices; HiCMA-PaRSEC has no TLR solve; STRUMPACK’s distributed BLR needs the dense matrix and offers only triangular solves |
| GPUs on compressed data | H2Opus (H², TLR), STRUMPACK (BLR sparse fronts) | Both are NVIDIA-only; H2Opus has been dormant since 2022 |
| Log-determinant | ButterflyPACK, MUMPS | hmat-oss returns wrong values after LDLᵀ, and H2Pack’s overflows or underflows at realistic sizes |
| Hermitian factorizations | HLIBpro, H2Lib (complex builds), Htool-DDM, HMatrices.jl, PaStiX | hmat-oss, MUMPS and ButterflyPACK’s H² solver treat complex matrices as complex symmetric |
| Built-in Krylov or domain-decomposition solvers | Htool-DDM (HPDDM with GenEO), ButterflyPACK, PaStiX, STRUMPACK, HLIBpro, H2Lib | MUMPS offers only iterative refinement |
| Oscillatory kernels | ButterflyPACK (butterflies), H2Lib (directional H²), HLIBpro (frequency-aware admissibility) | — |
| Fast O(N) kernel matvecs | H2Pack (one node), H2Opus (multi-GPU) | H2Pack needs symmetric kernels in up to 3D; its proxy-point builds also assume translation invariance |
| Sparse direct solver with compression | MUMPS (BLR), PaStiX (BLR), STRUMPACK (BLR, HSS, HODLR, butterfly fronts) | MUMPS and PaStiX use absolute tolerances by default |

## Construction compared

Entry callbacks are the common route; matvec-only construction is rare, and only HLIBpro and H2Lib turn a sparse matrix into a user-visible H-matrix. Marks: ✓ provided, ◐ limited, ✗ absent.

| Package | Entry callback | Kernel + points | Matvecs only | Dense array | Sparse matrix |
| --- | --- | --- | --- | --- | --- |
| STRUMPACK | ✓ sequential BLR, ButterflyPACK formats; HSS efficient only with coordinates; MPI BLR only from a dense ScaLAPACK matrix | ✓ HSS, HODLR (kernel class) | ✓ ButterflyPACK formats | ✓ | Solver only |
| ButterflyPACK | ✓ all formats | Through the callback | ✓ HODLR/HODBF; H and BLR with low-rank blocks | Through a callback | ✗ |
| H2Opus | TLR only, O(N²) evaluations | ✓ H² by Chebyshev interpolation | ✓ symmetric H², one node | ✗ | ✗ |
| HiCMA | ◐ every tile formed densely | ✓ built-in kernels (STARS-H) | ✗ | ◐ through STARS-H | ✗ |
| Htool-DDM | ✓ | Through a generator | ✗ | Through a generator | ✗ |
| HLIBpro | ✓ ACA, ACA+, HCA | ✓ built-in BEM and Matérn | ✗ | ✓ C API | ✓ |
| H2Lib | ◐ one block at a time | ✓ BEM kernels; kernel + points on master | ✗ | ✓ | ✓ FEM-type |
| hmat-oss | ✓ | Through a callback | ✗ | Through a callback | ✗ |
| HMatrices.jl | ✓ | ✓ `KernelMatrix` | ✗ | Through `AbstractMatrix` | ✗ |
| H2Pack | ✗ | ✓ only route | ✗ | ✗ | ✗ |
| MUMPS | ✗ | ✗ | ✗ | ✗ | Solver only |
| PaStiX | ✗ | ✗ | ✗ | ✗ | Solver only |

Conversions between formats are scarce everywhere. H2Lib converts among dense, H and H² (H to H² needs a strict block tree); HLIBpro converts H to H² but not back; H2Pack converts H² to an SPD HSS. ButterflyPACK rebuilds one of its formats as another through extraction or matvec callbacks. The rest offer at most dense extraction, recompression on the same tree, or precision changes.

## Operations compared

None of the twelve provides square roots or other matrix functions, QR, or an eigendecomposition of a compressed matrix; ButterflyPACK’s PARPACK and dense-conversion eigensolvers are the only eigenvalue routines. Marks are for each package’s best-supported format.

**Apply and solve**

| Package | Matvec | Transposed matvec | Direct solve | Package-provided iterative solve | Log-determinant |
| --- | --- | --- | --- | --- | --- |
| STRUMPACK | ✓ | ✓ (not BLR under MPI) | ✓ HSS, BLR, HODLR/HODBF | ✓ GMRes, BiCGStab with any format’s mult | ✗ |
| ButterflyPACK | ✓ | ✓ (not H²) | ✓ (not SHNBF or tensor formats) | ✓ GMRES, CG, TFQMR, BiCGSTAB | ✓ |
| H2Opus | ✓ | ✓ on one node | ◐ TLR only | ✗ | ✗ |
| HiCMA | ✗ | ✗ | ✓ HiCMA only, SPD | ✗ | ✗ |
| Htool-DDM | ✓ | ◐ | ✓ on one node | ✓ HPDDM with GenEO | ✗ |
| HLIBpro | ✓ | ✓ | ✓ | ✓ | ✗ (user code sums the factor diagonal) |
| H2Lib | ✓ | ◐ adjoint only | ✓ | ✓ CG, GMRES, preconditioned | ✗ |
| hmat-oss | ✓ | ✓ (wrong for complex symmetric) | ✓ | ✗ | ◐ wrong values |
| HMatrices.jl | ✓ | ◐ adjoint only | ✓ one vector | ✗ | ✗ |
| H2Pack | ✓ | ✓ (symmetric, same as matvec) | ✓ HSS, one vector | ◐ example code | ◐ overflows or underflows |
| MUMPS | ✗ | ✗ | ✓ | ◐ refinement | ✓ |
| PaStiX | ✗ | ✗ | ✓ | ✓ GMRES, CG, BiCGSTAB | ✗ |

**Arithmetic and factorizations**

| Package | A + B | A + σI | A · B | Inversion | LU | Cholesky or LDLᵀ |
| --- | --- | --- | --- | --- | --- | --- |
| STRUMPACK | ✗ | ◐ HSS only | ◐ BLR, dense result | ◐ HODLR | ✓ BLR; HSS ULV | ✗ dense formats |
| ButterflyPACK | ✗ | ✗ | ✗ | ✓ HODLR, HSS-BF | ✓ H, BLR | ◐ HODLR and H² symmetric variants |
| H2Opus | ✗ | ✗ | ◐ TLR only | ✗ | ✗ | ◐ TLR only |
| HiCMA | ✗ | ✗ | ✓ TLR×TLR (HiCMA) | ✗ | ◐ complex, pivots discarded | ✓ Cholesky |
| Htool-DDM | ◐ low-rank + H; symmetric storage takes only symmetric updates | ✗ | ◐ internal; wrong on symmetric storage | ✗ | ✓ | ✓ Cholesky; LDLᵀ unreleased |
| HLIBpro | ✓ | ✓ | ✓ | ✓ | ✓ LU, LDU | ✓ Cholesky, LDLᴴ |
| H2Lib | ✓ H; ◐ H² | ✗ | ✓ H and H² | ✓ H and H² | ✓ H and H² | ✓ Cholesky (H, H²) |
| hmat-oss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ LLᵀ, LDLᵀ |
| HMatrices.jl | ✗ | ✓ | ◐ internal | ✗ | ✓ no pivoting | ✓ Cholesky |
| H2Pack | ✗ | ◐ inside factorization | ✗ | ✗ | ✓ HSS | ✓ HSS Cholesky |
| MUMPS | ✗ | ✗ | ✗ | ◐ selected entries | ✓ | ✓ LDLᵀ |
| PaStiX | ✗ | ✗ | ✗ | ✗ | ✓ static pivoting | ✓ LLᴴ, LDLᵀ, LDLᴴ |

## HPC compared

The largest compressed runs are sparse (MUMPS, 310M unknowns) or tile low-rank (HiCMA, 52.6M); among dense hierarchical formats, H2Opus has the strongest multi-node evidence. ButterflyPACK’s largest run compresses STRUMPACK’s sparse fronts.

| Package | Threads | MPI | GPU | Largest documented run |
| --- | --- | --- | --- | --- |
| STRUMPACK | OpenMP tasks | HSS, BLR, ButterflyPACK formats, sparse solver | Sparse fronts on NVIDIA, AMD and Intel GPUs; BLR fronts since 8.0, NVIDIA only; dense formats CPU only | GPU BLR sparse solve on 32 Perlmutter nodes, 13.8× faster than the exact CPU solver; problem size not stated |
| ButterflyPACK | OpenMP in compression and butterfly kernels | All operations; H² needs (2^d)^k ranks | MAGMA butterfly matvec only; CUDA H² and HODLR on unreleased branches | 21.2M-unknown 3D Helmholtz with butterfly fronts (through STRUMPACK) on 32 Cori nodes |
| H2Opus | OpenMP through Thrust | H² matvec, recompression and construction; symmetric, power-of-two rank counts | NVIDIA only: H² and TLR | Near-ideal scaling to 1,024 V100 GPUs on Summit; 16M-unknown fractional diffusion on 64 GPUs |
| HiCMA | StarPU or PaRSEC tasks | All operations | CUDA for dense-output tasks (HiCMA-PaRSEC) | Sparse TLR Cholesky of 52.6M points on 2,048 Shaheen II nodes in 36 min |
| Htool-DDM | OpenMP: assembly, matvec, LU, Cholesky | Assembly, matvec, DDM solve | ✗ | 524k-point assembly in 4.5 s on 16 threads (June 2026 benchmarks); no MPI timings published |
| HLIBpro | TBB task graphs | Claimed, undocumented | Dense SVD and QR only | 2M unknowns with H-Cholesky on a 20-core workstation |
| H2Lib | OpenMP for BEM assembly, H² recompression, the DH² matvec and dense LU/Cholesky; H and H² arithmetic sequential | ✗ | OpenCL for BEM matrix entries only | About 240k boundary unknowns (micromagnetics) |
| hmat-oss | ✗; BLAS pinned to one thread | ✗ | ✗ | This review’s test: N = 100,000 covariance on one core: assembly 2.8 s, LLᵀ 24 s, solve 0.2 s; residual 6.9e-6 |
| HMatrices.jl | Assembly and matvec | ✗ | ✗ | 100k points in the README, no timings |
| H2Pack | OpenMP and SIMD, all operations | ✗ | ✗ | 15.4M unknowns (periodic RPY) on 24 cores: matvec ≈8.6 s |
| MUMPS | OpenMP and tree parallelism | Factorization and solve | Offload since 5.9; BLR coverage unknown | 310M complex unknowns with BLR in 54 min on 100 Adastra nodes |
| PaStiX | Threads, StarPU, PaRSEC | BLR with the internal schedulers | Full-rank blocks only | Published BLR results are single-node: up to 4.4× less memory |

## Package summaries

**[STRUMPACK](packages/strumpack.md)** is LBNL’s sparse multifrontal solver with a dense structured-matrix interface: HSS, BLR and, through ButterflyPACK, HODLR, HODBF and butterfly. All offer matvec, and all but butterfly offer factorization and solve; there is no H-arithmetic, and only HSS has a diagonal shift. It is strongest at scale as a sparse solver and preconditioner, with NVIDIA GPUs for BLR-compressed fronts since 8.0.

**[ButterflyPACK](packages/butterflypack.md)** is LBNL’s MPI library for dense matrices compressed with low-rank and butterfly blocks, in HODLR, H, BLR and nested formats, plus a symmetric H² solver and tensor variants. It builds from entries or matvecs, factors every format except SHNBF and the tensor formats, and returns log-determinants, but has no arithmetic between compressed matrices. With default options, factorization drops the forward operator, and its released GPU support is one MAGMA butterfly-matvec kernel; CUDA backends are on branches.

**[H2Opus](packages/h2opus.md)** is KAUST’s GPU-oriented H² library: single- and multi-GPU matvecs, recompression and low-rank updates, construction from kernels or symmetric matvecs, and a tile low-rank module with Cholesky and LDLᵀ. It has no H² factorization or solve. It is NVIDIA-only and has had no commits since October 2022.

**[HiCMA and HiCMA-PaRSEC](packages/hicma.md)** form KAUST’s task-based tile low-rank stack, scaling TLR Cholesky of SPD kernel matrices to tens of millions of unknowns on thousands of CPU nodes. HiCMA (StarPU) adds TLR multiplication and triangular solves; HiCMA-PaRSEC adds mixed precision and sparse TLR but offers only Cholesky. Neither has a matvec, and GPUs run only dense tile tasks.

**[Htool-DDM](packages/htool-ddm.md)** is a header-only C++ H-matrix library built around domain decomposition: MPI-distributed assembly and matvec, HPDDM Krylov methods with GenEO coarse spaces, and threaded H-LU and H-Cholesky on one node. It powers PETSc’s MATHTOOL and FreeFEM’s BEM plugin. Its H-arithmetic is limited to undocumented internal routines; on symmetric storage, H×H and non-symmetric low-rank updates silently give wrong results.

**[HLIBpro](packages/hlibpro.md)** is Ronald Kriemann’s closed-source C++ library with the broadest H-arithmetic of the twelve: addition, multiplication, inversion, LU, LDU, Cholesky and LDLᴴ on any block structure, threaded with TBB. It builds from entries or sparse matrices but not from matvecs; its H² type supports only matvec, iterative solves and norm estimation, and MPI support is undocumented. Memory limits a 1 TB node to roughly 7–22 million unknowns with H-LU.

**[H2Lib](packages/h2lib.md)** is Steffen Börm’s open-source C library and goes deepest on H²: conversions among dense, H and H², H² multiplication, inversion, LU and Cholesky (experimental), and directional H² for high-frequency Helmholtz. It also ships preconditioned CG and GMRES. Its arithmetic is sequential, it has no MPI, and its last release dates from 2016, though master was updated in June 2026.

**[hmat-oss](packages/hmat-oss.md)** is Airbus’s open-source, strictly sequential H-matrix library with a C API: H×H, addition, shift, truncation, inversion, and LU, LLᵀ, LDLᵀ and HODLR factorizations. It underpins OpenTURNS’s covariance matrices, while Airbus keeps the parallel version proprietary. Testing found several silent wrong results and a buggy RRQR compressor.

**[HMatrices.jl](packages/hmatrices-jl.md)** is a pure-Julia H-matrix package for boundary integral equations and the backend of Inti.jl. It offers threaded assembly and matvec, shift and sparse addition, and serial LU and Cholesky without pivoting. Its distributed path appears unable to run.

**[H2Pack](packages/h2pack.md)** is Georgia Tech’s shared-memory library for symmetric kernel matrices in up to 3D, with linear-time H² construction by proxy points and very fast H² matvecs. Only its HSS format can be factored, and the HSS log-determinant overflows or underflows. It was updated in September 2026 after three quiet years.

**[MUMPS](packages/mumps.md)** is a widely used sparse direct solver with BLR compression: MPI + OpenMP LU and LDLᵀ, nearly every solver service on compressed factors, and runs up to 310M unknowns. Compression is internal, so there is no compressed matrix to apply or combine.

**[PaStiX](packages/pastix.md)** is Inria’s supernodal sparse solver, with internal BLR compression under three timing strategies, package-provided GMRES, CG and BiCGSTAB, and StarPU and PaRSEC runtimes. Compressed factorization never runs on GPUs and is MPI-distributed only with the internal schedulers. There has been no release since July 2024.

## Sources

Every claim here is summarized from the package docs, each of which lists its own sources (repositories, documentation and papers):

- [STRUMPACK](packages/strumpack.md), [H2Opus](packages/h2opus.md), [HLIBpro](packages/hlibpro.md), [H2Lib](packages/h2lib.md)
- [ButterflyPACK](packages/butterflypack.md), [Htool-DDM](packages/htool-ddm.md), [hmat-oss](packages/hmat-oss.md), [H2Pack](packages/h2pack.md)
- [HMatrices.jl](packages/hmatrices-jl.md), [HiCMA and HiCMA-PaRSEC](packages/hicma.md), [MUMPS](packages/mumps.md), [PaStiX](packages/pastix.md)
- [Survey of further H-matrix packages](survey.md), which covers the selection and the packages left out
