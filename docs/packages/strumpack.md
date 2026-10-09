# STRUMPACK H-Matrix Capabilities

> Written in October 2026 by Nick Alger. Reflects STRUMPACK v8.1.0 (source, released 5 Oct 2026); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

STRUMPACK compresses matrices in six rank-structured formats, plus floating-point (ZFP) compression; HSS, BLR, HODLR and HODBF can then be factored and solved, while Butterfly and LR support only matvecs. It has no H-matrix arithmetic, no matrix functions, and no GPU path of its own for its hierarchical formats. Its strongest HPC component is the sparse multifrontal solver.

Findings were checked against the [v8.1.0 source](https://github.com/pghysels/STRUMPACK), released Oct 5, 2026. The [v7.1.0 manual](https://portal.nersc.gov/project/sparse/strumpack/v7.1.0/index.html) and parts of the in-code documentation are out of date; where they disagree, this doc follows the code. The most reliable capability reference is the dispatch code in `src/structured/StructuredMatrix.cpp`.

## Formats in the package

Seven formats sit behind the common `structured::StructuredMatrix` interface. HODLR, HODBF, Butterfly and LR are wrappers around [ButterflyPACK](https://github.com/liuyangzhuan/ButterflyPACK) and require an MPI build.

| Type | Class | Structure | Sequential/threaded | MPI |
| --- | --- | --- | --- | --- |
| HSS | `HSS::HSSMatrix`, `HSS::HSSMatrixMPI` | Weak admissibility, nested bases (interpolative decompositions) | ✓ | ✓ (needs ≥ 2 ranks) |
| BLR | `BLR::BLRMatrix`, `BLR::BLRMatrixMPI` | Flat tiling with low-rank off-diagonal tiles; optional per-tile admissibility mask | ✓ | ✓ (native class only) |
| HODLR | `HODLR::HODLRMatrix` | Weak admissibility, no nested bases; square only | — | ✓ (1 rank is fine) |
| HODBF | `HODLR::HODLRMatrix` | HODLR with butterfly off-diagonal blocks; the generic interface uses butterflies at all levels, but native HODLR options can set fewer | — | ✓ |
| BUTTERFLY | `HODLR::ButterflyMatrix` | One butterfly-compressed block, can be rectangular | — | ✓ |
| LR | `HODLR::ButterflyMatrix` | One global low-rank factorization | — | ✓ |
| LOSSY, LOSSLESS | `LossyMatrix` | ZFP/SZ3 floating-point compression of a dense matrix | ✓ | — |

LOSSY and LOSSLESS are included for completeness; they compress floating-point values, not ranks. The package has no strong-admissibility H-matrices and no H² format other than HSS. ButterflyPACK itself has an H-matrix format with H-LU, but STRUMPACK's wrapper uses only its HODLR/HODBF and single-block modes.

## Construction

Most formats can be built from an entry oracle, only the ButterflyPACK formats from matvecs alone, and nothing from a sparse matrix.

**From a sparse matrix: no.** No routine turns a CSR matrix into a `StructuredMatrix`. Sparse input only goes to the multifrontal solver (`SparseSolver`, `SparseSolverMPIDist`). That solver compresses the dense frontal matrices of its LU factorization with HSS, BLR, HODLR/HODBF, BLR+HODLR mixes, or ZFP.

The result is an approximate factorization behind `solve()`, used directly or as the preconditioner in the built-in refinement, GMRES or BiCGStab. The compressed blocks are internal; you never get an H-matrix of A or A⁻¹ to operate on.

**From an entry oracle, without forming the dense matrix: yes, for most formats.**

- **BLR (sequential):** runs ACA or BACA on each admissible tile.
- **ButterflyPACK formats (MPI):** HODLR, HODBF, Butterfly and LR use ButterflyPACK's entry-based algorithms.
- **HSS:** efficient only if you also supply point coordinates, or use the `kernel::Kernel` class, which drives an approximate-nearest-neighbor (ANN) sampling scheme. Without coordinates, every sampling pass evaluates all N² entries; they are never stored, but the code warns about the O(N²) cost.

**From matrix-vector products: only the ButterflyPACK formats.** HODLR, HODBF, Butterfly and LR can be built purely from products; the callback must supply products with both A and Aᵀ/Aᴴ. HSS needs products with A and Aᴴ plus entry access ("partially matrix-free"). BLR cannot use matvecs at all.

| Format | Dense in memory | Entries only | Matvec only | Matvec + entries | Points + kernel |
| --- | --- | --- | --- | --- | --- |
| HSS, sequential | ✓ randomized, O(N²r) | ✓ ANN-based with coordinates; O(N²) evaluations without | ✗ | ✓ | ✓ |
| HSS, MPI | ✓ | ✓ O(N²) through the generic interface; ANN only via Points + kernel | ✗ | ✓ | ✓ |
| BLR, sequential | ✓ RRQR or ACA per tile | ✓ ACA/BACA per admissible tile | ✗ | ✓ (matvec ignored) | — |
| BLR, MPI | ✓ (`from_ScaLAPACK` only) | ✗ | ✗ | ✗ | — |
| HODLR, HODBF | ✓ | ✓¹ | ✓ | ✗ | ✓ |
| Butterfly, LR | ✓ | ✓¹ | ✓ | ✗ | — |
| Lossy | ✓ | ✗ | ✗ | ✗ | — |

"Points + kernel" means the `kernel::Kernel` class: points plus a kernel function, with λI added on the diagonal. Its `eval(i, j)` is virtual, so any entry oracle that comes with point coordinates can use the ANN algorithms on this path.

¹ Through `StructuredMatrix`, per-entry callbacks work, but the distributed block-extraction overload throws "Not implemented yet". The native `compress()` overloads on `HODLRMatrix` and `ButterflyMatrix` do accept multi-block extraction.

Other construction details:

- **Cluster trees:** every constructor takes an optional `ClusterTree`; `binary_tree_clustering` builds one from coordinates using 2-means, k-d tree, PCA or cobble partitioning.
- **Factored BLR:** `construct_and_factor_from_dense` and `construct_and_factor_from_elements` compress and LU-factor in one pass. Through the generic interface, this is the only way to get a factored BLR.
- **C and Fortran interfaces:** these support dense and entry-based construction only, not matvec-based.

## Conversion between formats

There are no direct converters between any pair of formats; conversions go through dense or through another format's construction callbacks.

| Conversion | Provided? |
| --- | --- |
| Any format → dense | ✓ `dense()` on HSS, BLR, HODLR and Butterfly (the MPI classes return ScaLAPACK matrices); `to_ScaLAPACK()` on BLR-MPI; `decompress()` on Lossy |
| Dense → any format | ✓ `construct_from_dense`, `BLRMatrixMPI::from_ScaLAPACK` |
| BLR with dense tiles → compressed BLR | ✓ `compress()` |
| Format X → format Y, any pair | ✗ |
| Sequential ↔ MPI, same format | ✗ |

Beyond going through dense, the only route is to pass one object's `mult` or entry extraction as the callback to another format's constructor. The signatures line up, but no routine does this for you. HODLR objects also carry an internal permutation (`perm()`/`iperm()`) that you would have to account for.

## Operations

Beyond matvecs, solves, an HSS-only diagonal shift and BLR-only products formed in dense tiles, the formats support no H-matrix operations: no sums, matrix functions, QR or eigendecompositions.

| Operation | HSS (seq & MPI) | BLR seq | BLR MPI | HODLR / HODBF | Butterfly / LR | Lossy |
| --- | --- | --- | --- | --- | --- | --- |
| op(A)·X, op ∈ {N, T, C} | ✓ (complex T is applied as C) | ✓ | `gemv`, N only² | ✓ | ✓ | ✗ |
| Factorization | ULV-type, general nonsymmetric | LU, pivoting only within diagonal tiles; done together with compression | Same LU, as a separate `factor()` | Hierarchical inversion (ButterflyPACK) | ✗ | ✗ |
| A⁻¹·X | ✓ (no transposed solve) | ✓ (no transposed solve) | No `solve()`; `laswp`, `trsv` and `trsm` provided (N only) | ✓, plus op(A)⁻¹ via `inv_mult` | Via built-in GMRES/BiCGStab (1 RHS) | ✗ |
| A + σI | ✓ `shift()`; refactor afterwards | ✗ | ✗ | ✗ | ✗ | ✗ |
| A + B | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| A·B, both compressed | ✗ | ✓ result is dense, not recompressed | ✓ into a dense-tile BLR (N only), then `compress()` | ✗ | ✗ | ✗ |
| A^½, A^−½, other f(A) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| QR, eigendecomposition, SVD | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Determinant / log-det | ✗ | ✗ | ✗ | ✗³ | ✗ | ✗ |
| Entry/submatrix extraction | ✓ | ✓ | Dense tiles only | ✓ | ✓ | ✗ |

² Vectors are stored as `BLRMatrixMPI` objects. BLR-MPI implements none of the `StructuredMatrix` operations, so `mult` and `solve` through the generic interface are unavailable.

³ ButterflyPACK computes log|det| during its HODLR factorization, but STRUMPACK doesn't expose it.

Further details on operations:

- **Generic interface:** `StructuredMatrix` exposes only `mult`, `factor`, `solve` and `shift`.
  - HSS has all four.
  - Sequential BLR has `mult` and `solve`; `factor()` throws, so use `construct_and_factor_*` instead.
  - HODLR/HODBF have `mult`, `factor` and `solve`; Butterfly/LR have `mult` only.
  - Lossy only answers memory and rank queries.
  - The capability table in `StructuredMatrix.hpp` marks `shift` as "?" for BLR, HODLR and HODBF; in the code, all three throw.
- **Factors:** BLR LU factors stay in BLR tile form with an accessible pivot vector, usable through BLR `trsm`/`trsv`. The HSS ULV factors and HODLR inverse factors are internal.
- **Symmetric variants:** none of the dense formats has a Cholesky or LDLᵀ variant.
- **Kernel matrices:** λI is added at construction, and `fit_HSS`/`fit_HODLR` compress, factor and solve in one call.
- **Krylov solvers:** `strumpack::iterative` provides `GMRes` and `BiCGStab` (and MPI versions) that take any matvec and preconditioner, for one right-hand side. The dense examples precondition them with HSS, BLR or HODLR `solve()`.

## HPC suitability

STRUMPACK is strong at scale as a sparse direct solver and preconditioner, including on GPUs. Its dense hierarchical formats run on CPUs, apart from some butterfly products that a MAGMA-enabled ButterflyPACK can offload, and support a narrow operation set.

**Distributed memory**

- **HSS:** `HSSMatrixMPI` uses the ScaLAPACK 2D block-cyclic layout. It maps the HSS tree onto shrinking process subgrids, switching to threaded sequential HSS for subtrees owned by one rank. Compression, ULV factorization, solve and matvec are all distributed.
- **BLR:** `BLRMatrixMPI` distributes tiles block-cyclically over a 2D process grid and does a distributed LU. The generic interface can't create it, it only takes dense input, and it is mainly a building block for the sparse solver.
- **ButterflyPACK formats:** HODLR, HODBF, Butterfly and LR are distributed through ButterflyPACK, using 1D block-row vectors with helpers to convert to and from 2D layouts. STRUMPACK's own README points users with large dense problems to ButterflyPACK directly.
- **Sparse solver:** the most mature part at scale. It takes distributed CSR input, uses ParMETIS/PT-Scotch orderings, and stores distributed fronts in dense (ScaLAPACK/SLATE), HSS, BLR or HODLR form. It is also available through PETSc, Trilinos and MFEM.
- **Threads:** the native C++ formats use OpenMP task parallelism throughout.

**GPUs**

- **Dense formats:** the `StructuredMatrix` interface and STRUMPACK's HSS and HODLR code have no GPU path; ButterflyPACK built with MAGMA can offload some butterfly products. The BLR classes have undocumented public GPU factorization routines (`construct_and_partial_factor_gpu`, `partial_factor_gpu`) built for the sparse solver.
- **Sparse solver, dense fronts:** these run on NVIDIA (CUDA, cuBLAS, cuSOLVER), AMD (HIP/ROCm) and Intel (SYCL, oneMKL) GPUs. Options include MAGMA, SLATE for distributed fronts, and GPU-aware MPI; there is also a CUDA-only SPD Cholesky path. Inputs and outputs must be in host memory.
- **Sparse solver, compressed fronts:** since v8.0, BLR-compressed fronts also run on NVIDIA GPUs, using KBLAS batched randomized compression or, without KBLAS, per-tile cuSOLVER SVD. On HIP and SYCL builds this fallback is an unimplemented stub. HSS- and HODLR-compressed fronts still run on the CPU.
- **Documentation gap:** the manual's GPU page still says no rank-structured solver is GPU-accelerated, which is out of date.
- **Published results:** a 2025 [IJHPCA paper](https://www.osti.gov/pages/biblio/2499469-graphics-processing-unit-accelerated-sparse-direct-solver-preconditioner-block-low-rank-compression) reports runs on Perlmutter, Frontier and Aurora. On 32 Perlmutter nodes, for a high-frequency wave problem, the exact GPU factorization was about 6.5× faster than the CPU-only solver. The GPU BLR solver was about 13.8× faster than the exact CPU solver.

**Maturity**

- **Out-of-date documentation:** the [project website](https://portal.nersc.gov/project/sparse/strumpack/) still lists v7.1.0 as the latest release, and the capability table in `StructuredMatrix.hpp` disagrees with the code.
- **Unimplemented paths:** several MPI paths in `StructuredMatrix` throw "Not implemented yet".
- **Release pace:** v8.0.0 came out in November 2024 and v8.1.0 in October 2026, the latter described as mostly bug fixes.
- **Apparent bug:** in the dense interface, LOSSLESS builds the same lossy object as LOSSY (ZFP fixed-accuracy mode, tolerance 1e-3).

## Sources

- [STRUMPACK on GitHub](https://github.com/pghysels/STRUMPACK) (v8.1.0 source and CHANGELOG)
- [STRUMPACK v7.1.0 manual](https://portal.nersc.gov/project/sparse/strumpack/v7.1.0/index.html)
- [STRUMPACK website](https://portal.nersc.gov/project/sparse/strumpack/)
- [A GPU-accelerated sparse direct solver and preconditioner with block low rank compression](https://www.osti.gov/pages/biblio/2499469-graphics-processing-unit-accelerated-sparse-direct-solver-preconditioner-block-low-rank-compression) (IJHPCA, via OSTI)
- [ButterflyPACK on GitHub](https://github.com/liuyangzhuan/ButterflyPACK)
