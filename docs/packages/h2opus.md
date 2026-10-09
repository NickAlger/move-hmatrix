# H2Opus H-Matrix Capabilities

> Written in October 2026 by Nick Alger. Reflects H2Opus at its final commit, cbc7985 (17 Oct 2022); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

H2Opus is a GPU-oriented H² library for fast matrix-vector products, recompression and low-rank updates, plus a tile low-rank (TLR) module with symmetric factorizations. It has no H² factorization or solve, and its only general H² arithmetic is reconstruction from matvecs. Development stopped in October 2022.

Findings were checked against the final source on [GitHub](https://github.com/ecrc/h2opus): 59 commits, no tagged releases, last commit Oct 17, 2022. PETSc's `MATH2OPUS` wrapper is maintained separately and adds a few capabilities; this doc marks those as PETSc-only.

## Formats in the package

H2Opus has three representations: single-node H², distributed H², and TLR.

| Format | Class | Structure | Single node (CPU / NVIDIA GPU) | MPI |
| --- | --- | --- | --- | --- |
| H² | `HMatrix`, `HMatrix_GPU` | Nested row/column bases plus coupling blocks. Weak, strong (η-based geometric) or user-defined admissibility. Square only; the block structure always comes from a k-d tree over a point set | ✓ | — |
| Distributed H² | `DistributedHMatrix`, `DistributedHMatrix_GPU` | Same, distributed by block rows; a single basis tree, so symmetric only | — | ✓ |
| TLR | `TTLR_Matrix<T, hw>` | Uniform tiles: dense diagonal tiles and independently compressed U·Vᵀ off-diagonal tiles. Nonsymmetric, symmetric or triangular | ✓ | — |

- **Symmetry:** single-node H² handles nonsymmetric matrices through separate U and V trees, but several operations are symmetric-only.
- **Arithmetic:** real only, in single or double precision chosen at build time; no complex support.
- **Not included:** HSS, HODLR, BLR or butterfly formats, and H-matrices without nested bases.

## Construction

H² is built from a kernel function plus points, or from matvecs for symmetric matrices; only TLR accepts an entry oracle, and nothing accepts a sparse matrix.

**From a sparse matrix: no, not in H2Opus itself.** PETSc's `MatCreateH2OpusFromMat` can sample any PETSc matrix, including a sparse one, through H2Opus's matvec-based construction. It only works on one MPI rank; without coordinates it clusters by index order.

**From an entry oracle: TLR only.**

- **TLR:** `construct_tlr_matrix` takes any A(i, j) functor. It evaluates every entry, one tile column at a time, and compresses each tile with pivoted QR. The dense matrix is never stored, but the cost is O(N²) evaluations.
- **H²:** `buildHMatrix` needs a kernel function κ(x, y) plus point coordinates, not an entry oracle. It evaluates κ at Chebyshev nodes inside each k-d tree box, so κ must accept arbitrary points, not just data points.
  - Initial ranks are pᵈ for p nodes per dimension (125 for p = 5 in 3D).
  - It is therefore normally followed by algebraic recompression with `hcompress`.

**From matrix-vector products: H² only, with restrictions.** `hara` (hierarchical adaptive randomized approximation) builds an H² from a black-box sampler that returns A·X.

- **Symmetric only:** the sampler never supplies Aᵀ·X, so the result must be symmetric. PETSc's source has a commented-out check stating that nonsymmetric sampling doesn't work.
- **Sibling blocks must be low rank:** HARA samples sibling blocks level by level, as if the matrix were weakly admissible. It then writes the result into any structure built beforehand from point coordinates with `buildHMatrixStructure`. The shipped example and PETSc both use a strong (box-center) admissibility structure.
- **Single node:** it runs on one CPU or one GPU, never distributed.
- **TLR:** has no matvec-based construction.

| Format | Kernel κ(x, y) + points | Entry oracle A(i, j) | Matvecs | Dense matrix | Sparse matrix |
| --- | --- | --- | --- | --- | --- |
| H² (single node) | ✓ `buildHMatrix` (Chebyshev interpolation) | ✗ | ✓ `hara` (symmetric; sibling blocks must be low rank) | ✗ | ✗ (PETSc: via sampling, 1 rank) |
| Distributed H² | ✓ `buildDistributedHMatrix` | ✗ | ✗ | ✗ | ✗ |
| TLR | — | ✓ `construct_tlr_matrix`, O(N²) evaluations | ✗ | ✗ (no dedicated routine) | ✗ |

Every H² needs point coordinates, even for `hara`, because the cluster tree is a median-split k-d tree. Kernel-based construction and structure building run on the CPU, and the result is copied to the GPU. H² vectors live in a permuted ordering exposed through `index_map`.

## Conversion between formats

H2Opus converts only between CPU and GPU copies and from H² to dense; it has no converter between H², distributed H² and TLR.

| Conversion | Provided? |
| --- | --- |
| CPU ↔ GPU (H², distributed H², TLR) | ✓ copy constructors |
| H² → dense | ✓ `expandHmatrix` (a debug utility, CPU only) |
| TLR → dense | ✗ (only in example code) |
| Dense → H² or TLR | ✗ |
| H² ↔ TLR | ✗ |
| Single-node H² ↔ distributed H² | ✗ |

The README presents `hara` as the general way to evaluate products and matrix expressions. Its sampler can be any black box, including an H² matvec, so it can rebuild another operator as a symmetric H², provided its sibling blocks are numerically low rank. It is not a dedicated converter.

## Operations

H² supports matvecs, recompression and low-rank updates but no factorization, solve or shift; TLR adds symmetric Cholesky and LDLᵀ factorizations and a recompressed product.

| Operation | H² (single node) | Distributed H² | TLR |
| --- | --- | --- | --- |
| op(A)·X, op ∈ {N, T} | ✓ `hgemv`, multiple vectors | ✓ `distributed_hgemv` (symmetric, so no transpose argument) | ✓ `tlr_gemv` |
| Basis orthogonalization and recompression | ✓ `horthog`, `hcompress` (absolute tolerance) | ✓ `distributed_horthog`, `distributed_hcompress` | — |
| A + s·U·Vᵀ (global low rank) | ✓ `hlru_global`, `hlru_sym_global`; ranks grow until you recompress | ✗ | ✗ |
| Local low-rank and dense-block updates | ✓ `hlru_sym` (symmetric only), `hlru_dense_block_update` | ✗ | ✗ |
| A + σI | ✗ | ✗ | ✗ |
| A + B, both compressed | Only by rebuilding with `hara` | ✗ | ✗ |
| A·B, both compressed | Only by rebuilding with `hara` | ✗ | ✓ `tlr_gemm`, recompressed to a tolerance |
| Factorization | ✗ | ✗ | Cholesky (`tlr_potrf`) and LDLᵀ (`tlr_sytrf`), symmetric only |
| A⁻¹·X | ✗ | ✗ | ✓ `tlr_potrs`, `tlr_sytrs`; `tlr_trsm` (lower triangular, left side) |
| A^½, A^−½, other f(A) | ✗ | ✗ | ✗ |
| QR, eigendecomposition, SVD | ✗ | ✗ | ✗ |
| Determinant / log-det | ✗ | ✗ | ✗ |
| Entry/submatrix extraction | ✗ | ✗ | ✗ |

Further details on operations:

- **`hara` as arithmetic:** the README presents `hara` as the way to multiply H² matrices and evaluate algebraic expressions. You write a sampler that applies, for example, x ↦ A(Bx) with two `hgemv` calls.
  - The examples ship samplers for A², Newton–Schulz inversion and a pseudo-inverse iteration, but these are example code, not library routines.
  - The result is always symmetric and single-node, and the operator's sibling blocks must be numerically low rank.
- **Inversion:** there is no H² inverse or solve. Approximate inverses come from Newton–Schulz iterations built on `hara`, in the `ns` example and in PETSc's `PCH2OPUS` preconditioner.
- **Shift:** there is no shift routine. The closest primitive is `hlru_dense_block_update`, which adds dense updates to the dense leaf blocks.
- **TLR factorizations:** both are left-looking, using batched adaptive randomized approximation for the block-column updates.
  - Cholesky has an optional Schur-compensation mode, plus a block-pivoted variant that runs only on the CPU.
  - LDLᵀ is unpivoted.
  - Both print timing tables to stdout unconditionally.
  - `tlr_gemm` is not exercised by any shipped example.
- **Eigenvalues:** none in H2Opus; one example calls SLEPc through the PETSc matrix type.
- **PETSc layer:** `MATH2OPUS` adds `MatMult`, `MatMultTranspose`, `MatMultAdd`, `MatScale` and `MatNorm`.
  - `MatMultTranspose` works in parallel only for symmetric matrices.
  - It also supports products with dense matrices, and wraps orthogonalization, compression and low-rank updates; the low-rank update is single-rank only.
  - It adds no shift, sum or factorization.

## HARA speed

HARA builds a symmetric operator with modest ranks in seconds on one P100 GPU at N ≈ 260,000. It is slow as the engine of iterative schemes such as Newton–Schulz inversion: about 30 minutes for a 2D problem at N = 65,536.

Published results, on NVIDIA P100 GPUs unless noted, from the [2019 HARA paper](https://epubs.siam.org/doi/10.1137/18M1210101) and the [2020 fractional-diffusion paper](https://scholarworks.aub.edu.lb/items/e83bf253-330c-4d20-82d6-3c732c41c477):

| Experiment | Size | Reported cost |
| --- | --- | --- |
| Rebuild a 2D covariance matrix from its matvecs, tolerance 10⁻² vs 10⁻⁴ | N up to 262,144 | Up to \~800 samples. Times only plotted (log axis 0.1–10 s); 10⁻⁴ takes about 7× longer |
| Build A² (two matvecs per sample) | N up to 262,144 | Up to \~250 samples. Plotted times span roughly 0.3–3 s; 10⁻⁴ takes at most 2× longer |
| Schur complement of a Poisson operator, 2D and 3D | S of size 1,024–4,096 | 34–397 samples. Sampling (sparse triangular solves) 3.5–141 s; HARA's own compression 0.03–0.8 s |
| Newton–Schulz inverse, 1D fractional diffusion | N = 131,072 | About 8 iterations, each under 2 s |
| Newton–Schulz inverse, 2D fractional diffusion | N = 65,536 | 5 iterations, about 1,771 s total; last iteration 763 s (hardware not stated) |
| Newton–Schulz inverse, 2D, three fractional orders | N = 25,921 | 421–529 s |

For scale, the same papers report one H² matvec at N ≈ 2.1 million taking under 40 ms on a P100.

**Cost structure (from the source)**

- **Peeling:** each sampling step applies the operator to a block of 16 or 32 random vectors, then subtracts the coarser levels already built, using an `hgemv` on the partial matrix.
- **Sample count:** about twice the rank of each level's sibling blocks, which can far exceed the final H² rank, plus one leaf-size batch for the diagonal blocks. That is O(k log N) in total; the HARA example runs below used about 270–2,000 samples.
- **Compression:** after every 16–32 columns of low-rank update, the whole matrix is re-orthogonalized and recompressed (`horthog` + `hcompress`). This O(k²N log N) work dominates at tight tolerances.
- **Ranks:** each level uses one uniform rank, the maximum over its blocks, and the randomized range finder overestimates ranks somewhat.

**Practical implications**

- **Black-box cost:** the operator is applied to hundreds or thousands of vectors, in blocks of 16 or 32. If it is expensive, sampling dominates; blocking the samples gave a 12× speedup in the Schur-complement case.
- **Builds versus matvecs:** estimated from the plots and the matvec timing, one build costs hundreds of matvec-equivalents.
- **Iterated use compounds:** every Newton–Schulz iteration is a full build. Each sample applies an order-8 to order-16 polynomial in A, and the iterates' ranks grow.
- **Dimension matters:** sibling-block ranks grow with interface size in 2D and 3D. At equal size (S = 1,024), the 3D Schur complement needed 152 samples versus 34 in 2D.
- **No scale-out:** HARA is single-node and symmetric-only. The authors' 2022 multi-GPU paper solved its fractional-diffusion problem with algebraic-multigrid-preconditioned Krylov rather than a HARA-built inverse.

## HARA accuracy and error compounding

A loose tolerance gives a proportionally less accurate approximation, not a catastrophic one. The achieved error rises with tree depth and problem size; the runs below do not establish the cause, and there is no worst-case guarantee.

**What the literature says**

- **The HARA paper:** it uses one threshold for every level and acknowledges that peeling errors could accumulate down the tree. It leaves the analysis to future work; its only evidence is a qualitative remark that thresholds were met.
- **Theory for classical peeling:** [Chen, Keles, Halikias, Musco, Musco and Persson (2024)](https://arxiv.org/abs/2407.04686) note that existing analyses cover only matrices that are exactly HODLR.
  - For range-finder-based peeling, which is what HARA uses, they report evidence of hard instances where the error roughly doubles per level. Doubling at each of the 12 levels below the root, at leaf size 64 and N = 2¹⁸, would give a factor of about 4,000.
  - Their robust variant, using generalized Nyström and "perforated" sketches, bounds the growth to (1+β) per level at the cost of more matvecs. With only O(k log N) matvecs, the guarantee is merely polynomial in N.

**Measured behavior**

H2Opus's own HARA example was run on its final commit (CPU build). It rebuilds a 2D exponential covariance matrix A and its square A² from matvecs, then estimates the spectral-norm error.

![Two panels: HARA's achieved divided by requested error against tree depth (5 to 11 levels) for tolerances 1e-1 to 1e-4, rebuilding A and building A squared](../figures/h2opus-hara-accuracy.svg)

*Measured with H2Opus's examples/hara/test\_hara.cpp (final commit, CPU build) · 2D grids, N = 1,024–65,536 · values from the run logs*

<details>
<summary>Data behind the figure</summary>

| Experiment | Tolerance | Tree levels | N | Sample cap | Achieved error | Achieved ÷ requested |
| --- | --- | --- | --- | --- | --- | --- |
| Rebuild A from its matvecs | 10⁻¹ | 5 | 1,024 | 128 | 2.193e-02 | 0.22 |
| Rebuild A from its matvecs | 10⁻¹ | 7 | 4,096 | 128 | 2.734e-02 | 0.27 |
| Rebuild A from its matvecs | 10⁻¹ | 9 | 16,384 | 128 | 8.970e-02 | 0.90 |
| Rebuild A from its matvecs | 10⁻¹ | 11 | 65,536 | 128 | 1.034e-01 | 1.03 |
| Rebuild A from its matvecs | 10⁻² | 5 | 1,024 | 128 | 1.355e-03 | 0.14 |
| Rebuild A from its matvecs | 10⁻² | 7 | 4,096 | 128 | 2.513e-03 | 0.25 |
| Rebuild A from its matvecs | 10⁻² | 9 | 16,384 | 128 | 4.551e-03 | 0.46 |
| Rebuild A from its matvecs | 10⁻² | 11 | 65,536 | 128 | 6.873e-03 | 0.69 |
| Rebuild A from its matvecs | 10⁻³ | 5 | 1,024 | 128 | 2.578e-04 | 0.26 |
| Rebuild A from its matvecs | 10⁻³ | 7 | 4,096 | 128 | 1.783e-04 | 0.18 |
| Rebuild A from its matvecs | 10⁻³ | 9 | 16,384 | 128 | 2.019e-04 | 0.20 |
| Rebuild A from its matvecs | 10⁻⁴ | 5 | 1,024 | 128 | 1.563e-05 | 0.16 |
| Rebuild A from its matvecs | 10⁻⁴ | 7 | 4,096 | 128 | 1.697e-05 | 0.17 |
| Rebuild A from its matvecs | 10⁻⁴ | 9 | 16,384 | 256, raised | 1.862e-05 | 0.19 |
| Rebuild A from its matvecs | 10⁻⁴ | 9 | 16,384 | 128, rerun at the default cap | 1.529e-04 | 1.53 |
| Build A² (two matvecs per sample) | 10⁻¹ | 5 | 1,024 | 128 | 4.288e-02 | 0.43 |
| Build A² (two matvecs per sample) | 10⁻¹ | 7 | 4,096 | 128 | 4.141e-02 | 0.41 |
| Build A² (two matvecs per sample) | 10⁻¹ | 9 | 16,384 | 128 | 8.544e-02 | 0.85 |
| Build A² (two matvecs per sample) | 10⁻¹ | 11 | 65,536 | 128 | 9.476e-02 | 0.95 |
| Build A² (two matvecs per sample) | 10⁻² | 5 | 1,024 | 128 | 2.207e-03 | 0.22 |
| Build A² (two matvecs per sample) | 10⁻² | 7 | 4,096 | 128 | 3.424e-03 | 0.34 |
| Build A² (two matvecs per sample) | 10⁻² | 9 | 16,384 | 128 | 7.231e-03 | 0.72 |
| Build A² (two matvecs per sample) | 10⁻² | 11 | 65,536 | 128 | 1.456e-02 | 1.46 |
| Build A² (two matvecs per sample) | 10⁻³ | 5 | 1,024 | 128 | 2.384e-04 | 0.24 |
| Build A² (two matvecs per sample) | 10⁻³ | 7 | 4,096 | 128 | 1.870e-04 | 0.19 |
| Build A² (two matvecs per sample) | 10⁻³ | 9 | 16,384 | 128 | 5.594e-04 | 0.56 |
| Build A² (two matvecs per sample) | 10⁻⁴ | 5 | 1,024 | 128 | 1.749e-05 | 0.17 |
| Build A² (two matvecs per sample) | 10⁻⁴ | 7 | 4,096 | 128 | 3.709e-05 | 0.37 |
| Build A² (two matvecs per sample) | 10⁻⁴ | 9 | 16,384 | 256, raised | 3.261e-05 | 0.33 |
| Build A² (two matvecs per sample) | 10⁻⁴ | 9 | 16,384 | 128, rerun at the default cap | 4.898e-05 | 0.49 |

</details>

- **Tolerance scaling:** at fixed N, the error tracks the tolerance proportionally. A loose 10⁻¹ request gave an error near 10⁻¹, not garbage.
- **Depth growth:** with leaf size fixed, depth and N grow together. From 5 to 11 levels, achieved ÷ requested error rose from 0.14–0.43 to 0.69–1.46 at 10⁻¹ and 10⁻²; four points cannot distinguish additive from multiplicative growth. At 10⁻³ and 10⁻⁴, measured to 9 levels, A barely grows but A² does (0.24× to 0.56× at 10⁻³).
- **1D check:** in 1D (10⁻² only), the relative error was 2–3 × 10⁻⁷ at 5–7 levels and 0.8–1.3× tolerance at 9–13 levels; with OpenBLAS the 13-level run reached 1.9×. Exactly low-rank sibling blocks did not prevent tolerance-level error, so these runs do not establish the cause of the 2D growth.
- **Sample-cap failure:** the hollow markers are the same run at the default per-level sample cap of 128. Only the rebuild of A overshot (1.53×); cap 256 gave 0.19×. A² stayed at 0.49×, and two other plotted runs hit the cap without overshooting.
- **Scope:** one kernel, regular grids, one random seed and reference BLAS. These runs show typical behavior, not a bound.

**Where errors enter in H2Opus's implementation**

- **Peeling residual:** each level is sampled from A·X minus the coarser levels' partial approximation, so their errors enter the new samples as additive noise.
- **Projection noise:** the projection pass applies the residual to all blocks' bases at once, so coarse-level errors acting on other blocks' bases enter each block's coefficients directly.
- **Fixed per-level tolerance:** the same absolute tolerance is used at every level, so errors add up with depth. A per-level budget, tolerance ÷ (depth − level), exists in the code but is commented out.
- **Repeated recompression:** after every 16–32 columns of update, the whole matrix is recompressed, so coarse levels are re-truncated many times.
- **Dense diagonal blocks:** these are sampled last with stacked identity blocks, so each diagonal leaf absorbs the summed residual errors of its entire block row.
- **Safety margins:** the range-finder threshold is about tolerance ÷ 8 and the truncation threshold about tolerance ÷ 16. Ranks are set to each level's maximum. In the 2D runs above, these margins kept the error within tolerance up to 9 levels, apart from the sample-cap case, but not at 11 (up to 1.46×).

**Practical implications**

- **No warnings:** neither tolerance overshoot nor sample-cap saturation produces a warning.
- **Checking:** H2Opus provides an a posteriori error estimator (`sampler_difference`, used in its example), and PETSc exposes `-mat_h2opus_check`. In verbose output, per-level sample counts at twice the cap reveal saturation.

## HPC suitability

H2Opus has demonstrated multi-GPU scaling for symmetric H² matvecs and recompression, but it is NVIDIA-only and has been dormant since 2022, which makes deployment on current systems a real risk.

**Distributed memory**

- **What runs distributed:** H² matvec, orthogonalization, recompression and kernel-based construction, on CPUs or NVIDIA GPUs. Each rank picks a GPU device.
- **Restrictions:** symmetric matrices only, block-row distribution, and the MPI rank count must be a power of two (the construction code asserts this).
- **What doesn't run distributed:** `hara`, all low-rank updates, the whole TLR module, and Newton–Schulz inversion.
- **Published results:** the [paper](https://arxiv.org/abs/2109.05451), published in *Advances in Computational Mathematics* in 2022, reports runs on Summit.
  - Near-ideal scaling to 1024 V100 GPUs.
  - Over 2.3 Tflop/s per GPU for matvecs and about 670 Gflop/s per GPU for compression.
  - A 2D fractional diffusion problem with up to 16 million unknowns, solved on 64 GPUs.

**GPUs**

- **Backend:** NVIDIA only, through CUDA, MAGMA and [KBLAS](https://github.com/ecrc/kblas-gpu) (ECRC's batched linear algebra library). The hipcc and DPC++ build configurations produce CPU-only builds; there is no AMD or Intel GPU backend.
- **Design:** operations are reorganized into batched kernels over tree levels, which is what drives the GPU performance. The [TLR paper](https://arxiv.org/abs/2108.11932) reports TLR Cholesky above 1.2 Tflop/s on one V100.
- **Toolchain risk:** the INSTALL file says the code was tested with CUDA 9.0 through 10.2 and that CUDA 11 may need changes. KBLAS was last updated in March 2023. PETSc's CI builds and tests its pinned April 2022 H2Opus commit with CUDA 12.0; newer CUDA releases are untested.

**Other factors**

- **CPUs:** OpenMP through Thrust, with MKL, ESSL, AOCL or generic BLAS. There are also builds for NEC SX vector engines and IBM XL.
- **Interfaces:** C++ only, plus PETSc and SLEPc integration. The Python binding is a prototype that builds one fixed 2D example and exports its data.
- **Maturity:** no commits since October 2022, no releases, and the examples are the only documentation. PETSc still ships the wrapper but pins an April 2022 commit.

Where STRUMPACK centers on factorizations, H2Opus centers on fast H² application and updates.

## Sources

- [H2Opus on GitHub](https://github.com/ecrc/h2opus)
- [H2Opus: A distributed-memory multi-GPU software package for non-local operators](https://arxiv.org/abs/2109.05451) (arXiv 2109.05451)
- [KAUST article: H2Opus, a performance-oriented library for hierarchical matrices](https://cemse.kaust.edu.sa/articles/2020/09/16/h2opus-performance-oriented-library-hierarchical-matrices)
- [H2OPUS-TLR: High Performance Tile Low Rank Symmetric Factorizations using Adaptive Randomized Approximation](https://arxiv.org/abs/2108.11932) (arXiv 2108.11932)
- [Randomized GPU algorithms for the construction of hierarchical matrices from matrix-vector operations](https://epubs.siam.org/doi/10.1137/18M1210101) (SIAM J. Sci. Comput., 2019; [open-access copy](https://scholarworks.aub.edu.lb/server/api/core/bitstreams/0cde3f78-27c2-4b46-bf19-200aa4e703ed/content))
- [Hierarchical matrix approximations for space-fractional diffusion equations](https://scholarworks.aub.edu.lb/items/e83bf253-330c-4d20-82d6-3c732c41c477) (CMAME, 2020)
- [Near-optimal hierarchical matrix approximation from matrix-vector products](https://arxiv.org/abs/2407.04686) (arXiv 2407.04686)
- [Fast construction of hierarchical matrix representation from matrix-vector multiplication](https://dukespace.lib.duke.edu/items/c885c482-d59b-430e-b0e1-a0db8df2d9c6) (Lin, Lu and Ying; the original peeling algorithm)
- [PETSc MATH2OPUS manual page](https://petsc.org/release/manualpages/Mat/MATH2OPUS/)
- [PETSc source mirror](https://github.com/petsc/petsc) (MATH2OPUS and PCH2OPUS implementations)
- [KBLAS on GitHub](https://github.com/ecrc/kblas-gpu)
