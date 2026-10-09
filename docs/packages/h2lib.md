# H2Lib: capabilities review

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects the H2Lib `community` branch at 920662c (13 Feb 2020); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

H2Lib goes deepest on H² of the twelve packages in the overview, but it is a mostly sequential, single-node library. It compresses dense, H and H² matrices into H², multiplies, inverts and factorizes (LU, Cholesky) H²-matrices, and adds directional H² (DH²) for high-frequency Helmholtz. The H² arithmetic module is labelled experimental, construction is built around boundary-element (BEM) problems, and nothing builds a matrix from matvecs.

All H and H² arithmetic and the H and H² matvecs run sequentially. OpenMP covers BEM assembly, H² recompression, the DH² matvec and dense LU/Cholesky; OpenCL covers only BEM quadrature, and there is no MPI.

H2Lib comes from Steffen Börm's group at Kiel University and is open source under LGPL-3.0 ([GitHub](https://github.com/H2Lib/H2Lib/tree/community)). It is plain C with mostly C++-compatible headers, builds with make or CMake, and fixes the scalar type (real or complex, single or double) at compile time.

The last tagged release is v3.0.1 (December 2016). The `community` branch reviewed here was last changed in February 2020. `master` is newer: it adds H² approximation of generic kernel matrices (April 2020), a 1D integral-equation example and `scale_hmatrix` (2022). It also fixes complex-valued CG/PCG and pivoted-QR pivots (2020) and adds missing C++ linkage declarations (2026). The tables describe `community` at commit 920662c; `master` differs only by these fixes and additions and has no CMake build.

This review reads the community-branch source, its build, its test programs and a small benchmark run. h2lib.org blocks automated fetching, so the website was not read.

## Formats

H2Lib has separate data structures for H, H² and directional H², all built over cluster and block trees from user-chosen admissibility conditions.

| Format | Types | Notes |
| --- | --- | --- |
| H-matrix | `hmatrix` tree; leaves `rkmatrix` (A·Bᴴ) and `amatrix` (dense) | The block tree comes from an admissibility callback. Built-in conditions are strong-type (Euclidean, max-norm, sphere, minimum), plus domain-decomposition admissibility for finite-element matrices. No built-in weak or HODLR condition; you can pass your own callback |
| H² | `h2matrix`; leaves `uniform` (V·S·Wᴴ); nested `clusterbasis` with transfer matrices | Orthogonal or non-orthogonal bases; `clusteroperator` objects hold basis weights for compression and updates |
| Directional H² (DH²) | `dh2matrix`, `duniform`, `dclusterbasis` over directional cluster and block trees | For high-frequency Helmholtz; parabolic or standard admissibility |
| Sparse | `sparsematrix` (compressed rows) with `sparsepattern` | FEM assembly for P1 and RT0 elements on triangles and tetrahedra |
| Dense, low-rank | `amatrix`, `rkmatrix` | — |

Not present: HSS, HODLR or BLR as dedicated formats, butterfly, and mixed or compressed precision.

## Construction

H2Lib builds H-matrices from FEM-type sparse matrices, from BEM kernels on a surface mesh, or from dense arrays; a generic entry oracle drives ACA only one block at a time, and nothing builds a matrix from matvecs.

| Target | Sparse | Generic entry oracle | BEM kernel and mesh | Dense array | Matvecs |
| --- | --- | --- | --- | --- | --- |
| H | Yes (FEM-type, needs coordinates) | Per-block ACA only | Yes: ACA, HCA, Green, Green-hybrid, interpolation | Yes | No |
| H² | Only via H, then compression | Only via H, then compression (`master`: kernel function plus points) | Yes: interpolation, Green-hybrid, hierarchical recompression | Yes | No |
| DH² | No | No | Yes: 3D directional interpolation | Yes | No |

- **From a sparse matrix.** `copy_sparsematrix_hmatrix` copies the nonzeros into an existing H-matrix whose admissible blocks hold no nonzeros. The cluster tree comes from domain-decomposition clustering (`build_adaptive_dd_cluster`), which needs coordinates; there is no purely graph-based clustering. H² comes only via H, then `compress_hmatrix_h2matrix`.
- **From a generic entry oracle.** The `matrixentry_t` callback fills a sub-block for given row and column index lists and drives partial-pivoting ACA one block at a time (`decomp_partialaca_rkmatrix`). The community branch has no driver that assembles a whole H-matrix from a callback.
- **From BEM objects.** `bem2d` and `bem3d` (bem2d: Laplace; bem3d: Laplace and Helmholtz; piecewise-constant or piecewise-linear elements) need the surface mesh and kernel, not just entries:
  - H: ACA with full or partial pivoting, HCA, Green, Green-hybrid (Green plus ACA) and interpolation, optionally coarsened on the fly.
  - H²: interpolation, Green-hybrid (including an orthogonal variant), and hierarchical recompression during assembly.
  - DH²: directional interpolation, with orthogonalizing and recompressing variants.
- **Generic kernels (`master` only).** `kernelmatrix` builds an H² from a kernel function k(x, y) and a point set in any dimension, by interpolation.
- **From a dense array.** All three compressed formats need the full matrix in memory: to H, `add_amatrix_hmatrix` truncates it into a given block structure; to H², `compress_amatrix_h2matrix`; to DH², the `dh2compression` routines.
- **From matvecs.** No peeling, randomized sketching or probing.

## Conversion

H2Lib converts freely among dense, H and H², and from dense into DH²; it has no path between H or H² and DH².

| From → to | Provided | Functions |
| --- | --- | --- |
| Sparse → H | Yes | `copy_sparsematrix_hmatrix` (admissible blocks must be empty) |
| Sparse → dense | Yes | `add_sparsematrix_amatrix` |
| Dense → H | Yes | `add_amatrix_hmatrix` (blockwise truncation into an existing structure) |
| Dense → H² | Yes | `compress_amatrix_h2matrix`; `build_projected_amatrix_h2matrix` (given bases) |
| Dense → DH² | Yes | `buildrow_amatrix_dclusterbasis` and `buildcol_amatrix_dclusterbasis`, then `projectdense_dh2matrix` |
| H → dense | Yes | `add_hmatrix_amatrix` |
| H → coarser H | Yes | `coarsen_hmatrix` (merges sibling blocks into one low-rank block when that saves storage) |
| H → one low-rank block | Yes | `add_hmatrix` into a rank-0 leaf (`new_rk_hmatrix`); `merge_hmatrix_rkmatrix` (one level) |
| H → H² | Yes | `compress_hmatrix_h2matrix` (adaptive bases; needs a strict block tree, else 10–40% error without warning); `project_hmatrix_h2matrix` and `build_projected_hmatrix_h2matrix` (given bases); `unify_h2matrix` (merges H² submatrices) |
| H² → H | Yes | `convert_h2matrix_hmatrix` |
| H² → dense | Yes | `convert_h2matrix_amatrix` |
| H² → H² (recompress or change bases) | Yes | `compress_h2matrix_h2matrix`, `compress_symmetric_h2matrix_h2matrix`, `recompress_inplace_h2matrix`, `project_inplace_h2matrix` |
| H² leaf block ↔ low-rank block | Yes | `convert_uniform_rkmatrix`, `convert_rkmatrix_uniform` |
| DH² → dense | Yes | `expand_dh2matrix` |
| DH² → DH² | Yes | `compress_dh2matrix_dh2matrix`, `build_projected_dh2matrix` |
| H or H² ↔ DH² | No | — |

## Operations

H-matrices get the full arithmetic set and H²-matrices get an experimental one; DH² supports only matvec, recompression, norms and the package's Krylov solvers.

| Operation | H | H² | DH² |
| --- | --- | --- | --- |
| Matvec | Yes: y += αAx or αAᴴx; symmetric variant using the lower triangle | Yes: same, including the symmetric variant | Yes, including a parallel version |
| Product with a dense block | Yes: `addmul_hmatrix_amatrix_amatrix` | Yes, from either side | No |
| Inverse apply | Yes: LU and Cholesky solves; CG, PCG, GMRES and PGMRES; triangular solves with dense or H right-hand sides (`triangularinvmul_*`) | Yes: `lrsolve_h2matrix_avector`, `cholsolve_h2matrix_avector`; CG, PCG, GMRES and PGMRES; triangular solves with dense or H² right-hand sides (some transpose combinations not implemented) | Krylov only: CG, PCG, GMRES and PGMRES |
| √A, A^(−1/2), other matrix functions | No | No | No |
| A + B | Yes: `add_hmatrix`, truncated; cluster trees must match, block structures may differ. Dense or low-rank into H also works | Partly: an H-matrix into an H² (`add_hmatrix_h2matrix`) and low-rank updates (`rkupdate_h2matrix`); no direct H² + H² | No |
| A + λI | No dedicated function (`identity_hmatrix` builds I as an H-matrix) | No | No |
| A·B | Yes: `addmul_hmatrix` (Z += αXY), accumulator version `addmul2_hmatrix`, and a lower-triangle-only version | Yes: `addmul_h2matrix` (C += αAB; bases adapted by low-rank updates) and `mul_h2matrix` | No |
| Inversion | Yes: `invert_hmatrix` | Yes: `invert_h2matrix` (not covered by the shipped tests) | No |
| Triangular factorizations | Yes: LU and Cholesky (LLᴴ in complex builds), standard and accumulator-based; no LDLᵀ, no pivoting | Yes: LU and Cholesky (LLᴴ in complex builds; experimental) | No |
| QR | No (dense matrices only) | No | No |
| Eigendecomposition | No (dense only: symmetric, generalized, SVD) | No | No |
| Recompression, coarsening | Yes: coarsening; truncation by relative, absolute or blockwise error in the Euclidean or Frobenius norm | Yes: recompress, re-project, unify | Yes: recompress |
| Norms | Yes: spectral norm, differences between any two formats, and ‖A − LU‖ and ‖I − (LU)⁻¹A‖ | Yes | Yes |

- **Experimental H² arithmetic.** `h2arith.h` says "This is an experimental module, not all functions have been thoroughly tested, some special cases are not yet implemented." Several triangular solves with matrix right-hand sides are marked "not tested yet" or lack some transpose combinations.
- **Test coverage.** The shipped tests check H² LU and Cholesky only on a 579-unknown 2D problem; they passed in this build.
- **No pivoting.** LU is unpivoted block LU, down to unpivoted LU on the dense leaves.
- **H² factorization interface.** You must pre-build the empty triangular factors with their own cluster bases and basis-weight objects (`prepare_row_clusteroperator`, `prepare_col_clusteroperator`), so the interface is low-level.
- **Dense matrices only.** LU, Cholesky, LDLᵀ, pivoted QR, SVD and symmetric eigensolvers.
- **Krylov solvers.** [`krylovsolvers.h`](https://github.com/H2Lib/H2Lib/blob/community/Library/krylovsolvers.h) has CG, PCG, GMRES and PGMRES drivers for dense, sparse, H, H² and DH² matrices, plus versions taking any matvec callback. Preconditioners are callbacks; `krylov.h` notes that `lrsolve_hmatrix_avector` and `cholsolve_hmatrix_avector` can be cast to one, so an H-LU or H-Cholesky factor plugs in directly. `krylov.h` also has step-wise BiCG, BiCGStab and Uzawa iterations.

## Spot check: H vs H² on a 3D BEM problem

In runs made for this review, H² kept storage per unknown flat and its Cholesky factor was 1.2–1.7× smaller than H's. H²-Cholesky, however, took 11–12× longer and was 4.5–9× less accurate at the same ε, the accuracy gap growing with n.

![Two panels: Cholesky factorization time and matrix storage per unknown for H and H² in H2Lib, at 2,048, 8,192 and 32,768 unknowns](../figures/h2lib-h-vs-h2.svg)

*Runs made for this review on the H2Lib community branch · 3 sizes, 2,048 to 32,768 unknowns*

<details>
<summary>Data behind the figure</summary>

| Unknowns n | H storage (MiB) | H² storage, incl. bases (MiB) | H per unknown (KiB) | H² per unknown (KiB) | H-Cholesky (s) | H²-Cholesky (s) | Time ratio H² ÷ H | Storage ratio H ÷ H² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,048 | 17.1 | 10.7 | 8.6 | 5.3 | 0.82 | 8.81 | 10.7 | 1.6 |
| 8,192 | 109.6 | 51.4 | 13.7 | 6.4 | 5.81 | 71.55 | 12.3 | 2.1 |
| 32,768 | 586.3 | 209.9 | 18.3 | 6.6 | 36.75 | 427.18 | 11.6 | 2.8 |

</details>

The gap in factorization time held steady while the storage advantage grew with n, from 1.6× to 2.8×. Other measurements from the same runs:

| Unknowns | H build (ACA) | H→H² compression | Matvec, H / H² | Cholesky solve error, H / H² | Cholesky factor (lower triangle), H / H² |
| --- | --- | --- | --- | --- | --- |
| 2,048 | 0.23 s | 0.09 s | 3.1 / 8.1 ms | 7.7e-5 / 3.5e-4 | 7.5 / 6.3 MiB |
| 8,192 | 1.2 s | 0.57 s | 22 / 15 ms | 8.7e-5 / 5.6e-4 | 45 / 31 MiB |
| 32,768 | 6.0 s | 3.5 s | 118 / 54 ms | 7.1e-5 / 6.6e-4 | 228 / 131 MiB |

- **Setup.** Laplace single-layer operator on the octahedral sphere (`new_sphere_macrosurface3d`, split 16/32/64), piecewise-constant elements, quadrature orders 2/4. Strict block tree (`build_strict_block`, `admissible_2_cluster`, η = 1.4), leaf size 32. Partial-pivoting ACA at ε = 1e-4, then `compress_hmatrix_h2matrix` and both Cholesky factorizations with relative Euclidean truncation at the same ε. The H² factor starts from empty bases (`build_from_block_lower_h2matrix`, `init_cholesky_h2matrix`). Built with `OPT=1` and OpenMP on a 2-core Xeon VM with reference BLAS; the arithmetic is sequential either way.
- **Measures.** Solve error is the relative error of a solve with the factor for a random solution. H² storage includes the cluster bases.
- **Accumulators.** The accumulator-based H-Cholesky took 30 s instead of 37 s at 32,768 unknowns.

## HPC suitability

H2Lib is a shared-memory library whose arithmetic runs on one core: OpenMP parallelizes BEM assembly, H² recompression, the DH² matvec and dense LU/Cholesky; OpenCL only computes BEM matrix entries; there is no MPI.

- **Parallel with OpenMP (optional build flag).** BEM assembly and quadrature, cluster-basis construction, H²-to-H² recompression, the DH² matvec, and dense LU and Cholesky (task-based variants). H→H² and dense→H² compression are sequential, and the parallel unification routine is labelled experimental. The nesting depth is set with the `H2_PARDEPTH` environment variable.
- **Sequential.** The H and H² matvecs, and all H and H² arithmetic: addition, multiplication, inversion, LU and Cholesky. Parallel basis transforms exist, but the default H² matvec does not use them. [Carratalá-Sáez et al.](https://arxiv.org/pdf/1906.00874) call it "this sequential library".
- **Outside parallelization.** Carratalá-Sáez et al. task-parallelized H2Lib's H-LU with OmpSs-2 in their own code. They got 18–22× speedup on 24 cores of a Xeon Platinum 8160 node for 3D BEM at about 30k unknowns, where sequential H2Lib took 204–1,099 s.
- **GPU.** OpenCL computes BEM matrix entries (near-field quadrature and Green-hybrid far-field entries) for 3D Laplace and Helmholtz with piecewise-constant elements only, through a CPU/GPU task scheduler. [Börm and Christophersen](https://arxiv.org/abs/1510.07244v1) report 19–30× faster setup with two consumer GPUs than with a quad-core CPU. There is no GPU matvec or arithmetic; SSE2 and AVX kernels also speed up BEM quadrature.
- **Distributed memory.** None; the code has no MPI.
- **Scale in practice.** [Hertel, Christophersen and Börm](https://arxiv.org/pdf/1811.05731) used H2Lib (HCA, then H→H² recompression) for micromagnetics with about 240k boundary unknowns. The H² matrix stayed under 1 GB and was up to 15× smaller than the H-matrix.
- **Limits.** Indices are 32-bit unsigned (sizes use `size_t`), and the scalar type is fixed per build.
- **Maintenance.** The last release was in 2016 ([releases](https://github.com/H2Lib/H2Lib/releases)). The community branch was last changed in 2020; `master` has small updates through June 2026.

## Sources

- **H2Lib:** [GitHub, community branch](https://github.com/H2Lib/H2Lib/tree/community) · [Releases](https://github.com/H2Lib/H2Lib/releases) · [MaRDI portal entry](https://portal.mardi4nfdi.de/wiki/H2Lib)
- **Papers using or describing H2Lib:**
  - [Carratalá-Sáez et al., Exploiting nested task-parallelism in the H-LU factorization, arXiv 1906.00874](https://arxiv.org/pdf/1906.00874) ([ar5iv](https://ar5iv.arxiv.org/html/1906.00874))
  - [Börm & Christophersen, Approximation of boundary element matrices using GPGPUs and nested cross approximation, arXiv 1510.07244](https://arxiv.org/abs/1510.07244v1)
  - [Hertel, Christophersen & Börm, Large-scale magnetostatic field calculation in finite element micromagnetics with H2-matrices, arXiv 1811.05731](https://arxiv.org/pdf/1811.05731)
  - [Börm, Hierarchical matrix arithmetic with accumulated updates, arXiv 1703.09085](https://arxiv.org/abs/1703.09085)
  - [Börm & Reimer, Efficient arithmetic operations for rank-structured matrices based on hierarchical low-rank updates, arXiv 1402.5056](https://arxiv.org/abs/1402.5056)
  - [Börm & Christophersen, GCA-H² matrix compression for electrostatic simulations, arXiv 1810.08429](https://ar5iv.labs.arxiv.org/html/1810.08429)
- **Benchmark:** runs made for this review on the community branch, described under Spot check.
