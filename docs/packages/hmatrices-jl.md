# HMatrices.jl deep dive

> Written in October 2026 by Nick Alger. Reflects HMatrices.jl main at efc8d50 (27 Feb 2026) and release 0.2.13; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

HMatrices.jl is a small, pure-Julia H-matrix library for boundary integral equations, and the H-matrix backend of Inti.jl. It assembles one format, a geometric H-matrix with low-rank (ABᴴ) and dense leaves, from any entry or block oracle plus point coordinates, using partial ACA or truncated SVD. Assembly and matvec are threaded.

It adds A + σI, addition of a sparse matrix, an internal serial H×H, and serial hierarchical LU (no pivoting) and Cholesky with vector solves and H-triangular solves. There is no H², HSS or BLR, no construction from matvecs or sparse matrices, and no H + H, inversion, determinant, LDLᵀ, QR, eigensolver or matrix functions.

Factorization and H×H are single-threaded, and there is no MPI or GPU support. The documented Distributed.jl path is marked experimental and, from reading the code, cannot run. Apart from experimental tensor-valued kernels, it is functionally a subset of the H-matrix layers of H2Lib and HLIBpro.

## Snapshot

HMatrices.jl is essentially a one-author package with sporadic bursts of maintenance (none since 27 Feb 2026) and three registered dependents.

| Item | Value |
| --- | --- |
| Release | 0.2.13, registered in the Julia General registry on 28 Aug 2025 |
| Last commit | 27 Feb 2026 on main. The 7 commits after 0.2.13 are unreleased: `ACAWithRecompression`, a faster tensor-kernel pivot computation, code formatting and a CI action bump |
| Activity | 90 commits on main since 2024 (70 excluding merges, 5 of them by Dependabot) |
| Licence | MIT |
| Language | Pure Julia, no other bindings. `Project.toml` declares Julia 1.6, but the code needs 1.8 or later (`const` fields in a mutable struct). CI runs the current Julia release on Ubuntu x64 only |
| Dependencies | Standard libraries Distributed, LinearAlgebra, Printf, SparseArrays and Statistics; packages StaticArrays and RecipesBase |
| Maintainers | Luiz M. Faria (POEMS, ENSTA Paris and Inria) wrote about 95% of the commits; about 6 human contributors. 49 stars, 7 forks, 5 open issues and 10 open pull requests on 8 Oct 2026 |
| Documentation | A [Documenter site](https://integralequations.github.io/HMatrices.jl/stable/) with three narrative pages and an API page that includes internals. Examples run at build time for n ≤ 10⁴ with no timings. Parts are stale: LU is described as the only factorization, Cholesky is absent from the narrative pages, and the distributed page documents a path that appears broken |
| Users | [Inti.jl](https://github.com/IntegralEquations/Inti.jl) (package extension: `Inti.assemble_hmatrix`, plus `axpy!` for sparse near-field corrections); IrregularSpectra.jl (kernel matrix LU as a Krylov preconditioner); AcousticScattering.jl |

## Construction

HMatrices.jl builds H-matrices from anything that returns entries, plus a cluster tree built from point coordinates. There is no sparse or matvec-only path.

| Access | Formats | Algorithms | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | ✗ | — | — | SparseArrays is used only to add a sparse matrix to an existing HMatrix. A sparse input would go through the entry path, undocumented and untested |
| Single entries | HMatrix; `Hermitian{HMatrix}` | Admissible leaves use `PartialACA(; atol, rtol, rank)`, partial-pivot ACA with a Frobenius-norm stopping estimate, or `TSVD(; atol, rtol, rank)` on the dense block. Main only: `ACAWithRecompression`. Inadmissible leaves are dense | `assemble_hmatrix([T,] K, rowtree, coltree; adm, comp, global_index, threads, distributed)` | K is any `AbstractMatrix` with `getindex` and `size`, and must be thread-safe because threaded assembly is the default. Tolerances belong to the compressor; there is no near-field or sparse-correction argument |
| Blocks | Same | Same | Overload `HMatrices.getblock!(out, K, irange, jrange)` | With the default `global_index = true`, K is wrapped in a permuted matrix that falls back to per-entry access, so a vectorized `getblock!` helps only with `global_index = false` |
| Kernel + coordinates | HMatrix | Partial ACA; defaults are a geometric splitter with 50-point leaves, strong admissibility with η = 3, and rtol = √eps | `KernelMatrix(f, X, Y)` then `assemble_hmatrix(K; atol, rtol, rank, …)` | Elements can be points or any object defining `HMatrices.center`. Tensor kernels (Stokes, elasticity) choose pivots by the largest smallest singular value; the docs call this experimental |
| Matvecs only | ✗ | — | — | Internally `hmul!` compresses sums of products by ACA from H-matvecs, but this is not exposed for user operators |
| Dense matrix | HMatrix | ACA or TSVD on its blocks | `assemble_hmatrix(M, Xclt, Yclt; comp)`; compressors applied to a dense block return a low-rank block | Coordinates are still needed; not specifically tested |
| Hermitian half storage | `Hermitian{HMatrix}` | As above | `assemble_hmatrix(Hermitian(K), Xclt, Xclt; …)` | Only upper-triangle leaves are assembled; required for Cholesky |
| Distributed | DHMatrix | One assembly per worker | `assemble_hmatrix(…; distributed = true)` | Experimental and apparently non-functional (see HPC) |

- **Cluster tree.** `ClusterTree(points, splitter; copy_elements, threads)` takes a vector of `SVector` points; there is no algebraic or graph clustering.
- **Splitters.** `GeometricSplitter` bisects the largest box axis (the default since 0.2.7) and `CardinalitySplitter` splits at the median; `PrincipalComponentSplitter` exists but is not exported, and users can add their own. Child boxes shrink to fit their points.
- **Admissibility.** `StrongAdmissibilityStd(eta = 3)` admits blocks with min(diam) < η · dist. `WeakAdmissibilityStd()` admits disjoint boxes, which with bisection gives a HODLR-like partition in the same HMatrix type. Any callable `adm(Xnode, Ynode)` also works.
- **Shapes.** Rectangular matrices on different point sets work for assembly and matvec; LU, Cholesky and A + σI need square matrices with identical row and column trees.

## Conversions

Only dense extraction, lazy adjoints and leaf recompression exist; there are no other formats.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| HMatrix (also adjoint and Hermitian wrappers) → dense | `Matrix(H; global_index = true)` | Exact | `global_index = false` returns cluster ordering. Tested for HMatrix only; triangular views such as `F.L` have no method |
| Low-rank block → dense | `Matrix(R)` | Exact | — |
| Dense or any `AbstractMatrix` → HMatrix | `assemble_hmatrix(M, Xclt, Yclt; comp)` | Approximation by ACA or TSVD | Needs coordinates |
| HMatrix → HMatrix, looser tolerance, same blocks | `HMatrices.compress!(H, TSVD(…))` | Recompressing (QR + SVD per low-rank leaf) | Internal and not exported; no block merging |
| HMatrix → adjoint | `H'` | Exact and lazy | Usable in matvecs and as an `hmul!` operand |
| General ↔ Hermitian storage; element type changes | — | ✗ | Only by reassembly; `assemble_hmatrix(T, K, …)` chooses the stored type |
| HMatrix → H², HODLR, HSS, BLR; DHMatrix ↔ HMatrix | — | ✗ | No other formats exist |

## Operations

The HMatrix supports products, LU, shift and sparse addition; the Hermitian wrapper adds Cholesky. Neither provides inversion, LDLᵀ, QR, eigendecomposition, SVD, determinant, norms or matrix functions, and the distributed DHMatrix has no operation that runs (see HPC).

| Operation | HMatrix | `Hermitian{HMatrix}` |
| --- | --- | --- |
| Matvec | ✓ \[1\] | ✓ \[1\] |
| Transpose or adjoint matvec | ◐ adjoint only \[2\] | ✓ self-adjoint |
| Block of vectors | ◐ column loop \[3\] | ◐ \[3\] |
| Direct solve | ✓ by LU, one vector \[4\] | ✓ by Cholesky, one vector \[5\] |
| Package-provided iterative solve | ✗ \[6\] | ✗ |
| Triangular solve | ✓ \[7\] | ✓ on Cholesky factors \[7\] |
| A + B | ✗ (sparse + H only) \[8\] | ✗ |
| A + σI | ✓ \[8\] | ✗ \[8\] |
| Scaling | ◐ in place \[9\] | ✗ |
| A · B | ◐ internal `hmul!` \[10\] | ◐ |
| LU | ✓ no pivoting \[4\] | ✗ |
| Cholesky | Needs the Hermitian wrapper | ✓ \[5\] |
| Recompression | ◐ internal \[11\] | ✗ |

1. **Matvec.** `H*x` and `mul!(y, H, x, α, β; global_index, threads)`, also for triangular views and `SVector` densities. Tested serial, threaded (CI uses one thread) and Hermitian.
2. **Adjoint.** `H'*x` uses a lazy, threaded adjoint. `transpose(H)*x` has no method and errors, so complex matrices offer only the conjugate transpose.
3. **Block of vectors.** `H*X` loops over columns, calling the vector matvec (marked FIXME in the code).
4. **LU.** `lu(H; atol, rtol, rank)` and `lu!` run a block-recursive H-LU with no pivoting, even in dense leaves, using H-triangular solves and `hmul!` updates compressed by partial ACA. `F\b` and `ldiv!(F, b)` exist for a vector only; a matrix right-hand side falls to LinearAlgebra’s generic path, which from reading would skip the cluster permutation. Tested at n = 5,000 with atol 1e-10.
5. **Cholesky.** `cholesky(Hs; atol, rtol, rank)` on `Hermitian{HMatrix}` only, without pivoting, storing the upper factor, with `F\b` for a vector. Available since 0.2.7 and tested at n = 5,000, but missing from the narrative docs.
6. **Iterative solvers.** None; the docs use `gmres!` from the external IterativeSolvers.jl.
7. **Triangular solves.** `ldiv!` and `rdiv!` with lower or upper triangular views, for dense, low-rank or H right-hand sides (the latter with a compressor). They work in cluster ordering without permutation; upper `ldiv!` with low-rank or H right-hand sides and `rdiv!` by a lower factor are missing.
8. **Shift and sparse addition.** `axpy!(σ, I, H)`, `H + σI` and `I + H` add σ to the dense diagonal leaves and are untested. The Hermitian wrapper has no such method. `axpy!(a, S, H)` and `S + H` add a sparse matrix, appending each nonzero in an admissible leaf as a rank-1 term without recompression. In admissible leaves the scalar `a` is never applied, so `axpy!(2, S, H)` adds S there.
9. **Scaling.** `rmul!(H, α)` and `zero(H)`; there is no `α*H` or `lmul!`.
10. **H×H.** `HMatrices.hmul!(C, A, B, α, β, comp)` computes C ← αAB + βC into a preallocated C with a compatible block structure, compressing each target block’s products once by ACA from H-matvecs. It is serial, tested, not exported, and needs an ACA-type compressor; `H1*H2` is not provided.
11. **Recompression.** `HMatrices.compress!(H, TSVD(…))` is internal. The main-only `ACAWithRecompression` would, from reading, fail inside `lu!`, `cholesky!` and `hmul!`.
12. **Other.** `compression_ratio(H)`, plotting of matrices and factors, and global switches `use_threads()` and `use_global_index()`, changed by redefining the methods.

## HPC suitability

Only assembly, matvec and opt-in cluster-tree construction run in parallel, as Julia tasks on one node. Everything else is serial, and there is no working distributed or GPU path.

- **Threading.** Assembly spawns one task per thread, each taking leaves from an atomic counter with its own ACA buffer. The matvec schedules leaves the same way, with a private full-length output per task and a locked reduction, so each product needs O(p · n) extra memory. `lu!` and `cholesky!` accept a `threads` keyword that is never used.
- **Distributed memory.** Distributed.jl only, no MPI. `distributed = true` builds a DHMatrix of column slabs, one HMatrix per worker, with matvec outputs reduced on the master. The docs call it experimental, and its test has been commented out since 2021.
- **Why the distributed path appears unusable.** DHMatrix defines no `rowtree`, `coltree`, `children` or `data` methods, yet `size`, `leaves` and the column partition call them. The distributed `mul!` also calls `Trees.isleaf` from a package dropped in 0.2.0. Julia was not available to run it.
- **GPU and runtimes.** No GPU code. A DataFlowTasks.jl task-graph LU exists only on unmerged 2023–24 branches; a leftover example on main cannot run.

| Operation | Threads | Distributed | GPU |
| --- | --- | --- | --- |
| Cluster tree | ◐ opt-in | ✗ | ✗ |
| Assembly | ✓ by default | ✗ (DHMatrix appears broken) | ✗ |
| Matvec, including adjoint and triangular views | ✓ by default | ✗ | ✗ |
| Block of vectors | ◐ each column threaded | ✗ | ✗ |
| H×H, LU, Cholesky, solves, shift, sparse addition, scaling, recompression | ✗ serial | ✗ | ✗ |

Largest documented runs, none with timings:

- **README:** n = 100,000 random points on the unit sphere, 3D Laplace kernel, atol 1e-6. Compression ratio about 25 and pointwise matvec error about 2e-7; `lu(H; atol = 1e-6)` “may take a few minutes on a reasonable machine”.
- **Documentation examples** (generated in CI): 2D Laplace on a circle at n = 10,000 and atol 1e-6 compressed 21.8×, with matvec error 5.0e-6 and LU solve error 2.6e-8. A 3D Helmholtz case at n = 10,000 had ranks 20–86 and compressed 2.47×.
- **Benchmark suite:** N = 50,000 Laplace on a cylinder, timing assembly, matvec and LU with and without threads. It runs only on a self-hosted machine for labelled pull requests, and no results are published.

Practical limits: one node, serial factorization and H×H, no pivoting, a thread-safe oracle required, and per-block error control whose truncation errors the docs warn can accumulate.

## Caveats and surprises

- **Errors instead of silent densification.** `getindex` is disabled on purpose for HMatrix and low-rank blocks, so `H1*H2`, `H1 + H2`, `inv`, `det`, `norm`, `transpose(H)*x`, `α*H` and `lu(Hermitian(H))` all raise errors.
- **Ordering pitfalls.** Data are stored in cluster order. `mul!`, `Matrix`, `F\b` and sparse `axpy!` apply the permutation, but triangular `ldiv!`/`rdiv!`, `hmul!` and `getcol` work in cluster ordering, and `F\B` with a matrix probably misses it.
- **Sparse addition** ignores its scalar in admissible blocks and grows ranks without recompression.
- **Spurious ACA warnings.** Because assembly wraps K in a permuted matrix, ACA’s kernel-matrix check never matches, and genuinely zero admissible blocks always trigger “aca possibly failed” after scanning every row.
- **Tensor kernels.** The ACA stopping estimate is marked `@test_broken` for `SMatrix` entries.
- **Exported names that do not exist.** `DyadicSplitter` and `GeometricMinimalSplitter` are exported but defined nowhere, in 0.2.13 and on main.
- **Naming.** `WeakAdmissibilityStd` means disjoint bounding boxes, not the textbook “all off-diagonal blocks”.
- **Easily misread status.** The February 2026 commits are unreleased; 0.2.13 offers only `PartialACA` and `TSVD` (no full-pivot ACA); and the distributed code covers only assembly and matvec, apparently non-functional.

## Verification

No Julia code was run: Julia was not installed in the author’s sandbox, and the proxy refused its download hosts. Every behaviour claim comes from reading the code, the package’s own tests and the outputs in its built documentation.

- **Code read:** all of `src/`, `test/`, `docs/`, `benchmark/` and the CI workflows at main efc8d50, plus the 0.2.13 sources. The only semantic differences between them are `ACAWithRecompression`, a buffer allocator and the faster tensor pivot.
- **History:** tags, unmerged branches (a DataFlowTasks LU, an inexact H-matvec), authors, and the history of `dhmatrix.jl`.
- **Checks:** a Python regex script confirmed the two exported names that are defined nowhere. A separate re-read confirmed that `_lu!` is serial and that DHMatrix lacks the tree methods its own code calls.
- **Dependents:** a search of the Julia General registry for the package’s UUID found Inti, IrregularSpectra and AcousticScattering; the Inti and IrregularSpectra extensions were read.
- **Web:** the repository page and stable and dev docs. Papers, talks and Discourse threads had no HMatrices.jl timings. The GitHub API and issue list were not accessible; star, fork, issue and PR counts come from the repository page header.

## Sources

- [HMatrices.jl repository](https://github.com/IntegralEquations/HMatrices.jl) at main efc8d50 and tag v0.2.13 (54ba356). Files read: `src/HMatrices.jl`, `hmatrix.jl`, `kernelmatrix.jl`, `compressor.jl`, `clustertree.jl`, `splitter.jl`, `rkmatrix.jl`, `multiplication.jl`, `triangular.jl`, `lu.jl`, `cholesky.jl`, `dhmatrix.jl`, `utils.jl`; `test/`; `benchmark/benchmarks.jl`; `.github/workflows/`; `README.md`; `Project.toml`; `docs/src/`
- [Stable documentation](https://integralequations.github.io/HMatrices.jl/stable/), including the [kernel matrix page](https://integralequations.github.io/HMatrices.jl/stable/kernelmatrix/) and [API reference](https://integralequations.github.io/HMatrices.jl/stable/references/)
- [Julia General registry](https://github.com/JuliaRegistries/General) entries for HMatrices, Inti, IrregularSpectra and AcousticScattering
- [Inti.jl](https://github.com/IntegralEquations/Inti.jl) (`ext/IntiHMatricesExt.jl`) and [IrregularSpectra.jl](https://github.com/cgeoga/IrregularSpectra.jl)
- [Chaillat, Desiderio, Ciarlet, JCP 2017](https://arxiv.org/abs/1706.09384), the tensor-kernel ACA pivoting reference
- [DataFlowTasks.jl talk, JuliaCon 2023](https://eindhoven2023.pydata.org/juliacon/talk/SV3D7V)
