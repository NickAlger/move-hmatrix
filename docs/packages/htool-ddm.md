# Htool-DDM deep dive

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects Htool-DDM main at 0ee4cac (6 Oct 2026) and release 1.0.2; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

Htool-DDM is a header-only C++ library (MIT) that builds H-matrices from a block-entry callback and distributes them over MPI as a row-block operator. It solves dense systems with HPDDM’s Krylov methods and one- or two-level Schwarz preconditioners (GenEO coarse space), making it mainly a domain-decomposition tool for BEM and kernel systems. It is the engine behind PETSc’s MATHTOOL type and FreeFEM’s BEM plugin.

On one node it threads assembly, matvec and H×dense products, and runs H-LU and H-Cholesky as OpenMP tasks. An H-LDLᵀ is on the main branch but unreleased. Only assembly, matvec and the DDM solve use MPI; factorization stays on one node, and there is no GPU support.

There is one format: no H², HODLR, HSS or BLR, and no construction from sparse matrices or matvecs. It has no H + H, shift, inversion, matrix functions, QR, eigensolver or determinant. H×H and triangular H-solves exist only as undocumented routines working in cluster numbering (the cluster tree’s permuted order).

## Snapshot

Htool-DDM is actively developed by a small Inria-led team, and PETSc already ships code from its unreleased main branch.

| Item | Value |
| --- | --- |
| Release | 1.0.2, tagged 14 Feb 2026 (1.0.0 in Sep 2025, 1.0.1 in Jan 2026); [JOSS paper](https://doi.org/10.21105/joss.09279) published 17 Feb 2026 |
| Last commit | 6 Oct 2026 on main, adding the HMatrix LDLᵀ; branch `improve_task_based_parallelism` updated 8 Oct 2026 |
| Activity | 69 commits on main since 2024 (pull requests are squash-merged), 136 across all branches; 42 on main since October 2025 |
| Licence | MIT |
| Language and bindings | Header-only C++: the README says C++14, but the CMake target requires C++17. Python bindings ([htool\_python](https://github.com/htool-ddm/htool_python), pybind11 + mpi4py, double and complex double); on PyPI only as a pre-release sdist (Htool-DDM 1.0.1rc2, July 2026) |
| Dependencies | BLAS and LAPACK. CMake requires MPI, although the HMatrix-only headers compile without it. Optional: OpenMP, HPDDM (needed for DDM), MKL |
| Maintainers | Pierre Marchand (Inria, POEMS; about 890 of 965 commits), Pierre-Henri Tournier (LJLL), Pierre Jolivet (LIP6); about 8 contributors |
| Documentation | [Sphinx and Doxygen site](http://htool-ddm.pages.math.cnrs.fr/) with quickstarts and an API reference. It covers build, matvec, H×dense products, LU, Cholesky and solves only; H×H, triangular solves, scaling, recompression and LDLᵀ are undocumented |
| Users | FreeFEM 4.5 and later (BEM plugin); PETSc 3.16 and later through [MATHTOOL](https://petsc.org/release/manualpages/Mat/MATHTOOL/), used with PCHPDDM. PETSc 3.26.0 and main pin post-1.0.2 commits |

## Construction

Htool-DDM builds one format, a standard-admissibility H-matrix, from a block-entry callback plus coordinates. It cannot start from a sparse matrix or from matvecs.

| Access | Formats | Algorithms | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | ✗ | — | — | — |
| Entry or block callback | H | `sympartialACA` (default) and `partialACA` with partial pivoting and Frobenius-norm stopping; `fullACA` and SVD on the whole block; optional QR + SVD recompression; fixed-rank mode. A block reverts to dense when rank · (m + n) > mn | `HMatrixTreeBuilder(eps, eta, sym, UPLO, reqrank, lrgen, admissibility).build([policy,] gen, target_tree, source_tree)`; `DefaultApproximationBuilder` for MPI | `copy_submatrix(M, N, rows, cols, ptr)` takes arbitrary index lists in user numbering; partial ACA requests one row or column at a time. Dense leaves can be generated in one batch |
| Kernel + coordinates | H | As above. The unreleased `interpolation` branch adds Chebyshev hybrid cross approximation (HCA) through the external Theia library | No kernel API in Htool; PETSc’s `MatCreateHtoolFromKernel` wraps one | Coordinates are needed anyway for clustering |
| Matvecs only | ✗ | — | — | No randomized or peeling construction |
| Dense matrix | H | As above | Only through a generator; a testing header wraps a stored matrix | Compression is recomputed, and coordinates are still needed |
| Other | H; DistributedOperator | User-supplied low-rank blocks; user admissibility conditions | `set_low_rank_generator`, `set_admissibility_condition`, `CustomApproximationBuilder` | A DistributedOperator sums local operators: H, dense block rows or user-defined ones such as an external FMM |

- **Cluster tree.** `ClusterTreeBuilder::create_cluster_tree(n, dim, coords, …)` bisects geometrically, along the largest extent or a PCA axis, down to a maximum leaf size. It is built sequentially and replicated on every rank.
- **Admissibility.** The default is Rjasanow–Steinbach, 2 · min(r\_t, r\_s) < η · dist; user conditions plug in.
- **Symmetry.** Storage can be general (N), symmetric (S, real or complex) or Hermitian (H, complex), with the lower or upper triangle; only one block triangle is built.

## Conversions

Htool-DDM extracts dense data and recompresses leaves, but has no other formats to convert into.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| H → dense | `copy_to_dense`, `copy_to_dense_in_user_numbering` | Exact | The user-numbering version also expands symmetric storage. Python `to_dense`; PETSc `MatConvert` to dense |
| H → diagonal or sub-block | `copy_diagonal(_in_user_numbering)`, `get_sub_hmatrix` | Exact | Used to extract each rank’s diagonal block for the DDM |
| Low-rank block → dense | `LowRankMatrix::copy_to_dense` | Exact | — |
| Dense → H | Only through a generator | Recompressing | See Construction |
| H → H, same tree, lower ranks | `recompression`, `openmp_recompression` | Recompressing (QR + SVD per low-rank leaf) | No coarsening or merging of blocks |
| H → transpose | None in Htool; only operation flags on products. PETSc `MatTranspose` reassembles from a transposed kernel | Recompressing (PETSc) | — |
| H → H², HODLR, BLR or another block tree; precision changes | — | ✗ | The scalar type is a template parameter, but there is no cast between precisions |

## Operations

The H-matrix can be applied, factored and solved on one node; the MPI operator adds distributed products and HPDDM’s solvers. No storage provides matrix functions, A + σI, inversion, QR, eigendecomposition, SVD or a determinant.

| Operation | HMatrix, full storage | HMatrix, symmetric or Hermitian storage | DistributedOperator (MPI) |
| --- | --- | --- | --- |
| Matvec | ✓ \[1\] | ✓ \[1\] | ✓ \[10\] |
| Transpose or adjoint matvec | ◐ \[2\] | ◐ \[2\] | ✓ \[10\] |
| Block of vectors | ✓ \[3\] | ✓ \[3\] | ✓ \[10\] |
| Direct solve | ✓ \[4\] | ✓ \[4\] | ✗ |
| Package-provided iterative solve | ✗ | ✗ | ✓ \[11, 12\] |
| Triangular solve | ◐ internal \[6\] | ◐ internal \[6\] | ✗ |
| A + B | ◐ low-rank + H only \[7\] | ◐ symmetric low-rank + H only \[7\] | ✗ |
| Scaling | ✓ \[8\] | ✓ \[8\] | ✗ |
| A · B | ◐ internal \[9\] | ✗ silently wrong \[9\] | ✗ |
| LU | ✓ \[4\] | ✗ (needs full storage) | ✗ |
| Cholesky | ✓ \[4\] | ✓ \[4\] | ✗ |
| LDLᵀ | ◐ unreleased \[5\] | ◐ unreleased \[5\] | ✗ |
| Recompression | ✓ | ✓ | ✗ |

✓ provided; ◐ partial, internal or unreleased; ✗ absent or unusable; \[n\] = note n below.

1. **Matvec.** `add_hmatrix_vector_product([policy,] trans, α, A, x, β, y)` works in user numbering; `internal_…` variants work in cluster numbering, sequentially or with OpenMP. Symmetric storage uses symv or hemv on diagonal leaves and applies off-diagonal leaves twice. Verified by a run.
2. **Transposes.** T and C work in cluster numbering, including rectangular matrices. The user-numbering wrapper permutes with the source and target clusters regardless of `trans`, so it is right only when they coincide; a rectangular matrix gives a heap-buffer overflow (ASan, 1.0.2, unchanged on main). Symmetric storage logs an error for C on S and T on H.
3. **Dense block.** `add_hmatrix_matrix_product([policy,] transa, transb, α, A, B, β, C)`, with OpenMP kernels.
4. **Factorizations,** in place on the same block structure. `lu_factorization([policy,] A)` needs full storage; `cholesky_factorization([policy,] UPLO, A)` takes full (trusted symmetric), symmetric or Hermitian storage, and logs an error but continues on complex-symmetric storage. `lu_solve(trans, A, X)` and `cholesky_solve(UPLO, A, X)` take several columns; pivoting happens only inside dense diagonal leaves.
5. **LDLᵀ.** `ldlt_factorization(sym, UPLO, A)` and `ldlt_solve` use Bunch–Kaufman inside dense leaves and run sequentially; they exist only on main (compiling against 1.0.2 fails). LU and Cholesky date from 0.9.0, and 1.0.0 added their OpenMP-task versions.
6. **Triangular solves.** `internal_triangular_hmatrix_{matrix,hmatrix,lrmat}_solve(side, UPLO, trans, diag, α, A, B)` work in cluster numbering; they are undocumented but tested.
7. **Addition.** Only low-rank + H (`internal_add_lrmat_hmatrix`); there is no H + H.
8. **Scaling.** `scale(α, H)` and `openmp_scale`.
9. **H×H.** `internal_add_hmatrix_hmatrix_product(transa, transb, α, A, B, β, C)` accumulates into an existing C (H, low-rank or dense), truncated by SVD to C’s tree and tolerance. A task-based variant needs a precomputed partition. It reproduced the dense product to 1.5e-6 at ε = 1e-6; a symmetric-storage operand silently drops the unstored triangle (52% error in a test).
10. **DistributedOperator.** Global-to-global and local-to-local vector and matrix products support N, T and C on rectangular matrices. Each rank stores one block row (local rows × all columns), split by one level of the target cluster tree; products use `MPI_Allgatherv`, `MPI_Allreduce` or `MPI_Alltoallv`.
11. **Iterative solver (package-provided).** `DDM::solve` calls HPDDM: GMRES, block GMRES, CG, block CG, GCRODR and variants, or Richardson. Preconditioners: one-level Schwarz (none, RAS or ASM) and two-level GenEO, which solves dense generalized eigenproblems on each densified local matrix and reduces the coarse operator to rank 0.
12. **DDM local solvers.** Dense LAPACK, or the H-matrix factored by H-LU, H-Cholesky (only when `is_positive_definite` is set) or H-LDLᵀ. In 1.0.2, symmetric matrices always used H-Cholesky, which fails for indefinite ones.
13. **Other.** Diagonal extraction, sub-H views, copies, rank statistics and visualization. PETSc’s MATHTOOL adds `MatGetDiagonal`, `MatCreateSubMatrices` and lazy `MatScale`/`MatShift`, which are PETSc features, not Htool’s.

## HPC suitability

Htool-DDM threads assembly, products and factorizations on one node, and uses MPI only for assembly, products and the DDM solve. There is no GPU code on any branch.

- **Threading.** OpenMP only. The `par` policy selects `omp parallel for` loops (assembly, matvec, H×dense) or OpenMP tasks with dependencies over at most 64 top blocks (LU and Cholesky). Recompression and scaling are threaded only by `openmp_recompression` and `openmp_scale`. No `std::execution` algorithm is ever called, and the Python module is built without OpenMP flags.
- **Distributed memory.** MPI covers DistributedOperator assembly (one block row per rank, cluster tree and coordinates replicated), its products (global collectives, O(N) traffic each), HPDDM’s Krylov and Schwarz solvers, and GenEO coarse assembly. There is no distributed factorization or arithmetic, and PETSc’s MATHTOOL factorization needs a single process.
- **GPU and runtimes.** No CUDA, HIP, SYCL or Kokkos code; no runtime system beyond OpenMP tasks.

| Operation | Threads | MPI | GPU |
| --- | --- | --- | --- |
| Cluster and block trees | ✗ sequential | Cluster tree replicated; block tree per rank | ✗ |
| Assembly | ✓ OpenMP loops or tasks (`DefaultApproximationBuilder` and PETSc build each rank’s block row sequentially) | ✓ one block row per rank | ✗ |
| Matvec, H×dense | ✓ OpenMP loops with a reduction | ✓ collectives | ✗ |
| Recompression, scaling | ✓ | ✗ | ✗ |
| H×H, triangular H-solves | Sequential; task variants are internal | ✗ | ✗ |
| H-LU, H-Cholesky | ✓ OpenMP tasks | ✗ | ✗ |
| H-LDLᵀ (unreleased) | ✗ sequential; task version on a branch | ✗ | ✗ |
| Factor solves | ✗ sequential (BLAS may thread) | ✗ | ✗ |
| DDM local factorization | ✗ sequential per rank | ✓ one subdomain per rank | ✗ |
| GenEO eigenproblems, coarse operator | Dense LAPACK per rank | ✓ broadcast, then reduce to rank 0 | ✗ |
| Krylov solve | — | ✓ HPDDM | ✗ |

✓ provided; ✗ absent; — not applicable.

Benchmark data are shared-memory only. Except the June 2026 bullet, they come from the [htool\_benchmarks](https://github.com/PierreMarchand20/htool_benchmarks) reference set (pre-1.0 code, April 2024, 2 × Xeon Gold 6230). The kernel was 1/(|x − y| + 1e-5) on points on a cylinder, with symmetric storage and η = 10.

- **Assembly**, n = 524,288 at ε = 1e-10: 61.7 s on 1 thread; 10.0 s with tasks or 14.7 s with loops on 16 threads; compression ratio 158. At ε = 1e-4: 11 s falling to 2.2 s, ratio 502.
- **Thirty matvecs** at that size: 88 s on 1 thread, 16.4 s on 16 threads.
- **H-LU**, n = 262,144 (η = 100, 1 thread): 345 s at ε = 1e-4 to 1,610 s at 1e-10. H-Cholesky took 175–828 s; 30 solves took 17–58 s after H-LU and 17–49 s after H-Cholesky.
- **June 2026 set** (htool\_benchmarks c348f93, AMD nodes, same kernel; the docs’ [Benchmark Explorer](https://pierremarchand20.github.io/htool-benchmarks-app/app.html), data at [htool-benchmarks-app 374b213](https://github.com/PierreMarchand20/htool-benchmarks-app/tree/374b213a719601a10c430a4571062fbf198f4556)). At ε = 1e-10, 1 vs 16 threads: assembly at n = 524,288, 24.6 s vs 4.5 s; 30 matvecs at that size, 60 s vs 23.5 s; task-based H-Cholesky at n = 16,384, 25.8 s vs 9.0 s.
- **No MPI or DDM timings** were found: the JOSS paper has no performance data, and 2021 slides with N = 10⁶ MPI runs are illegible.

Practical limits: indices are 32-bit, every product moves O(N) data through collectives, and GenEO densifies each local matrix (O(n\_p³)) and builds a dense coarse matrix on rank 0. DDM local factorizations are single-threaded, and the matvec reduction allocates one full-length vector per thread.

## Caveats and surprises

- **Rectangular transposed matvec corrupts the heap** in user numbering (ASan on 1.0.2; the code is unchanged on main). DistributedOperator transposes are fine.
- **Errors are logged, not thrown.** H-Cholesky on an indefinite matrix, for example, prints errors and continues.
- **The coarse space hard-codes `MPI_COMM_WORLD`** (`DDM::build_coarse_space`).
- **H arithmetic is undocumented** and works in cluster numbering; H×H needs a pre-built result matrix, and there is no H + H, shift or inverse.
- **Python bindings** (pinned to a March 2026 commit) expose N products, LU and Cholesky with solves, recompression, the DistributedOperator, the DDM solvers and GenEO, but not LDLᵀ, H×H or transposed products.
- **Easily misread history.** LU and Cholesky arrived in 0.9.0 (September 2024), not 1.0. PETSc main already uses the unreleased LDLᵀ for non-SPD “Cholesky”.
- **Unreleased branches** add Chebyshev HCA construction (`interpolation`), a task-based LDLᵀ and Python threading.

## Verification

This deep dive read Htool main at 0ee4cac and 1.0.2 (63b23d0), plus htool\_python c1d53b9, HPDDM 3c5c250, htool\_benchmarks c348f93, FreeFEM 4.17’s BEM plugin (754dae8) and PETSc main’s Htool interface. Test programs ran on a 2-core container against dense references.

| Test | Result |
| --- | --- |
| Probe, n = 3,000, ε = 1e-6, η = 10, kernel 1/(0.1 + \|x − y\|) + 10δ on a sphere, rectangle 3,000 × 2,000 (g++ 13, OpenMP); relative 2-norm errors (solves: against x) | Assembly used 1 thread with `seq` and 2 with `par`. Matvec N and T 7e-7; rectangular T in cluster numbering 1.4e-6; H×dense (4 columns) 4.5e-7; H×H 1.5e-6; H-LU solve 1.2e-6, also for T |
| Symmetric storage, same probe | H-Cholesky 2e-6; H-LDLᵀ 2e-6 (SPD) and 7e-5 (indefinite) |
| ASan build against 1.0.2 | Heap-buffer overflow in the rectangular user-numbering T matvec |
| BLAS interposition | `par` LU, Cholesky and H×dense ran on 2 threads; `seq`, `lu_solve` and LDLᵀ on 1. Speed-ups could not be measured on the throttled container |
| MPI, 2 ranks (MPICH from a PyPI wheel) | Each rank stores a 2,000 × 4,000 block row; global-to-global N and T, local-to-local, 3 right-hand sides and symmetric storage all gave errors near 1e-6 (symmetric matrix, so T is checked against the N result) |
| Copy of `examples/use_ddm_solver.cpp` with n cut from 10,000 to 4,000 (ε = 1e-3, η = 100), 2 ranks | CG with ASM (block Jacobi) and local H-LDLᵀ converged in 3 iterations; error 5e-9 against x for b = A\_H·x (excludes compression error) |

gitlab.com was refused by the network proxy, so PETSc was read from its GitHub mirror.

## Sources

- [Htool-DDM repository](https://github.com/htool-ddm/htool) at main 0ee4cac and 1.0.2 (63b23d0). Files read: `include/htool/hmatrix/` (`hmatrix.hpp`, `tree_builder/`, `interfaces/`, `lrmat/`, `linalg/`, `utils/recompression.hpp`, `execution_policies.hpp`), `include/htool/distributed_operator/`, `include/htool/solvers/` (`ddm.hpp`, `utility.hpp`, `local_solvers/`, `geneo/`), `CHANGELOG.md`, `CMakeLists.txt`
- [htool\_python](https://github.com/htool-ddm/htool_python) and [HPDDM](https://github.com/hpddm/hpddm)
- [Documentation](http://htool-ddm.pages.math.cnrs.fr/) and [JOSS paper](https://doi.org/10.21105/joss.09279)
- PETSc [MATHTOOL](https://petsc.org/release/manualpages/Mat/MATHTOOL/) and [MatCreateHtoolFromKernel](https://petsc.org/release/manualpages/Mat/MatCreateHtoolFromKernel/) manual pages; `src/mat/impls/htool/` in the PETSc GitHub mirror
- FreeFEM sources, `plugin/mpi/bem.hpp`
- [htool\_benchmarks](https://github.com/PierreMarchand20/htool_benchmarks) and the [benchmark hardware](https://docs.idcs.mesocentre.ip-paris.fr/cholesky/hardware_description/)
- [Marchand, JCJC 2021 slides](https://jcjcdeveloppement.pages.math.cnrs.fr/2021/slides/pierre-marchand.pdf)
