# ButterflyPACK deep dive

> Written in October 2026 by Nick Alger. Reflects ButterflyPACK master at a8cf81b (17 Sep 2026), just after release 5.0.0; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

ButterflyPACK is an MPI + OpenMP library from Lawrence Berkeley National Laboratory for building and directly factoring dense matrices with low-rank and butterfly blocks. It is strongest on oscillatory kernels, such as integral equations and Helmholtz fronts inside STRUMPACK, and on black-box construction from entries or randomized matvecs. It offers HODLR/HODBF inversion, H-LU and BLR-LU with log-determinants, all distributed over MPI; version 5.0.0 adds a separate H² solver and extends the H-tensor format introduced in 4.1.0.

There is no arithmetic between compressed matrices: no A + B, A·B, shift, scaling or recompression, and no matrix functions, Cholesky, QR or SVD. The H² solver handles only transpose-symmetric matrices, needs an MPI build, and its `c_bpack_mult` accepts only `trans` = N (Aᵀ = A here). H and BLR solve only untransposed systems, and SHNBF and tensor formats have no direct solver.

GPU support in the release is limited to an optional MAGMA kernel for butterfly matvecs; CUDA backends for H² and HODLR sit on unreleased branches.

## Snapshot

ButterflyPACK is actively developed, mostly by one maintainer, and its newest features are documented only in the changelog, option strings and examples.

| Item | Value |
| --- | --- |
| Release | 5.0.0, released 22 Aug 2026 (the git tag is dated 24 Aug); 4.1.0 in Jan 2026, 4.0.0 in Jul 2025 |
| Last commit | 17 Sep 2026 on master: 5 commits after 5.0.0, adding a C interface for the `BP_*` routines. Unreleased branches `H2_optimization` and `gpu_hodlr` (to 6 Oct 2026) carry GPU work |
| Activity | 332 commits on master since 2024 (51 in 2024, 87 in 2025, 194 in 2026 to mid-September); 1,335 in total |
| Licence | BSD-3-Clause with the LBNL enhancements clause |
| Languages and bindings | Fortran core in four precisions (s/d/c/z); header-only C++20 H² solver; C API; templated C++ wrapper; Python since 4.1.0 (not on PyPI) |
| Dependencies | CMake, BLAS, LAPACK; MPI and ScaLAPACK for parallel builds. Optional: OpenMP, (P)ARPACK, ZFP, FFTW, MAGMA, PaRSEC, pybind11 |
| Maintainers | Yang Liu (LBNL) is effectively the sole maintainer; about 10 contributors in all, including Tianyu Liang for the H² solver |
| Documentation | Hosted Doxygen is still v3.2.0 and covers formats 1–5 only. H², Python, PaRSEC and the iterative solvers are documented mainly by the changelog, option help strings and example drivers |
| Users | STRUMPACK (HODLR/HODBF compression of dense matrices and sparse fronts); AWS Palace through STRUMPACK; xSDK, E4S and Spack; GPTune examples |

## Construction

Every format can be built from an entry or block callback. HODLR/HODBF, and H and BLR with low-rank blocks, can also be built from matvecs alone; no format accepts a sparse matrix or a dense array directly.

| Access | Formats | Algorithms | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | ✗ | — | — | Sparse problems go through STRUMPACK, which calls ButterflyPACK for its fronts |
| Entry or block callback | HODLR/HODBF (format 1), H-LR/H-BF (2), HSS-BF/SHNBF (3), BLR/B-BF (5), H² (7); single LR or butterfly blocks; rectangular H and BLR | Low-rank blocks: SVD, RRQR, ACA, BACA, pseudo-skeleton, parallel ACA and hierarchical H-BACA. Butterflies: O(N^1.5) or O(N log N) interpolative-decomposition butterfly with proxy sampling. H²: ID skeletonization with sparse sketching and optional proxies | `BPACK_construction_Element` / `c_bpack_construct_element_compute`; `BF_*` and `BP_*` variants | 1-based indices in original order; single entries or lists of 2D-distributed sub-blocks (`elem_extract` 0, 1 or 2). Dense near-field blocks can be ZFP-compressed |
| Kernel + coordinates | Same | Coordinates drive clustering, admissibility and proxy selection; the kernel is always the user’s entry callback | Same | No built-in kernels; examples supply RBF, Matérn, Laplace, EFIE/CFIE, VIE and Radon kernels |
| Matvecs only (A and Aᵀ or Aᴴ) | HODLR (low-rank and butterfly); H and BLR with low-rank blocks; single blocks | Level-by-level peeling with randomized low-rank or randomized butterfly reconstruction; block-colored peeling for strong admissibility; adaptive rank growth | `BPACK_construction_Matvec` / `c_bpack_construct_matvec_compute` | H-BF aborts with “not yet implemented”. Not available for HSS-BF/SHNBF, H² or tensors |
| Dense matrix | Through callbacks | — | — | Example drivers wrap a dense or 2D block-cyclic matrix as an entry or matvec callback |
| Existing compressed matrix | 1, 2, 3, 5 | Entry extraction or matvec of the source | `c_bpack_extractelement`, `c_bpack_mult` | See Conversions |
| Tensor entries | HSS-BF-MD/SHNBF-MD (4), H-tensor (6), single tensor butterfly | Tensor butterfly with Tucker-type ID; optional TT/QTT and ZFP compression of middle-level cores; translational-invariance reuse; FFT for inadmissible blocks | `c_bpack_md_construct_element_compute` | Entry callbacks only |

Geometry is optional for formats 1–6 and required for H²:

- **Formats 1–6** cluster from coordinates (KD-tree or “cobble” partition), from entries alone (Gram distance; symmetric or Hermitian matrices, via `c_bpack_construct_init_gram`), from a user distance and admissibility callback, or from user nearest-neighbour lists. A user leaf-size tree can also be passed. Admissibility is weak (`near_para` = 0) or strong (about 2).
- **H²** needs 1–3D coordinates and builds a uniform 2^d tree in Morton order.
- **Accuracy depends on sampling.** In testing, HODBF built from entries reached only 2.4e-5 matvec error at `tol_comp` = 1e-9 with default sampling; built from matvecs it reached 1.4e-9.

## Conversions

Formats do not convert into one another. A different format or tolerance is rebuilt from an existing compressed matrix through its own extraction or matvec callbacks.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| Formats 1, 2, 3, 5 → dense | `BPACK_Convert2Dense` (Fortran only) | Exact | Applies the matrix to identity columns into a ScaLAPACK 2D block-cyclic array; used by the dense eigensolver |
| Formats 1, 2, 3, 5 → another of these, or another tolerance | Rebuild from `c_bpack_extractelement` or `c_bpack_mult` callbacks | Recompressing | The documented pattern in `EXAMPLE/InterfaceTest.cpp`; since 3.2.0 the sketching and target formats may differ |
| Double complex → single complex | `z2c_BPACK_copy`, `z2c_BPACK_Solution` | Rounded to single | HODLR and H/BLR only. Used for a double-precision iterative solve preconditioned by a single-precision factorization |
| H ↔ H² (no in-place conversion) | — | Recompressing (H² → 1, 2 or 5 only) | H² cannot be read by `c_bpack_extractelement` or built from matvecs; its N-only `c_bpack_mult` (Aᵀ = A) can feed a matvec rebuild into formats 1, 2 or 5. |

## Operations

ButterflyPACK applies, solves and factors, but offers no arithmetic on compressed matrices. No format provides matrix functions, A + B, A + σI, scaling, A·B, Cholesky, QR, user-callable recompression or public triangular solves.

| Operation | HODLR/HODBF (1) | H-LR/H-BF (2) | BLR/B-BF (5) | HSS-BF (3) | SHNBF (3) | H² (7) | Tensor (4, 6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matvec | ✓ \[3\] | ◐ \[3\] | ◐ \[3\] | ✓ \[3\] | ✓ | ✓ \[2\] | ✓ \[11\] |
| Transpose or adjoint matvec | ✓ \[3\] | ◐ \[3\] | ◐ \[3\] | ✓ \[3\] | ✓ | ✗ | ✗ |
| Block of vectors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Direct solve | ✓ N, T, C \[4\] | ◐ N only \[4\] | ◐ N only \[4\] | ✓ N, T, C | ✗ \[10\] | ✓ N | ✗ |
| Own iterative solve \[5\] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ BiCGSTAB | ✓ |
| Inversion | ✓ randomized \[6\] | ✗ (LU instead) | ✗ (LU instead) | ✓ partitioned | ✗ | ✗ | ✗ |
| LU | ✗ | ✓ H-LU \[7\] | ✓ block LU \[7\] | ✗ | ✗ | ✗ | ✗ |
| LDLᵀ | ◐ `sym=1`, low-rank only \[6\] | ✗ | ✗ | ✗ | ✗ | ◐ RS-S, symmetric only \[8\] | ✗ |
| Eigenvalues | ◐ \[13\] | ◐ | ◐ | ◐ | ◐ regular mode, dense | ✗ | ✗ |
| Log-determinant \[9\] | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ RS-S only | ✗ |
| Entry extraction \[12\] | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |

1. **Matvec.** `BPACK_Mult` / `c_bpack_mult(trans, …)` dispatches to the HODLR, H/BLR and HSS kernels; each handles N, T and C, and the number of vectors is an argument (`SRC/BPACK_solve_mul.f90:3364–3390`). Verified with three vectors for formats 1, 2, 3 and 5.
2. **H² matvec.** `c_bpack_mult` with format 7 rejects any `trans` other than N (`SRC/C_BPACK_wrapper.cpp:1212–1218`, verified). It multiplies with either the RS-S factors or a compression-only H².
3. **Forward operator retention.** H and BLR keep a copy of the forward blocks only if `ErrSol` = 1 or `precon` ≠ 1. With the defaults, `c_bpack_mult` on an H matrix segfaulted in testing; HODLR and HSS-BF also drop their forward blocks during factorization.
4. **Direct solve.** `BPACK_Solution` / `c_bpack_solve` with `precon` = 1, and `c_bpack_inv_mult(trans)`. For H and BLR, T and C stop with “XxU^-1 with MPI is not yet implemented” (`SRC/BPACK_solve_mul.f90:5316–5318`, verified); `IR_HODLR` > 0 adds iterative refinement inside the HODLR inverse apply. For H² use `c_bpack_solve`, because `c_bpack_inv_mult` has no format-7 branch.
5. **Package-provided iterative solvers.** `precon` = 2 runs without a preconditioner and `precon` = 3 uses the factorization as preconditioner; `iter_solver` selects GMRES (default), TFQMR, IR or CG, one right-hand side at a time. Black-box variants accept user matvec and preconditioner callbacks; H² uses BiCGSTAB on the compression-only H².
6. **HODLR/HODBF inversion.** Leaves are LU-inverted, off-diagonal blocks are updated bottom-up by randomized low-rank or butterfly reconstruction, and each 2×2 level uses a partitioned inverse. `sym` = 1 is a symmetric variant (Bunch–Kaufman leaves plus rank-r Schur corrections) that needs format 1, `LRlevel` = 0 and no ZFP.
7. **H and BLR LU.** A right-looking block LU over a 2^`Dist_level` grid of top-level blocks in 2D block-cyclic layout; each block is a recursive H-LU with randomized or deterministic low-rank and butterfly arithmetic. `ILU` = 1 factors only the dense diagonal blocks.
8. **H² factorization.** Strong recursive skeletonization (RS-S) with Bunch–Kaufman dense kernels, for transpose-symmetric matrices only: with `sym` = 0 the code calls `MPI_Abort` (verified). Hermitian matrices are not covered.
9. **Log-determinant.** `c_bpack_logdet`, accumulated during factorization from leaf pivots and Schur factors. HSS-BF stops with “not yet implemented”; H² needs an RS-S factorization; Python exposes it as `bpack_logdet`.
10. **SHNBF.** Construction and matvec work, but its factorization (the HSS-BF path) crashed in testing, so solves are iterative only.
11. **Tensor formats.** `c_bpack_md_mult` applies the matrix but not its transpose; `c_bpack_md_solve` is iterative only.
12. **Entry extraction.** `c_bpack_extractelement` returns arbitrary sub-blocks of the compressed matrix for formats 1, 2, 3 and 5.
13. **Eigenvalues.** `BPACK_Eigen` (Fortran only, PARPACK, double complex only) runs regular, shift-invert or generalized modes; shift-invert needs a separately built and factored shifted matrix. `BPACK_Eigen_Dense` converts to dense and uses ScaLAPACK; there is no SVD.
14. **Interfaces.** Fortran reaches everything except H², which only the C dispatchers reach. The C++ wrapper covers construction, extraction, mult, factor and solve; Python covers block-entry construction, factor, solve, transposed mult and log-det.

## HPC suitability

Every format is distributed over MPI for construction, matvec, factorization and solve. Threading lives mostly inside compression and butterfly kernels, and the release has almost no GPU code.

- **Threading.** Optional OpenMP in low-rank and butterfly compression, entry evaluation and butterfly matvec kernels, plus optional batched GEMM for butterfly matvecs (`BPACK_BF_MVP_BATCH`). The H-matrix matvec recursion is serial within a rank. Top-level OpenMP tasking for H and BLR LU (`enable_toplevel_openmp`) is off by default and marked experimental.
- **Distributed memory.** MPI with BLACS/ScaLAPACK over a binary process tree. HODLR/HODBF and HSS-BF distribute each block over a process group; H and BLR place top-level blocks 2D block-cyclically, one rank per block. H² distributes a Morton-ordered uniform tree with graph-coloring elimination and optional communication-avoiding levels.
- **GPU.** In the release, only an optional MAGMA path that offloads butterfly block matvecs (`magma_gemm_vbatched`), copying data per call (read in code, not run). Unreleased branches from September–October 2026 add CUDA backends for H² and HODLR construction, factorization, multiplication and solve.
- **Runtime systems.** PaRSEC, for the H and BLR factorization only: build with `-Denable_parsec=ON` and set `use_parsec` = 1. The task graph covers the top-level block LU.

| Operation | Threads | MPI | GPU in the release |
| --- | --- | --- | --- |
| Construction from entries (formats 1, 2, 3, 5, tensors) | ✓ | ✓ | ✗ |
| Construction from entries (H²) | ✓ | ✓ | ✗ (unreleased branch) |
| Construction from matvecs (1, 2, 5) | ◐ kernels | ✓ (user matvec must be distributed) | ◐ MAGMA butterfly kernel |
| Matvec | ◐ butterfly kernels and BLAS | ✓ | ◐ MAGMA butterfly kernel |
| HODLR/HODBF and HSS-BF inversion | ✓ | ✓ | ◐ MAGMA kernel in randomized products |
| H and BLR LU | ◐ BLAS; experimental top-level tasks | ✓ one rank per top-level block; optional PaRSEC | ✗ |
| H² factorization, solve, log-det | ✓ | ✓ | ✗ (unreleased branch) |
| Direct solves | ◐ | ✓ | ✗ |
| Iterative and eigen solvers | Through matvec | ✓ | Through matvec (eigen: ✗) |

Largest documented runs:

- **Butterfly fronts in STRUMPACK**, 3D heterogeneous Helmholtz (Marmousi2), N = 21,176,640 on 32 Cori Haswell nodes. At ε = 1e-3 the factorization took 674–1,037 s with 0.58–0.77 TB of factors and 59–63 GMRES iterations; at 1e-6 it took 1,657 s. The exact multifrontal solver took 660 s and 1.97 TB, so compression saved memory, not time ([Liu, Ghysels, Claus, Li, SISC 2021](https://arxiv.org/pdf/2007.00202), Table 6.2).
- **Matvec-only butterfly construction**, 3D Helmholtz, n = 1,510,734 at ε = 1e-4: 1,470 s and 10.4 GB on 2 Cori nodes with 64 ranks ([Liu et al., SISC 2021](https://arxiv.org/pdf/2002.03400); arXiv Table 8.1).
- **2D EFIE preconditioner** (H-IDBF-LU): “up to 100 million unknowns”; the paper does not say ButterflyPACK produced these runs ([Liu and Yang, arXiv:1901.11371](https://arxiv.org/abs/1901.11371)).
- **Tensor butterfly**: 6-mode Green’s-function tensor up to 512 wavelengths (2,048 points) per mode at ε = 1e-2 on 4 Perlmutter CPU nodes; at 64 wavelengths, 200× faster and 30× less memory than matrix butterflies ([Kielstra et al.](https://arxiv.org/abs/2411.03029v2)).
- **H²**: no published numbers for the 5.0.0 solver. Notes on an unreleased branch report compressing and factoring 7.1M points (3D Laplace, 1e-4) in 11.8 s on 64 A100s.

Practical limits: HODLR ranks grow quickly for 3D volume problems, and H/BLR LU parallelism is bounded by the number of top-level blocks. H² needs `sym` = 1, MPI, 1–3D coordinates, a uniform tree and (2^d)^k MPI ranks (1, 8, 64, … in 3D).

## Caveats and surprises

- **Default options drop the forward operator.** With `precon` = 1 and `ErrSol` = 0, H and BLR keep no forward copy, and HODLR and HSS-BF delete theirs during factorization. `c_bpack_mult` on an H matrix then segfaults; set `ErrSol` = 1 (adds a test solve) or `precon` = 3 (makes solves iterative), or for HODLR `IR_HODLR` > 0, to keep the operator.
- **No transposed solve for H and BLR.** A T or C solve stops the program; HODLR and HSS-BF support both.
- **SHNBF cannot be factored.** The factorization crashed in testing, so SHNBF is iterative only.
- **Iterative solves for formats 1–6 take one right-hand side at a time.** H² BiCGSTAB and the direct HODLR, H and H² solves take many in one call.
- **Errors usually end the program.** Fortran paths call `stop` (exit status 0) and H² paths call `MPI_Abort` instead of returning an error code; some H² argument checks throw C++ exceptions.
- **Easily misread features.** The log-det also exists for HODLR, H and BLR, not just H². The H² solver is reachable only from C, C++ and Python, needs MPI, and its “symmetric” means transpose-symmetric. PaRSEC accelerates only the H and BLR factorization.
- **Unreleased GPU work.** The CUDA H² and HODLR backends that landed on branches after 5.0.0 should be treated as experimental.
- **Same group, different code.** The H² recursive-skeletonization solver of [Boukaram, Keyes, Li, Liu and Turkiyyah](https://arxiv.org/html/2509.11152v1) is not ButterflyPACK’s.

## Verification

This deep dive read master at commit a8cf81b (17 Sep 2026), which differs from tag v5.0.0 in 10 files, mainly the new C `c_bp_*` API. Two builds were tested on a 2-core container against dense LAPACK references.

Sequential build (gfortran 13.3 with OpenMP, no MPI): N = 1,500 random 2D points, kernel exp(−r/0.2) + 0.5I, `tol_comp` = 1e-9, three right-hand sides. All but the symmetric HODLR runs add 0.3(xᵢ − xⱼ)exp(−r/0.2), where x is the first coordinate, so A ≠ Aᵀ.

| Format | Results (matvec: relative error; solve: relative residual ‖Ax − b‖/‖b‖; log-det: relative error of log\|det\|) |
| --- | --- |
| HODLR | Matvec N and T ≈1e-9; log-det 1.2e-13; solve 1.4e-8; T solve 1.7e-8; built from 88 black-box matvec calls (blocks of vectors) at 1.2e-9; unpreconditioned GMRES 1.4e-8 |
| HODLR, `sym` = 1 | Log-det 2.9e-13; solve 2.9e-8 |
| HODBF | From entries, matvec 2.4e-5; from matvecs, 1.4e-9 |
| H-LR | Matvec segfaults with default options. With `ErrSol` = 1: matvec 5e-10, log-det 9e-15, solve 7e-9; T solve stops; matvec construction 1.9e-8 |
| H-BF | Matvec 4.8e-7; log-det 3e-11; solve 4.7e-6 |
| BLR | Matvec 3.4e-10; log-det 4e-15; solve 8e-9; matvec construction 1.2e-9 |
| HSS-BF | Matvec 3.5e-5; solve 4.3e-4; T solve 3.5e-4; log-det stops |
| SHNBF | Matvec 4.6e-6; factorization crashes; GMRES 4.9e-5 |

MPI build (OpenMPI 4.1.6, ScaLAPACK 2.2.1), H² with 1 rank × 2 threads on a 16³ grid (N = 4,096), symmetric kernel exp(−r/0.2) + 0.5I, tolerance 1e-6.

| Case | Results |
| --- | --- |
| RS-S factorization | Matvec 1.2e-6; solve with 3 right-hand sides, residual 1.4e-5; log-det 5.4e-10; T matvec rejected |
| Compression only (`precon` = 2) | Matvec 1.3e-6; BiCGSTAB residual 1.4e-5 |
| `sym` = 0 | `MPI_Abort` asking for `sym` = 1 |

Not exercised: multi-rank runs, PaRSEC, MAGMA, Python, tensor formats and the PARPACK eigensolvers.

## Sources

- [ButterflyPACK repository](https://github.com/liuyangzhuan/ButterflyPACK) at master a8cf81b and tag v5.0.0 (747b05a). Files read: `CHANGELOG`, `README.md`, `license.txt`, `SRC/BPACK_defs.f90`, `SRC/BPACK_constr.f90`, `SRC/BPACK_randomized.f90`, `SRC/BPACK_factor.f90`, `SRC/BPACK_solve_mul.f90`, `SRC/BPACK_wrapper.f90`, `SRC/C_BPACK_wrapper.cpp`, `SRC/Bplus_compress.f90`, `SRC/Bplus_utilities.f90`, `SRC/parsec_hmat_factorization_ptg.jdf`, `SRC/Py_BPACK_wrapper.{cpp,py}`, `h2_parallel/*.hpp`, `CPP_INTERFACE/CPPWrapper.hpp`, `CPP_INTERFACE/m_BPACK_utilities.f90`, `EXAMPLE/InterfaceTest.cpp`, `doc/doxygen/pages/*.txt`
- [Hosted documentation, v3.2.0](https://portal.nersc.gov/project/sparse/butterflypack/)
- [Liu et al., Butterfly factorization via randomized matrix-vector multiplications, SISC 2021](https://arxiv.org/pdf/2002.03400)
- [Liu, Ghysels, Claus, Li, Sparse approximate multifrontal factorization with butterfly compression, SISC 2021](https://arxiv.org/pdf/2007.00202)
- [Liu and Yang, H-IDBF-LU preconditioner, arXiv:1901.11371](https://arxiv.org/abs/1901.11371)
- [Liu et al., H-BACA](https://arxiv.org/pdf/1901.06101)
- [Kielstra, Shi, Luo, Qian, Liu, tensor butterfly](https://arxiv.org/abs/2411.03029v2)
- [T. Liang, dissertation abstract, 2026](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/32302.html)
- [Palace installation docs](https://awslabs.github.io/palace/stable/install/)
