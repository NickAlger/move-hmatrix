# hmat-oss deep dive

> Written in October 2026 by Nick Alger. Reflects hmat-oss master at 45a2fdb (25 Sep 2026) and tag 1.11.1; later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

hmat-oss is a small, strictly sequential C++ H-matrix library with a C API, first developed by Airbus. It assembles H-matrices from entry or block callbacks with ACA or SVD. It offers the broadest arithmetic of the eight packages reviewed here: H×H, addition, identity shift, truncation and in-place inversion.

Factorizations cover H-LU for general real or complex matrices, H-LLᵀ and LDLᵀ for real symmetric or complex-symmetric ones, and a symmetric HODLR factorization A = WWᵀ with a log-determinant. OpenTURNS uses it for Gaussian-process covariance matrices.

The open-source code has no threads, MPI or GPU, and even pins BLAS to one thread; Airbus keeps its parallel StarPU version proprietary. There are no H², HSS or BLR formats, no Hermitian storage, no construction from dense, sparse or matvec input, and no QR, eigensolver or matrix functions. Several unsupported combinations return wrong results silently, all confirmed by running the code.

## Snapshot

hmat-oss is maintained by Airbus engineers, released as git tags only, and widely packaged.

| Item | Value |
| --- | --- |
| Release | 1.11.1, a git tag of 24 Nov 2025 (1.11.0 Nov 2025, 1.10.0 Jul 2024, 1.9.0 Sep 2023); no GitHub releases |
| Last commit | 25 Sep 2026 on master, 23 unreleased commits after 1.11.1: floating-point compression through composyx (ZFP, SZ, SZ3), octree clustering, matrix printing and Windows fixes. Active unmerged branches: `gs/master` (BLAS threads in workers, 7 Oct 2026), `ed/ddm_gemm` (8 Oct 2026), `gs/cuda` and `gs/cuda2` (Jan–Feb 2026) |
| Activity | 50 commits on master since 2024 (26 by Guillaume Sylvand, Airbus), 230 across all branches; 1,195 on master since November 2014 |
| Licence | GPL v2 or later |
| Language and bindings | C++11 core with a C99 API in `include/hmat/hmat.h`, the only installed header; single, double, complex and double-complex scalars. No Python or Fortran bindings; Python access only through OpenTURNS |
| Dependencies | CMake 3.13 or later, BLAS with CBLAS, LAPACKE or MKL. Optional: OpenMP (used only to switch threads off) and jemalloc. Release builds enable `-ffast-math` by default |
| Maintainers | Jérôme Robert (owner and original author), Guillaume Sylvand (Airbus Central R&T, main committer 2024–26), Denis Barbier; about 22 contributors from Airbus, IMACS, Inria and Phimeca |
| Documentation | Low to moderate: Doxygen comments in `hmat.h`, a README architecture overview rewritten in 2026, and C examples. No manual, hosted API reference or paper |
| Users | [OpenTURNS](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HMatrix.html) (HMatrix class; kriging and Gaussian-process algorithms with `LinearAlgebra = HMAT`); Airbus’s proprietary HMAT, which shares the C interface; Inria’s H-Chameleon. Packaged by Debian, Ubuntu, Fedora, FreeBSD, AUR, conda and nixpkgs |

## Construction

hmat-oss assembles one block-tree format, with dense and low-rank (ABᵀ) leaves, from an entry or block callback and point coordinates. The admissibility choice yields standard H-matrices, HODLR, or all-low-rank and all-dense tilings.

| Access | Formats | Algorithms | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | ✗ | — | — | The enum value `hmat_block_sparse` is never used in the source |
| Single-entry callback | H tree, any admissibility; general or lower-symmetric storage | SVD, full ACA and RRQR use the whole block; partial ACA, ACA+ and ACA-random need only rows and columns. Blocks with max(m, n) < 100 (setting `compressionMinLeafSize`) always use SVD, and every new low-rank block is truncated by QR + SVD at σᵢ ≤ ε·σ₁ | `assemble_generic` with `simple_compute`; compressors from `hmat_create_compression_{svd,aca_full,aca_partial,aca_plus,aca_random,rrqr}(ε)` | Verified. The callback sees original indices and must return double or complex double even for single precision. RRQR is buggy (see Caveats) |
| Block callback | Same | Same; multi-material sums (“strata”) are compressed per stratum, summed and recompressed | `assemble_generic` with `prepare` plus `block_compute` (legacy) or `advanced_compute` | `prepare` can declare null blocks, null rows or columns, memory needs and strata; compute is called for a full block, one row or one column |
| Kernel + coordinates | H | As above | `hmat_create_cluster_tree(coord, dim, n, algo)` plus a callback | Coordinates are mandatory; there are no built-in kernels, interpolation or proxy methods |
| Matvecs only | ✗ | — | — | — |
| Dense matrix | ✗ | — | — | Only by wrapping it in a callback |
| Other | H | — | `copy`, `copy_struct` (zero matrix on the same tree), `read_struct` and `read_data`, `gemm` into an existing structure; `ctx.factorization` assembles then factorizes | No helper assembles from another H-matrix or from a dense array |

- **Clustering.** Median, geometric, hybrid and NTilesRecursive algorithms, with `max_dof` and `span` wrappers (span = degrees of freedom with spatial extent); octree on master only. `hmat_create_cluster_tree_generic` adds spans, group indices and per-level algorithms.
- **Admissibility.** Standard Hackbusch, min(diam) ≤ η·dist on bounding boxes, with an aspect `ratio` and `max_width`; HODLR (every off-diagonal block low-rank); “always” (all low-rank, tiles bounded in size) and “never” (all dense).
- **Rectangular matrices** use different row and column trees.

## Conversions

hmat-oss extracts dense data, truncates and transposes, but has no other format to convert into and no precision cast.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| H → dense sub-block | `get_block` | Exact | Contiguous ranges in hmat numbering, returning index maps; mirrored only for a whole-matrix request (an upper sub-block of lower storage returns 0 and writes nothing); rejected on HODLR-factorized matrices. Verified |
| H → selected entries | `get_values` | Exact | 1-based original indices. Verified |
| H → diagonal | `extract_diagonal`, `extract_diagonal_block` | Exact | After LDLᵀ they return D |
| H(ε₁) → H(ε₂) | `set_low_rank_epsilon` + `truncate` | Recompressing (QR + SVD per low-rank leaf) | Verified: 1e-8 → 1e-3 shrank storage from 1.70M to 1.18M entries, matvec error 2.3e-5 |
| H → coarser H | `coarsening` setting, applied at assembly | Recompressing | Merges sibling low-rank leaves when cheaper |
| H → Hᵀ | `transpose`, in place | Exact | Verified |
| H → copy or zero matrix | `copy`, `copy_struct` | Exact or zero | Verified |
| H ↔ byte stream | `write_struct`/`write_data`, `read_struct`/`read_data` | Exact | Through user I/O callbacks; not run |
| H → lossy compressed leaves (master only) | `FPcompress` / `FPdecompress` | Lossy (ZFP, SZ, SZ3) | Storage only; arithmetic needs `FPdecompress` first |
| Dense or sparse → H; H ↔ H², HSS, BLR; general ↔ symmetric storage; precision changes | — | ✗ | HODLR is chosen at creation through admissibility, not by conversion |

## Operations

hmat-oss has full sequential H-arithmetic and three factorization families. A HODLR-factorized matrix is a separate state with a narrower set of operations.

| Operation | General storage | Symmetric lower storage | HODLR-factorized |
| --- | --- | --- | --- |
| Matvec | ✓ \[1\] | ✓ \[1\] | ◐ applies W \[7\] |
| Transpose or adjoint matvec | ✓ T, C \[1\] | ◐ C wrong for complex \[2\] | ◐ applies Wᵀ \[7\] |
| Block of vectors | ✓ H·X and X·H \[1\] | ✓ | ◐ \[7\] |
| Direct solve | ✓ LU \[3\] | ✓ LLᵀ, LDLᵀ \[3\] | ✓ dense right-hand side \[7\] |
| Package-provided iterative solve | ✗ | ✗ | ✗ |
| Triangular solve | ◐ \[4\] | ✓ \[4\] | ◐ `hodlrsym` only \[7\] |
| Matrix functions | ✗ | ✗ (applies the Cholesky factor only \[5\]) | ✗ (applies W only \[5\]) |
| A + B | ✓ \[6\] | ✓ \[6\] | ✗ |
| A + σI | ✓ \[6\] | ✓ \[6\] | ✗ |
| Scaling | ✓ \[6\] | ✓ \[6\] | ✗ |
| A · B | ✓ \[8\] | ✓ \[8\] | ✗ |
| Inversion | ✓ \[9\] | ✗ \[9\] | ✗ |
| LU | ✓ \[3\] | ✗ (accepted but wrong) | — |
| Cholesky | ✗ \[10\] | ✓ \[10\] | ✓ `hodlrsym`, real only \[7\] |
| LDLᵀ | ✗ | ✓ \[10\] | — |
| QR, eigen, SVD | ✗ \[11\] | ✗ | ✗ |
| Log-determinant | ✗ \[12\] | ◐ \[12\] | ◐ \[12\] |
| Recompression | ✓ \[13\] | ✓ \[13\] | ✗ |

1. **Matvec.** `gemv(trans, α, H, x, β, y, nrhs)` works in original numbering and is marked deprecated. `gemm_dense(trans_b, trans_x, side, …)` does H·X or X·H with N, T or C in hmat numbering, and `full_gemm` does dense × H. Verified: N and T at 2.7e-10, three columns, complex C on general storage at 9.7e-10.
2. **Adjoint on complex-symmetric storage.** Mirrored blocks map C to N, so the adjoint product is wrong (relative error 0.11, verified) while N is right (9.9e-10).
3. **Direct solve.** `factorize_generic` takes `lu`, `ldlt`, `llt`, `hodlr` or `hodlrsym`; solves go through `solve_systems`, `solve_dense` or `solve_generic`, and `solve_mat` returns an H solution for an H right-hand side. H-LU pivots only inside dense diagonal leaves, and LU has no transposed solve. Verified: LU 4.7e-10 with three right-hand sides; LLᵀ and LDLᵀ 4.4e-10.
4. **Triangular solves.** `trsm(side, uplo, trans, diag, m, n, α, A, is_b_hmat, B)` works on LU factors with dense or H right-hand sides; LUT, RUT and RLN raise errors, and α is ignored (verified). `solve_lower_triangular(_dense)` applies L⁻¹ or L⁻ᵀ after LLᵀ (2.4e-16). The `upper` flag of `solve_generic` actually solves LᵀX = B, not UX = B as documented.
5. **Factor application, not matrix functions.** After LLᵀ, `gemv` applies L or Lᵀ (verified L(Lᵀv) = Av at 3.4e-10); after `hodlrsym`, W or Wᵀ with A = WWᵀ. OpenTURNS samples Gaussian processes this way.
6. **Addition, shift and scaling.** `axpy(a, X, Y)`, `add_identity` and `scale`, verified at 1e-15 to 2e-10 on both storages. Adding a lower-stored X into a general Y reports success but is wrong (0.115).
7. **HODLR factorizations.** Both need lower storage, HODLR structure and SPD diagonal leaves, and neither recompresses. `hodlr` (A = diag(A₀₀, A₁₁)(I + UVᵀ)) only solves dense right-hand sides, and `solve_mat` on it segfaults; `hodlrsym` (A = WWᵀ, real only) solves at 3.0e-9 and supports W applies, W⁻¹ solves and log-det.
8. **Multiplication.** `gemm(transA, transB, α, A, B, β, C)` computes H×H into a pre-structured C, whose block tree need not match. Verified NN, TN and TT (an internal header says TT is unsupported; it works at 1.4e-9); rejected for HODLR-factorized operands.
9. **Inversion.** `inverse` is an in-place recursive H-inversion, verified at 3.0e-10; it took 4.8 s against 1.0 s for LU at N = 1,500. It refuses symmetric storage.
10. **Cholesky and LDLᵀ.** Real types use LAPACK potrf; complex types get an unconjugated complex-symmetric LLᵀ, and LDLᵀ is unconjugated and unpivoted. Hermitian matrices therefore need general storage and LU.
11. **Eigenvalues.** Only an internal, unexported power method; OpenTURNS implements its own over `gemv`.
12. **Log-determinant.** `logdet` returns log det L = ½ log det A after LLᵀ or `hodlrsym` (verified ratio 0.500000) and silently returns 0 after LDLᵀ. It raises an error after LU or `hodlr`, and for complex types writes a complex value into a `double*`.
13. **Recompression.** `truncate` with `set_low_rank_epsilon`; the same ε drives recompression inside gemm, axpy and the factorizations.
14. **Other.** Frobenius `norm` (verified), JSON tree dumps, sub-block views, leaf walkers (C++ only) and progress callbacks.

## HPC suitability

The open-source library is single-core by design: no operation uses threads, MPI or a GPU, and it actively pins BLAS to one thread.

- **Threading.** None in released or master code: no OpenMP pragmas, `std::thread`, TBB or StarPU. `DisableThreadingInBlock` sets MKL, OpenMP and OpenBLAS to one thread, each only if detected at configure time, during assembly, factorization, inversion, products, solves, scaling, truncation, axpy and extraction (`src/disable_threading.cpp`). An unmerged work-in-progress branch adds `HMAT_BLAS_NUM_THREADS` so that workers can use BLAS threads.
- **Distributed memory and GPU.** No MPI code. Unmerged branches `gs/cuda` and `gs/cuda2` (Jan–Feb 2026) offload only low-rank truncation to one GPU through cuBLAS and cuSOLVER.
- **The parallel version is proprietary.** Airbus’s HMAT provides a StarPU engine behind the same C API (`hmat_init_starpu_interface`), which OpenTURNS can call when built against it. A 2021 thesis calls it “the Airbus proprietary C/C++ library for H-Matrices”; no public terms for obtaining it were found.
- **External parallel use.** Inria’s H-Chameleon runs hmat-oss kernels inside Chameleon tiles on StarPU, in shared memory only.

Largest documented runs:

- **hmat-oss alone:** no published timings.
- **This deep dive’s test,** the `c-cholesky` example (exponential covariance on a 3D cylinder point cloud, ACA+ at ε = 1e-4, η = 2), ran on a 2-vCPU VM. Its runtime BLAS and thread count were not recorded. At N = 100,000 it stored 137.9M entries (≈1.1 GB) against 5.0G for the dense lower half. A phase-timed rerun with OpenBLAS on one thread took 2.8 s to assemble, 24 s to factor by LLᵀ and 0.2 s to solve. At N = 20,000 these three phases took 3.2 s in total. The example’s 301 s wall time is mostly its O(N²) dense residual check. The relative residual ‖Ax − b‖/‖b‖ is 6.9e-6; the example prints ‖Ax − b‖²/‖b‖ = 9.9e-10 under that label.
- **H-Chameleon** with hmat-oss kernels: N up to 200,000 on up to 35 threads, with numbers only in figures ([Carratalá-Sáez thesis, 2021](https://www.tdx.cat/bitstream/handle/10803/671577/2021_Tesis_Carratala%20Saez_Rocio.pdf)).
- **Proprietary HMAT** (not hmat-oss): at most “a few million unknowns” in the 2010s and “today limited to about 10⁷ unknowns” (translated from an undated offer), with distributed memory an open goal ([Airbus/Inria PhD offer](https://calcul.math.cnrs.fr/job_offers/job_3a2ba479a7338bdb8b93ecac9888316a_attachment.pdf)). Coupled FEM/BEM runs reached several million unknowns on StarPU ([Sylvand, WONAPDE 2019](https://www.ci2ma.udec.cl/wonapde2019/abstract/Sylvand.pdf)).

Practical limits: one core, 32-bit `int` indices and sizes in the API, a global settings singleton rather than per-matrix settings, `-ffast-math` by default, and SVD compression forced on blocks with max(m, n) < 100 by default.

## Caveats and surprises

- **Silent wrong results**, all verified on 1.11.1 and master:
  - LU accepted on symmetric storage gives a solve error of 0.12.
  - Adding a lower-stored X into a general Y gives an error of 0.115.
  - The adjoint matvec on complex-symmetric storage gives an error of 0.11.
  - `logdet` after LDLᵀ returns 0.
  - `trsm` ignores α.
  - `get_block` on an upper sub-block of lower storage returns 0 and writes nothing.
- **RRQR compression fails at tight tolerances.** At N = 1,500 and ε = 1e-6, LAPACK `geqrf` returns info = −4, and assembly returns 1 with a matrix whose relative error is 0.72. The code writes past a buffer sized min(m, n) and uses a wrong pivot index (`src/scalar_array.cpp:1164–1236` at 1.11.1; 1197–1269 on master); it works at 1e-2 and 1e-4.
- **Crash.** `solve_mat` after `hodlr` segfaults; other misuse prints a backtrace and returns 1.
- **No Hermitian support.** Symmetric storage mirrors blocks by transposition, and complex LLᵀ and LDLᵀ are not conjugated.
- **Factorization changes what `gemv` means.** After LLᵀ or `hodlrsym` it applies the factor, by design; after LU there is no guard.
- **Misleading header comments.** `hmat.h` calls `logdet` the matrix log-determinant and says the `upper` flag solves with U; both are wrong, as is the internal `src/hmat_cpp_interface.hpp` note that TT gemm is unsupported. `luDecomposition` is commented “Do not use. Doesn’t work”, yet LU works in every test.
- **Version string.** A master build reports 1.11.1 although it is 23 commits ahead.
- **Narrow HODLR factorizations.** `hodlr` and `hodlrsym` accept only symmetric lower-stored HODLR matrices with SPD leaves; a non-symmetric HODLR matrix must use plain H-LU or inversion.

## Verification

The review read and built both tag 1.11.1 (f28023d) and master (45a2fdb), and every “verified” claim above comes from test runs against dense LAPACK and CBLAS references.

- **Code read:** `hmat.h`, `c_wrapping.hpp`, `default_engine.cpp`, `h_matrix.*`, `hodlr.cpp`, `compression.*`, `assembly.cpp`, `admissibility.cpp`, `scalar_array.cpp`, `disable_threading.*`, `fromdouble.cpp`, `CMakeLists.txt`, the examples, and the `gs/cuda2` and `gs/master` branch diffs.
- **Builds:** gcc 13.3 and CMake 3.28, with a minimal LAPACKE built from Reference-LAPACK 3.12.0, linked against the system `libblas.so.3` and `liblapack.so.3`. These were switched from netlib 3.12 to OpenBLAS 0.3.26 between the master and 1.11.1 builds, and the runtime BLAS and thread count were not recorded. The 4 ctest tests passed on both versions.
- **Test programs:** four drivers at N = 1,500 covering assembly, products, all factorizations, solves, arithmetic, inversion, log-det and RRQR. Both versions behaved identically.
- **Timing:** `c-cholesky` at N = 20,000 and 100,000 on a 2-vCPU VM, then a phase-timed rerun with OpenBLAS on one thread.
- **Not done:** threading behaviour with MKL or OpenBLAS (read in code only), serialization, the C++ assembly path, floating-point compression and the CUDA branches.
- **Not accessible:** HAL, riunet and gitlab.inria.fr were behind bot protection, so the 2020 H-Chameleon paper was not read; the 2021 thesis was used instead.

## Sources

- [hmat-oss repository](https://github.com/jeromerobert/hmat-oss) at tag 1.11.1 (f28023d) and master (45a2fdb). Files read: `include/hmat/hmat.h`; `src/c_wrapping.hpp`, `default_engine.cpp`, `hmat_cpp_interface.{hpp,cpp}`, `h_matrix.{hpp,cpp}`, `hodlr.cpp`, `compression.{hpp,cpp}`, `assembly.cpp`, `fromdouble.cpp`, `scalar_array.cpp`, `admissibility.cpp`, `disable_threading.{hpp,cpp}`, `iengine.hpp`; `CMakeLists.txt`; `examples/hodlrvsllt.c`, `examples/c-cholesky.c`
- OpenTURNS sources on GitHub (`HMatrixImplementation.cxx`, `HMatrixFactory.cxx`, `ResourceMap.cxx`, `GaussianProcess.cxx`) and the [OpenTURNS HMatrix documentation](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HMatrix.html)
- [Debian package tracker](https://tracker.debian.org/pkg/hmat-oss) and [Fedora package](https://packages.fedoraproject.org/pkgs/hmat-oss/hmat-oss)
- [Carratalá-Sáez, PhD thesis, 2021](https://www.tdx.cat/bitstream/handle/10803/671577/2021_Tesis_Carratala%20Saez_Rocio.pdf)
- [Sylvand, WONAPDE 2019 abstract](https://www.ci2ma.udec.cl/wonapde2019/abstract/Sylvand.pdf) and [CORE abstract on coupled FEM/BEM solvers](https://core.ac.uk/works/138966487)
- [Airbus/Inria CIFRE PhD offer](https://calcul.math.cnrs.fr/job_offers/job_3a2ba479a7338bdb8b93ecac9888316a_attachment.pdf)
