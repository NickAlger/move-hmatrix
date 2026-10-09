# H2Pack deep dive

> Written in October 2026 by Nick Alger with Claude Opus 5.5. Reflects H2Pack master at 2aa58a6 (20 Sep 2026); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

H2Pack is a shared-memory C library (OpenMP and SIMD) for symmetric kernel matrices K(X, X) on one point set in up to three dimensions. Its strength is O(N) H² construction by the proxy-point method, with fast H² matvecs and products with blocks of vectors, stored or recomputed on the fly. For repeated Laplace matvecs it beats PVFMM and FMM3D at comparable accuracy.

Direct solvers exist only for its HSS format: ULV LU or Cholesky of A + σI, a one-right-hand-side solve and a log-determinant, plus a conversion from H² to an SPD HSS preconditioner. There is no H² factorization and no H-arithmetic. Non-symmetric and rectangular matrices, construction from entries, dense, sparse or matvec input, and MPI and GPU code are all absent.

Testing exposed three defects: the HSS log-det overflows or underflows to ±inf at realistic sizes, a non-symmetric kernel is accepted silently and gives wrong products, and the Python bindings do not compile against NumPy 2.

## Snapshot

H2Pack is a research-group library that was quiet from 2023 until a construction-speed update in September 2026.

| Item | Value |
| --- | --- |
| Release | No releases or tags; pyh2pack’s `setup.py` says 1.0.0. Reviewed master at 2aa58a6 (20 Sep 2026) |
| Last commit | 20 Sep 2026: “Faster H2 construction: task-parallel partitioning, cheaper per-node ID, parallel teardown”. The commit before it dates from 18 Jul 2023 |
| Activity | 5 commits since 2024, all in September 2026; 356 in total (164 in 2020, the peak year). Branch `sample_pt_opt` holds 2 unmerged commits from June 2023 |
| Licence | MIT |
| Language and bindings | C (GNU C11 with OpenMP); the “C++ interface” is only `extern "C"` guards. Python: pyh2pack, a C extension with built-in kernels only, double precision only. A separate [MATLAB prototype](https://github.com/xinxing02/H2Pack-Matlab), last changed in February 2021 |
| Dependencies | BLAS and LAPACKE (OpenBLAS or MKL), OpenMP and the header-only ASTER SIMD wrapper (AVX, AVX2, AVX-512, SVE, NEON). Plain makefiles; file I/O calls Python helper scripts through `system()` |
| Maintainers | Hua Huang, Xin Xing and Edmond Chow (Georgia Tech); 3 people in the git log |
| Documentation | README, a [GitHub wiki](https://github.com/scalable-matrix/H2Pack/wiki) of about 17 pages (last edited April 2023), well-commented headers and examples. No API reference, test suite or CI |
| Users | The group’s own papers. [H2P-ERI](https://github.com/scalable-matrix/H2P-ERI) from the same group began on H2Pack but has carried its own copy since 2023 (last commit November 2023). Not on PyPI or Spack |

## Construction

Every H2Pack matrix starts from point coordinates and a symmetric kernel callback; there is no entry-oracle, dense, sparse or matvec input.

| Access | Formats | Algorithm | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix, entry oracle by index, dense matrix, matvecs only | ✗ | — | — | Only the MATLAB prototype converts dense to H². SPD HSS construction uses partial matvecs internally but needs an H2Pack H² (see Conversions) |
| Kernel + coordinates | H² | Proxy-point method: one proxy set per level, shifted to box centres; a bottom-up interpolative decomposition (partial-pivoted QR) of K(skeleton candidates, proxies) yields nested bases and skeletons. B and D blocks are kernel evaluations | `H2P_init` → `H2P_calc_enclosing_box` → `H2P_partition_points` → `H2P_generate_proxy_point_ID_file` or `H2P_generate_proxy_point_surface` → `H2P_build` | Numerical proxy points are random candidates in annular layers, sketched and ID-selected; they depend only on kernel, tolerance and box size and can be reused from a file. Proxy surfaces are 2D/3D only. Assumes translation invariance; O(N) once proxies exist |
| Kernel + coordinates | HSS | Same tree with weak admissibility; hybrid compression against the skeletons of near nodes plus proxies, with a Gaussian sketch for wide blocks, level by level | `H2P_run_HSS(h)` before partitioning, then the same calls | 3D ranks are large: 755 for HSS against 134 for H² on the same points |
| Kernel + coordinates, data-driven | H² | Sample points chosen per node from the data through anchor grids (Cai, Huang, Chow, Xi, SISC 2023), then ID | `H2P_select_sample_point` → `H2P_build_with_sample_point` | No box-centre shift, so no translation-invariance assumption; rejects variable-radius Rotne–Prager–Yamakawa (RPY) kernels |
| Kernel + coordinates, periodic | Periodic H² | Proxy points plus an Ewald-summed periodic kernel | `H2P_run_RPY_Ewald`, `H2P_partition_points_periodic`, `H2P_build_periodic` | On-the-fly mode only; in practice RPY-Ewald only |
| File | H², HSS | Reads tree, bases, B and D blocks and coordinates from a JSON + binary draft format | `H2P_read_from_file`, `H2P_store_to_file` | Symmetric only; without the auxiliary file only `H2P_matvec` is safe. Needs Python scripts in the working directory |

- **Kernel callback.** `kernel_eval_fptr(coord0, ld0, n0, coord1, ld1, n1, param, mat, ldm)` “must be symmetric”; an optional bi-matvec callback speeds up the on-the-fly mode.
- **Tree.** Built internally as an adaptive 2^d-ary box tree, bisecting each dimension down to `max_leaf_points` (400 in 3D, 200 in 2D by default). Users cannot supply a tree or block partition.
- **Admissibility.** H² boxes must be separated by at least one box width; HSS accepts any non-overlapping boxes. Both are compile-time constants.
- **Accuracy.** `H2P_init(&h, pt_dim, krnl_dim, stop_type, &param)`, with `stop_type` one of `QR_REL_NRM`, `QR_ABS_NRM` or `QR_RANK`, sets a relative or absolute threshold, or a fixed maximum rank, for the interpolative decomposition.

## Conversions

The one structural conversion turns an H² matrix into an SPD HSS approximation of A + σI, which can then be factored.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| H² → SPD HSS (≈ A + σI) | `H2P_SPDHSS_H2_build(max_rank, reltol, shift, h2, &hss)` | Recompressing | Pushes max\_rank + 10 Gaussian vectors through level-wise partial H² matvecs, then compresses each node with Cholesky scaling, column-pivoted QR and an eigendecomposition. The “scaling-and-compression” scheme guarantees an SPD result at O(rN log N) cost. Needs `H2P_HSS_calc_adm_inadm_pairs` first; refuses HSS input and RPY-Ewald |
| H² or HSS ↔ files | `H2P_store_to_file`, `H2P_read_from_file` | Exact | Refuses RPY, periodic and H2P-ERI objects |
| Anything else | — | ✗ | No H² → dense, dense → H², HSS → H² or H² re-truncation. Switching between stored and on-the-fly modes needs a rebuild or a file round trip (`H2P_read_from_file` takes `BD_JIT`). Precision is a compile-time choice: double by default, single with `-DDTYPE_SIZE=4` |

## Operations

H² matrices can only be applied; HSS matrices can also be factored and solved. No format provides matrix functions, A + B, scaling, A·B, inversion, LDLᵀ, QR, eigendecomposition or SVD.

| Operation | H² (proxy or sample points) | Periodic H² (RPY-Ewald) | HSS |
| --- | --- | --- | --- |
| Matvec | ✓ \[1\] | ✓ \[2\] | ✓ \[1\] |
| Transpose or adjoint matvec | Same as matvec \[3\] | Same as matvec | Same as matvec |
| Block of vectors | ✓ \[4\] | ✓ \[2\] | ✓ \[4\] |
| Direct solve | ✗ \[5\] | ✗ | ✓ one right-hand side \[6\] |
| Package-provided iterative solve | ◐ example code \[7\] | ◐ | ◐ |
| Triangular solve | ✗ | ✗ | ✓ \[8\] |
| A + σI | ◐ parameter only \[9\] | ✗ | ◐ parameter only \[9\] |
| LU | ✗ | ✗ | ✓ \[6\] |
| Cholesky | ✗ | ✗ | ✓ \[6\] |
| Log-determinant | ✗ | ✗ | ◐ over- or underflows \[10\] |
| Recompression | ◐ via SPD HSS only | ✗ | ✗ |

1. **Matvec.** `H2P_matvec(h, x, y)` runs the upward sweep, B stage, downward sweep and dense D stage. Each off-diagonal block is stored once and applied as B and Bᵀ. Stored mode keeps B and D; on-the-fly mode recomputes them from skeleton coordinates. One real vector per call, and the object is not re-entrant.
2. **Periodic.** `H2P_matvec_periodic` and `H2P_matmul_periodic`, on-the-fly mode only.
3. **Transpose.** Matrices are symmetric by construction, so there is no transpose API, and there is no complex arithmetic.
4. **Block of vectors.** `H2P_matmul(h, layout, n_vec, X, ldx, Y, ldy)` processes up to 128 columns at a time with BLAS-3 per block, in both modes. Sixteen vectors took 2.84 s against 15.7 s for 16 matvecs (n = 40,000, 2 threads). The README’s “O(N) matrix-matrix multiplication” means this, not H²×H².
5. **No H² factorization.** The ULV routines refuse non-HSS objects (“H2Pack is not running in HSS mode!”, verified). The only route from H² is conversion to SPD HSS.
6. **HSS ULV.** `H2P_HSS_ULV_LU_factorize(h, shift)` and `H2P_HSS_ULV_Cholesky_factorize(h, shift)` factor A + σI node by node (Householder QR of the basis, then `getrf` or `potrf`), with `*_solve(h, op, b, x)` for one right-hand side. LU handles symmetric indefinite matrices: at σ = −0.5 its solve matched dense to 1.4e-5, while Cholesky correctly reported “not SPD”. At n = 3,000 (Gaussian, tolerance 1e-8) solves matched dense to 1.6e-8 (σ = 1) and 9.6e-8 (σ = 0.1).
7. **Iterative solvers.** The library has no Krylov solver; preconditioned CG ships only as example code (`examples/PCG/pcg.c`). Library preconditioners are the SPD HSS with ULV Cholesky and `AFN_precond_build`/`apply` (Nyström or adaptive factorized Nyström) for K(X, X) + μI.
8. **Factor solves.** `op` = 1 and 2 apply the two ULV factors separately for split preconditioning; 1 then 2 reproduces the full solve. The Cholesky header comment describes the two the other way round.
9. **Shift.** σ is only a parameter: the ULV factors A + σI while the stored matrix and `H2P_matvec` still give A, and the SPD HSS conversion adds σ to its diagonal blocks.
10. **Log-determinant.** After ULV, `h->HSS_logdet` (Python `HSSMat.logdet()`) holds log|det(A + σI)|, accumulated as the log of each node’s pivot product, which overflows or underflows. At n = 3,000 and σ = 1, Cholesky matched dense to 9 digits but LU gave +inf; at σ = 0.1 both gave −inf (dense: −3.12e3), and at n = 10,000 Cholesky gave +inf. LU returns no sign.
11. **Other.** File storage, proxy-point file reuse, `H2P_print_statistic` (ranks, storage, timers), and pyh2pack dense kernel evaluation with separate source and target sets.

## HPC suitability

H2Pack is shared-memory only: every operation uses OpenMP and SIMD, and no MPI or GPU version of H2Pack exists in the repository or its organization.

- **Threading.** OpenMP, with the thread count fixed at `H2P_init`. Built-in kernels are vectorized through ASTER (AVX to AVX-512, SVE, NEON), and user kernels can use the same wrappers. BLAS is pinned to one thread inside the basis-construction loops and reset afterwards, so ULV levels, root included, run with BLAS threading enabled.
- **Distributed memory and GPU.** None; the wiki answers “Can I use GPU / MPI to accelerate H2Pack? For the moment, no.” The organization’s other repositories are unrelated MPI matrix-multiplication codes, and H2P-ERI is a shared-memory application.
- **Runtimes.** None; the H² basis build uses a custom lock-free task queue, and partitioning uses OpenMP tasks.

| Operation | Threading (MPI and GPU: none) |
| --- | --- |
| Partitioning | OpenMP tasks (new in September 2026); the admissible-pair recursion is serial |
| Numerical proxy points | Threaded kernel evaluation and ID within each level; levels in sequence |
| H² basis build | OpenMP over nodes through the task queue, children before parents |
| HSS basis build | OpenMP over each level’s nodes, level by level |
| Stored B and D blocks | OpenMP with a static NUMA first-touch partition |
| Matvec | OpenMP; sweeps level by level; B and D stages with thread-private outputs and a reduction |
| Block of vectors | OpenMP over nodes, BLAS-3 per block |
| ULV factorization and solve | OpenMP over each level’s nodes; levels in sequence; the root in a one-thread OpenMP team |
| SPD HSS conversion | OpenMP per level, with accumulated partial matvecs |

Largest documented runs (papers and wiki; none reproduced here):

| Source | Problem | Accuracy | Hardware | Results |
| --- | --- | --- | --- | --- |
| [Xing, Huang, Chow, JCP 2022](https://faculty.cc.gatech.edu/~echow/pubs/xing-huang-chow-jcp-2021.pdf), Table 4 | Periodic RPY (3×3 tensor kernel), 5.12M particles, n = 15.4M | 1e-4 | 2 × Xeon Gold 6226, 24 cores | Build ≈40 s, matvec ≈8.6 s |
| Same paper, Table 2 | Non-periodic RPY, 1.28M particles, n = 3.84M | 1e-6 | Same | Build 47–49 s, matvec 2.41 s, 21 GB |
| [Huang, Xing, Chow, TOMS 2020](https://faculty.cc.gatech.edu/~echow/pubs/h2pack.pdf) | 3D Laplace, 1.6M points | ≈1e-5 (sphere), ≈1e-11 (ball) | 2 × Xeon Platinum 8160, 48 cores | Sphere: setup 0.44 s, matvec 0.075 s. Ball: setup 4.06 s, matvec 0.63 s, 5.5 GB storage (9.4 GB peak) |
| [Xing, Huang, Chow, SIMAX 2021](https://arxiv.org/pdf/2011.07632) | SPD HSS conversion up to N = 2.56M (Matérn) | Ranks 100 and 200 | 24-core Xeon Gold 6226 | N = 320,000, rank 100: build 21–24 s, apply ≈0.045 s |
| [Wiki](https://github.com/scalable-matrix/H2Pack/wiki) | 1.6M-point Coulomb, on-the-fly | 1e-6 | 24 cores | Build 1.99 s, matvec 0.54 s, 2.7 GB. A separate 800,000-point test gives 55–58% parallel efficiency at 24 cores |

The TOMS paper also reports matvecs 1.5–5× faster than PVFMM and 5–25× faster than FMM3D, with the largest gains on sphere point sets. Its on-the-fly matvec reached 89% parallel efficiency on a 64-core Knights Landing but 33% (42% stored) on a 48-core Skylake node.

Practical limits:

- **One node only**, with a full-length output buffer per thread in the matvec and 32-bit sizes.
- **Stored mode is limited by B and D storage.** At n = 40,000 (Gaussian, 1e-6) the bases took 52 MB, B 1.31 GB and D 261 MB, so large N needs on-the-fly mode.
- **Proxy-point selection can dominate**: 34 s at n = 40,000 on 2 cores; store and reuse the points.
- **3D HSS is expensive.** At n = 10,000 the build took 21 s and the ULV 23 s (rank 755); n = 40,000 ran out of 7 GB. The top ULV levels have few nodes, so they rely on threaded BLAS.

## Caveats and surprises

- **Non-symmetric kernels are accepted silently.** With K(x, y) = e^(−0.5|x − y|²)(1 + 0.5x₁) at n = 3,000, `H2P_build` gave no warning and the matvec error was 3.1e-2, against 7e-16 for the symmetric kernel. Rectangular or two-point-set matrices are unsupported; pyh2pack’s separate-set kernel calls are dense evaluations.
- **The README’s limits hold.** Kernels must be at most 3D, symmetric, translation-invariant and non-oscillatory, on one point set. Both proxy-point schemes assume translation invariance; the sample-point build does not.
- **The log-determinant overflows or underflows** at realistic sizes (see Operations, note 10).
- **Python bindings fail to compile against NumPy 2** (`PyArray_DOUBLE` undeclared; fine with NumPy 1.26 headers) and still use distutils. They offer only built-in kernels: Gaussian, exponential, Matérn, quadratic, Coulomb, 2D Laplace, Stokes and RPY.
- **A declared function does not exist.** `H2P_generate_proxy_point_ID` is in the public header but defined only in `extra/src-obsolete`, so calls fail at link time.
- **No application code.** `extra/` holds test drivers and obsolete sources only; the block-Lanczos K^(1/2)z of the JCP paper is not in the repository.
- **SPD HSS conversion is not black-box.** It needs an H2Pack H² object and a prior call to `H2P_HSS_calc_adm_inadm_pairs`.
- **Smaller surprises.** HSS uses the same 2^d-ary tree as H², not a binary tree. Reusing a proxy-point file enlarges the root box and changed the tree for the same points (4 against 5 levels). Fixed-rank mode leaves the proxy-selection tolerance at 0, with unclear effect.
- **Fragile packaging.** File I/O calls Python scripts that must sit in the working directory and be made executable, and internal headers are installed. The MATLAB prototype adds dense ↔ H² conversions and a Cholesky HSS log-det that sums log-pivots, so it does not overflow.

## Verification

The review read H2Pack at 2aa58a6 with its ASTER submodule, the wiki, H2Pack-Matlab and H2P-ERI, and built `libH2Pack` in 15 s (GCC 13.3, OpenBLAS 0.3.26, LAPACKE 3.12). Two unpublished C drivers ran on a 2-core, 7 GB container (OMP\_NUM\_THREADS=2, OPENBLAS\_NUM\_THREADS=1) with uniform random points in a 3D cube of side n^(1/3), the Gaussian e^(−0.5r²) and relative tolerances. References were dense at n = 3,000 and 1,000 directly computed rows at n ≥ 10,000. The first n = 3,000 driver used 100-point leaves; the other runs halved the proxy candidates (H2P\_GEN\_PP\_X0\_SIZE=1000, H2P\_GEN\_PP\_Y0\_LSIZE=2000).

| Size and setting | Results |
| --- | --- |
| n = 3,000, Gaussian, tolerance 1e-8 | H² matvec and matmul, stored and on the fly: ≈1e-15 (tree too shallow to compress). HSS matvec 9e-9; HSS ULV solves against dense (σ = 1e-2) 7e-7. SPD HSS at rank 100: matvec 4.6e-2, solve 0.195, log-det within 2%. ULV on an H² object rejected; non-symmetric kernel 3.1e-2 error |
| n = 40,000, tolerance 1e-6 | Proxy points 34 s; stored build 6.6 s; matvec 1.17 s with error 8e-10 on 1,000 directly computed rows; ranks up to 105 (average 79). On the fly: build 2.4 s, matvec 1.79 s. Sixteen-vector matmul 2.84 s against 15.7 s for 16 matvecs. HSS build ran out of memory |
| n = 10,000, tolerance 1e-6, σ = 1 | HSS build 21.5 s; HSS matvec 7.2e-6; ULV Cholesky 23 s; solve 0.74 s with residual 2.8e-15 against the HSS matrix itself; log-det +inf |

Compile checks: pyh2pack fails with 7 errors against NumPy 2.5.3 headers and passes against 1.26.4; all sources compile in single precision. GitHub issue pages and the GitHub API were not accessible.

## Sources

- [H2Pack repository](https://github.com/scalable-matrix/H2Pack) at 2aa58a6. Files read: `src/H2Pack_typedef.h`, `H2Pack_config.h`, `H2Pack_partition.{h,c}`, `H2Pack_gen_proxy_point.{h,c}`, `H2Pack_build.{h,c}`, `H2Pack_build_with_sample_point.{h,c}`, `H2Pack_build_periodic.{h,c}`, `H2Pack_matvec.{h,c}`, `H2Pack_matmul.{h,c}`, `H2Pack_HSS_ULV.{h,c}`, `H2Pack_SPDHSS_H2.{h,c}`, `H2Pack_file_IO.{h,c}`, `H2Pack_ID_compress.c`, `H2Pack_utils.c`, `DAG_task_queue.c`, `AFN_precond.{h,c}`; `examples/`, `extra/`, `pyh2pack/`
- [H2Pack wiki](https://github.com/scalable-matrix/H2Pack/wiki) and the [scalable-matrix organization](https://github.com/orgs/scalable-matrix/repositories)
- Related code: [H2P-ERI](https://github.com/scalable-matrix/H2P-ERI), [H2Pack-Matlab](https://github.com/xinxing02/H2Pack-Matlab), [ASTER](https://github.com/huanghua1994/ASTER)
- [Huang, Xing, Chow, H2Pack, ACM TOMS 47(1), 2020](https://faculty.cc.gatech.edu/~echow/pubs/h2pack.pdf)
- [Xing, Huang, Chow, SPD HSS preconditioner, SIMAX 42(2), 2021](https://arxiv.org/pdf/2011.07632)
- [Xing, Huang, Chow, Brownian dynamics with H², JCP 448, 2022](https://faculty.cc.gatech.edu/~echow/pubs/xing-huang-chow-jcp-2021.pdf)
- [Zhao, Xu, Huang, Chow, Xi, AFN preconditioner](https://arxiv.org/abs/2304.05460)
- [Hua Huang, PhD thesis, 2024](https://repository.gatech.edu/entities/publication/35d42a43-f1c4-4523-827c-541817293907)
