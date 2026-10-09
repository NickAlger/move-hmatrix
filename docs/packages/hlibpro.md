# HLIBpro: capabilities review

> Written in October 2026 by Nick Alger. Reflects the HLIBpro 3.2 documentation and published papers (closed source; no code was downloaded); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

HLIBpro 3.2 has broad H-matrix arithmetic, but in practice it is a single-node CPU library: MPI is claimed but undocumented, and GPUs serve only dense SVD and QR. It adds, multiplies, inverts and factorizes (LU, LDU, Cholesky, LDL) H-matrices of any block structure. It builds from matrix entries or sparse matrices, never from matvecs, and has no QR, eigensolver or matrix functions.

Extrapolating published runs (HLIBpro’s 2D covariance data and 3D figures from Kriemann's research code HLR), a 1 TB node holds about 7–22 million unknowns with H-LU, and 15–46 million for build and matvec only. The largest published HLIBpro run is 2 million unknowns with H-Cholesky on a 128 GB workstation.

HLIBpro is developed by Ronald Kriemann at the Max Planck Institute for Mathematics in the Sciences, Leipzig. It is closed source: binaries are free for academic use, and commercial licenses are by request ([download page](https://www.hlibpro.com/download/)). It is a C++ library, templated over float, double and complex since 3.0, with a C API; no Python or Fortran interface is documented.

No release date is given for 3.2. Version 3.1.2 shipped on 2024-06-06, and the 3.2 install notes target macOS 15, Boost 1.87 and TBB 2022.1, so 3.2 is from 2025 or later. This review uses only public documentation, release notes and papers; no HLIBpro code was downloaded, because of the license.

## Formats

HLIBpro has one general H-matrix type; HODLR and MBLR are block layouts of it, chosen through the admissibility condition or the cluster tree. A separate H² type exists but supports little beyond matvec.

| Format | Classes | Notes |
| --- | --- | --- |
| H-matrix | `THMatrix` over a `TBlockMatrix` tree; leaves `TRkMatrix` (A·Bᴴ) and `TDenseMatrix` | Strong geometric admissibility (`TStrongGeomAdmCond`, η = 2 by default) is the standard choice |
| Weak admissibility | Same classes, `TWeakGeomAdmCond` | Vertex, edge or face variants; all dimensions since 3.2 |
| HODLR | Same classes, `TOffDiagAdmCond` | Every off-diagonal block admissible |
| Oscillatory kernels | Same classes, `THiLoFreqGeomAdmCond`, `TParabolicGeomAdmCond` | Caps wavelengths per cluster to bound ranks at high frequency |
| Sparse-matrix structure | Same classes, `TStdAlgAdmCond`, `TWeakAlgAdmCond` | Graph-based admissibility on the sparsity pattern |
| MBLR | Same classes over a `TMBLRBSPCTBuilder` cluster tree (2.7+) | Clustering option only; no separate BLR or TLR data structure |
| H² | `TH2Matrix`; leaves `TUniformMatrix` (V·S·Wᴴ); nested `TClusterBasis` | Bases from `THClusterBasisBuilder` (from an H-matrix) or `TDenseClusterBasisBuilder` |
| Sparse | `TSparseMatrix` (CRS); optional sparse-leaf H-matrix | Sparse leaves support matvec only |
| Dense, global low-rank | `TDenseMatrix`, `TRkMatrix` | — |
| Lazy operators | `TMatrixSum`, `TMatrixProduct`, `TPermMatrix`, factor inverses (`TLUInvMatrix`, `TLLInvMatrix`, `TLDLInvMatrix`, `TLDUInvMatrix`) | Apply to vectors only |

Clustering options: geometric binary space partitioning (geometrically balanced, cardinality balanced, PCA), algebraic graph partitioning (BFS, multilevel, METIS, Scotch, Chaco, Mongoose), nested dissection, and space-filling curves (via CGAL, 3.1+).

Not present: HSS, butterfly/HODBF, uniform-H with non-nested bases, and compressed floating-point storage. Kriemann's compressed-storage work (AFLP, BFL, DFL, adaptive precision) lives in his separate open-source research library HLR ([arXiv 2405.03456](https://arxiv.org/html/2405.03456v1)).

## Construction

HLIBpro builds H-matrices from matrix entries or from a sparse matrix; nothing builds one from matvecs.

| Target | Sparse matrix | Entry oracle (matrix never dense) | Dense array | Matvecs only |
| --- | --- | --- | --- | --- |
| H (strong, weak, HODLR, MBLR) | Yes: `TSparseMatBuilder`, exact | Yes: `TDenseMatBuilder` with ACA, ACA+ or HCA | Yes, C API only: `matrix_build_dense` | No |
| H with sparse leaves | Yes: sparse mode | — | — | No |
| H² | Only via an H-matrix, then conversion | Only via an H-matrix, then conversion¹ | Cluster bases only: `TDenseClusterBasisBuilder` | No |

¹ `TH2MatBuilder` is an abstract base for builders over given cluster bases: you implement the leaf construction, and no ready-made subclass is documented.

- **From a sparse matrix.** `TSparseMatBuilder` copies the entries into an H-matrix over an algebraic cluster tree (C API: `matrix_build_sparse`, `matrix_import_crs`, `matrix_import_ccs`). Admissible blocks normally come out empty, so the examples build with exact accuracy. In sparse mode the leaves stay sparse, and the result then supports only matvec.
- **From an entry oracle.** `TDenseMatBuilder` takes a `TCoeffFn` callback that fills a sub-block for given row and column index lists and declares the matrix unsymmetric, symmetric or Hermitian; `TPermCoeffFn` handles the permutation. Each admissible block then gets a low-rank method:
  - ACA, ACA+ (the recommended default) and HCA touch about k(m+n) entries of an m×n block; HCA also needs a kernel-generator function.
  - ACA-Full, SVD, RRQR and randomized SVD evaluate each admissible block in full, one block at a time.
- **Built-in entry oracles.** BEM bilinear forms with SIMD kernels (Laplace, Helmholtz, Maxwell EFIE/MFIE, acoustic, exponential), Matérn covariance, and a plain C callback (`matrix_build_coeff`).
- **Build behavior.** ACA falls back to dense evaluation of a block when it does not converge. Recompression is on by default, coarsening is optional, and blocks with rank of at least half their smaller dimension are stored dense.
- **Accuracy control.** `TTruncAcc` sets a fixed rank, a relative or absolute precision, and a rank cap; Frobenius-norm truncation arrived in 3.1. `TBlockTruncAcc` sets accuracy per block region.
- **From a dense array.** Only the C API documents it: `matrix_build_dense` approximates the far-field blocks, and `matrix_import_dense` imports without approximation.
- **From matvecs.** No peeling or randomized sketching from a black-box operator. Releases 2.7 and 2.8 add low-rank approximation of sums of matrices and operators, used inside the arithmetic; these produce one low-rank block and are not a documented construction route.
- **Assembly.** `assemble_block` or a custom builder joins separately built matrices into one block matrix; `TIdMatBuilder` and `TZeroMatBuilder` build identity and empty matrices.

## Conversion

Conversions into the H-matrix format are well covered; conversions out of it go to dense, a single low-rank block or parts of the matrix, and H² has no documented way back.

| From → to | Provided | How |
| --- | --- | --- |
| Sparse → H | Yes | `TSparseMatBuilder`, exact |
| Dense → H | Yes (C API) | `matrix_build_dense` |
| H → H² | Yes, per the feature list | `THClusterBasisBuilder::build(M, acc)` returns nested row and column bases; the features page and release notes list H→H² conversion (parallel since 2.0), but the 3.2 reference does not show the conversion call |
| H² → H | No | Not documented |
| H → H at a new accuracy | Yes | `copy(acc)`, `copy_to(acc)`, `truncate`, by SVD, RRQR or randomized SVD |
| H → coarser H | Yes | `TCoarsen` or `copy(acc, coarsen)` merges neighboring blocks into one low-rank or dense block when that saves memory |
| H → another admissibility (e.g. strong → HODLR) | Indirect | No converter; `add(1, A, 1, C, acc)` into an empty C from `TZeroMatBuilder` over the target block tree |
| H → dense | Yes (C API) | `matrix_to_dense` |
| H or dense → one low-rank block | Yes (C API) | `matrix_to_rank`, `matrix_approx_rank` |
| H → parts of itself | Yes (C API) | `copy_nearfield`, `restrict_nearfield`, `nearfield_to_crs` (exports to CRS), `copy_blockdiag` |
| Several matrices → one block H | Yes | `assemble_block` |
| Between value types | Yes | C API value-type conversions; `restrict_re`, `restrict_im` (3.1) |

Functions marked C API are documented only there, with a value-type prefix (e.g. `hpro_d_matrix_to_dense`); the C++ reference lists no matching free functions, so their C++ spellings are unconfirmed.

## Operations

All arithmetic works on H-matrices of any block layout. H² supports only matvec, iterative solves and spectral-norm estimation, and sparse-leaf matrices support only matvec.

| Operation | H-matrix (any admissibility) | H² (`TH2Matrix`) |
| --- | --- | --- |
| Matvec | Yes: `mul_vec`, y = α·op(A)x + βy with op ∈ {A, Aᵀ, Aᴴ}; the conjugate alone is not supported. `apply_add` accepts a dense block of vectors | Yes: `mul_vec` |
| Inverse apply | Yes: `factorise_inv`, `lu_inv`, `ll_inv`, `ldl_inv`, `ldu_inv` return an inverse operator (A⁻¹, A⁻ᵀ, A⁻ᴴ); an explicit H-inverse also works. Triangular solves on a dense block of right-hand sides: `solve_lower_left`, `solve_upper_left`, `solve_diag_left` Krylov solvers (`TCG`, `TMINRES`, `TBiCGStab`, `TGMRES`, `TTFQMR`, `TAutoSolver`) also solve iteratively, preconditioned by a factor inverse, `TJacobi`, `TGaussSeidel` or `TSOR`. | Iterative only: the Krylov solvers accept any `TLinearOperator` |
| √A, A^(−1/2), other matrix functions | No | No |
| Log-determinant | No routine; the [log-likelihood tutorial](https://www.hlibpro.com/doc/3.2/TUTORLoglikelyhood.html) sums the logarithms of the Cholesky factor's diagonal in user code | No |
| A + B | Yes: `add(α, A, β, C, acc)` with truncation. Row and column cluster trees must match; block structures may differ. `add(list, acc)` sums many matrices | Not documented (inherited from the block-matrix base) |
| A + λI | Yes: `add_identity`, exact; needs dense diagonal leaf blocks | No |
| A·B | Yes: `multiply(α, op(A), op(B), β, C, acc)`, op ∈ {N, T, H}, into an existing C or returning a new matrix; also `multiply_accu`, `multiply_diag` (A·D·B) and H × dense. Cluster trees must be compatible | No |
| Inversion | Yes: `invert` (in place, task-graph parallel), `gauss_elim`, `invert_ll` and `invert_ur` (triangular parts), `inverse_diag` (diagonal of A⁻¹ only), WAZ factorization of the inverse (2.4+) | No |
| Triangular factorizations | Yes: LU and LDU (unsymmetric); Cholesky and LDLᴴ (symmetric or Hermitian, lower triangle stored); all in place | No |
| QR | No (dense QR exists only in the BLAS layer, also on GPU, and inside truncation) | No |
| Eigendecomposition | No (dense eigen wrappers exist only in the BLAS layer) | No |
| Truncation, coarsening | Yes | Listed in the class reference (`copy(acc)`, `truncate`); behavior not documented |
| Norms | Yes: `norm_2`, `norm_F`, `diff_norm_2`, `inv_approx_2` (‖I − BA‖₂), `condest` | Yes: `TSpectralNorm` (any linear operator) |

Factorization details:

- **No pivoting.** The docs say pivoting is "only possible to a very small degree for H-matrices" and is not implemented.
- **Block-wise mode (the default)** inverts the dense diagonal leaves, so it breaks down only on a singular diagonal block; `pseudo_inversion` uses a pseudo-inverse there instead.
- **Point-wise mode** factorizes the diagonal leaves too and breaks down more easily. Cholesky is always point-wise, and the docs call LDL the more stable choice.
- **Storage.** Symmetric or Hermitian matrices store only the lower triangle, and LU or LDU throw on them. Factors overwrite A and are used through operator classes; the symmetric or Hermitian property must be passed explicitly, because factorization drops it.
- **Scheduling.** Task-graph (DAG) scheduling is on by default; accumulator and lazy arithmetic (`use_accu`, `lazy_eval`) reduce the number of truncations.

## HPC suitability

HLIBpro is a shared-memory library: TBB task parallelism is mature, MPI support is claimed but undocumented, and GPUs are used only for dense SVD and QR. Measured sizes and timings are under Practical size and rank limits.

- **Shared memory (its strength).** TBB runs construction, matvec, and H-LU and inversion scheduled as task graphs, following Kriemann's many-core H-LU paper ([MiS preprint 5/2014](https://www.mis.mpg.de/publications/preprints/2014/prepr2014-5.html)). BEM kernels are vectorized for SSE3 through AVX-512, MIC, NEON and VSX.
- **Distributed memory (unclear).** The docs introduction names MPI as the communication layer. The [features page](https://www.hlibpro.com/features/), dated 2019, says matvec and addition run distributed but multiplication does not, and that inversion and factorizations are "implemented for shared systems only".
- **Distributed memory, history.** Release 1.0 claimed MPI construction and factorization, and 2.0 an H-LU usable in distributed environments. Version 3.2 has distribution classes (`TBlockDistrBC`, `TNDDistrBC`, `TSFCDistrBC`, `TProcSet`, `TGhostMatrix`), but no MPI tutorial, no MPI API documentation and no MPI dependency in the install notes. Plan for a single node, and ask the author before counting on MPI.
- **GPU (minimal).** Version 3.2 adds "basic support for CUDA and HIP for SVD and QR (in case of large matrices)", which are dense kernels; the BLAS reference also lists SYCL versions. There is no GPU matvec, construction or H-arithmetic.
- **Deployment.** Academics get prebuilt binaries for Debian/Ubuntu, Fedora, CentOS Stream and macOS, depending on Boost, TBB, LAPACK, METIS, zlib, FFTW, HDF5, GSL and CGAL. You cannot rebuild against your cluster's own BLAS, MPI or CUDA stack, and the site does not say whether a commercial license includes source.
- **Precision.** Float, double and complex value types; the solver classes support mixed precision since 3.0.

## Practical size and rank limits

One node's memory, not any index or API limit, sets HLIBpro's ceiling. Published HLIBpro runs reach 2 million unknowns with H-Cholesky on a 128 GB workstation; extrapolation gives about 7–22 million unknowns on a 1 TB node with H-LU, and 15–46 million for build and matvec only.

### Published runs

HLR is Kriemann's open-source research library; its runs stand in where HLIBpro has none published.

| Run | Unknowns | Accuracy | Memory per unknown | Time | Hardware |
| --- | --- | --- | --- | --- | --- |
| [HLIBCov](https://arxiv.org/abs/1709.08625) (HLIBpro 2.6), 2D Matérn covariance | 2.0M | ε = 1e-5 | matrix 7.4 kB, Cholesky about 10 kB | build 229 s, Cholesky 473 s | 2× Xeon E5-2680 v2 (20 cores), 128 GB |
| [Litvinenko et al.](https://arxiv.org/abs/1709.04419) (HLIBpro), 2D Matérn covariance | 2.0M | ε = 1e-7 to 1e-8 | matrix 11–15 kB, Cholesky 14–18 kB | build 114–138 s, Cholesky 601–1,057 s | same workstation |
| [HLIBpro 3.2 tutorial](https://www.hlibpro.com/doc/3.2/TUTORApprox.html), 3D convection-diffusion (sparse) | 133k | error 2e-3 | 1.31 GB total for the H-LU run | H-LU 23–41 s | Intel i5-4670, 1 core |
| [HLIBpro 3.2 tutorial](https://www.hlibpro.com/doc/3.2/TUTORPrecond.html), Helmholtz BEM (complex) | 125k | ε = 1e-4 | not reported | ACA+ build 76 s, H-LU 248 s | Xeon E7-8857, 12 cores |
| [HLIBpro 3.2 tutorial](https://www.hlibpro.com/doc/3.2/TUTORApprox.html), Laplace BEM, 32,768 triangles | about 16k | error 1.5e-3 | 385 MB total for the H-LU run | H-LU 22–33 s | Intel i5-4670, 1 core |
| [Börm, Christophersen, Kriemann](https://arxiv.org/pdf/1911.07531) (HLR, HLIBpro 2.7.2), H-LU task graph only | 8.4M | — | — | graph of 4.2×10⁷ tasks built in 61–102 s on one core | 2× Xeon Gold 6148, 192 GB |
| [Kriemann 2024](https://arxiv.org/html/2405.03456v1) (HLR), matvec only | 2.1M | ε = 1e-6 | not reported | close to the memory-bandwidth limit | EPYC 9554 (64 cores), 384 GB |
| [Kriemann 2023](https://arxiv.org/abs/2308.10960) (HLR), 3D Laplace BEM | 1.05M | ε = 1e-3 to 1e-8 | matrix 15–40 kB, LU 17–46 kB | not reported | 2× EPYC 9554 (128 cores), 768 GB |
| Kriemann 2023 (HLR), Matérn covariance, weak admissibility | 1.05M | ε = 1e-3 to 1e-8 | matrix 8–28 kB, LU 11–38 kB | not reported | same |

### How memory and time grow with n

![Two panels: HLIBpro memory per unknown (matrix and Cholesky factor, with a log2(n/64) model line) and H-Cholesky time per unknown, from 2,000 to 2,000,000 unknowns](../figures/hlibpro-memory-time.svg)

*Litvinenko, Sun, Genton, Keyes, arXiv 1709.04419, Table 4 · 11 sizes from 2,000 to 2,000,000 unknowns*

<details>
<summary>Data behind the figure</summary>

| Unknowns n | Matrix (MiB) | Cholesky factor (MiB) | H-Cholesky (s) | Matrix per unknown (kB) | Factor per unknown (kB) | H-Cholesky per unknown (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 5.2 | 5.2 | 0.07 | 2.7 | 2.7 | 35 |
| 4,000 | 12.7 | 13 | 0.17 | 3.3 | 3.4 | 42 |
| 8,000 | 29.8 | 32 | 0.36 | 3.9 | 4.2 | 45 |
| 16,000 | 72 | 75 | 1.11 | 4.7 | 4.9 | 69 |
| 32,000 | 193 | 210 | 3.04 | 6.3 | 6.9 | 95 |
| 64,000 | 447 | 486 | 11 | 7.3 | 8.0 | 172 |
| 128,000 | 1,160 | 1,310 | 36.7 | 9.5 | 10.7 | 287 |
| 256,000 | 2,550 | 2,960 | 64 | 10.4 | 12.1 | 250 |
| 512,000 | 4,740 | 5,800 | 128 | 9.7 | 11.9 | 250 |
| 1,000,000 | 11,260 | 13,910 | 361 | 11.8 | 14.6 | 361 |
| 2,000,000 | 23,650 | 29,610 | 1001 | 12.4 | 15.5 | 500 |

</details>

Memory per unknown tracks the log₂(n/64) model within about 10% from 128k unknowns up, so the size limits below extrapolate it. H-Cholesky time per unknown kept rising, which is why the time estimates use n^1.2 to n^1.5. Each size used its own fitted covariance parameters, which explains the dips.

### Size limits from memory

These ceilings assume memory per unknown grows like log₂(n/64) from the measured points, which tracks the HLIBpro data within about 10% from 128k unknowns up. They also assume 75% of RAM is usable. Each cell is millions of unknowns: matvec only / with H-LU (H-Cholesky for the covariance row), keeping the original matrix.

| Problem (real, double precision) | 256 GB | 1 TB | 4 TB |
| --- | --- | --- | --- |
| 2D covariance, ε = 1e-7 | 13 / 6.2 | 46 / 22 | 170 / 79 |
| 3D BEM, ε = 1e-3 | 10 / 5.1 | 36 / 18 | 130 / 65 |
| 3D BEM, ε = 1e-6 (interpolated) | 5.4 / 2.7 | 19 / 9.5 | 70 / 34 |
| 3D BEM, ε = 1e-8 | 4.2 / 2.1 | 15 / 7.3 | 53 / 26 |

- Factoring in place, without keeping the original matrix, allows about 1.75× the H-LU figure.
- Complex-valued kernels roughly halve every entry; oscillatory kernels lower them further, because ranks grow with frequency.
- The 3D rows rest on HLR measurements at a single size (1.05M), so their growth with n is assumed, not measured.

### Rank limits

- **Per block.** A rank above half the block's smaller dimension saves nothing, and HLIBpro stores such blocks dense (`to_dense_ratio` = 0.5). ACA gives up at a quarter of the block size (`aca_max_ratio` = 0.25); its dense fallback then evaluates the rest of the block, cheap for small blocks but O(mn) for a large one. `TTruncAcc::set_max_rank` sets a hard cap.
- **Cost.** Memory grows linearly with the rank k; multiplication, H-LU and inversion grow like k², with k³ work per truncation. Doubling the rank roughly quadruples factorization time.
- **Strong admissibility (the default), smooth kernels.** Block ranks stay in the tens: in the HLIBCov paper, fixed ranks of 20–50 reached ‖C − C̃‖₂ of 1e-7 to 1e-13 (rank 10: about 1e-3). Memory runs out before rank does.
- **Weak admissibility (HODLR).** Rank is the binding limit. Storage is about 16·k·n·log₂(n/64) bytes in real double precision: 0.22, 2.8 and 33 GB per unit of rank at 10⁶, 10⁷ and 10⁸ unknowns. Dense 64×64 diagonal blocks add 512 bytes per unknown to each matrix; the table below includes them.

Largest average HODLR rank that fits, matvec only / with H-LU (75% of RAM, real double precision, factor assumed the size of the matrix):

| Unknowns | 256 GB | 1 TB | 4 TB |
| --- | --- | --- | --- |
| 10⁶ | 860 / 430 | 3,400 / 1,700 | 13,000 / 6,700 |
| 10⁷ | 68 / 33 | 270 / 130 | 1,100 / 540 |
| 10⁸ | 4 / 1 | 21 / 10 | 90 / 44 |

Weak-admissibility ranks also grow with block size for 2D and 3D geometry, roughly with the unknowns near the shared boundary: about √m for 2D domains and surfaces, m^(2/3) for 3D volumes. So HODLR and BLR layouts hit this limit first; in Kriemann's matvec study, BLR's largest problem was half the H-matrix's on the same 384 GB node. At 10⁶ unknowns the memory ceiling is in the thousands, so the k² arithmetic cost is what stops you.

### Time

- **Matvec** reads the whole matrix once, so it takes about matrix size ÷ memory bandwidth: 2–4 s for a 750 GB matrix at 200–390 GB/s. HLR's matvec runs close to that limit; HLIBpro's own efficiency is not published.
- **H-Cholesky, 2D covariance.** Published times (Litvinenko et al., Table 4: 1,001 s at 2M) grew like n^1.2 to n^1.5 between 128k and 2M unknowns. At the H-LU ceilings above, that gives about 1–1.5 h at 6M, 5–10 h at 22M and 23–70 h at 79M on the 2013 20-core workstation; a current 64–128-core node should be several times faster.
- **H-LU, complex Helmholtz.** The same growth applied to the tutorial run (248 s at 125k unknowns, 12 cores, wavenumber 16 held fixed) gives about 0.8–1.6 h at 10⁶ and 13–49 h at 10⁷. At 10⁷, matrix plus factor need about 1.1 TB (3D BEM storage interpolated to ε = 1e-4, doubled for complex values), so a 1.5–2 TB node.
- **Task graph.** At 8.4M unknowns the H-LU graph has 4.2×10⁷ tasks and 1.5–8.7×10⁸ dependencies, an estimated 4–12 GB at 64–128 bytes per task and 8 bytes per dependency. Building it took 61–102 s on one core, with parallel speedup capped near 3–4×.

### Hard limits and an accuracy trap

- **Index width does not bind.** Neither HLIBpro's index type (`idx_t`) nor its LAPACK integer width is documented; the C API passes sizes as `size_t`. Even 32-bit indices (2.1×10⁹) would sit more than ten times above the memory ceiling of a 4 TB node.
- **Loose tolerances can break factorization.** In Litvinenko et al., H-Cholesky of the 2M-unknown covariance matrix crashed at ε = 1e-4, and of the 128k one at ε = 6.4e-3. Their SPD runs needed ε ≤ 1e-5, which puts the factor at 8–18 kB per unknown; HLIBpro's docs also call LDL more stable than its always point-wise Cholesky.

## Open points

- **H→H² conversion call.** The documented route is cluster bases from `THClusterBasisBuilder`, then conversion, but the 3.2 reference does not show the conversion call.
- **MPI status.** The docs introduction, the features page and the 1.0 and 2.0 release notes disagree on what runs distributed.
- **Truncated reference pages.** The Algebra reference page cut off partway through multiplication, so some details, such as the WAZ function's name, come only from tutorials and release notes.
- **Index width.** Neither `idx_t` nor the LAPACK integer width is documented.
- **3.2 release date.** Not stated anywhere.
- **Performance on current nodes.** No HLIBpro timings are published for current 64–128-core nodes. HLR results stand in, on the assumption that HLIBpro's storage and arithmetic match its author's research code.

## Sources

- **HLIBpro site:** [Home](https://www.hlibpro.com/) · [Features](https://www.hlibpro.com/features/) · [Download](https://www.hlibpro.com/download/) · [News](https://www.hlibpro.com/news/)
- **3.2 manual:** [Index](https://www.hlibpro.com/doc/3.2/index.html) · [Changelog](https://www.hlibpro.com/doc/3.2/changelog.html) · [Install](https://www.hlibpro.com/doc/3.2/install.html) · [Parameters](https://www.hlibpro.com/doc/3.2/parameters.html) · [Initialization](https://www.hlibpro.com/doc/3.2/TUTORInit.html) · [BLAS interface](https://www.hlibpro.com/doc/3.2/TUTORBLAS.html)
- **3.2 tutorials:** [Construction](https://www.hlibpro.com/doc/3.2/TUTORMatConstr.html) · [Basic algebra](https://www.hlibpro.com/doc/3.2/TUTORAlgebraBasic.html) · [Factorization](https://www.hlibpro.com/doc/3.2/TUTORMatrixFac.html) · [Block cluster trees](https://www.hlibpro.com/doc/3.2/TUTORClusterBCT.html) · [Approximation](https://www.hlibpro.com/doc/3.2/TUTORApprox.html) · [Preconditioning](https://www.hlibpro.com/doc/3.2/TUTORPrecond.html) · [Many right-hand sides](https://www.hlibpro.com/doc/3.2/TUTORManyRHS.html) · [Block assembly](https://www.hlibpro.com/doc/3.2/TUTORBlocked.html) · [Sparse systems](https://www.hlibpro.com/doc/3.2/TUTORSparseAlg.html) · [Spectrum](https://www.hlibpro.com/doc/3.2/TUTORSpectrum.html) · [Log-likelihood](https://www.hlibpro.com/doc/3.2/TUTORLoglikelyhood.html) · [C example](https://www.hlibpro.com/doc/3.2/TUTORCIntEq.html)
- **3.2 reference:** [Class list](https://www.hlibpro.com/doc/3.2/annotated.html) · [Topics](https://www.hlibpro.com/doc/3.2/topics.html) · [Algebra](https://www.hlibpro.com/doc/3.2/group__Algebra__Module.html) · [Matrix](https://www.hlibpro.com/doc/3.2/group__Matrix__Module.html) · [Cluster](https://www.hlibpro.com/doc/3.2/group__Cluster__Module.html) · [BLAS](https://www.hlibpro.com/doc/3.2/group__BLAS__Module.html) · [C bindings](https://www.hlibpro.com/doc/3.2/group__CBinding__Module.html) · [C bindings, real double](https://www.hlibpro.com/doc/3.2/group__CBinding__Module__RFP64.html) · [C bindings, common types](https://www.hlibpro.com/doc/3.2/group__CBinding__Module__Common.html) · [TH2Matrix](https://www.hlibpro.com/doc/3.2/classHpro_1_1TH2Matrix.html) · [TH2MatBuilder](https://www.hlibpro.com/doc/3.2/classHpro_1_1TH2MatBuilder.html) · [THClusterBasisBuilder](https://www.hlibpro.com/doc/3.2/classHpro_1_1THClusterBasisBuilder.html) · [TIndexSet](https://www.hlibpro.com/doc/3.2/classHpro_1_1TIndexSet.html)
- **Older manuals:** [2.0](https://hlibpro.com/doc/2.0/index.html) · [1.2](https://hlibpro.com/doc/1.2/index.html)
- **Papers using HLIBpro:** [HLIBCov, arXiv 1709.08625](https://arxiv.org/abs/1709.08625) and its [code](https://github.com/litvinen/HLIBCov) · [Litvinenko, Sun, Genton, Keyes, arXiv 1709.04419](https://arxiv.org/abs/1709.04419)
- **Kriemann and HLR:** [H-LU Factorization on Many-Core Systems, MiS preprint 5/2014](https://www.mis.mpg.de/publications/preprints/2014/prepr2014-5.html) ([hgpu listing](https://hgpu.org/?p=11635)) · [Hierarchical Lowrank Arithmetic with Binary Compression, arXiv 2308.10960](https://arxiv.org/abs/2308.10960) · [Performance of H-Matrix-Vector Multiplication with Floating Point Compression, arXiv 2405.03456](https://arxiv.org/html/2405.03456v1) · [Semi-Automatic Task Graph Construction for H-Matrix Arithmetic, arXiv 1911.07531](https://arxiv.org/pdf/1911.07531)
- **Related work:** [Exploiting nested task-parallelism in the H-LU factorization, arXiv 1906.00874](https://arxiv.org/abs/1906.00874)
