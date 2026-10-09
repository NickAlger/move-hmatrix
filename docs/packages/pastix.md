# PaStiX deep dive

> Written in October 2026 by Nick Alger. Reflects the PaStiX 6.4.0 handbook and its source listings (not built); later versions may differ. [All background docs](../README.md) · [Overview of the twelve packages](../overview.md)

PaStiX 6.4.0 (July 2024, still the latest release) is Inria’s supernodal sparse direct solver. It offers LLᴴ/LLᵀ, LDLᵀ/LDLᴴ and LU with static pivoting (on the pattern of A + Aᵀ; `spmCheckAndCorrect` pads unsymmetric patterns with zeros) in four arithmetics, with optional double-to-single mixed precision. Its one compressed format is an internal, flat block low-rank representation of the off-diagonal blocks of wide supernodes, built during factorization.

Compression is controlled only by parameters: three timings (Minimal-Memory, Just-In-Time and an ILU(k)-guided hybrid), five kernels (SVD and four QR variants), and a tolerance that is absolute by default. The compressed factors support direct solves with multiple and distributed right-hand sides, transposed solves, triangular and diagonal subtask solves, and a dense Schur complement on one MPI process. Package-provided GMRES, CG, BiCGSTAB or simple refinement use them as a preconditioner.

There is no user-visible low-rank object, matvec, arithmetic, conversion, determinant, selected inverse or hierarchical format. Compressed factorization is threaded under every scheduler but MPI-distributed only with the internal ones; GPUs process only full-rank blocks, analysis is centralized, and every BLR result found is single-node.

## Snapshot

PaStiX released one to three versions a year until 2024 and none in the 27 months since; it remains widely packaged.

| Item | Value |
| --- | --- |
| Release | 6.4.0, about July 2024 (`api.h` is dated 5 Jul 2024; the conda-forge feedstock was created on 29 Jul 2024). As of October 2026, Spack, conda-forge, Arch, MacPorts, FreeBSD and PETSc main are all at 6.4.0 or older |
| Cadence | 6.2.1 (January 2022), 6.2.2, 6.3.0, 6.3.1 and 6.3.2 (by December 2023), 6.4.0 (July 2024). The 6.0 series (2018–19) was shared-memory only. The last commit date could not be checked, because gitlab.inria.fr was refused |
| Licence | LGPL-3.0, per Arch, conda-forge, MacPorts, FreeBSD and the LaBRI software page; the LICENSE file itself was not read |
| Languages and bindings | C library with CUDA kernels; Fortran 90 modules, a Python package and a Julia package (PaStiX.jl); the companion SpM library has the same bindings. Low-rank kernel headers are not installed |
| Dependencies | CBLAS and LAPACKE with a sequential BLAS; hwloc (highly recommended); Scotch (on by default, also needed for BLR clustering) and/or METIS; optional PT-Scotch, StarPU 1.3 or later, PaRSEC (a specific branch), CUDA (only with a runtime), MPI, EZTrace |
| Maintainers | Inria Bordeaux, LaBRI and Bordeaux INP: Faverge and Ramet (PIs), Pruvost (build); contributors include Pichon, Korkmaz, Delarue, Lisito and Nicolas |
| Documentation | A [Doxygen handbook](https://solverstack.gitlabpages.inria.fr/pastix/) with user API groups and browsable source; every IPARM and DPARM entry is documented with its default in `api.h`. Compression enums have one-line comments in `api.h`; the ILU(k) role of `PastixCompressWhenDuring` must be read from code and papers |
| Users | PETSc `MATSOLVERPASTIX` (distributed input; options for compression timing and tolerance); JOREK (fusion MHD) tested BLR as a preconditioner; packaged by Spack, conda-forge, Arch, MacPorts, FreeBSD and MSYS2 |

## Construction

The only input is an assembled sparse matrix. Compression happens during numerical factorization, with groups and flags fixed at analysis time.

| Access | Formats | Algorithm | Entry point | Notes |
| --- | --- | --- | --- | --- |
| Sparse matrix | Full-rank supernodal factors; BLR-compressed supernodal factors; ILU(k) factors | Nested-dissection ordering (Scotch or METIS on the gathered graph, PT-Scotch distributed, or a user permutation), symbolic factorization, block reordering and k-way clustering of separators, then compression during factorization | An SpM matrix (CSC, CSR or IJV; centralized, replicated or distributed), then `pastix_task_analyze` and `pastix_task_numfact`, or the `pastix_subtask_*` steps | Compression is off by default and is set before analysis by `IPARM_COMPRESS_*`, `DPARM_COMPRESS_*` and the `IPARM_SPLITTING_*` clustering options. No geometry is used |
| Entry oracle, kernel + coordinates, matvecs only, dense matrix | ✗ | — | — | A dense matrix can only be passed as a full sparse pattern. The only geometric helper is a nested-dissection ordering for regular-grid Laplacians |
| ILU(k) pattern | — | Level-of-fill symbolic factorization | `IPARM_INCOMPLETE` = 1, `IPARM_LEVEL_OF_FILL` = k | Combination with BLR not verified |

- **Storage.** Each block holds a rank, a maximum rank and factors U and V; rank −1 marks a dense block.
- **Which blocks.** Only supernodes at least 128 columns wide are candidates, and within them off-diagonal blocks at least 20 rows tall. L and U are compressed separately, and diagonal blocks stay dense.
- **Kernels** (`IPARM_COMPRESS_METHOD`, default PQRCP): SVD; PQRCP, a partial QR with column pivoting built on LAPACK GEQP3; RQRCP, randomized QRCP; TQRCP, truncated randomized QRCP (Duersch–Gu); RQRRT, randomized QR with rotation. Recompression after low-rank additions uses the matching kernel (PQRCP for RQRRT).
- **Tolerance.** `DPARM_COMPRESS_TOLERANCE` defaults to 0.01 and is absolute unless `IPARM_COMPRESS_RELTOL` = 1 makes it relative to each block’s Frobenius norm. The example program scales A by 1/‖A‖\_F first.
- **Rank caps.** A block stays dense above ρ·mn/(m + n) for Minimal-Memory or ρ·min(m, n)/4 otherwise, with ρ = `DPARM_COMPRESS_MIN_RATIO` (1.0).

| `IPARM_COMPRESS_WHEN` | Strategy | What happens |
| --- | --- | --- |
| `PastixCompressNever` | Off (default) | — |
| `PastixCompressWhenBegin` | Minimal-Memory | Eligible blocks are compressed as soon as A is scattered into the factor structure, so fill-in starts at rank 0; updates are low-rank products added with recompression |
| `PastixCompressWhenEnd` | Just-In-Time | Blocks stay dense while receiving updates and are compressed after the last one, just before the panel triangular solve, which then acts on V only |
| `PastixCompressWhenDuring` | ILU(k)-guided hybrid ([Korkmaz et al., HiPC 2021](https://doi.org/10.1109/HiPC53243.2021.00024)) | Blocks with ILU fill level above k (`IPARM_COMPRESS_ILUK`, 0 by default here) are compressed at the start, the rest just in time unless `IPARM_COMPRESS_PRESELECT` = 0. An explicit k applies the same split under every timing |

## Conversions

There is no public conversion between compressed formats, because only one internal format exists; the public extractions are the dense Schur complement and the factor diagonal.

| From → To | API | Exact or recompressing | Notes |
| --- | --- | --- | --- |
| Dense block → low-rank | Internal `core_*ge2lr_*` kernels | Recompressing at τ; dense above the rank cap | Used only inside factorization; no installed header |
| Low-rank + low-rank → low-rank | Internal `core_*rradd_*` kernels | Recompressing (orthogonalization, then RRQR or SVD) | Used in Minimal-Memory updates and MPI fan-in accumulation |
| Low-rank → dense | Internal uncompress routines | Exact | Not exposed |
| Compressed Schur blocks → dense S | Public `pastixGetSchur` | Exact expansion of the stored blocks | Column-major dense output |
| Full-rank factors → BLR after factorization | Internal `coeftabCompress` | Recompressing | No public trigger |
| Double → single-precision factors | `IPARM_MIXED` = 1 | Lossy cast with an overflow check | Factors stored and computed in single; refinement in double. Right-hand-side conversions are provided |
| To H, H², HODLR | — | ✗ | No hierarchical format exists; hierarchical compression appears only in 2016–17 talks and as future work in a 2022 thesis |

## Operations

Factor and solve calls are the same with or without compression; ILU(k) and mixed precision with BLR are unverified. No variant provides a matvec with the factors, matrix functions, A + B, A + σI, scaling of factors, A·B, inversion, QR, eigendecomposition, SVD, determinant, selected inverse, inertia, null space or user-callable recompression. SpM’s `spmMatVec` multiplies by the sparse A only.

| Operation | Full-rank | BLR, Minimal-Memory | BLR, Just-In-Time or hybrid |
| --- | --- | --- | --- |
| Direct solve | ✓ \[1\] | ✓ | ✓ |
| Package-provided iterative solve | ✓ \[2\] | ✓ | ✓ |
| Triangular and diagonal subtask solves | ✓ \[3\] | ✓ | ✓ |
| LU | ✓ \[4\] | ✓ | ✓ |
| Cholesky (LLᴴ; complex-symmetric LLᵀ) | ✓ \[4\] | ✓ | ✓ |
| LDLᵀ and LDLᴴ | ✓ \[4\] | ✓ | ✓ |
| Schur complement | ◐ \[5\] | ◐ | ◐ |
| ILU(k) factorization | ✓ \[6\] | ? | ? |
| Mixed-precision factorization | ✓ \[6\] | ? | ? |
| Factor diagonal | ◐ \[7\] | ◐ | ◐ |

✓ provided; ◐ partial (see note); ? not verified; ✗ absent.

1. **Direct solve.** `pastix_task_solve(data, m, nrhs, b, ldb)` takes multiple and distributed right-hand sides. Transposed solves (`IPARM_TRANSPOSE_SOLVE`) cover A and Aᵀ for LU and complex-symmetric LLᵀ, and A and Aᴴ for LLᴴ. Compressed factors are used directly, each low-rank block applied as two GEMMs through an rk × nrhs temporary.
2. **Iterative solvers (package-provided).** `pastix_task_refine` and `pastix_task_solve_and_refine` run GMRES (default, restart 25), CG, BiCGSTAB or simple refinement, preconditioned by the solve with the computed factors and multiplying by the original A. They stop at a relative residual of 1e-12 in double by default, handle one right-hand side at a time, and run unpreconditioned when no factorization exists.
3. **Subtask solves.** `pastix_subtask_trsm(data, side, uplo, trans, diag, B)` and `pastix_subtask_diag(data, B)` act on a right-hand side permuted with `pastix_subtask_applyorder`; the right-side case is not implemented.
4. **Factorizations.** `IPARM_FACTORIZATION` selects POTRF (LLᴴ), PXTRF (complex-symmetric LLᵀ), SYTRF (LDLᵀ), HETRF (LDLᴴ) or GETRF (LU, the default; unsymmetric patterns are padded to A + Aᵀ). Pivoting is static only: tiny pivots are replaced at a threshold scaled by ‖A‖\_F and counted. The handbook lists all five as compressible; the low-rank code paths were checked for LU and LDLᵀ.
5. **Schur complement.** `pastixSetSchurUnknownList` orders the chosen unknowns last and `pastixGetSchur` returns S as a dense column-major matrix. It returns −1 with more than one MPI process. Schur blocks are not excluded from compression, so S carries the compression error; with Minimal-Memory its own blocks are also stored compressed.
6. **ILU(k) and mixed precision.** `IPARM_INCOMPLETE` with a fill level, and `IPARM_MIXED` (single-precision factors for double input). Neither combination with BLR was verified.
7. **Factor diagonal.** `pastixGetDiag` copies the diagonal of the factored diagonal blocks in permuted order, although its documentation says it returns A(i, i). There is no determinant routine.

## HPC suitability

Compressed factorization and solve are threaded under every scheduler but MPI-distributed only with PaStiX’s internal schedulers, and GPUs never touch compressed blocks.

- **Threading.** An internal thread pool with hwloc binding runs the Sequential, Static or Dynamic (work-stealing, default) schedulers; BLAS must be sequential. Compression kernels run inside per-block tasks, and Minimal-Memory compression runs in parallel over supernodes.
- **Distributed memory.** MPI between processes, threads within. Analysis is centralized: the graph is gathered and the ordering is computed and broadcast from rank 0. Factorization uses proportional mapping with fan-in messages, and compressed supernodes are packed, sent and added with recompression.
- **GPU.** CUDA only, through StarPU or PaRSEC, for GEMM and TRSM updates. CUDA is removed from the allowed devices whenever a supernode is compressed (`codelet_blok_dgemmsp.c`, lines 384–388, re-checked); diagonal factorization and solves stay on the CPU.
- **Runtimes.** StarPU runs factorization and solve tasks, with MPI support and low-rank packing in its data interface. PaRSEC runs the factorization only: solves fall back to the static scheduler, and LDLᵀ and LDLᴴ silently switch to the Dynamic scheduler.

| Operation | Threads | MPI | GPU |
| --- | --- | --- | --- |
| Ordering, symbolic factorization, clustering, mapping | Scotch can be threaded; the rest is sequential | Centralized; PT-Scotch optional | ✗ |
| Full-rank factorization | ✓ all schedulers | ✓ internal, StarPU-MPI, PaRSEC | ✓ CUDA updates through StarPU or PaRSEC |
| BLR factorization | ✓ all schedulers | ✓ internal schedulers; PaRSEC refuses; StarPU full-rank only per handbook | ✗ forced to CPU |
| Full-rank solve | ✓ (PaRSEC falls back to static) | ✓; PaRSEC with MPI gathers all factors on rank 0 | ✗ |
| BLR solve | ✓ internal and StarPU | ✓ internal schedulers | ✗ |
| Refinement | ✓ | ✓ reductions, one right-hand side at a time | ✗ |
| Schur complement | ✓ | ✗ error with more than one process | Full-rank blocks only |

Published results, all single-node for BLR as far as could be accessed (full texts on HAL were blocked, so matrix sizes are unknown):

- **[Pichon et al., J. Comput. Sci. 2018](https://doi.org/10.1016/j.jocs.2018.06.007)** (and IPDPSW 2017): on one 24-thread node with 128 GB, real-life matrices needed up to 4.4× less memory with Minimal-Memory and solved up to 3.3× faster with Just-In-Time.
- **Korkmaz et al., HiPC 2021:** ILU(k) preselection, now the hybrid strategy, ran up to 8.5× faster than the original BLR version with under 1.49× the memory.
- **[Nies and Hoelzl, 2019](https://arxiv.org/abs/1907.13442):** as a JOREK preconditioner on one node, factor memory fell from 65.1 GB uncompressed to 47.6, 40.7, 32.0 and 17.4 GB at ε = 1e-16, 1e-12, 1e-8 and 1e-4. With PaStiX 6.0.2, GMRES failed at 1e-4 and in most 1e-8 runs; the authors say a 6.0.2 bug may partly explain this.

Practical limits: analysis is centralized and replicated; compressed factorization has no GPU or PaRSEC-with-MPI path; the Schur complement is shared-memory only; refinement takes one right-hand side at a time; and pivoting is static only. Scale A first, since the default tolerance of 0.01 is absolute.

## Caveats and surprises

- **One flat format.** The handbook’s “low-rank compression methods” are kernels and timings of a single BLR format, which the 2018 thesis calls “flat, non-hierarchical”.
- **Strategy names appear only in `api.h` comments.** Minimal-Memory is `PastixCompressWhenBegin`, and Just-In-Time is `PastixCompressWhenEnd`, which compresses each block after its last update, not at the end of factorization. The hybrid `PastixCompressWhenDuring` is not described in the handbook prose at all.
- **A setting that compresses nothing.** With the default `IPARM_COMPRESS_ILUK`, setting `IPARM_COMPRESS_PRESELECT` = 0 under Just-In-Time disables compression entirely.
- **The handbook’s feature table is optimistic.** It marks LDLᵀ and LDLᴴ as supported under PaRSEC, but the code silently runs them with the Dynamic scheduler. Conversely, under MPI it lists StarPU as full-rank only, although StarPU’s data interface packs low-rank blocks.
- **Small inconsistencies.** `pastixGetSchur` returns −1 under MPI rather than an error code. RQRRT has no recompression kernel of its own. The handbook cites the 2018 BLR paper in the wrong journal; it appeared in J. Comput. Sci. 27:255–270.
- **No new compression formats since 2018.** Work from 2022–2025 covers QR-variant kernels, StarPU automatic partitioning, new parallel features, and mixed precision against low rank.

## Verification

Nothing was built or run: the release tarball and GitLab were refused, and there are no PyPI wheels. The findings come from the handbook and about 35 of its Doxygen source listings, which are the library’s own code.

- **Refused hosts.** The release tarball server, gitlab.inria.fr and the handbook host were each refused once from the shell and not retried. No official GitHub mirror exists, and no third-party source copies were used.
- **Read through the web.** The handbook’s main page, feature table and build options; the API, user, solve, refinement, Schur and example groups; and source listings of the scheduler, compression, low-rank, solve, Schur, StarPU, CUDA and refinement code. Only the CUDA exclusion is cited by line; the other listings are named under Sources.
- **Packaging and integrations.** The conda-forge feedstock, the Spack recipe, PETSc’s PaStiX interface, and Arch, FreeBSD and MacPorts package pages.
- **Re-check.** The CUDA exclusion for compressed supernodes was re-read independently in `codelet_blok_dgemmsp.c`.
- **Blocked.** HAL’s bot protection hid the full texts of the 2018 paper, the research reports and both theses. ScienceDirect, theses.fr, IEEE Xplore and gitlab.com were also unreachable.

## Sources

- [PaStiX 6.4.0 handbook](https://solverstack.gitlabpages.inria.fr/pastix/), including its API, user, solve, refinement and Schur groups
- Handbook source listings (`*_source.html`): `api.h`, `pastix_task_sopalin.c`, `cand.c`, `pastix_subtask_order.c`, `pastix_subtask_symbfact.c`, `coeftab.c`, `coeftab_dinit.c`, `cpucblk_dcompress.c`, `cpucblk_dschur.c`, `cpucblk_dmpi_coeftab.c`, `core_dgetrfsp.c`, `core_dsytrfsp.c`, `core_dtrsmsp.c`, `core_dgemmsp.c`, `core_dgelrops.c`, `core_dpqrcp.c`, `core_drqrcp.c`, `core_dtqrcp.c`, `core_drqrrt.c`, `lowrank.c`, `pastix_lowrank.h`, `solve_dtrsmsp.c`, `sequential_dtrsm.c`, `starpu_sparse_matrix.c`, `pastix_starpu_interface.c`, [`codelet_blok_dgemmsp.c`](https://solverstack.gitlabpages.inria.fr/pastix/codelet__blok__dgemmsp_8c_source.html), `sopalin/schur.c`, `diag.c`, `pastix_task_refine.c`, `d_refine_gmres.c`, `bvec_dcompute.c`, `compress.c`, `refinement.c`
- [SpM handbook](https://solverstack.gitlabpages.inria.fr/spm/)
- [Spack recipe](https://raw.githubusercontent.com/spack/spack-packages/develop/repos/spack_repo/builtin/packages/pastix/package.py) and [conda-forge feedstock](https://github.com/conda-forge/pastix-feedstock)
- PETSc [PaStiX.py](https://raw.githubusercontent.com/petsc/petsc/main/config/BuildSystem/config/packages/PaStiX.py) and [pastix.c](https://raw.githubusercontent.com/petsc/petsc/main/src/mat/impls/aij/mpi/pastix/pastix.c)
- Packaging: [Arch](https://archlinux.org/packages/extra/x86_64/pastix/), [FreeBSD](https://www.freshports.org/science/pastix/), [MacPorts](https://ports.macports.org/port/pastix/details/), [LaBRI software page](https://www.labri.fr/en/valorisation/logiciels/pastix)
- [Pichon et al., J. Comput. Sci. 27, 2018](https://doi.org/10.1016/j.jocs.2018.06.007) and the [IPDPSW 2017 abstract](https://doi.org/10.1109/IPDPSW.2017.86)
- [Korkmaz et al., HiPC 2021 abstract](https://doi.org/10.1109/HiPC53243.2021.00024)
- [Nies and Hoelzl, 2019](https://arxiv.org/abs/1907.13442)
- [Ramet publication list](https://github.com/ramet/ramet.github.io), for theses and reports
