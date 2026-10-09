# Background docs

The studies behind [move-hmatrix](../README.md): what twelve hierarchical-matrix packages can
do, and how a compressed matrix moves from one to another. They were written in October 2026
from each package's source code, documentation and papers, and each names the versions it
read; packages change, so check a detail against the current release before relying on it.
Corrections are welcome as issues or pull requests.

## Start here

- [Overview and comparison](overview.md): the twelve packages side by side (formats, construction, operations, HPC reach) and which to choose for a given need.
- [Moving H-matrices between libraries](interoperability.md): the transfer routes, what each library exposes, measured costs and silent traps; the study this package implements.
- [Survey of further H-matrix packages](survey.md): how eight packages were added to the first four, and the candidates left out.

## Package deep dives

- [STRUMPACK](packages/strumpack.md): LBNL's sparse multifrontal solver, with HSS, BLR, HODLR and butterfly formats for dense matrices.
- [ButterflyPACK](packages/butterflypack.md): LBNL's MPI library for low-rank and butterfly formats, built from entries or matvecs.
- [H2Opus](packages/h2opus.md): KAUST's GPU-oriented H² and tile low-rank library, dormant since 2022.
- [HiCMA and HiCMA-PaRSEC](packages/hicma.md): KAUST's task-based tile low-rank Cholesky at scale, with STARS-H.
- [Htool-DDM](packages/htool-ddm.md): header-only C++ H-matrices with MPI assembly and domain-decomposition solvers.
- [HLIBpro](packages/hlibpro.md): the closed-source C++ library with the broadest H-arithmetic, reviewed from its documentation.
- [H2Lib](packages/h2lib.md): the open-source C library that goes deepest on H² arithmetic, mostly sequential.
- [hmat-oss](packages/hmat-oss.md): Airbus's sequential H-matrix library with a C API and broad arithmetic.
- [HMatrices.jl](packages/hmatrices-jl.md): pure-Julia H-matrices for boundary integral equations.
- [H2Pack](packages/h2pack.md): shared-memory H² and HSS for symmetric kernel matrices, built by proxy points.
- [MUMPS](packages/mumps.md): sparse direct solver with internal BLR compression of its factors.
- [PaStiX](packages/pastix.md): Inria's supernodal sparse solver with internal BLR compression.

The figures are SVGs rendered from the sources in [figures/src](figures/src/README.md).
