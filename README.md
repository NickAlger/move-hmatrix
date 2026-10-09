# hmatrix-bridges

Glue code that moves hierarchical matrices between seven open-source H-matrix libraries
(Htool-DDM, H2Lib, hmat-oss, ButterflyPACK, H2Opus, STARS-H and H2Pack), with tests that pin
down what each move costs and loses, and a catalogue of the library quirks that make such
moves fail silently.

It packages the prototypes of an interoperability study ("Moving H-matrices between
libraries"): the three transfer experiments and the import and export checks from the
library surveys, rewritten as adapters behind one neutral interface and tested against
pinned library versions.

## Three ways to move a matrix

| Route | What crosses | Cost, in source matvecs | Accuracy |
|---|---|---|---|
| **Block copy** | cluster trees, permutations, block tree and leaf factors, through a neutral in-memory view (`hmb_hmatrix`) | 1-11 | lossless (1e-16 to 1e-15) |
| **Rebuild from entries** | the entries the destination's compressor asks for (`hmb::EntrySource`) | 60-80 | both libraries' errors add |
| **Rebuild from matvecs** | products with blocks of random vectors (`hmb::BlackBox`) | 1,100-3,300 | can miss the tolerance several-fold |

Measured at N = 4,000-20,000 with a smooth kernel on points of a sphere, tolerance 1e-6, one
thread; RESULTS.md has the numbers and GOTCHAS.md the traps.

## What each library can do here

| Library | Block copy out | Block copy in | Built from entries | Built from matvecs | Matvec out | Also |
|---|---|---|---|---|---|---|
| Htool-DDM | `htool_export` (zero-copy; one-triangle storage mirrored) | `htool_import` (any tree, lossless) | `htool_build` | - | `htool_blackbox` | |
| H2Lib | `h2lib_export` (zero-copy) | `h2lib_import` (copy, borrow, strict) | - | - | `h2lib_blackbox` | H -> H² `h2lib_to_h2`, `h2lib_h2_blackbox` |
| hmat-oss | `hmatoss_export` (its serialization stream) | `hmatoss_import` (lossless) | `hmatoss_build` | - | `hmatoss_blackbox` | `hmatoss_entries` |
| STARS-H | `starsh_export` (zero-copy) | `starsh_import` (any partition) | `starsh_from_entries` (TLR) | - | `starsh_blackbox` | |
| H2Opus, TLR | `h2opus_tlr_export` (zero-copy) | `h2opus_tlr_import` (uniform tiles only) | `h2opus_tlr_from_entries` | - | `h2opus_tlr_blackbox` | |
| H2Opus, H² | - | - | - | `h2opus_from_matvec` (HARA; symmetric only) | `h2opus_blackbox` | |
| ButterflyPACK | - | - | `bpack_from_entries` (HODLR, H) | `bpack_from_matvec` (HODLR, H) | `bpack_blackbox` | `bpack_factor`, `bpack_solve_blackbox` |
| H2Pack | - | file writer `tools/h2pack/h2pack_write.py` (real symmetric H²) | - | - | - | file-based, see `tools/h2pack/` |

Everything is in namespace `hmb` (C APIs for H2Lib and STARS-H as `hmb_h2lib_*`,
`hmb_starsh_*`). So:

- **Any exporter into any importer.** All 48 pairs of the five block-copy libraries (eight
  sources, six destinations) are tested to be lossless by `tests/test_cross_copy.cpp`; H2Opus
  TLR refuses non-tile partitions with an explanation.
- **Any exported view feeds the rebuild routes.** `entries_from_neutral(view)` and
  `blackbox_from_neutral(view)` (`include/hmb/neutral_bridge.hpp`) turn any view into an
  `EntrySource` or a `BlackBox`, so a library without entry access (H2Lib, Htool-DDM) can
  still feed ButterflyPACK, Htool-DDM or hmat-oss from entries.
- **Chains.** `examples/chain_four_libraries.cpp` runs the study's "extreme example": Htool-DDM
  builds the matrix, H2Lib converts it to H², ButterflyPACK rebuilds and factors it, and
  H2Opus builds the inverse as an H² from ButterflyPACK's solves. The block copy into H2Lib
  costs 9 source matvecs and the conversion to H² 93; the two rebuilds cost 3,100 and 20,700
  (RESULTS.md).

## Quick start

```sh
sudo apt-get install build-essential gfortran cmake git pkg-config python3 python3-numpy \
                     libopenblas-dev liblapacke-dev
nice deps/build_deps.sh   # the seven libraries at pinned commits: ~5 min on 2 cores, ~230 MB
make -j2                  # adapters, examples and tests for every configured library
make check                # 27 test programs, ~2 min on 2 cores
```

`deps/build_deps.sh` writes `deps/install/config.mk`, which the Makefile uses when there is no
top-level `config.mk`. To use your own builds instead, copy `config.mk.example` to `config.mk`
and set the variables of the libraries you have: any subset works, and `make info` shows what
is enabled and what will be built. Linux and GCC only; no MPI, CUDA or MAGMA needed.

## Using it

Each library's neutral-facing header (`adapters/<lib>/hmb_<lib>.h*`) includes none of the
library's own headers, so any combination compiles in one file:

```cpp
#include "hmb_bpack.hpp"       // ButterflyPACK: rebuild from matvecs
#include "hmb_h2lib.hpp"       // H2Lib: import, H -> H2
#include "hmb_htool.hpp"       // Htool-DDM: build from entries
#include "hmb_htool_copy.hpp"  // Htool-DDM <-> neutral view

// The matrix to compress, as entries in your own (original) ordering.
hmb::EntrySource K;
K.n_rows = K.n_cols = n;
K.thread_safe = true;
K.get = [&](int M, int N, const int *rows, const int *cols, double *out) { // out is M x N, column-major
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) out[i + size_t(M) * j] = kernel(rows[i], cols[j]);
};

hmb::HtoolHandle A = hmb::htool_build(K, 3, xyz.data());              // Htool-DDM, ACA at 1e-6
std::shared_ptr<const hmb_hmatrix> view = hmb::htool_export(A);       // zero-copy neutral view
hmb::H2libHandle H = hmb::h2lib_import(*view, HMB_H2LIB_STRICT);       // lossless block copy
hmb::H2libH2Handle H2 = hmb::h2lib_to_h2(H, 1e-6);                     // H -> H2 inside H2Lib

hmb::BpackOptions opt;                                                 // rebuild in ButterflyPACK
opt.format = hmb::BpackFormat::HODLR;                                  // from H2 products
opt.tolerance = 1e-7;                                                  // 10x tighter than the target
hmb::BpackStats st;
hmb::BpackHandle B = hmb::bpack_from_matvec(hmb::h2lib_h2_blackbox(H2), 3, xyz.data(), opt, &st);
std::printf("%lld sampling vectors, error vs H2: %.1e\n", st.vectors, st.check_error);

hmb::BlackBox Bx = hmb::bpack_blackbox(B);                             // y = B x, original ordering
```

`xyz` holds the points (3 x n, point-major), which every library uses for its cluster tree.
Handles are `std::shared_ptr`s and every view or BlackBox keeps what it points into alive,
so the order of destruction does not matter. C++ functions throw `std::invalid_argument` or
`std::runtime_error` with the function's name and the reason.

To build your own program against the package, `make` leaves one archive per adapter in
`build/lib/` and `make info` prints what each needs on the link line:

```sh
g++ -std=c++17 -fopenmp -I$HMB/include -I$HMB/adapters/htool -I$HMB/adapters/h2lib \
    -I$HMB/adapters/butterflypack my.cpp \
    $HMB/build/lib/libhmb_htool.a $HMB/build/lib/libhmb_h2lib.a $HMB/build/lib/libhmb_bpack.a \
    $HMB/build/lib/libhmb.a  <H2LIB_LIBS> <BPACK_LIBS> <BLAS_LIBS>
```

The examples in `examples/` show every route end to end, with costs and checks; each header
documents its conventions, ownership, limits and the version it was tested with.

## What is tested

`make check` runs 27 programs with about 1,100 individual checks; each program prints every
measured value next to its threshold and must end with a `HMB_TEST_PASSED` line (a Fortran
STOP inside ButterflyPACK exits with status 0, so the status alone proves nothing).

- Each adapter against its library's own operations: orderings and permutations, 'N' and 'T'
  products, non-symmetric kernels, one-triangle storage, padded leading dimensions, rank-0
  leaves, unusual tree shapes, error paths, and a test for each quirk it works around.
- The neutral core (validation, reference products and entries) against dense matrices.
- Every block-copy exporter into every importer (`test_cross_copy`).
- Six examples that reproduce the study's measurements at small sizes, and at the original
  sizes through `results/run_reference.sh`.
- Both against fresh pinned builds (`deps/build_deps.sh`) and against the study's own builds;
  the adapters were also run under AddressSanitizer.

## Gotchas

GOTCHAS.md lists 40 library quirks, most of which return success with wrong numbers: hmat-oss's
1-based `get_values` (91% error), H2Lib's H -> H² on a non-strict tree (34-57%), Htool-DDM's
transposed product on two trees (142%), stale diagonal ranks in H2Opus's TLR construction,
STARS-H returning another block's factors, ButterflyPACK ending the program with exit status 0,
H2Pack's reader corrupting the heap. Code that handles one is marked `GOTCHA(<id>)`.

## Limits

- Real double precision only, one process (no MPI), CPU only.
- The neutral form has no nested bases: H² matrices move only by rebuilding (from matvecs, or
  H2Lib's own H -> H² after a block copy), or into H2Pack through its file format.
- Libraries studied but not bridged: HLIBpro (closed source; not downloaded), STRUMPACK,
  HMatrices.jl (Julia), HiCMA (its STARS-H is bridged), MUMPS and PaStiX (no compressed
  matrices in or out).
- Linux with GCC. The pinned H2Lib and H2Pack builds use `-march=native`.

## Layout

```
include/hmb/    neutral view (neutral.h), owners (cstore.h, neutral_store.hpp),
                BlackBox, EntrySource, neutral_bridge.hpp
src/            libhmb.a: validation, reference matvec and entries (needs only BLAS)
adapters/<lib>/ hmb_<lib>.h* (neutral-facing), <lib>_native.hpp (library types), sources
tools/h2pack/   H2Pack file writer (Python), reader check, reader fix
examples/       one self-checking program per route, plus the four-library chain
tests/          tests, harness (tests/common/hmb_test.hpp), runner, LSan suppressions
mk/             make fragments; deps/ pinned builds; results/ logs and notes
```

LAYOUT.md has the conventions for adding a library or a route.

## Versions and licences

| Library | Commit | Licence |
|---|---|---|
| Htool-DDM | 0ee4cac | MIT |
| H2Lib (community branch) | 920662c | LGPL-3.0 |
| hmat-oss | 45a2fdb | GPL-2.0-or-later |
| ButterflyPACK | a8cf81b | BSD-3-Clause (LBNL) |
| H2Opus (with Thrust 1.17.2) | cbc7985 | BSD-3-Clause (Thrust: Apache-2.0) |
| STARS-H | c115e75 | BSD-3-Clause |
| H2Pack | 2aa58a6 | MIT |

`deps/versions.txt` has the full commits, dates and build options. A program that links
hmat-oss is subject to its GPL. This package's own licence has not been chosen yet.
