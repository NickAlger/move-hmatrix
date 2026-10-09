# Agent A (deps): pinned dependency builds

**Summary.** `deps/build_deps.sh` fetches, builds and installs all seven libraries (plus Thrust)
at their pinned commits and writes `deps/install/config.mk`. A run from scratch takes
4m21s-6m19s on this machine, and a re-run skips every library (27 s). The package builds against
the result without a compiler warning and passes all 23 of its checks (`make B=build-deps
CONFIG=deps/install/config.mk check`). The top-level `config.mk` was not touched. Every compiled
object has the same instructions as the study's builds. Four new build quirks were found and
are handled.

## 1. Files created

- `deps/build_deps.sh`: fetches, verifies, builds and installs the libraries, writes
  `<prefix>/config.mk`, then compiles, links and runs one numerical check per library through it.
- `deps/versions.txt`: the pins the script reads. One line per library: id, name, repository,
  full commit, commit date, licence, build options; submodule lines (`thrust:dependencies/cub`,
  `starsh:cmake_modules/ecrc`, `h2pack:src/ASTER`) are checked.
- `deps/README.md`: usage, options, each build and why, the checks, using your own builds,
  updating a pin, known build problems and their fixes, measured times and disk use.
- `config.mk.example` (top level): every variable, with values for `deps/install` and for builds
  elsewhere, and what each build must be.
- Created by the script (gitignored): `deps/src/<id>`, `deps/build/<id>` with `<id>.log`,
  `smoke/` and `smoke.log`, and `deps/install/<id>/{include,lib,LICENSE*,BUILD_INFO,config.mk.part}`,
  `deps/install/thrust`, `deps/install/config.mk`.
- No patches (`deps/patches/` not needed).

## 2. Public interface

```
deps/build_deps.sh [--prefix DIR] [--only LIB,...] [--jobs N] [--skip-fetch] [--blas "FLAGS"] [--help]
```

- `--prefix DIR`: install directory (default `deps/install`). Sources always go to `deps/src`,
  build trees and logs to `deps/build`.
- `--only`: build and force-rebuild only these ids: `htool h2lib hmat bpack h2opus starsh h2pack`
  (long names accepted). Other installed libraries stay in `config.mk`.
- `--jobs N`: make/cmake parallelism (default 2).
- `--skip-fetch`: no network. The checkouts must already be at the pinned commits and unmodified.
- `--blas`: BLAS/LAPACK/CBLAS link flags. The default is `-lopenblas` if it links, else
  `-llapack -lblas`. LAPACKE is probed separately. H2Pack is skipped without OpenBLAS.
- Environment: `CC`, `CXX`, `FC` (MPI wrappers are refused).
- Exit status 0 only when every requested library is installed and passes its check.
- Idempotency: `<prefix>/<id>/BUILD_INFO` carries a signature that hashes the pinned commits and
  submodules, the build step's code (`declare -f build_<id>`; comments excluded), the compiler
  versions, cmake and the BLAS. A library is skipped when its signature matches and its files
  exist.
- `config.mk` variables: those of LAYOUT.md (toolchain, `<LIB>_CPPFLAGS`, `<LIB>_LIBS`, `PYTHON`)
  plus two helpers, `DEPS_PREFIX` and `DEPS_LAPACKE_LIBS`. The exact file is in section 9.

## 3. Checks

**Script checks.** One program per library, compiled and linked only through `config.mk`, the
way the package links (`$(CXX) $(OPENMP) obj <LIB>_LIBS $(BLAS_LIBS)`). A check passes only when
the program prints `SMOKE_OK <id>` and `ldd` shows no MPI library.

| check | verifies | measured | threshold | result |
|---|---|---|---|---|
| htool | H-matrix (ACA 1e-6), 2,000 sphere points, kernel 1/(0.1+r), matvec vs dense | 2.91e-06 | 1e-4 | PASS |
| h2lib | LAPACK LU of a 300x300 matrix and a solve; reads H2Lib's structs (checks that the `-D` flags match) | 2.11e-15 | 1e-10 | PASS |
| hmat | ACA+ assembly (1e-6) through the C API, gemv in the original ordering; shared library + rpath | 6.43e-07 | 1e-4 | PASS |
| bpack | HODLR from entries (tol 1e-6) with the MPI stub, mult with a 1-based new->old permutation | 7.00e-06 | 1e-4 | PASS |
| h2opus | H2 by Chebyshev interpolation (6^3), hgemv through `index_map` | 1.25e-04 | 1e-3 | PASS |
| starsh | TLR (tol 1e-3) of the generated exponential kernel, relative Frobenius error | 5.44e-04 | 1e-2 | PASS |
| h2pack | H2 with proxy points (QR_REL_NRM 1e-6) of 1/r on 16,000 points; matvec checked on 400 rows | 1.63e-07 | 1e-4 | PASS |

**AddressSanitizer** (`OPT='-O1 -g -fsanitize=address -fno-omit-frame-pointer'`): all seven
programs run without ASan errors. LeakSanitizer reports only two leaks, both inside the libraries
and already suppressed by other agents:
- hmat-oss: `hmat-node-leak`, `adapters/hmatoss/hmatoss_lsan.supp`.
- STARS-H: `starsh-false-far-leak`, `tests/test_starsh_lsan.supp`, which needs
  `ASAN_OPTIONS=fast_unwind_on_malloc=0`.

With those suppression files every program exits 0.

**Package against the fresh builds.** Clean `build-deps`, `make B=build-deps
CONFIG=deps/install/config.mk -j2`: no compiler warnings. The only linker message is ButterflyPACK's
"requires executable stack", the same as with the study's objects. Then `make ... check`:

```
check                                        result  seconds  detail
test_neutral                                 PASS        0.0  test_neutral: 136 checks; n=240
test_bpack_matvec                            PASS       10.3  test_bpack_matvec: 44 checks; n=2000 tol=1e-06
test_bpack_entries                           PASS       13.4  test_bpack_entries: 33 checks; n=2000 tol=1e-06
test_bpack_safety                            PASS        5.5  test_bpack_safety: 23 checks; n=1500
test_bpack_stop                              PASS        0.5  test_bpack_stop: 10 checks
test_h2lib                                   PASS        1.8  test_h2lib: 211 checks; n=2000
test_h2opus_hara                             PASS        4.1  test_h2opus_hara: 29 checks; n=2000 tol=1e-06
test_hmatoss_entries                         PASS        0.7  test_hmatoss_entries: 33 checks; n=1500 eps=1e-06
test_hmatoss_copy                            PASS        1.2  test_hmatoss_copy: 35 checks; n=2000 eps=1e-06
test_htool_build                             PASS        0.3  test_htool_build: 24 checks; n=2000
test_starsh                                  PASS        0.4  test_starsh: 82 checks; n=2000 tile=192 eps=1e-06
test_htool_copy                              PASS        0.5  test_htool_copy: 121 checks; n=2000
test_h2opus_tlr                              PASS        0.3  test_h2opus_tlr: 32 checks; n=2001 bs=256 eps=1e-06
test_h2opus_tlr_native                       PASS        0.1  test_h2opus_tlr_native: 6 checks; n=1000 bs=128
test_h2lib_htool                             PASS        0.9  test_h2lib_htool: 41 checks; n=2000
test_bpack_htool                             PASS       11.9  test_bpack_htool: 18 checks; n=2000 tol=1e-06
test_h2opus_hara_htool                       PASS       43.5  test_h2opus_hara_htool: 7 checks; n=2000 tol=1e-06
copy_htool_h2lib                             PASS        2.5  copy_htool_h2lib: 12 checks; n=4000: block copy 2.0-5.5, H->H2 65 sour
entries_hmatoss_htool                        PASS        7.6  entries_hmatoss_htool: 11 checks; N=4000 transfer 105.3 source matvecs
matvec_htool_bpack                           PASS       46.5  matvec_htool_bpack: 8 checks; n=4000 tol=1e-06
matvec_htool_h2opus                          PASS       12.9  matvec_htool_h2opus: 4 checks; n=4000 tol=1e-06
copy_tlr_starsh_h2opus                       PASS        2.1  copy_tlr_starsh_h2opus: 13 checks; N=4000 tile=300 eps=1e-06
test_h2pack_file                             PASS        1.5  test_h2pack_file: 16 checks; sphere n=2000 eta=2, line n=1024, tol 1e-
passed 23, failed 0
```

ButterflyPACK was rebuilt once more after this run. Only the comment text of its `config.mk`
section changed; the 35 objects are again identical to the study's. Its 6 package checks were
re-run and passed.

**Script behaviour tested:**
- A fresh run builds all seven; a re-run skips all seven.
- `--prefix` elsewhere; `--blas "-llapack -lblas"`: H2Pack is reported "skipped: needs OpenBLAS"
  and exits 1 when forced.
- `--skip-fetch` builds normally; with a modified checkout it fails with "local modifications"
  and the previous install stays; with a missing checkout it fails with "run without
  --skip-fetch".
- Missing compiler and `CC=mpicc` are refused, with the `apt-get` line.
- Everything still builds and runs with `deps/src` and `deps/build` moved away.
- `make CONFIG=config.mk.example info` enables all seven libraries.

## 4. Measured costs

2-core VM (Cooper Lake), GCC 13.3, OpenBLAS 0.3.26 pthreads, `--jobs 2`, `nice`. Runs from
scratch (`rm -rf deps/src deps/build deps/install && nice deps/build_deps.sh --jobs 2`); times
include the fetch. Run 1 had load average about 4 (other agents compiling), run 2 (the final
script) 1 to 4.

| library | run 1 | run 2 | sources MB | build MB | installed MB |
|---|---|---|---|---|---|
| htool | 0m02s | 0m02s | 4 | 0 | 1 |
| h2lib | 0m57s | 0m22s | 5 | 10 | 4 |
| hmat | 1m19s | 0m29s | 2 | 7 | 2 |
| bpack | 2m38s | 1m54s | 24 | 47 | 8 |
| h2opus + thrust | 0m17s | 0m29s | 43 | 3 | 11 |
| starsh | 0m21s | 0m17s | 9 | 3 | 1 |
| h2pack | 0m17s | 0m22s | 15 | 21 | 3 |
| total incl. checks | 6m19s | 4m21s | 101 | 98 | 30 |

A re-run with everything up to date takes 27 s (the checks). Blobless clones plus sparse
checkouts cut ButterflyPACK's sources from about 800 MB to 24 MB (415 MB of example data are left
out) and STARS-H's by 113 MB of meshes.

## 5. Gotchas

New ids, for GOTCHAS.md:

- **bpack-openblas-batch.** ButterflyPACK's CMake adds `-DHAVE_OPENBLAS_BATCH` whenever the BLAS
  library string contains "openblas". The code then calls `cblas_dgemm_batch`, which OpenBLAS
  0.3.26 lacks, so every program linking ButterflyPACK fails with "undefined reference to
  cblas_dgemm_batch" (reproduced by compiling `dMISC_DenseLA.f90` with the define).
  - Handled: `deps/build_deps.sh:390-393` probes the BLAS and appends `-UHAVE_OPENBLAS_BATCH`.
  - Covered by: the bpack check and every bpack test link.
- **starsh-codegen-cwd.** STARS-H's configure step generates 25 kernel sources with `python
  ../misc_scripts/...` relative to cmake's working directory, i.e. it assumes cmake runs in
  `<source>/build`. From anywhere else the generation fails, configure still succeeds, and
  `libstarsh.a` lacks `starsh_ssdata_block_*_kernel_*` (link errors). It also needs a `python`
  command.
  - Handled: `deps/build_deps.sh:482-506` starts cmake in `<source>/src` with a `python` ->
    python3 link on PATH, and `nm` checks two kernels.
  - Covered by: the starsh check, which uses the generated exponential kernel.
- **hmat-blas-threads.** Configured against OpenBLAS, hmat-oss defines `OPENBLAS_DISABLE_THREADS`:
  it sets OpenBLAS's global thread count to 1 inside most API calls (`DISABLE_THREADING_IN_BLOCK`
  in hmat_cpp_interface.cpp and the C wrappers, including `get_block`) and restores it. This
  races when several threads call hmat-oss.
  - Handled: `deps/build_deps.sh:349-368` pre-sets `HAVE_OPENBLAS_SET_NUM_THREADS=OFF
    HAVE_GOTO_GET_NUM_PROCS=OFF`, matching the study (whose build had it off by accident), and
    greps the generated `config.h`, which is identical to the study's.
- **h2opus-thrust-backend.** H2Opus's `h2opus_defs.h` picks `THRUST_DEVICE_SYSTEM` from
  `_OPENMP`, so a file compiled without `-fopenmp` sees other Thrust types than `libh2opus.a`.
  - Handled: documented in the generated `config.mk` (`deps/build_deps.sh:469`) and in
    `config.mk.example`. The package always compiles with `$(OPENMP)`; there is no automatic check.

Existing ids handled by the builds:

- **mpi-stub** (`deps/build_deps.sh:171`, `:388`, `:399`, `:1070`). ButterflyPACK is built with
  `enable_mpi=OFF` (`libmpi_dummy.a`) and STARS-H with `MPI=OFF`. MPI compiler wrappers are
  refused, and a check fails if `ldd` shows an MPI library. OpenMPI and ScaLAPACK are installed
  on this machine and are not picked up.
- **h2pack-blas-threads** (`:596-597`, `:543`). H2Pack needs `openblas_set_num_threads`; it is
  skipped with a message when the BLAS lacks it.
- **scalar-type** (`:332`). `H2LIB_CPPFLAGS` is derived from H2Lib's own effective `CFLAGS`, so it
  carries exactly the library's `-D` flags (real double).

Build problems with no gotcha id (all in deps/README.md):
- H2Opus `make all` runs unlimited `-j`.
- H2Opus's sample make.inc turns MKL on with `MKLROOT`.
- H2Lib's default options.inc has `DEBUG=1 USE_CAIRO=1`.
- H2Pack's make file has a bogus `-I`.
- ButterflyPACK's configure writes into its source tree.
- Debian's OpenBLAS has no LAPACKE.

## 6. Differences from the study's builds

Object-level comparison (`objdump -d`, symbolic labels ignored): the instructions are identical
for H2Lib 60/60, hmat-oss 26/26, ButterflyPACK double 35/35, H2Opus 13/13 (only `__FILE__` path
strings differ), STARS-H 64/64 and H2Pack 20/20. Htool is the same commit, header-only.

| library | study | now | why |
|---|---|---|---|
| ButterflyPACK | CMake at -O1 (shared, 4 precisions), then double sources recompiled at -O2 by hand into `libdbpack_O2.a` with `mpi_dummy.o` inside | CMake at -O2 directly, targets `dbutterflypack` + `mpi_dummy` only; `libdbutterflypack.a` + separate `libmpi_dummy.a` | cleanest equivalent; same machine code |
| hmat-oss | linked a hand-built static LAPACKE subset into `libhmat.so` (96 exported `LAPACKE_*`) against reference BLAS symlinks | links the system `liblapacke.so.3` and `libopenblas.so.0`; `BUILD_EXAMPLES=OFF` | system packages now exist; same objects and `config.h` |
| H2Opus | 13 files compiled by hand | H2Opus's own make with a generated `make.inc`, same flags; static `libh2opus.a` | preferred by the brief |
| Thrust | CUB submodule not initialised | CUB fetched at its recorded revision | the brief asks for it; unused by the CPU backend |
| STARS-H | examples and tests built | `EXAMPLES/TESTING/DOCS=OFF` | unchanged library (64 identical objects) |
| H2Lib | `locallib` BLAS symlinks, `BRIEF_OUTPUT=1` | neither | not needed; full commands in the log |
| BLAS on link lines | `-llapack -lblas` (resolved to OpenBLAS by Debian alternatives) | `-lopenblas` | same library; H2Pack needs the OpenBLAS symbols anyway |
| includes | source and build trees | per-library install dirs under the prefix (STARS-H: whole source `include/`; H2Pack: plus `examples/meta_*.py`) | `deps/src` and `deps/build` can be deleted |

## 7. Limitations

- Linux/GNU toolchain only, tested on Ubuntu 24.04 with GCC 13.3, CMake 3.28 and OpenBLAS 0.3.26.
  `--blas` with MKL is mapped (`BLA_VENDOR=Intel10_64lp_seq` for STARS-H) but untested.
- H2Lib and H2Pack use `-march=native`, as their make files do: the install is tied to this CPU type.
- Libraries are built one at a time, each with `-j N`. A failed rebuild removes the old install of
  that library.
- `--blas` assumes `cblas.h` and `lapacke.h` are on the compiler's search path (use `CPATH`).
- Sources and build trees are shared between prefixes (`deps/src`, `deps/build`).
- The checks are smoke tests of the configuration, not tests of the adapters.

## 8. Requested changes to files I do not own

**LAYOUT.md** (configuration table):
- `dC_BPACK_wrapper.h` does not exist; ButterflyPACK's C header is `dBPACK_wrapper.h`.
- `BPACK_LIBS` also needs `-lgomp -lm`.
- The pinned H2Lib build has a third `-D`.
- The helper variables should be mentioned.

```diff
--- a/LAYOUT.md
+++ b/LAYOUT.md
@@ -116,12 +116,16 @@
 |---|---|
 | `CC`, `CXX`, `FC`, `OPT`, `OPENMP`, `BLAS_LIBS`, `PYTHON` | toolchain |
 | `HTOOL_CPPFLAGS` | `-I` of the directory containing `htool/` (header-only) |
-| `H2LIB_CPPFLAGS`, `H2LIB_LIBS` | `-I.../Library` plus the `-D` flags libh2.a was built with (`-DUSE_BLAS -DUSE_OPENMP`); the library and its link dependencies |
+| `H2LIB_CPPFLAGS`, `H2LIB_LIBS` | `-I.../Library` plus every `-D` flag libh2.a was built with (pinned build: `-DUSE_BLAS -DUSE_OPENMP -DHARITH_RKMATRIX_QUICK_EXIT`); the library and its link dependencies |
 | `HMAT_CPPFLAGS`, `HMAT_LIBS` | hmat-oss include directories (source `include/` and the build directory with `hmat/config.h`); `-L... -lhmat` |
-| `BPACK_CPPFLAGS`, `BPACK_LIBS` | ButterflyPACK double-precision C interface (`dC_BPACK_wrapper.h`); `libdbutterflypack.a`, the MPI stub, `-lgfortran` |
+| `BPACK_CPPFLAGS`, `BPACK_LIBS` | ButterflyPACK double-precision C interface (`dBPACK_wrapper.h`); `libdbutterflypack.a`, the MPI stub `libmpi_dummy.a`, `-lgfortran -lgomp -lm` |
 | `H2OPUS_CPPFLAGS`, `H2OPUS_LIBS` | H2Opus include directories (with the generated `h2opusconf.h`, thrust and cub); the library |
 | `STARSH_CPPFLAGS`, `STARSH_LIBS` | STARS-H include directory; `libstarsh.a` and its BLAS/LAPACKE |
-| `H2PACK_CPPFLAGS`, `H2PACK_LIBS` | H2Pack include directory and defines; `libH2Pack.a` and its BLAS |
+| `H2PACK_CPPFLAGS`, `H2PACK_LIBS` | H2Pack include directory and defines (`-DUSE_OPENBLAS`); the static `libH2Pack.a` and its LAPACKE (OpenBLAS comes from `BLAS_LIBS`) |
+
+`deps/install/config.mk` also defines two helpers, `DEPS_PREFIX` (the install prefix) and
+`DEPS_LAPACKE_LIBS` (`-llapacke` where the BLAS lacks LAPACKE); fragments use only the variables
+of this table.
 
 ## Tests and examples
```

**GOTCHAS.md** (does not exist yet): add the four new ids of section 5:
`bpack-openblas-batch`, `starsh-codegen-cwd`, `hmat-blas-threads`, `h2opus-thrust-backend`.

**Top-level `config.mk`**: no change requested. `make CONFIG=deps/install/config.mk` gives the
same results.

## 9. The config.mk written (deps/install/config.mk)

```make
# hmatrix-bridges configuration for the pinned dependency builds under this directory.
# Written by deps/build_deps.sh on 2026-10-09 06:21 UTC; re-run the script rather than
# editing it. The top-level Makefile reads this file when there is no top-level config.mk; to use
# other builds, copy config.mk.example to config.mk and edit that. Commits, licences and build
# options of every library: deps/versions.txt; per-library build records: <id>/BUILD_INFO.

DEPS_PREFIX := /root/hmatrix-bridges/deps/install

# ---- toolchain ----------------------------------------------------------------------------------
# The compilers that built the libraries below. Keep them for the package: C++ ABI (H2Opus, hmat-oss),
# Fortran runtime (ButterflyPACK, STARS-H) and OpenMP runtime must match.
#   CC:  gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
#   FC:  GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
CC = gcc
CXX = g++
FC = gfortran
OPT = -O2
OPENMP = -fopenmp
# BLAS, LAPACK and CBLAS for every link line; detected by build_deps.sh (option --blas):
#   -lopenblas (OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=64)
BLAS_LIBS = -lopenblas
# LAPACKE (C interface of LAPACK), used by hmat-oss, STARS-H and H2Pack; empty when BLAS_LIBS has it.
DEPS_LAPACKE_LIBS = -llapacke
# Python 3 with numpy 2.5.3 (tools/h2pack)
PYTHON = /usr/bin/python3

# ---- Htool-DDM 0ee4cac (2026-10-06), header-only ---------------------------------------------
# Licence: MIT (https://github.com/htool-ddm/htool)
# Nothing is compiled: include/htool is copied. The package uses only Htool's sequential/OpenMP
# H-matrix and clustering headers, which need BLAS/LAPACK (BLAS_LIBS) but no MPI.
HTOOL_CPPFLAGS = -I$(DEPS_PREFIX)/htool/include

# ---- H2Lib 920662c (2020-02-13, branch community), static ------------------------------------
# Licence: LGPL-3.0 (https://github.com/H2Lib/H2Lib)
# H2Lib's own Makefile, target libh2.a, options.inc OPT=1 USE_BLAS=1 USE_OPENMP=1
# HARITH_RKMATRIX_QUICK_EXIT=1 and system.inc.linux, i.e. compiled with
#   gcc -Wall -O3 -march=native -funroll-loops -DUSE_BLAS -DUSE_OPENMP -fopenmp -DHARITH_RKMATRIX_QUICK_EXIT
# GOTCHA(scalar-type): this libh2.a is real double precision (no USE_COMPLEX, no USE_FLOAT).
# H2LIB_CPPFLAGS repeats every -D of that line: H2Lib's scalar type (field) and its struct layouts
# depend on USE_COMPLEX, USE_FLOAT, USE_SIMD, USE_BLAS and USE_OPENMP, so code compiled with other -D
# flags than libh2.a misreads its structs without any error. HARITH_RKMATRIX_QUICK_EXIT only
# matters inside harith.c; it is repeated so that the list is exactly the library's.
# libh2.a is compiled without -fPIC (link it into executables, not into shared libraries) and
# with -march=native (it runs only on CPUs with this machine's instruction set).
H2LIB_CPPFLAGS = -I$(DEPS_PREFIX)/h2lib/include -DUSE_BLAS -DUSE_OPENMP -DHARITH_RKMATRIX_QUICK_EXIT
# H2Lib links its own programs with -llapack -lblas -lgfortran -lm (BLAS here: BLAS_LIBS).
H2LIB_LIBS = $(DEPS_PREFIX)/h2lib/lib/libh2.a -lgfortran -lm

# ---- hmat-oss 45a2fdb (2026-09-25), shared library -------------------------------------------
# Licence: GPL-2.0-or-later (https://github.com/jeromerobert/hmat-oss)
# CMake: CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG; hmat-oss adds -fopenmp -ffast-math),
# BUILD_SHARED_LIBS=ON, BUILD_EXAMPLES=OFF, HMAT_GIT_VERSION=OFF, HMAT_DISABLE_OPENMP=ON (default:
# hmat-oss sets OpenMP threads to 1 inside its calls), BLAS_LIBRARIES=-lopenblas,
# HAVE_OPENBLAS_SET_NUM_THREADS=OFF and HAVE_GOTO_GET_NUM_PROCS=OFF (no switching of OpenBLAS's
# global thread count, as in the study's build). include/hmat holds hmat.h and the generated
# config.h/config-oss.h. hmat-oss is GPL-2.0-or-later: programs linking it fall under the GPL.
HMAT_CPPFLAGS = -I$(DEPS_PREFIX)/hmat/include
HMAT_LIBS = -L$(DEPS_PREFIX)/hmat/lib -Wl,-rpath,$(DEPS_PREFIX)/hmat/lib -lhmat

# ---- ButterflyPACK a8cf81b (2026-09-17), double precision, sequential, static ----------------
# Licence: BSD-3-Clause (LBNL variant, license.txt) (https://github.com/liuyangzhuan/ButterflyPACK)
# CMake with enable_mpi=OFF (sequential: ButterflyPACK's MPI stub MPI_DUMMY), enable_openmp=ON,
# BUILD_SHARED_LIBS=OFF, CMAKE_BUILD_TYPE=None and CMAKE_{C,CXX,Fortran}_FLAGS=-O2 (the study
# recompiled its -O1 CMake objects at -O2). Only the targets dbutterflypack (double precision)
# and mpi_dummy are compiled; libdbutterflypack.a archives the dbutterflypack objects. Fortran:
#   gfortran -DHAVE_FINAL -DHAVE_OPENBLAS_BATCH -DHAVE_TASKLOOP -DHAVE_OPENMP -fopenmp
#   -fno-range-check -fallow-argument-mismatch -cpp -ffree-line-length-none
#   -ffixed-line-length-none -fimplicit-none -lpthread -DNDEBUG -O2 -UHAVE_OPENBLAS_BATCH -fPIC
#   -DDAT=1
# -UHAVE_OPENBLAS_BATCH: ButterflyPACK enables cblas_dgemm_batch when the BLAS name contains
# "openblas", which this BLAS lacks (OpenBLAS 0.3.26 has none: undefined reference at link time).
# C interface: dBPACK_wrapper.h (without HAVE_MPI, MPI_Fint is int). libmpi_dummy.a provides the
# MPI routines ButterflyPACK calls: never link it into a program that also links a real MPI
# (GOTCHA(mpi-stub)). ButterflyPACK's C++ object needs libstdc++: link with $(CXX).
BPACK_CPPFLAGS = -I$(DEPS_PREFIX)/bpack/include
BPACK_LIBS = $(DEPS_PREFIX)/bpack/lib/libdbutterflypack.a $(DEPS_PREFIX)/bpack/lib/libmpi_dummy.a -lgfortran -lgomp -lm

# ---- H2Opus cbc7985 (2022-10-17) + Thrust 1.17.2 (1ac51f2), CPU only, static -----------------
# Licence: BSD-3-Clause (https://github.com/ecrc/h2opus)
# H2Opus's make with a generated make.inc: CPU only (no GPU, MAGMA, MKL or MPI), double precision,
# CXXFLAGS = -O2 -fopenmp -std=c++14 -fPIC -w (the study's flags; no H2OPUS_PROFILING_ENABLED, which
# changes inline code in H2Opus's headers), H2OPUS_FMANGLE_TYPE=add (Fortran BLAS names dgemm_).
# The generated include/h2opusconf.h defines H2OPUS_USE_DOUBLE_PRECISION and H2OPUS_FMANGLE_ADD, so
# no -D is needed. Thrust (tag 1.17.2; Apache-2.0, parts BSL-1.0/BSD-2-Clause) and its CUB submodule
# (BSD-3-Clause) share <prefix>/thrust/include.
# GOTCHA(h2opus-thrust-backend): H2Opus's headers choose Thrust's backend from _OPENMP (OpenMP
# with -fopenmp, serial without): compile every file that includes them with -fopenmp, like the
# library, or the Thrust types of the two sides differ.
H2OPUS_CPPFLAGS = -I$(DEPS_PREFIX)/h2opus/include -I$(DEPS_PREFIX)/thrust/include
H2OPUS_LIBS = $(DEPS_PREFIX)/h2opus/lib/libh2opus.a

# ---- STARS-H c115e75 (2023-11-20), static ----------------------------------------------------
# Licence: BSD-3-Clause (https://github.com/ecrc/stars-h)
# CMake: CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG), OPENMP=ON, MPI=OFF, STARPU=OFF, GSL=OFF, CUDA=OFF,
# KBLAS=OFF, BLA_VENDOR=Open, EXAMPLES/TESTING/DOCS=OFF. include/ is STARS-H's whole source
# include directory. STARS-H calls CBLAS (BLAS_LIBS), LAPACKE and, in its Fortran
# applications, the gfortran runtime.
STARSH_CPPFLAGS = -I$(DEPS_PREFIX)/starsh/include
STARSH_LIBS = $(DEPS_PREFIX)/starsh/lib/libstarsh.a $(DEPS_LAPACKE_LIBS) -lgfortran -lm

# ---- H2Pack 2aa58a6 (2026-09-20; ASTER 9bf3b07), static --------------------------------------
# Licence: MIT (https://github.com/scalable-matrix/H2Pack)
# H2Pack's make with src/GCC-OpenBLAS.make: gcc -Wall -g -std=gnu11 -O3 -fPIC -fopenmp -march=native
# -DUSE_OPENBLAS (-march=native: runs only on CPUs with this machine's instruction set).
# -DUSE_OPENBLAS makes linalg_lib_wrapper.h include <cblas.h> and <lapacke.h> and call
# openblas_set_num_threads, so H2Pack needs OpenBLAS in BLAS_LIBS (and it resets OpenBLAS's
# thread count: GOTCHA(h2pack-blas-threads)).
# examples/ holds H2Pack's metadata converters meta_json_to_txt.py and meta_txt_to_json.py.
H2PACK_CPPFLAGS = -I$(DEPS_PREFIX)/h2pack/include -DUSE_OPENBLAS
H2PACK_LIBS = $(DEPS_PREFIX)/h2pack/lib/libH2Pack.a $(DEPS_LAPACKE_LIBS) -lm
```
