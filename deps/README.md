# Pinned third-party builds

`deps/build_deps.sh` fetches the seven libraries hmatrix-bridges connects, at the commits pinned
in `deps/versions.txt`, builds each one the way the package uses it, installs it under
`deps/install/<id>` and writes `deps/install/config.mk`. The top-level Makefile reads that file
whenever there is no top-level `config.mk`, so a fresh checkout needs:

    sudo apt-get install build-essential gfortran cmake git pkg-config python3 python3-numpy \
                         libopenblas-dev liblapacke-dev
    nice deps/build_deps.sh            # about 5 minutes on 2 cores
    make -j2 && make check

The script ends by compiling, linking and running one small program per library through the new
`config.mk` (a numerical check, not just a link), so a configuration that it reports as complete
builds the package. Nothing needs MPI, CUDA or MAGMA, and nothing picks them up even when they
are installed (this machine has OpenMPI).

## Options

| Option | Meaning |
|---|---|
| `--prefix DIR` | install directory (default `deps/install`); sources always go to `deps/src/<id>`, build trees and logs to `deps/build/<id>`, `deps/build/<id>.log` |
| `--only LIB,...` | build only these libraries and rebuild them even if up to date; libraries built earlier stay in `config.mk` |
| `--jobs N` | parallel compile jobs (default 2) |
| `--skip-fetch` | no network: build from the existing checkouts in `deps/src`, which must be at the pinned commits and unmodified |
| `--blas "FLAGS"` | BLAS/LAPACK/CBLAS link flags; default `-lopenblas` when OpenBLAS links, else `-llapack -lblas`. LAPACKE is detected separately (`-llapacke` on Debian/Ubuntu, whose OpenBLAS has none). H2Pack needs OpenBLAS and is skipped with another BLAS. Headers must be on the compiler's search path (`CPATH`) |
| `--help` | usage |

Library ids: `htool`, `h2lib`, `hmat` (hmat-oss), `bpack` (ButterflyPACK), `h2opus` (with
Thrust), `starsh` (STARS-H), `h2pack`; `--only` also accepts the long names. The compilers come
from `CC`, `CXX`, `FC` (default gcc, g++, gfortran; MPI wrappers are refused). The prerequisite
check runs first and lists everything missing at once, with the `apt-get` line above.

## What the script does, per library

1. **Fetch**: `git clone --filter=blob:none --no-checkout`, then `git checkout --detach <commit>`
   and `git submodule update --init --recursive`. Blobless clones download history without file
   contents; ButterflyPACK's example data (`EXAMPLE/*_DATA`, 415 MB) and STARS-H's
   `SARS-CoV-2-meshes` (113 MB) are left out with `git sparse-checkout` because no build reads
   them. The checkout must be exactly the pinned commit, unmodified, with every submodule at the
   revision the commit records and `versions.txt` lists. The commit date is checked against
   `versions.txt`, and the remote branches holding the commit are recorded (H2Lib's is on
   `community`, not on `master`).
2. **Build** in `deps/build/<id>` (a copy of the sources when the library builds in its source
   tree), with every command in `deps/build/<id>.log`.
3. **Install** into `<prefix>/<id>` (plus `<prefix>/thrust`), with the licence, a
   `BUILD_INFO` record (commits, dates, branches, licence, compiler, BLAS, signature) and the
   library's section of `config.mk`.
4. **Skip if up to date**: the signature hashes the pinned commits and submodules, the build
   step's code in the script, the compiler, cmake and BLAS versions and flags. If `BUILD_INFO`
   carries the same signature and the installed files exist, the library is skipped. A failed
   build leaves the library out of `config.mk` (its old install is removed first). A failed
   fetch or verification leaves the previous install in place.
5. After all libraries: write `<prefix>/config.mk`, then the checks below, then disk use.

A second run right after a full build skips every library and only reruns the checks (about
30 seconds).

## The builds and why

The options reproduce the builds behind the study's measurements unless noted. The machine code
was compared object by object with the study's builds (`objdump -d`, symbolic labels ignored):
the instructions of every object of H2Lib (60), hmat-oss (26), ButterflyPACK's double precision
(35), H2Opus (13), STARS-H (64) and H2Pack (20) are identical (H2Opus's objects differ only in
the source path strings embedded by `__FILE__`).

**Htool-DDM 0ee4cac** (MIT). Header-only: `include/htool` is copied. Htool's CMake requires MPI,
but the H-matrix and clustering headers the package uses compile without it; they need
BLAS/LAPACK only.

**H2Lib 920662c** (LGPL-3.0, branch `community`). H2Lib's own Makefile, target `libh2.a`, with
the study's `options.inc` (`OPT=1 USE_BLAS=1 USE_OPENMP=1 HARITH_RKMATRIX_QUICK_EXIT=1`; the
default `options.inc.default` has `DEBUG=1` and `USE_CAIRO=1`) and `system.inc.linux`
(`gcc -Wall -O3 -march=native -funroll-loops -fopenmp`). The script asks H2Lib's Makefile for
its effective `CFLAGS` and writes every `-D` of them into `H2LIB_CPPFLAGS`
(`-DUSE_BLAS -DUSE_OPENMP -DHARITH_RKMATRIX_QUICK_EXIT`): H2Lib's scalar type and struct layouts
depend on `USE_COMPLEX`, `USE_FLOAT`, `USE_SIMD`, `USE_BLAS` and `USE_OPENMP`, and code compiled
with other flags misreads its structs silently. Real double precision. No `-fPIC` (fine for
executables); `-march=native` ties the archive to this CPU type. The study's `locallib`
symlinks are not needed now that the BLAS packages provide the `.so` links.

**hmat-oss 45a2fdb** (GPL-2.0-or-later: a program linking it is covered by the GPL). CMake,
`CMAKE_BUILD_TYPE=Release` (-O3 -DNDEBUG; hmat-oss adds `-fopenmp -ffast-math`),
`BUILD_SHARED_LIBS=ON`, `HMAT_DISABLE_OPENMP=ON` (default), `HMAT_GIT_VERSION=OFF` (library
`libhmat.so.1.11.1`, soname `libhmat.so.4`), `BUILD_EXAMPLES=OFF` (the study built them; they
do not change the library). `HAVE_OPENBLAS_SET_NUM_THREADS=OFF HAVE_GOTO_GET_NUM_PROCS=OFF` are
pre-set: configured against OpenBLAS, hmat-oss sets OpenBLAS's global thread count to 1 at the
start of most API calls and restores it at the end, which races when several threads call it;
the study's build had it compiled out by accident (it was configured against reference BLAS
before OpenBLAS was installed), and its private `config.h` is identical to the new one.
Difference: the study linked a hand-built static LAPACKE subset into `libhmat.so` (it exported 96
`LAPACKE_*` symbols); the new library links the system `liblapacke.so.3` and
`libopenblas.so.0`. `HMAT_LIBS` carries an rpath.

**ButterflyPACK a8cf81b** (BSD-3-Clause with LBNL's enhancement clause). The study configured the
CMake build at -O1 (shared, four precisions) and recompiled the generated double-precision
sources at -O2 by hand. Here CMake does it directly: `enable_mpi=OFF` (sequential, with
ButterflyPACK's MPI stub `MPI_DUMMY`), `enable_openmp=ON`, `BUILD_SHARED_LIBS=OFF`,
`CMAKE_BUILD_TYPE=None`, `CMAKE_{C,CXX,Fortran}_FLAGS=-O2`, and only the targets
`dbutterflypack` (an object library) and `mpi_dummy` are built. `libdbutterflypack.a` archives
the 35 double-precision objects; `libmpi_dummy.a` is separate (the study put `mpi_dummy.o` into
its archive). The configure step writes `SRC_DOUBLE`, `SRC_SINGLE`, ... into the source tree, so
it runs on a copy. The C interface is `dBPACK_wrapper.h` (`MPI_Fint` is `int` without
`HAVE_MPI`). `-UHAVE_OPENBLAS_BATCH` is appended to the Fortran flags when the BLAS has no
`cblas_dgemm_batch` (see below).

**H2Opus cbc7985** (BSD-3-Clause) with **Thrust 1.17.2** (Apache-2.0) and its CUB submodule
(BSD-3-Clause). H2Opus's own make, with a generated `make.inc`: CPU only (no GPU, MAGMA, MKL or
MPI), double precision, `CXXFLAGS = -O2 -fopenmp -std=c++14 -fPIC -w` (the study's flags; the
sample `make.inc.cpu` also sets `-DH2OPUS_PROFILING_ENABLED`, which changes inline code in the
headers), `H2OPUS_FMANGLE_TYPE=add`, static library only. The generated `h2opusconf.h` carries
`H2OPUS_USE_DOUBLE_PRECISION` and `H2OPUS_FMANGLE_ADD`. The headers pick Thrust's OpenMP backend
from `_OPENMP`, so every file including them must be compiled with `-fopenmp`. The study used
Thrust without initialising CUB; CUB is fetched now (the CPU backend does not use it).

**STARS-H c115e75** (BSD-3-Clause; its `ecrc_cmake` submodule is CeCILL-C). CMake, Release,
`OPENMP=ON MPI=OFF STARPU=OFF GSL=OFF CUDA=OFF KBLAS=OFF BLA_VENDOR=Open`, `EXAMPLES`,
`TESTING` and `DOCS` off (the study built the examples and tests; the library is the same 64
objects). `include/` holds STARS-H's whole source include directory, as the study's
`-I<source>/include` did.

**H2Pack 2aa58a6** (MIT; ASTER submodule MIT). H2Pack's make with `src/GCC-OpenBLAS.make`
(`gcc -Wall -g -std=gnu11 -O3 -fPIC -fopenmp -march=native -DUSE_OPENBLAS`), static library.
`INCS=` drops the make file's `-I../../OpenBLAS-git/install/include`. `-DUSE_OPENBLAS` makes
H2Pack call `openblas_set_num_threads`, so it needs OpenBLAS. `examples/` holds H2Pack's
`meta_json_to_txt.py` and `meta_txt_to_json.py`, which the package's H2Pack file bridge needs.

## The checks

One program per library, compiled with `<LIB>_CPPFLAGS` and linked like the package's
executables (`$(CXX) $(OPENMP) <objects> <LIB>_LIBS $(BLAS_LIBS)`), sources and Makefile in
`deps/build/smoke`, log `deps/build/smoke.log`. Each program prints its error next to its
threshold and passes only with the line `SMOKE_OK <id>` (a Fortran STOP would exit with status
0); `ldd` must show no MPI library.

| Library | Program | Measured | Threshold |
|---|---|---|---|
| Htool-DDM | H-matrix (ACA 1e-6) of 1/(0.1+r) on 2,000 sphere points, matvec vs dense | 2.9e-6 | 1e-4 |
| H2Lib | LU (LAPACK) of a 300 x 300 matrix and a solve; reads H2Lib's structs directly | 2.1e-15 | 1e-10 |
| hmat-oss | ACA+ assembly (1e-6) through the C API, gemv vs dense | 6.4e-7 | 1e-4 |
| ButterflyPACK | HODLR from entries (tol 1e-6), mult in its own ordering vs dense | 7.0e-6 | 1e-4 |
| H2Opus | H2 by Chebyshev interpolation (6^3 points per box), hgemv vs dense | 1.3e-4 | 1e-3 |
| STARS-H | TLR (tol 1e-3) of its generated exponential kernel, Frobenius error | 5.4e-4 | 1e-2 |
| H2Pack | H2 (proxy points, 1e-6) of 1/r on 16,000 points, matvec on 400 rows | 1.6e-7 | 1e-4 |

To run them under AddressSanitizer: in a copy of `deps/build/smoke`,
`make smoke_<id> CONFIG=<prefix>/config.mk OPT='-O1 -g -fsanitize=address -fno-omit-frame-pointer'
OPENMP='-fopenmp -fsanitize=address'`. All seven run without ASan errors. LeakSanitizer reports
two leaks inside the libraries, both known to the adapters: hmat-oss leaks one small object per
block-tree node (GOTCHA(hmat-node-leak)), and STARS-H's TLR approximation leaks about 104 bytes
per far-field tile when tiles need more than `maxrank` (GOTCHA(starsh-false-far-leak)). Both are
suppressed in `tests/lsan.supp`, which needs `ASAN_OPTIONS=fast_unwind_on_malloc=0`; with
`LSAN_OPTIONS=suppressions=$PWD/tests/lsan.supp` all seven pass (`tests/run_tests.sh` sets
both by default).

## Using your own builds

Copy `config.mk.example` to `config.mk` at the top level and edit it (or pass
`make CONFIG=file`). It lists every variable with the values for `deps/install` and for builds
elsewhere, and says what each build must be (H2Lib's exact `-D` flags, ButterflyPACK sequential
double precision with its MPI stub, H2Opus CPU double compiled with OpenMP, H2Pack with OpenBLAS,
...). `deps/install/config.mk`, written by the script, explains every flag of the pinned builds.
A library whose `<LIB>_CPPFLAGS` is empty is left out of the build.

## Moving a library to another commit

Edit its line in `versions.txt` (full 40-digit hash, its commit date, licence), run
`deps/build_deps.sh --only <id>`, then `make check`. A new commit can move the things the
package's adapters depend on: re-read the library's notes in `results/notes/` and the
`GOTCHA(...)` comments in its adapter. The submodule lines are checked, not used for fetching:
update them to what the new parent commit records.

## Known build problems and their fixes

| Problem | Fix in build_deps.sh |
|---|---|
| ButterflyPACK enables `cblas_?gemm_batch` whenever the BLAS library name contains "openblas" (`-DHAVE_OPENBLAS_BATCH`); OpenBLAS 0.3.26 has no `cblas_dgemm_batch`, so every program linking ButterflyPACK fails with an undefined reference | probe the BLAS; if the symbol is missing, append `-UHAVE_OPENBLAS_BATCH` to the Fortran flags (GOTCHA(bpack-openblas-batch)) |
| ButterflyPACK's configure step writes generated sources into its source tree | configure a copy in `deps/build/bpack/src` |
| STARS-H generates its spatial, electrostatics and electrodynamics kernels at configure time with `python ../misc_scripts/...`, relative to the directory cmake was started in (it expects `<source>/build`). Elsewhere the generation fails, configure succeeds anyway and `libstarsh.a` lacks 25 kernel files (undefined references at link time). It also needs a `python` command | start cmake in `<source>/src` with a `python` link to python3 on `PATH`; check that the kernels are in the archive (GOTCHA(starsh-codegen-cwd)) |
| hmat-oss switches OpenBLAS's global thread count in its calls when its configure step finds `openblas_set_num_threads` | pre-set `HAVE_OPENBLAS_SET_NUM_THREADS=OFF HAVE_GOTO_GET_NUM_PROCS=OFF`; checked in `config.h` (GOTCHA(hmat-blas-threads)) |
| H2Opus's `make` (target `all`) runs `make -j$(MAKE_NP)`: unlimited parallel jobs when `MAKE_NP` is unset | build target `lib` with `-j<jobs>` |
| H2Opus's sample `make.inc.cpu` switches MKL on whenever `MKLROOT` is set | write our own `make.inc`, unset `MKLROOT` and the `H2OPUS_USE_*` variables |
| H2Lib's default options (`options.inc.default`) build with `DEBUG=1 USE_CAIRO=1` | write `options.inc` |
| H2Pack's make file adds `-I../../OpenBLAS-git/install/include` | `INCS=` on the make command line |
| `ld: warning: dBplus_randomized.f90.o: requires executable stack` when linking ButterflyPACK | none needed: gfortran trampolines for internal procedures, same as in the study's objects; the programs run normally |
| OpenMPI and ScaLAPACK installed system-wide (GOTCHA(mpi-stub)) | every MPI option off (ButterflyPACK `enable_mpi=OFF`, STARS-H `MPI=OFF`); MPI compiler wrappers refused; the checks fail if a program links an MPI library |
| Debian/Ubuntu's OpenBLAS has no LAPACKE | detect it and add `-llapacke` (`DEPS_LAPACKE_LIBS` in `config.mk`) |

## Build times and disk space (measured here)

2-core VM (Intel Cooper Lake), GCC 13.3, OpenBLAS 0.3.26 (pthreads), `--jobs 2`, under `nice`,
from scratch (`rm -rf deps/src deps/build deps/install && nice deps/build_deps.sh`), times
including the fetch. Four other agents were compiling at the same time: load average about 4
during the first run, 1 to 4 during the second (the final script).

| Library | First run | Second run | Sources (MB) | Build tree (MB) | Installed (MB) |
|---|---|---|---|---|---|
| Htool-DDM | 0m02s | 0m02s | 4 | 0 | 1 |
| H2Lib | 0m57s | 0m22s | 5 | 10 | 4 |
| hmat-oss | 1m19s | 0m29s | 2 | 7 | 2 |
| ButterflyPACK | 2m38s | 1m54s | 24 | 47 | 8 |
| H2Opus + Thrust | 0m17s | 0m29s | 43 | 3 | 11 |
| STARS-H | 0m21s | 0m17s | 9 | 3 | 1 |
| H2Pack | 0m17s | 0m22s | 15 | 21 | 3 |
| checks (7 programs) | about 40 s | about 25 s | | | |
| total | 6m19s | 4m21s | 101 | 98 | 30 |

A re-run with everything up to date takes 27 s (the checks). Sources include the git metadata of
the blobless clones (Thrust's CUB submodule is a full clone). `deps/src` and `deps/build` can be
deleted after a build: `config.mk` and the installed files refer only to the prefix.
