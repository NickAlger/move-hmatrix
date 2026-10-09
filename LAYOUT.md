# Layout and conventions

How the package is organised, and the rules every adapter, example and test follows.
Read this before adding a library or a route.

## Directories

```
include/hmb/          library-neutral interfaces (no third-party headers)
  neutral.h           the neutral in-memory H-matrix view (hmb_hmatrix) + utilities
  cstore.h            growable owner for views built in C
  neutral_store.hpp   owner for views built in C++ (hmb::NeutralStore)
  blackbox.hpp        hmb::BlackBox: matvec operator (matvec route)
  entry_source.hpp    hmb::EntrySource: entry oracle (entry route)
  neutral_bridge.hpp  any neutral view as a BlackBox or an EntrySource
src/                  libhmb.a: implementation of the neutral utilities (no dependencies)
adapters/<lib>/       one directory per library (htool, h2lib, hmatoss, butterflypack,
                      h2opus, starsh); see "Adapters" below
tools/h2pack/         file-based bridge for H2Pack (writer of H2Pack's on-disk format)
examples/             one self-checking program per route; they reproduce RESULTS.md
tests/                unit and round-trip tests; tests/common/hmb_test.hpp is the harness
mk/                   make fragments, one per library or route
deps/                 pinned versions and the script that builds them
results/              measured numbers behind RESULTS.md, and the raw logs
```

## The three routes

| Route | Interface | Cost (measured) | Accuracy |
|---|---|---|---|
| Block copy | `hmb_hmatrix` (exporter -> importer) | ~1-4 source matvecs | lossless |
| Rebuild from entries | `hmb::EntrySource` | ~70 source matvecs | recompression at the destination's tolerance |
| Rebuild from matvecs | `hmb::BlackBox` | ~900-4,800 source matvecs | randomized construction at the destination's tolerance |

`include/hmb/neutral_bridge.hpp` turns any exported view into an EntrySource or a BlackBox,
so every exporter also feeds the two rebuild routes.

## Neutral form

`include/hmb/neutral.h` is the contract; read its header comment. In short: real double,
0-based, column-major, always the full matrix (exporters of symmetric storage mirror the
missing half), `perm[k]` = original index at permuted position k, clusters are contiguous
permuted ranges, block tree with SUPER / DENSE / LOWRANK nodes, low-rank leaves `U*V`
(vtrans 0, V rank x cols) or `U*V^T` (vtrans 1, V cols x rank). The view allocates nothing;
whoever builds it owns the storage (`hmb_cstore` in C, `hmb::NeutralStore` in C++).

`BlackBox` and `EntrySource` always use the ORIGINAL (user) ordering, 0-based indices,
column-major blocks.

## Adapters

Each library gets `adapters/<lib>/`. Two kinds of header:

- **Neutral-facing headers** `hmb_<lib>.h` (C) or `hmb_<lib>*.hpp` (C++) include only
  `hmb/` headers and the standard library, never the library's own headers. Library objects
  cross the interface as opaque handles (forward-declared structs for C libraries, e.g.
  `struct _hmatrix` for H2Lib; `std::shared_ptr` handles for C++ libraries). Any number of
  neutral-facing headers can be included in one translation unit.
- **Native headers** `<lib>_native.hpp` (or `.h`) include the library and expose its types,
  for users who already hold that library's objects. Never include the native headers of two
  libraries in one translation unit: H2Lib defines macros named `I`, `real` and `UINT_MAX`;
  Htool and H2Opus declare the same Fortran BLAS prototypes with different `const`.

Implementation files (`.c` / `.cpp`) in the adapter directory are compiled one per
translation unit with that library's flags. Every adapter directory is on the include path,
so file names there start with `hmb_` or the library's name (`h2lib_`, `htool_`, ...) and
never shadow one of the library's own headers.

Naming: C functions `hmb_<lib>_<verb>` (e.g. `hmb_h2lib_import`); C++ functions in namespace
`hmb`, named `<lib>_<verb>` (e.g. `hmb::htool_build`). Exporters are `*_export`, block-copy
importers `*_import`, rebuild importers `*_from_entries` / `*_from_matvec`, and
`*_blackbox` / `*_entries` expose a library's matrix through the common interfaces.

Errors: C functions return 0 on success and a negative code otherwise, with a message in
an optional `char *msg, size_t msglen` argument; C++ functions throw `std::runtime_error` (or
`std::invalid_argument`) whose message starts with the function name, e.g.
`"hmb::bpack_from_matvec: ..."`.

Ownership: an exporter never copies leaf data it can alias; it says in its header comment
how long the source must stay alive. An importer copies by default; a borrowing mode, if
offered, says so in its name or in a mode flag.

Gotchas: wherever code works around a library quirk, mark it with a comment
`// GOTCHA(<short-id>): <one line>` (C: `/* GOTCHA(<short-id>): ... */`), using the ids
listed in GOTCHAS.md, so the explanation and the code can be found from each other.

## Build

GNU make. `Makefile` reads the configuration (`config.mk`, else
`deps/install/config.mk`), sets `HAVE_<LIB>` for each configured library, then includes
`mk/*.mk` in sorted order:

| Fragment | Contents |
|---|---|
| `mk/10-core.mk` | tests of the neutral utilities |
| `mk/20-<lib>.mk` | the library's adapter objects (`<LIB>_OBJS`) and single-library tests |
| `mk/50-<route>.mk` | examples and tests that combine several libraries |

Rules for fragments:

- Wrap everything in `ifeq ($(HAVE_<LIB>),1)` (or nested ifs for several libraries) so a
  partial configuration still builds.
- Add compile flags as pattern-specific variables on object files, e.g.
  `$(OBJ)/adapters/h2lib/%.o: EXTRA_CFLAGS += $(H2LIB_CPPFLAGS)`. Never put flags on an
  executable target (make would pass them down to every prerequisite object).
- Objects: `$(OBJ)/<path without extension>.o` comes from `<path>.c` or `<path>.cpp` through
  the generic rules, so a source file needs no compile rule of its own.
- Executables go to `$(BIN)/`; link with `$(CXX) $(HMB_LDFLAGS) <objects> <adapter objects>
  $(LIBHMB) <library LIBS> $(BLAS_LIBS) -o $@`.
- Register executables with `TESTS += ...` or `EXAMPLES += ...`; scripts with
  `SCRIPT_CHECKS += ...`. `make check` runs all of them.

Configuration variables (all optional; a library is enabled when its `_CPPFLAGS` is set):

| Variable | Meaning |
|---|---|
| `CC`, `CXX`, `FC`, `OPT`, `OPENMP`, `BLAS_LIBS`, `PYTHON` | toolchain |
| `HTOOL_CPPFLAGS` | `-I` of the directory containing `htool/` (header-only) |
| `H2LIB_CPPFLAGS`, `H2LIB_LIBS` | `-I.../Library` plus the `-D` flags libh2.a was built with (`-DUSE_BLAS -DUSE_OPENMP`); the library and its link dependencies |
| `HMAT_CPPFLAGS`, `HMAT_LIBS` | hmat-oss include directories (source `include/` and the build directory with `hmat/config.h`); `-L... -lhmat` |
| `BPACK_CPPFLAGS`, `BPACK_LIBS` | ButterflyPACK double-precision C interface (`dC_BPACK_wrapper.h`); `libdbutterflypack.a`, the MPI stub, `-lgfortran` |
| `H2OPUS_CPPFLAGS`, `H2OPUS_LIBS` | H2Opus include directories (with the generated `h2opusconf.h`, thrust and cub); the library |
| `STARSH_CPPFLAGS`, `STARSH_LIBS` | STARS-H include directory; `libstarsh.a` and its BLAS/LAPACKE |
| `H2PACK_CPPFLAGS`, `H2PACK_LIBS` | H2Pack include directory and defines; `libH2Pack.a` and its BLAS |

## Tests and examples

- C++17, using `tests/common/hmb_test.hpp` (test problems, error measures, timers,
  `hmb::test::Checker`). Tests of C adapters call the C API from C++; if a test needs a
  library's own headers, put that part in a separate C file.
- Every test and example runs with no arguments in well under a few minutes on 2 cores
  (default sizes around N = 2,000-4,000) and ends with `return C.finish(...)`, which prints
  `HMB_TEST_PASSED ...` only when every check passed. `tests/run_tests.sh` requires both
  exit status 0 and that line: a Fortran STOP inside ButterflyPACK exits with status 0.
- Examples take `--n`, `--eps`, ... to reproduce the larger runs of RESULTS.md, and print the
  measured costs (time, memory, number of source matvecs or entries) next to the checks.
- Accuracy thresholds are stated relative to the tolerances used (e.g. `<= 10 * eps`), and
  every check prints the measured value.
