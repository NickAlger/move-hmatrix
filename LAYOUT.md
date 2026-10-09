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
src/                  libhmb.a: implementation of the neutral utilities (needs only BLAS)
adapters/<lib>/       one directory per library (htool, h2lib, hmatoss, butterflypack,
                      h2opus, starsh); see "Adapters" below
tools/h2pack/         file-based bridge for H2Pack (writer of H2Pack's on-disk format)
examples/             one self-checking program per route; they reproduce RESULTS.md
tests/                unit and round-trip tests; tests/common/hmb_test.hpp is the harness,
                      tests/run_tests.sh the runner, tests/lsan.supp the leak suppressions
mk/                   make fragments, one per library or route
deps/                 pinned versions and the script that builds them
results/              RESULTS.md's raw logs (logs/), the script that re-runs them
                      (run_reference.sh), and the notes of each part's authors (notes/)
```

## The three routes

| Route | Interface | Cost (measured) | Accuracy |
|---|---|---|---|
| Block copy | `hmb_hmatrix` (exporter -> importer) | ~1-11 source matvecs | lossless |
| Rebuild from entries | `hmb::EntrySource` | ~60-80 source matvecs into ACA destinations (Htool-DDM, hmat-oss); ButterflyPACK and TLR destinations request O(N²) entries | both libraries' errors add |
| Rebuild from matvecs | `hmb::BlackBox` | ~1,100-3,300 source matvecs | can miss the tolerance several-fold |

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
  libraries in one translation unit: H2Lib defines the macros `I` and `UINT_MAX` and a type `real`;
  Htool and H2Opus declare the same Fortran BLAS prototypes with different `const`.

Implementation files (`.c` / `.cpp`) in the adapter directory are compiled one per
translation unit with that library's flags. Every adapter directory is on the include path,
so file names there start with `hmb_` or the library's name (`h2lib_`, `htool_`, ...) and
never shadow one of the library's own headers.

Naming: C functions `hmb_<lib>_<verb>` (e.g. `hmb_h2lib_import`); C++ functions in namespace
`hmb`, named `<lib>_<verb>` (e.g. `hmb::htool_build`). Exporters are `*_export`, block-copy
importers `*_import`, rebuild importers `*_from_entries` / `*_from_matvec`, and
`*_blackbox` / `*_entries` expose a library's matrix through the common interfaces
(`htool_build` and `hmatoss_build` are the entry-route builders of those two libraries). C++ types
are `<Lib>Handle` (a `std::shared_ptr` to the library's matrix), `<Lib>Options`,
`<Lib>Stats`, with only the first letter of the library capitalized (`H2opusTlrHandle`).

C++ signatures follow one pattern: an exporter returns `std::shared_ptr<const hmb_hmatrix>`,
an aliasing pointer whose control block owns everything the view points into (the store, the
source handle), so the view stays valid on its own; an importer takes `const hmb_hmatrix &`
and returns a handle; `*_blackbox` returns a `BlackBox` that keeps its matrix alive.

Errors: C functions return 0 on success and a negative code otherwise, with a message in
an optional `char *msg, size_t msglen` argument; C++ functions throw `std::invalid_argument`,
`std::runtime_error` or (for an operation the matrix's state does not allow, e.g. a product
with a ButterflyPACK H matrix built without its forward operator) `std::logic_error`, whose
message starts with the function name, e.g. `"hmb::bpack_from_matvec: ..."`. One failure
cannot be caught: ButterflyPACK ends the process when its sampling does not converge (the
adapter turns the exit status 0 into 70 and prints why, GOTCHA(fortran-stop)).

Ownership: an exporter never copies leaf data it can alias; it says in its header comment
how long the source must stay alive. An importer copies by default; a borrowing mode, if
offered, says so in its name or in a mode flag.

Gotchas: wherever code works around a library quirk, mark it with a comment
`// GOTCHA(<short-id>): <one line>` (C: `/* GOTCHA(<short-id>): ... */`), using the ids
listed in GOTCHAS.md, so the explanation and the code can be found from each other.

## Build

GNU make. `Makefile` reads the configuration (`config.mk`, else
`deps/install/config.mk`), records it in `$(B)/config.stamp` (every object depends on that
file, which changes only when a configuration value or a configured static library does),
sets `HAVE_<LIB>` for each configured library, then includes `mk/*.mk` in sorted order:

| Fragment | Contents |
|---|---|
| `mk/10-core.mk` | tests of the neutral utilities |
| `mk/20-<lib>.mk` | the library's adapter objects (`<LIB>_OBJS`) and single-library tests |
| `mk/21-htool-copy.mk`, `mk/22-h2opus-tlr.mk` | second parts of the Htool-DDM and H2Opus adapters |
| `mk/50-<route>.mk` | examples and tests that combine several libraries |
| `mk/60-cross.mk` | every block-copy exporter into every block-copy importer |
| `mk/90-archives.mk` | one archive per adapter, `build/lib/libhmb_<lib>.a`, and their link lines in `make info` |

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
  `SCRIPT_CHECKS += ...`. `make check` runs all of them. Helper programs that a script runs
  go in `TOOLS += ...` (built by `make`, not run by `make check`).

Configuration variables (all optional; a library is enabled when its `_CPPFLAGS` is set):

| Variable | Meaning |
|---|---|
| `CC`, `CXX`, `FC`, `OPT`, `OPENMP`, `BLAS_LIBS`, `PYTHON` | toolchain (`FC` must be the Fortran compiler that built ButterflyPACK: `adapters/butterflypack/bpack_flush.f90` is compiled with it; `PYTHON` needs numpy) |
| `HTOOL_CPPFLAGS` | `-I` of the directory containing `htool/` (header-only) |
| `H2LIB_CPPFLAGS`, `H2LIB_LIBS` | `-I` of H2Lib's headers plus exactly the `-D` flags libh2.a was built with (struct layouts depend on them; the pinned build: `-DUSE_BLAS -DUSE_OPENMP -DHARITH_RKMATRIX_QUICK_EXIT`); libh2.a and its link dependencies |
| `HMAT_CPPFLAGS`, `HMAT_LIBS` | hmat-oss include directories (source `include/` and the build directory with `hmat/config.h`); `-L... -lhmat` with an rpath |
| `BPACK_CPPFLAGS`, `BPACK_LIBS` | ButterflyPACK double-precision C interface (`dBPACK_wrapper.h`); `libdbutterflypack.a`, the MPI stub `libmpi_dummy.a`, `-lgfortran -lgomp -lm` |
| `H2OPUS_CPPFLAGS`, `H2OPUS_LIBS` | H2Opus include directories (with the generated `h2opusconf.h`, thrust and cub); the library |
| `STARSH_CPPFLAGS`, `STARSH_LIBS` | STARS-H include directory (plus that of the CBLAS and LAPACKE headers if not on the default path); `libstarsh.a` and LAPACKE |
| `H2PACK_CPPFLAGS`, `H2PACK_LIBS` | H2Pack include directory and `-DUSE_OPENBLAS`; a STATIC `libH2Pack.a` (the reader fix wraps one of its internal calls at link time) and LAPACKE |
| `H2PACK_META_SCRIPT` | optional: H2Pack's `examples/meta_json_to_txt.py` (default: found next to the include directory) |

`deps/build_deps.sh` also writes two helpers into its config.mk, `DEPS_PREFIX` (the install
prefix) and `DEPS_LAPACKE_LIBS`, which the variables above refer to.

## Tests and examples

- C++17, using `tests/common/hmb_test.hpp` (test problems, error measures, timers,
  `hmb::test::Checker`). Tests of C adapters call the C API from C++; if a test needs a
  library's own headers, put that part in a separate C file.
- Every test and example runs with no arguments in about a minute or less on 2 cores
  (default sizes around N = 2,000-4,000) and ends with `return C.finish(...)`, which prints
  `HMB_TEST_PASSED ...` only when every check passed. `tests/run_tests.sh` requires both
  exit status 0 and that line: a Fortran STOP inside ButterflyPACK exits with status 0.
- AddressSanitizer: `make B=build-asan OPT='-O1 -g -fsanitize=address
  -fno-omit-frame-pointer' HMB_LDFLAGS='-fopenmp -fsanitize=address' check`. The runner
  applies `tests/lsan.supp` (leaks inside hmat-oss and STARS-H) unless `LSAN_OPTIONS` is set.
- Examples take `--n`, `--eps`, ... to reproduce the larger runs of RESULTS.md, and print the
  measured costs (time, memory, number of source matvecs or entries) next to the checks.
- Accuracy thresholds are stated relative to the tolerances used (e.g. `<= 10 * eps`), and
  every check prints the measured value.
