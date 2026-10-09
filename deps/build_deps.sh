#!/usr/bin/env bash
# deps/build_deps.sh: fetch, build and install the third-party libraries of hmatrix-bridges at
# the commits pinned in deps/versions.txt, write <prefix>/config.mk for the top-level Makefile
# (it reads deps/install/config.mk when there is no top-level config.mk), and check that file
# by compiling, linking and running one small program per library.
#
#   deps/build_deps.sh [--prefix DIR] [--only LIB,...] [--jobs N] [--skip-fetch]
#                      [--blas "<link flags>"] [--help]
#
# Libraries (ids): htool (Htool-DDM), h2lib (H2Lib), hmat (hmat-oss), bpack (ButterflyPACK),
# h2opus (H2Opus, with Thrust), starsh (STARS-H), h2pack (H2Pack). Sources go to deps/src/<id>,
# build trees to deps/build/<id>, build logs to deps/build/<id>.log, installs to <prefix>/<id>.
# No MPI, CUDA or MAGMA is used. Details, build times and disk use: deps/README.md.
set -euo pipefail

DEPS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$DEPS_DIR")
SRC_DIR=$DEPS_DIR/src
BUILD_DIR=$DEPS_DIR/build
PINS=$DEPS_DIR/versions.txt
LIBS_ALL=(htool h2lib hmat bpack h2opus starsh h2pack)

# Paths left out of the checkout (git sparse-checkout): example data the builds never read.
declare -A SPARSE_EXCLUDE=(
    [bpack]='/EXAMPLE/FULLMAT_DATA/ /EXAMPLE/KRR_DATA/ /EXAMPLE/EM3D_DATA/'
    [starsh]='/SARS-CoV-2-meshes/'
)

usage() {
    cat <<'EOF'
Usage: deps/build_deps.sh [options]

Fetches the libraries pinned in deps/versions.txt (git clone + checkout of the exact commit,
submodules at their recorded revisions), builds them the way hmatrix-bridges uses them, installs
them under the prefix and writes <prefix>/config.mk. A library already built at its pinned commit
with the same options is skipped. Finally every configured library is checked with a small
program compiled and linked through config.mk (log: deps/build/smoke.log).

Options:
  --prefix DIR       install directory (default: deps/install). Sources always go to deps/src,
                     build trees and logs to deps/build.
  --only LIB,...     build only these libraries, and rebuild them even if up to date. Ids: htool,
                     h2lib, hmat (hmat-oss), bpack (butterflypack), h2opus (thrust), starsh
                     (stars-h), h2pack. Libraries built earlier stay in config.mk.
  --jobs N           parallel compile jobs (default: 2).
  --skip-fetch       no network: use the existing checkouts in deps/src (they must be at the
                     pinned commits).
  --blas "FLAGS"     BLAS/LAPACK/CBLAS link flags (default: -lopenblas if OpenBLAS links, else the
                     reference "-llapack -lblas"). LAPACKE is added as needed. H2Pack needs
                     OpenBLAS (openblas_set_num_threads) and is skipped otherwise. Headers
                     (cblas.h, lapacke.h) must be on the compiler's search path (set CPATH).
  -h, --help         this text.

Environment: CC, CXX, FC choose the compilers (default gcc, g++, gfortran; not MPI wrappers).
Run it under `nice` on a shared machine. Exit status: 0 when every requested library is
installed and passes its check.
EOF
}

# ---- options ----------------------------------------------------------------------------
PREFIX=$DEPS_DIR/install
JOBS=2
ONLY=
SKIP_FETCH=0
BLAS=
BLAS_GIVEN=0
while [ $# -gt 0 ]; do
    case $1 in
        --prefix) PREFIX=${2:?--prefix needs a directory}; shift 2 ;;
        --prefix=*) PREFIX=${1#*=}; shift ;;
        --only) ONLY=${2:?--only needs a list of libraries}; shift 2 ;;
        --only=*) ONLY=${1#*=}; shift ;;
        --jobs) JOBS=${2:?--jobs needs a number}; shift 2 ;;
        --jobs=*) JOBS=${1#*=}; shift ;;
        --skip-fetch) SKIP_FETCH=1; shift ;;
        --blas) BLAS=${2?--blas needs link flags}; BLAS_GIVEN=1; shift 2 ;;
        --blas=*) BLAS=${1#*=}; BLAS_GIVEN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "build_deps.sh: unknown option '$1' (see --help)" >&2; exit 2 ;;
    esac
done
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "build_deps.sh: --jobs needs a positive integer" >&2; exit 2; }
mkdir -p "$PREFIX" "$SRC_DIR" "$BUILD_DIR"
PREFIX=$(cd "$PREFIX" && pwd)

canon_id() {
    case ${1,,} in
        htool|htool-ddm) echo htool ;;
        h2lib) echo h2lib ;;
        hmat|hmat-oss|hmatoss) echo hmat ;;
        bpack|butterflypack) echo bpack ;;
        h2opus|thrust) echo h2opus ;;
        starsh|stars-h) echo starsh ;;
        h2pack) echo h2pack ;;
        *) return 1 ;;
    esac
}
declare -A FORCE=()
if [ -n "$ONLY" ]; then
    SELECTED=()
    for name in ${ONLY//,/ }; do
        id=$(canon_id "$name") || { echo "build_deps.sh: unknown library '$name' (ids: ${LIBS_ALL[*]})" >&2; exit 2; }
        [ -n "${FORCE[$id]:-}" ] || SELECTED+=("$id")
        FORCE[$id]=1
    done
else
    SELECTED=("${LIBS_ALL[@]}")
fi

CC=${CC:-gcc}
CXX=${CXX:-g++}
FC=${FC:-gfortran}

say() { printf '%s\n' "$*"; }
die() { printf 'build_deps.sh: %s\n' "$*" >&2; exit 1; }

# ---- pins (deps/versions.txt) ---------------------------------------------------------------
# PIN_<field>[id]: NAME library, URL repository, COMMIT, DATE, LICENCE; SUBMODULES[id]: the
# "<path>=<commit>" pairs of the submodule lines "<id>:<path>".
declare -A PIN_NAME=() PIN_URL=() PIN_COMMIT=() PIN_DATE=() PIN_LICENCE=() SUBMODULES=()
trim() { local v=$1; v=${v#"${v%%[![:space:]]*}"}; printf '%s' "${v%"${v##*[![:space:]]}"}"; }
while IFS='|' read -r f1 f2 f3 f4 f5 f6 f7; do
    k=$(trim "$f1")
    case $k in ''|'#'*) continue ;; esac
    [ -n "$f7" ] || { echo "build_deps.sh: $PINS: line '$k' needs 7 fields" >&2; exit 1; }
    PIN_NAME[$k]=$(trim "$f2"); PIN_URL[$k]=$(trim "$f3"); PIN_COMMIT[$k]=$(trim "$f4")
    PIN_DATE[$k]=$(trim "$f5"); PIN_LICENCE[$k]=$(trim "$f6")
    if [[ $k == *:* ]]; then SUBMODULES[${k%%:*}]+="${k#*:}=${PIN_COMMIT[$k]} "; fi
done < "$PINS"
pinned_submodules() {  # lines "<path> <commit>" of the submodules pinned for repository $1
    local e
    for e in ${SUBMODULES[$1]:-}; do echo "${e%%=*} ${e#*=}"; done
}
# repositories a library is built from (H2Opus needs Thrust)
repos_of() { if [ "$1" = h2opus ]; then echo "h2opus thrust"; else echo "$1"; fi; }

for id in "${LIBS_ALL[@]}" thrust; do
    [ -n "${PIN_COMMIT[$id]:-}" ] || die "no line for '$id' in $PINS"
done
for id in "${!PIN_COMMIT[@]}"; do
    [[ ${PIN_COMMIT[$id]} =~ ^[0-9a-f]{40}$ ]] || die "$PINS: commit of '$id' must be a full 40-digit hash"
done

# ---- prerequisites --------------------------------------------------------------------------
PROBE_DIR=$BUILD_DIR/probe
probe_link() {  # probe_link "<flags>" sym...: do these symbols link with the flags?
    local flags=$1; shift
    mkdir -p "$PROBE_DIR"
    {
        for s in "$@"; do echo "extern void $s(void);"; done
        echo 'int main(void) { void (*volatile p)(void) = 0;'
        for s in "$@"; do echo "  p = $s;"; done
        echo '  return p == 0; }'
    } > "$PROBE_DIR/link.c"
    # shellcheck disable=SC2086
    "$CC" "$PROBE_DIR/link.c" -o "$PROBE_DIR/link" $flags -lm > "$PROBE_DIR/link.log" 2>&1
}
probe_header() {  # probe_header <header>: does #include <header> compile?
    mkdir -p "$PROBE_DIR"
    printf '#include <%s>\nint main(void) { return 0; }\n' "$1" > "$PROBE_DIR/hdr.c"
    "$CC" -c "$PROBE_DIR/hdr.c" -o "$PROBE_DIR/hdr.o" > "$PROBE_DIR/hdr.log" 2>&1
}
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

check_prereqs() {
    local problems=() c v
    for c in "$CC" "$CXX" "$FC" make cmake git python3 pkg-config ar tar; do
        command -v "$c" > /dev/null 2>&1 || problems+=("command not found: $c")
    done
    if [ ${#problems[@]} -eq 0 ]; then
        for c in "$CC" "$CXX" "$FC"; do  # GOTCHA(mpi-stub): an MPI wrapper would pull in a real MPI
            if "$c" -show > /dev/null 2>&1 || "$c" --showme > /dev/null 2>&1; then
                problems+=("$c is an MPI compiler wrapper: set CC/CXX/FC to the plain compilers (no MPI is used)")
            fi
        done
        v=$(cmake --version | awk 'NR == 1 { print $3 }')
        version_ge "$v" 3.14 || problems+=("cmake $v is too old (ButterflyPACK needs 3.14 or newer)")
        python3 -c 'import numpy' > /dev/null 2>&1 || problems+=("python3 cannot import numpy (needed by tools/h2pack)")
        if [ $BLAS_GIVEN = 0 ]; then
            if probe_link "-lopenblas" dgemm_ dgeqrf_ cblas_dgemm; then BLAS="-lopenblas"
            elif probe_link "-llapack -lblas" dgemm_ dgeqrf_ cblas_dgemm; then BLAS="-llapack -lblas"
            else problems+=("no BLAS/LAPACK with CBLAS found (tried -lopenblas, then -llapack -lblas)"); fi
        elif ! probe_link "$BLAS" dgemm_ dgeqrf_ cblas_dgemm; then
            problems+=("--blas \"$BLAS\" does not provide dgemm_, dgeqrf_ and cblas_dgemm (see $PROBE_DIR/link.log)")
        fi
        probe_header cblas.h || problems+=("cblas.h not found (CBLAS header, e.g. from libopenblas-dev)")
        probe_header lapacke.h || problems+=("lapacke.h not found (LAPACKE header, liblapacke-dev)")
        if [ -n "$BLAS" ]; then
            if probe_link "$BLAS" LAPACKE_dgeqrf; then LAPACKE=""
            elif probe_link "-llapacke $BLAS" LAPACKE_dgeqrf; then LAPACKE="-llapacke"
            else problems+=("LAPACKE does not link (tried $BLAS, then -llapacke $BLAS)"); fi
        fi
    fi
    if [ ${#problems[@]} -gt 0 ]; then
        printf 'build_deps.sh: prerequisites not met:\n' >&2
        printf '  - %s\n' "${problems[@]}" >&2
        printf 'On Debian/Ubuntu:\n  sudo apt-get install build-essential gfortran cmake git pkg-config python3 python3-numpy libopenblas-dev liblapacke-dev\n' >&2
        exit 1
    fi
}
LAPACKE=
check_prereqs
# What the chosen BLAS offers (it decides two build options and whether H2Pack can be built):
HAVE_OPENBLAS_THREADS=0; probe_link "$BLAS" openblas_set_num_threads && HAVE_OPENBLAS_THREADS=1
HAVE_GEMM_BATCH=0; probe_link "$BLAS" cblas_dgemm_batch && HAVE_GEMM_BATCH=1
BLAS_DESC="$BLAS"
if [ $HAVE_OPENBLAS_THREADS = 1 ]; then
    printf '#include <stdio.h>\nchar *openblas_get_config(void);\nint main(void) { puts(openblas_get_config()); return 0; }\n' > "$PROBE_DIR/cfg.c"
    # shellcheck disable=SC2086
    if "$CC" "$PROBE_DIR/cfg.c" -o "$PROBE_DIR/cfg" $BLAS -lm > /dev/null 2>&1; then
        BLAS_DESC="$BLAS ($("$PROBE_DIR/cfg"))"
    fi
fi
PYTHON_BIN=$(command -v python3)
NUMPY_VERSION=$(python3 -c 'import numpy; print(numpy.__version__)')
tool_version() { "$1" --version 2> /dev/null | head -n1; }

# ---- helpers used inside the build steps (they run with set -x into the library's log) ------
copy_tree() {  # copy_tree SRC DST: fresh copy of a checkout, without .git
    rm -rf "$2"
    mkdir -p "$2"
    tar -C "$1" --exclude=.git -cf - . | tar -C "$2" -xf -
}
short() { printf '%s' "${1:0:7}"; }
run() { echo "+ $*"; "$@"; }  # run a command, echoing it into the log

# Commit details of a checked-out repository: "<date>" and the remote branches holding it.
commit_date() { git -C "$SRC_DIR/$1" show -s --format=%cs HEAD; }
commit_branches() {
    git -C "$SRC_DIR/$1" branch -r --contains HEAD 2> /dev/null | sed 's/^ *//; /->/d' | tr '\n' ' ' | sed 's/ $//'
}

# fetch_repo <repo id>: clone (blobless) and check out the pinned commit, then the submodules.
fetch_repo() {
    local r=$1 url commit dir pat p
    url=${PIN_URL[$r]}; commit=${PIN_COMMIT[$r]}; dir=$SRC_DIR/$r
    if [ ! -e "$dir/.git" ]; then
        [ ! -e "$dir" ] || { echo "$dir exists but is not a git checkout: remove it"; return 1; }
        run git clone --filter=blob:none --no-checkout "$url" "$dir"
        if [ -n "${SPARSE_EXCLUDE[$r]:-}" ]; then
            pat=('/*')
            for p in ${SPARSE_EXCLUDE[$r]}; do pat+=("!$p"); done
            run git -C "$dir" sparse-checkout set --no-cone "${pat[@]}"
        fi
    fi
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2> /dev/null; then
        run git -C "$dir" fetch origin "$commit" || run git -C "$dir" fetch --tags origin
    fi
    run git -C "$dir" -c advice.detachedHead=false checkout --detach "$commit"
    run git -C "$dir" submodule update --init --recursive
}

# verify_repo <repo id>: the checkout is the pinned commit, unmodified, submodules as pinned.
verify_repo() {
    local r=$1 commit dir head date want path sub got
    commit=${PIN_COMMIT[$r]}; dir=$SRC_DIR/$r
    [ -e "$dir/.git" ] || { echo "no checkout of $r in $dir (run without --skip-fetch)"; return 1; }
    head=$(git -C "$dir" rev-parse HEAD)
    [ "$head" = "$commit" ] || { echo "$dir is at $head, pinned: $commit"; return 1; }
    git -C "$dir" diff --quiet HEAD -- || { echo "$dir has local modifications (git -C $dir diff)"; return 1; }
    date=$(commit_date "$r"); want=${PIN_DATE[$r]}
    [ "$date" = "$want" ] || echo "WARNING: commit date of $r is $date, deps/versions.txt says $want"
    while read -r path sub; do
        [ -n "$path" ] || continue
        got=$(git -C "$dir/$path" rev-parse HEAD 2> /dev/null) || { echo "submodule $r/$path not checked out"; return 1; }
        [ "$got" = "$sub" ] || { echo "submodule $r/$path is at $got, pinned: $sub"; return 1; }
        git -C "$dir/$path" diff --quiet HEAD -- || { echo "submodule $r/$path has local modifications"; return 1; }
        date=$(git -C "$dir/$path" show -s --format=%cs HEAD); want=${PIN_DATE[$r:$path]}
        [ "$date" = "$want" ] || echo "WARNING: commit date of $r:$path is $date, deps/versions.txt says $want"
    done < <(pinned_submodules "$r")
    local subs
    subs=$(git -C "$dir" submodule status --recursive)
    if grep -q '^[^ ]' <<< "$subs"; then
        echo "submodules of $r are not at the recorded revisions:"; echo "$subs"
        return 1
    fi
    echo "verified $r: $commit ($(commit_date "$r"); $(commit_branches "$r"))"
}

# Library section of config.mk: each build step writes <prefix>/<id>/config.mk.part, starting
# with config_header <id> <title...> (a title line and the licence).
config_header() {
    local id=$1; shift
    local line="# ---- $* "
    printf '%s%s\n' "$line" "$(printf '%*s' $((96 - ${#line})) '' | tr ' ' '-')"
    printf '# Licence: %s (%s)\n' "${PIN_LICENCE[$id]}" "${PIN_URL[$id]}"
}

# ---- build steps (one per library; run in a subshell, output to deps/build/<id>.log) --------
build_htool() {
    local src=$SRC_DIR/htool dst=$PREFIX/htool
    mkdir -p "$dst/include"
    cp -a "$src/include/htool" "$dst/include/"
    cp "$src/LICENSE" "$dst/"
    {
        config_header htool "Htool-DDM $(short "${PIN_COMMIT[htool]}") (${PIN_DATE[htool]}), header-only"
        cat <<EOF
# Nothing is compiled: include/htool is copied. The package uses only Htool's sequential/OpenMP
# H-matrix and clustering headers, which need BLAS/LAPACK (BLAS_LIBS) but no MPI.
HTOOL_CPPFLAGS = -I\$(DEPS_PREFIX)/htool/include
EOF
    } > "$dst/config.mk.part"
}

build_h2lib() {
    local src=$SRC_DIR/h2lib b=$BUILD_DIR/h2lib dst=$PREFIX/h2lib cflags defs
    copy_tree "$src" "$b"
    # The study's options.inc (BRIEF_OUTPUT off so that this log shows the compile commands).
    # Without a system.inc, H2Lib's Makefile uses system.inc.linux.
    cat > "$b/options.inc" <<'EOF'
# H2Lib build options written by hmatrix-bridges' deps/build_deps.sh
OPT=1
USE_BLAS=1
USE_OPENMP=1
HARITH_RKMATRIX_QUICK_EXIT=1
EOF
    printf 'print-cflags:\n\t@echo $(CFLAGS)\n' > "$b/print-cflags.mk"
    cflags=$(make -s -C "$b" -f Makefile -f print-cflags.mk CC="$CC" GCC="$CC" print-cflags)
    make -C "$b" -j"$JOBS" CC="$CC" GCC="$CC" libh2.a
    mkdir -p "$dst/include" "$dst/lib"
    cp "$b"/Library/*.h "$dst/include/"
    cp "$b/libh2.a" "$dst/lib/"
    cp "$src/LICENSE" "$dst/"
    # every -D the library was compiled with
    defs=$(for f in $cflags; do case $f in -D*) printf ' %s' "$f" ;; esac; done)
    {
        config_header h2lib "H2Lib $(short "${PIN_COMMIT[h2lib]}") (${PIN_DATE[h2lib]}, branch community), static"
        cat <<EOF
# H2Lib's own Makefile, target libh2.a, options.inc OPT=1 USE_BLAS=1 USE_OPENMP=1
# HARITH_RKMATRIX_QUICK_EXIT=1 and system.inc.linux, i.e. compiled with
#   $CC $cflags
# GOTCHA(scalar-type): this libh2.a is real double precision (no USE_COMPLEX, no USE_FLOAT).
# H2LIB_CPPFLAGS repeats every -D of that line: H2Lib's scalar type (field) and its struct layouts
# depend on USE_COMPLEX, USE_FLOAT, USE_SIMD, USE_BLAS and USE_OPENMP, so code compiled with other -D
# flags than libh2.a misreads its structs without any error. HARITH_RKMATRIX_QUICK_EXIT only
# matters inside harith.c; it is repeated so that the list is exactly the library's.
# libh2.a is compiled without -fPIC (link it into executables, not into shared libraries) and
# with -march=native (it runs only on CPUs with this machine's instruction set).
H2LIB_CPPFLAGS = -I\$(DEPS_PREFIX)/h2lib/include$defs
# H2Lib links its own programs with -llapack -lblas -lgfortran -lm (BLAS here: BLAS_LIBS).
H2LIB_LIBS = \$(DEPS_PREFIX)/h2lib/lib/libh2.a -lgfortran -lm
EOF
    } > "$dst/config.mk.part"
}

build_hmat() {
    local src=$SRC_DIR/hmat b=$BUILD_DIR/hmat dst=$PREFIX/hmat
    rm -rf "$b"
    # GOTCHA(hmat-blas-threads): against OpenBLAS, hmat-oss resets its global thread count per call
    # HAVE_OPENBLAS_SET_NUM_THREADS/HAVE_GOTO_GET_NUM_PROCS are pre-set so that this is compiled
    # out, as in the study's build (configured against reference BLAS): the switch is racy when
    # hmat-oss is called from several threads (e.g. as the entry source of a parallel Htool build).
    cmake -S "$src" -B "$b" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_INSTALL_PREFIX="$dst" \
        -DBUILD_EXAMPLES=OFF \
        -DHMAT_GIT_VERSION=OFF \
        -DHMAT_DISABLE_OPENMP=ON \
        -DBLAS_LIBRARIES="${BLAS// /;}" \
        -DLAPACKE_LIBRARIES="${LAPACKE:+$LAPACKE;}${BLAS// /;}" \
        -DHAVE_OPENBLAS_SET_NUM_THREADS=OFF -DHAVE_GOTO_GET_NUM_PROCS=OFF
    cmake --build "$b" -j "$JOBS"
    cmake --install "$b"
    cp "$src/LICENSE.md" "$dst/"
    # the pre-set values reached hmat-oss's private config.h (OPENBLAS_DISABLE_THREADS stays off)
    grep -q '^/\* #undef HAVE_OPENBLAS_SET_NUM_THREADS \*/' "$b/config.h"
    {
        config_header hmat "hmat-oss $(short "${PIN_COMMIT[hmat]}") (${PIN_DATE[hmat]}), shared library"
        cat <<EOF
# CMake: CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG; hmat-oss adds -fopenmp -ffast-math),
# BUILD_SHARED_LIBS=ON, BUILD_EXAMPLES=OFF, HMAT_GIT_VERSION=OFF, HMAT_DISABLE_OPENMP=ON (default:
# hmat-oss sets OpenMP threads to 1 inside its calls), BLAS_LIBRARIES=$BLAS,
# HAVE_OPENBLAS_SET_NUM_THREADS=OFF and HAVE_GOTO_GET_NUM_PROCS=OFF (no switching of OpenBLAS's
# global thread count, as in the study's build). include/hmat holds hmat.h and the generated
# config.h/config-oss.h. hmat-oss is GPL-2.0-or-later: programs linking it fall under the GPL.
HMAT_CPPFLAGS = -I\$(DEPS_PREFIX)/hmat/include
HMAT_LIBS = -L\$(DEPS_PREFIX)/hmat/lib -Wl,-rpath,\$(DEPS_PREFIX)/hmat/lib -lhmat
EOF
    } > "$dst/config.mk.part"
}

build_bpack() {
    local src=$SRC_DIR/bpack b=$BUILD_DIR/bpack dst=$PREFIX/bpack fflags=-O2 objs ff
    # ButterflyPACK's configure step generates SRC_DOUBLE, SRC_SINGLE, ... inside its source
    # tree, so it is configured from a copy.
    # GOTCHA(mpi-stub): sequential build (enable_mpi=OFF): ButterflyPACK's own MPI stub, no real MPI
    copy_tree "$src" "$b/src"
    # GOTCHA(bpack-openblas-batch): a BLAS name containing "openblas" turns on cblas_dgemm_batch calls
    # OpenBLAS 0.3.26 has no cblas_dgemm_batch (undefined reference at link time), so the define
    # is undone when the BLAS lacks it.
    [ "$HAVE_GEMM_BATCH" = 1 ] || fflags="-O2 -UHAVE_OPENBLAS_BATCH"
    cmake -S "$b/src" -B "$b/build" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=None \
        -DCMAKE_C_FLAGS=-O2 -DCMAKE_CXX_FLAGS=-O2 -DCMAKE_Fortran_FLAGS="$fflags" \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_Fortran_COMPILER="$FC" \
        -DBUILD_SHARED_LIBS=OFF \
        -Denable_mpi=OFF -Denable_openmp=ON -Denable_toplevel_openmp=OFF \
        -Denable_doc=OFF -Denable_python=OFF -Denable_fftw=OFF -Denable_parsec=OFF \
        -DTPL_BLAS_LIBRARIES="${BLAS// /;}" -DTPL_LAPACK_LIBRARIES="${BLAS// /;}" \
        -DBUILD_TESTING=OFF
    # Only the double-precision object library and the MPI stub (not the four precisions of
    # libbutterflypack, the C++ interface or the examples).
    cmake --build "$b/build" -j "$JOBS" --target dbutterflypack mpi_dummy
    ff=$(sed -n 's/^Fortran_FLAGS = //p' "$b/build/SRC_DOUBLE/CMakeFiles/dbutterflypack.dir/flags.make")
    objs=("$b"/build/SRC_DOUBLE/CMakeFiles/dbutterflypack.dir/*.o)
    [ ${#objs[@]} -ge 30 ] || { echo "only ${#objs[@]} objects in dbutterflypack"; return 1; }
    mkdir -p "$dst/include" "$dst/lib"
    rm -f "$dst/lib/libdbutterflypack.a"
    ar rcs "$dst/lib/libdbutterflypack.a" "${objs[@]}"
    cp "$b/build/MPI_DUMMY/libmpi_dummy.a" "$dst/lib/"
    cp "$b"/src/SRC_DOUBLE/*.h "$b"/src/SRC_DOUBLE/*.fi "$b/src/MPI_DUMMY/mpi_dummy.fi" "$dst/include/"
    find "$b/build/SRC_DOUBLE" "$b/build/MPI_DUMMY" -maxdepth 1 -name '*.mod' -exec cp {} "$dst/include/" \;
    cp "$src/license.txt" "$dst/"
    {
        config_header bpack "ButterflyPACK $(short "${PIN_COMMIT[bpack]}") (${PIN_DATE[bpack]}), double precision, sequential, static"
        cat <<EOF
# CMake with enable_mpi=OFF (sequential: ButterflyPACK's MPI stub MPI_DUMMY), enable_openmp=ON,
# BUILD_SHARED_LIBS=OFF, CMAKE_BUILD_TYPE=None and CMAKE_{C,CXX,Fortran}_FLAGS=-O2 (the study
# recompiled its -O1 CMake objects at -O2). Only the targets dbutterflypack (double precision)
# and mpi_dummy are compiled; libdbutterflypack.a archives the dbutterflypack objects. Fortran:
$(printf '%s\n' "$FC $ff" | fold -s -w 92 | sed 's/ *$//; s/^/#   /')$([ "$HAVE_GEMM_BATCH" = 1 ] || printf '\n%s' "# -UHAVE_OPENBLAS_BATCH: ButterflyPACK enables cblas_dgemm_batch when the BLAS name contains
# \"openblas\", which this BLAS lacks (OpenBLAS 0.3.26 has none: undefined reference at link time).")
# C interface: dBPACK_wrapper.h (without HAVE_MPI, MPI_Fint is int). libmpi_dummy.a provides the
# MPI routines ButterflyPACK calls: never link it into a program that also links a real MPI
# (GOTCHA(mpi-stub)). ButterflyPACK's C++ object needs libstdc++: link with \$(CXX).
BPACK_CPPFLAGS = -I\$(DEPS_PREFIX)/bpack/include
BPACK_LIBS = \$(DEPS_PREFIX)/bpack/lib/libdbutterflypack.a \$(DEPS_PREFIX)/bpack/lib/libmpi_dummy.a -lgfortran -lgomp -lm
EOF
    } > "$dst/config.mk.part"
}

build_h2opus() {
    local src=$SRC_DIR/h2opus b=$BUILD_DIR/h2opus dst=$PREFIX/h2opus th=$PREFIX/thrust
    # Thrust and CUB headers (H2Opus's CPU backend) in one include directory
    rm -rf "$th"
    mkdir -p "$th/include"
    cp -a "$SRC_DIR/thrust/thrust" "$SRC_DIR/thrust/dependencies/cub/cub" "$th/include/"
    cp "$SRC_DIR/thrust/LICENSE" "$th/LICENSE"
    cp "$SRC_DIR/thrust/dependencies/cub/LICENSE.TXT" "$th/LICENSE.cub"
    copy_tree "$src" "$b"
    # H2Opus's make with our make.inc (variables from the environment could switch on GPU/MKL/MPI)
    unset H2OPUS_USE_GPU H2OPUS_USE_MPI H2OPUS_USE_MKL H2OPUS_USE_SINGLE_PRECISION MKLROOT
    cat > "$b/make.inc" <<EOF
# H2Opus build options written by hmatrix-bridges' deps/build_deps.sh: CPU only, double precision,
# the study's compiler flags (no -DH2OPUS_PROFILING_ENABLED).
CXX = $CXX
CXXFLAGS = -O2 -fopenmp -std=c++14 -fPIC -w
LDFLAGS = -fopenmp
BLAS_LIBS = $BLAS
H2OPUS_FMANGLE_TYPE = add
THRUST_INCDIR = $th/include
H2OPUS_DISABLE_SHARED = 1
EOF
    # target lib, not all: 'all' runs make -j$(MAKE_NP), i.e. unlimited jobs when MAKE_NP is unset
    make -C "$b" -j"$JOBS" lib
    make -C "$b" install H2OPUS_INSTALL_DIR="$dst"
    cp "$src/LICENSE" "$dst/"
    {
        config_header h2opus "H2Opus $(short "${PIN_COMMIT[h2opus]}") (${PIN_DATE[h2opus]}) + Thrust 1.17.2 ($(short "${PIN_COMMIT[thrust]}")), CPU only, static"
        cat <<EOF
# H2Opus's make with a generated make.inc: CPU only (no GPU, MAGMA, MKL or MPI), double precision,
# CXXFLAGS = -O2 -fopenmp -std=c++14 -fPIC -w (the study's flags; no H2OPUS_PROFILING_ENABLED, which
# changes inline code in H2Opus's headers), H2OPUS_FMANGLE_TYPE=add (Fortran BLAS names dgemm_).
# The generated include/h2opusconf.h defines H2OPUS_USE_DOUBLE_PRECISION and H2OPUS_FMANGLE_ADD, so
# no -D is needed. Thrust (tag 1.17.2; Apache-2.0, parts BSL-1.0/BSD-2-Clause) and its CUB submodule
# (BSD-3-Clause) share <prefix>/thrust/include.
# GOTCHA(h2opus-thrust-backend): H2Opus's headers choose Thrust's backend from _OPENMP (OpenMP
# with -fopenmp, serial without): compile every file that includes them with -fopenmp, like the
# library, or the Thrust types of the two sides differ.
H2OPUS_CPPFLAGS = -I\$(DEPS_PREFIX)/h2opus/include -I\$(DEPS_PREFIX)/thrust/include
H2OPUS_LIBS = \$(DEPS_PREFIX)/h2opus/lib/libh2opus.a
EOF
    } > "$dst/config.mk.part"
}

build_starsh() {
    local src=$SRC_DIR/starsh b=$BUILD_DIR/starsh dst=$PREFIX/starsh vendor=Generic
    case $BLAS in *openblas*) vendor=Open ;; *mkl*) vendor=Intel10_64lp_seq ;; esac
    rm -rf "$b"
    # GOTCHA(starsh-codegen-cwd): kernel sources are generated relative to cmake's working directory
    # STARS-H's configure step generates its kernel sources (spatial, electrostatics,
    # electrodynamics) by running `python ../misc_scripts/...` in the directory cmake was started
    # from, i.e. it assumes cmake runs in <source>/build. Elsewhere the generation fails, the
    # configure step still succeeds and libstarsh.a silently lacks the kernels. So cmake starts in
    # a subdirectory of the source tree (nothing is written there), with `python` = python3.
    mkdir -p "$b/bin"
    ln -sf "$(command -v python3)" "$b/bin/python"
    (
        cd "$src/src"
        PATH=$b/bin:$PATH cmake -S "$src" -B "$b" -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="$CC" -DCMAKE_Fortran_COMPILER="$FC" \
            -DCMAKE_INSTALL_PREFIX="$dst" \
            -DOPENMP=ON -DMPI=OFF -DSTARPU=OFF -DGSL=OFF -DCUDA=OFF -DKBLAS=OFF \
            -DEXAMPLES=OFF -DTESTING=OFF -DDOCS=OFF \
            -DBLA_VENDOR="$vendor"
    )
    cmake --build "$b" -j "$JOBS"
    cmake --install "$b"
    # the generated kernels are in the archive (25 files: exp, sqrexp, sin, cos, Coulomb x 1d-4d, nd)
    local syms
    syms=$(nm -g --defined-only "$dst/lib/libstarsh.a")
    grep -q ' T starsh_ssdata_block_exp_kernel_2d$' <<< "$syms"
    grep -q ' T starsh_esdata_block_coulomb_potential_kernel_3d$' <<< "$syms"
    # also the headers the study's -I<source>/include exposed beyond the installed public ones
    tar -C "$src/include" --exclude=CMakeLists.txt -cf - . | tar -C "$dst/include" -xf -
    cp "$src/LICENSE" "$dst/"
    {
        config_header starsh "STARS-H $(short "${PIN_COMMIT[starsh]}") (${PIN_DATE[starsh]}), static"
        cat <<EOF
# CMake: CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG), OPENMP=ON, MPI=OFF, STARPU=OFF, GSL=OFF, CUDA=OFF,
# KBLAS=OFF, BLA_VENDOR=$vendor, EXAMPLES/TESTING/DOCS=OFF. include/ is STARS-H's whole source
# include directory. STARS-H calls CBLAS (BLAS_LIBS), LAPACKE and, in its Fortran
# applications, the gfortran runtime.
STARSH_CPPFLAGS = -I\$(DEPS_PREFIX)/starsh/include
STARSH_LIBS = \$(DEPS_PREFIX)/starsh/lib/libstarsh.a \$(DEPS_LAPACKE_LIBS) -lgfortran -lm
EOF
    } > "$dst/config.mk.part"
}

build_h2pack() {
    local src=$SRC_DIR/h2pack b=$BUILD_DIR/h2pack dst=$PREFIX/h2pack
    copy_tree "$src" "$b"
    # INCS= drops the make file's -I../../OpenBLAS-git/install/include (a path that does not
    # exist here); cblas.h and lapacke.h come from the compiler's search path.
    make -C "$b/src" -f GCC-OpenBLAS.make -j"$JOBS" CC="$CC" INCS=
    mkdir -p "$dst/lib" "$dst/examples"
    cp "$b/lib/libH2Pack.a" "$dst/lib/"
    cp -a "$b/include" "$dst/"
    # H2Pack's file reader runs examples/meta_json_to_txt.py; the package looks for it next to
    # the include directory (mk/20-h2pack.mk, H2PACK_META_SCRIPT)
    cp "$src/examples/meta_json_to_txt.py" "$src/examples/meta_txt_to_json.py" "$dst/examples/"
    cp "$src/LICENSE" "$dst/"
    {
        config_header h2pack "H2Pack $(short "${PIN_COMMIT[h2pack]}") (${PIN_DATE[h2pack]}; ASTER $(short "${PIN_COMMIT[h2pack:src/ASTER]}")), static"
        cat <<EOF
# H2Pack's make with src/GCC-OpenBLAS.make: $CC -Wall -g -std=gnu11 -O3 -fPIC -fopenmp -march=native
# -DUSE_OPENBLAS (-march=native: runs only on CPUs with this machine's instruction set).
# -DUSE_OPENBLAS makes linalg_lib_wrapper.h include <cblas.h> and <lapacke.h> and call
# openblas_set_num_threads, so H2Pack needs OpenBLAS in BLAS_LIBS (and it resets OpenBLAS's
# thread count: GOTCHA(h2pack-blas-threads)).
# examples/ holds H2Pack's metadata converters meta_json_to_txt.py and meta_txt_to_json.py.
H2PACK_CPPFLAGS = -I\$(DEPS_PREFIX)/h2pack/include -DUSE_OPENBLAS
H2PACK_LIBS = \$(DEPS_PREFIX)/h2pack/lib/libH2Pack.a \$(DEPS_LAPACKE_LIBS) -lm
EOF
    } > "$dst/config.mk.part"
}

# Files whose presence (with BUILD_INFO) means a library is installed.
artifacts() {
    case $1 in
        htool) echo htool/include/htool/htool.hpp ;;
        h2lib) echo h2lib/lib/libh2.a h2lib/include/hmatrix.h ;;
        hmat) echo hmat/lib/libhmat.so hmat/include/hmat/hmat.h hmat/include/hmat/config.h ;;
        bpack) echo bpack/lib/libdbutterflypack.a bpack/lib/libmpi_dummy.a bpack/include/dBPACK_wrapper.h ;;
        h2opus) echo h2opus/lib/libh2opus.a h2opus/include/h2opusconf.h h2opus/include/h2opus.h thrust/include/thrust/version.h thrust/include/cub/cub.cuh ;;
        starsh) echo starsh/lib/libstarsh.a starsh/include/starsh.h ;;
        h2pack) echo h2pack/lib/libH2Pack.a h2pack/include/H2Pack.h h2pack/examples/meta_json_to_txt.py ;;
    esac
}
installed() {
    local f
    [ -f "$PREFIX/$1/BUILD_INFO" ] && [ -f "$PREFIX/$1/config.mk.part" ] || return 1
    for f in $(artifacts "$1"); do [ -e "$PREFIX/$f" ] || return 1; done
}

# signature <id>: hash of everything a build depends on: pinned commits (with submodules), the
# build step itself, compilers, cmake and the BLAS. A library whose BUILD_INFO carries the same
# signature is up to date.
SIG_TOOLS="cc=$(tool_version "$CC")|cxx=$(tool_version "$CXX")|fc=$(tool_version "$FC")|$(cmake --version | head -n1)|blas=$BLAS|lapacke=$LAPACKE|gemm_batch=$HAVE_GEMM_BATCH"
signature() {
    local r
    {
        echo "$1"
        for r in $(repos_of "$1"); do echo "${PIN_COMMIT[$r]}"; pinned_submodules "$r"; done
        declare -f "build_$1"
        echo "$SIG_TOOLS"
    } | sha256sum | cut -c1-16
}
up_to_date() { installed "$1" && grep -qx "signature: $2" "$PREFIX/$1/BUILD_INFO"; }

fmt_time() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }
du_mb() { local t=0 d; for d in "$@"; do [ -e "$d" ] && t=$((t + $(du -sk "$d" | cut -f1))); done; echo $(((t + 512) / 1024)); }

# ---- build ----------------------------------------------------------------------------------
say "build_deps.sh: prefix $PREFIX, $JOBS jobs, BLAS: $BLAS_DESC${LAPACKE:+, LAPACKE: $LAPACKE}"
say "  compilers: $(tool_version "$CC") / $(tool_version "$CXX" | cut -d' ' -f1) / $(tool_version "$FC" | cut -d' ' -f1-2); $(cmake --version | head -n1)"
printf '%-7s %-8s %-28s %8s  %s\n' library commit status time detail
FAILED=()
declare -A STATUS=()
for id in "${SELECTED[@]}"; do
    commit=${PIN_COMMIT[$id]}
    log=$BUILD_DIR/$id.log
    # GOTCHA(h2pack-blas-threads): H2Pack calls openblas_set_num_threads, so it needs OpenBLAS
    if [ "$id" = h2pack ] && [ "$HAVE_OPENBLAS_THREADS" = 0 ]; then
        STATUS[$id]="skipped: needs OpenBLAS"
        printf '%-7s %-8s %-28s %8s  %s\n' "$id" "$(short "$commit")" "${STATUS[$id]}" - "H2Pack calls openblas_set_num_threads; --blas \"$BLAS\" lacks it"
        [ -z "${FORCE[$id]:-}" ] || FAILED+=("$id")
        continue
    fi
    sig=$(signature "$id")
    if [ -z "${FORCE[$id]:-}" ] && up_to_date "$id" "$sig"; then
        STATUS[$id]="up to date (skipped)"
        printf '%-7s %-8s %-28s %8s  %s\n' "$id" "$(short "$commit")" "${STATUS[$id]}" - "$PREFIX/$id"
        continue
    fi
    t0=$(date +%s)
    : > "$log"
    set +e
    (
        set -euo pipefail
        exec >> "$log" 2>&1
        echo "== $id: ${PIN_NAME[$id]} $commit, $(date -u '+%Y-%m-%d %H:%M UTC'), build_deps.sh --jobs $JOBS"
        for r in $(repos_of "$id"); do
            if [ "$SKIP_FETCH" = 0 ]; then echo "== fetch $r"; fetch_repo "$r"; fi
            verify_repo "$r"
        done
        echo "== build $id"
        rm -rf "${PREFIX:?}/$id"
        mkdir -p "$PREFIX/$id"
        PS4='+ '
        set -x
        "build_$id"
        set +x
        for f in $(artifacts "$id"); do [ -e "$PREFIX/$f" ] || { echo "missing after install: $PREFIX/$f"; exit 1; }; done
        {
            echo "# Build record of ${PIN_NAME[$id]} for hmatrix-bridges, written by deps/build_deps.sh."
            echo "# The script skips this library while 'signature' matches the pins and options."
            for r in $(repos_of "$id"); do
                echo "repository: $r ${PIN_URL[$r]} ${PIN_COMMIT[$r]} ($(commit_date "$r"); $(commit_branches "$r"))"
                echo "licence:    ${PIN_LICENCE[$r]}"
                while read -r path sub; do
                    [ -z "$path" ] || echo "submodule:  $r/$path $sub ($(git -C "$SRC_DIR/$r/$path" show -s --format=%cs HEAD))"
                done < <(pinned_submodules "$r")
            done
            echo "signature: $sig"
            echo "built: $(date -u '+%Y-%m-%d %H:%M UTC') with $(tool_version "$CC"), BLAS $BLAS_DESC${LAPACKE:+, LAPACKE $LAPACKE}"
        } > "$PREFIX/$id/BUILD_INFO"
        echo "== done"
    )
    rc=$?
    set -e
    dt=$(($(date +%s) - t0))
    if [ $rc -eq 0 ]; then
        STATUS[$id]=built
        printf '%-7s %-8s %-28s %8s  %s\n' "$id" "$(short "$commit")" built "$(fmt_time $dt)" "log: ${log#"$REPO_DIR"/}"
    else
        STATUS[$id]=FAILED
        FAILED+=("$id")
        printf '%-7s %-8s %-28s %8s  %s\n' "$id" "$(short "$commit")" FAILED "$(fmt_time $dt)" "see $log"
        tail -n 12 "$log" | sed 's/^/        | /'
    fi
done

# ---- config.mk ------------------------------------------------------------------------------
CONFIG=$PREFIX/config.mk
STALE=()
{
    cat <<EOF
# hmatrix-bridges configuration for the pinned dependency builds under this directory.
# Written by deps/build_deps.sh on $(date -u '+%Y-%m-%d %H:%M UTC'); re-run the script rather than
# editing it. The top-level Makefile reads this file when there is no top-level config.mk; to use
# other builds, copy config.mk.example to config.mk and edit that. Commits, licences and build
# options of every library: deps/versions.txt; per-library build records: <id>/BUILD_INFO.

DEPS_PREFIX := $PREFIX

# ---- toolchain ----------------------------------------------------------------------------------
# The compilers that built the libraries below. Keep them for the package: C++ ABI (H2Opus, hmat-oss),
# Fortran runtime (ButterflyPACK, STARS-H) and OpenMP runtime must match.
#   CC:  $(tool_version "$CC")
#   FC:  $(tool_version "$FC")
CC = $CC
CXX = $CXX
FC = $FC
OPT = -O2
OPENMP = -fopenmp
# BLAS, LAPACK and CBLAS for every link line; detected by build_deps.sh (option --blas):
#   $BLAS_DESC
BLAS_LIBS = $BLAS
# LAPACKE (C interface of LAPACK), used by hmat-oss, STARS-H and H2Pack; empty when BLAS_LIBS has it.
DEPS_LAPACKE_LIBS = $LAPACKE
# Python 3 with numpy $NUMPY_VERSION (tools/h2pack)
PYTHON = $PYTHON_BIN
EOF
    for id in "${LIBS_ALL[@]}"; do
        echo
        if installed "$id"; then
            cat "$PREFIX/$id/config.mk.part"
            if [ "${STATUS[$id]:-}" != built ] && ! grep -qx "signature: $(signature "$id")" "$PREFIX/$id/BUILD_INFO"; then
                STALE+=("$id")
                echo "# WARNING: built with other pins or options than build_deps.sh would use now; rebuild it"
                echo "# with deps/build_deps.sh --only $id"
            fi
        else
            echo "# ---- ${PIN_NAME[$id]}: not built (deps/build_deps.sh --only $id; log deps/build/$id.log)"
        fi
    done
} > "$CONFIG.tmp"
mv "$CONFIG.tmp" "$CONFIG"
say "wrote $CONFIG"
[ ${#STALE[@]} -eq 0 ] || say "WARNING: built with other pins or options (rebuild with --only): ${STALE[*]}"

# ---- smoke tests: one program per library, compiled and linked only through config.mk ------
SMOKE=$BUILD_DIR/smoke
SMOKE_LOG=$BUILD_DIR/smoke.log
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
: > "$SMOKE_LOG"

cat > "$SMOKE/Makefile" <<'EOF'
# Link check of a config.mk written by deps/build_deps.sh: each program is compiled with its
# library's <LIB>_CPPFLAGS and linked the way the package links its executables:
#   $(CXX) $(OPENMP) <objects> <LIB>_LIBS $(BLAS_LIBS)
include $(CONFIG)
CSTD = -std=gnu11 $(OPT) $(OPENMP)
CXXSTD = -std=c++17 $(OPT) $(OPENMP)
smoke_htool: smoke_htool.cpp
	$(CXX) $(CXXSTD) $(HTOOL_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(BLAS_LIBS) -o $@
smoke_h2lib: smoke_h2lib.c
	$(CC) $(CSTD) $(H2LIB_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(H2LIB_LIBS) $(BLAS_LIBS) -o $@
smoke_hmat: smoke_hmat.c
	$(CC) $(CSTD) $(HMAT_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(HMAT_LIBS) $(BLAS_LIBS) -o $@
smoke_bpack: smoke_bpack.c
	$(CC) $(CSTD) $(BPACK_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(BPACK_LIBS) $(BLAS_LIBS) -o $@
smoke_h2opus: smoke_h2opus.cpp
	$(CXX) $(CXXSTD) $(H2OPUS_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@
smoke_starsh: smoke_starsh.c
	$(CC) $(CSTD) $(STARSH_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(STARSH_LIBS) $(BLAS_LIBS) -o $@
smoke_h2pack: smoke_h2pack.c
	$(CC) $(CSTD) $(H2PACK_CPPFLAGS) -c $< -o $@.o
	$(CXX) $(OPENMP) $@.o $(H2PACK_LIBS) $(BLAS_LIBS) -o $@
EOF

# Shared by the C programs: points on the unit sphere (Fibonacci lattice, point-major) and the
# smooth kernel 1 / (0.1 + |x - y|) of the study.
cat > "$SMOKE/smoke_common.h" <<'EOF'
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static void sphere_points(int n, double *p) {
    for (int i = 0; i < n; i++) {
        double z = 1 - (2.0 * i + 1) / n, r = sqrt(1 - z * z), t = 2.399963229728653 * i;
        p[3 * i] = r * cos(t), p[3 * i + 1] = r * sin(t), p[3 * i + 2] = z;
    }
}
static double kernel(const double *p, int i, int j) {
    double dx = p[3 * i] - p[3 * j], dy = p[3 * i + 1] - p[3 * j + 1], dz = p[3 * i + 2] - p[3 * j + 2];
    return 1.0 / (0.1 + sqrt(dx * dx + dy * dy + dz * dz));
}
/* relative 2-norm error of y against the dense product of the kernel with x */
static double kernel_matvec_error(int n, const double *p, const double *x, const double *y) {
    double e = 0, r = 0;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += kernel(p, i, j) * x[j];
        e += (y[i] - s) * (y[i] - s), r += s * s;
    }
    return sqrt(e / r);
}
static int report(const char *lib, const char *what, double value, double threshold) {
    printf("smoke %s: %s %.2e <= %.0e %s\n", lib, what, value, threshold, value <= threshold ? "ok" : "FAILED");
    if (!(value <= threshold)) return 1;
    printf("SMOKE_OK %s\n", lib);
    return 0;
}
EOF

cat > "$SMOKE/smoke_htool.cpp" <<'EOF'
// Htool-DDM: H-matrix of the kernel on 2,000 points (ACA, tolerance 1e-6), matvec in the user
// ordering against the dense product.
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/interfaces/virtual_generator.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <cmath>
#include <vector>
#include "smoke_common.h"

struct Kernel : htool::VirtualGenerator<double> {
    const double *p;
    explicit Kernel(const double *q) : p(q) {}
    void copy_submatrix(int M, int N, const int *rows, const int *cols, double *ptr) const override {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) ptr[i + size_t(M) * j] = kernel(p, rows[i], cols[j]);
    }
};
int main() {
    const int n = 2000;
    std::vector<double> p(3 * n), x(n), y(n);
    sphere_points(n, p.data());
    for (int i = 0; i < n; i++) x[i] = std::sin(0.37 * i);
    htool::ClusterTreeBuilder<double> ctb;
    htool::Cluster<double> t = ctb.create_cluster_tree(n, 3, p.data(), 2, 2);
    Kernel K(p.data());
    htool::HMatrixTreeBuilder<double> tb(1e-6, 10.0, 'N', 'N');
    htool::HMatrix<double> A = tb.build(K, t, t);
    htool::add_hmatrix_vector_product('N', 1.0, A, x.data(), 0.0, y.data());
    return report("htool", "H-matrix matvec rel. error (eps 1e-6)", kernel_matvec_error(n, p.data(), x.data(), y.data()), 1e-4);
}
EOF

cat > "$SMOKE/smoke_h2lib.c" <<'EOF'
/* H2Lib: LU factorization (LAPACK, USE_BLAS) of a dense 300 x 300 matrix and a solve, reading
   H2Lib's structs directly: checks that H2LIB_CPPFLAGS matches libh2.a. */
#include "basic.h"
#include "amatrix.h"
#include "avector.h"
#include "factorizations.h"
#include "smoke_common.h"
int main(int argc, char **argv) {
    init_h2lib(&argc, &argv);
    const uint n = 300;
    pamatrix a = new_amatrix(n, n), lu = new_amatrix(n, n);
    pavector x = new_avector(n), b = new_avector(n);
    random_amatrix(a);
    for (uint i = 0; i < n; i++) a->a[i + i * a->ld] += n;
    copy_amatrix(false, a, lu);
    uint info = lrdecomp_amatrix(lu);
    random_avector(x);
    clear_avector(b);
    mvm_amatrix_avector(1.0, false, a, x, b);
    lrsolve_amatrix_avector(false, lu, b);
    double e = 0;
    for (uint i = 0; i < n; i++) e = fmax(e, fabs(b->v[i] - x->v[i]));
    int layout_ok = sizeof(field) == sizeof(double) && b->dim == n && a->rows == n && a->ld >= n;
    del_avector(b), del_avector(x), del_amatrix(lu), del_amatrix(a);
    int leaked = getactives_amatrix() + getactives_avector();
    uninit_h2lib();
    if (info != 0 || !layout_ok || leaked != 0) {
        printf("smoke h2lib: lrdecomp info %u, struct layout %s, %d objects leaked: FAILED\n", info, layout_ok ? "ok" : "WRONG", leaked);
        return 1;
    }
    return report("h2lib", "LU solve max error", e, 1e-10);
}
EOF

cat > "$SMOKE/smoke_hmat.c" <<'EOF'
/* hmat-oss: assemble the kernel on 2,000 points with ACA+ (1e-6) through the C interface, then
   gemv against the dense product (checks HMAT_CPPFLAGS, libhmat.so and its rpath). */
#include <hmat/hmat.h>
#include "smoke_common.h"
static void interaction(void *data, int i, int j, void *result) { *(double *)result = kernel((const double *)data, i, j); }
int main(void) {
    const int n = 2000;
    double *p = malloc(sizeof(double) * 3 * n), *x = malloc(sizeof(double) * n), *y = calloc(n, sizeof(double));
    sphere_points(n, p);
    for (int i = 0; i < n; i++) x[i] = sin(0.37 * i);
    hmat_interface_t hmat;
    hmat_settings_t settings;
    hmat_get_parameters(&settings);
    hmat_init_default_interface(&hmat, HMAT_DOUBLE_PRECISION);
    hmat_set_parameters(&settings);
    if (hmat.init() != 0) { printf("smoke hmat: init FAILED\n"); return 1; }
    hmat_clustering_algorithm_t *algo = hmat_create_clustering_median();
    hmat_cluster_tree_t *tree = hmat_create_cluster_tree(p, 3, n, algo);
    hmat_delete_clustering(algo);
    hmat_admissibility_t *adm = hmat_create_admissibility_standard(2.0);
    hmat_matrix_t *A = hmat.create_empty_hmatrix_admissibility(tree, tree, 0, adm);
    hmat_delete_admissibility(adm);
    hmat.set_low_rank_epsilon(A, 1e-6); /* recompression tolerance; hmat-oss's default is looser */
    hmat_assemble_context_t ctx;
    hmat_assemble_context_init(&ctx);
    ctx.user_context = p;
    ctx.compression = hmat_create_compression_aca_plus(1e-6);
    ctx.simple_compute = interaction;
    ctx.lower_symmetric = 0;
    hmat.assemble_generic(A, &ctx);
    hmat_delete_compression(ctx.compression);
    double one = 1, zero = 0;
    hmat.gemv('N', &one, A, x, &zero, y, 1); /* vectors in the original ordering */
    double e = kernel_matvec_error(n, p, x, y);
    hmat.destroy(A);
    hmat_delete_cluster_tree(tree);
    hmat.finalize();
    free(p), free(x), free(y);
    return report("hmat", "H-matrix gemv rel. error (eps 1e-6)", e, 1e-4);
}
EOF

cat > "$SMOKE/smoke_bpack.c" <<'EOF'
/* ButterflyPACK: HODLR compression of the kernel on 2,000 points from its entries (sequential
   build with the MPI stub), one matvec against the dense product. ButterflyPACK works in its
   own ordering: perm is 1-based new -> old, entries are requested with 1-based original indices. */
#include "dBPACK_wrapper.h"
#include "smoke_common.h"
static double *P;
static void entry(int *m, int *n, double *val, C2Fptr q) { (void)q; *val = kernel(P, *m - 1, *n - 1); }
static void entry_block(int *a, int *b, int *c, int64_t *d, int *e, int *f, double *g, int *h, int *i, int *j, int *k, int *l, C2Fptr q) {}
static void dist(int *m, int *n, double *val, C2Fptr q) {}
static void nearfar(int *m, int *n, int *v, C2Fptr q) {}
int main(void) {
    int n = 2000, one = 1, groups[1] = {0}, dim = 3, nlevel = 0, tree[1] = {n}, nloc = 0, nvec = 1;
    MPI_Fint comm = 0;
    F2Cptr bmat, option, stats, msh, ker, ptree;
    int *perm = malloc(sizeof(int) * n);
    double *x = malloc(sizeof(double) * n), *y = malloc(sizeof(double) * n);
    double *xb = malloc(sizeof(double) * n), *yb = malloc(sizeof(double) * n);
    P = malloc(sizeof(double) * 3 * n);
    sphere_points(n, P);
    for (int i = 0; i < n; i++) x[i] = sin(0.37 * i);
    d_c_bpack_createptree(&one, groups, &comm, &ptree);
    d_c_bpack_createoption(&option);
    d_c_bpack_createstats(&stats);
    d_c_bpack_set_I_option(&option, "format", 1); /* HODLR */
    d_c_bpack_set_D_option(&option, "tol_comp", 1e-6);
    d_c_bpack_set_I_option(&option, "Nmin_leaf", 64);
    d_c_bpack_set_I_option(&option, "cpp", 1);
    d_c_bpack_set_I_option(&option, "verbosity", -1);
    d_c_bpack_construct_init(&n, &dim, P, NULL, &nlevel, tree, perm, &nloc, &bmat, &option, &stats, &msh, &ker, &ptree, dist, nearfar, NULL);
    d_c_bpack_construct_element_compute(&bmat, &option, &stats, &msh, &ker, &ptree, entry, entry_block, NULL);
    for (int i = 0; i < n; i++) xb[i] = x[perm[i] - 1];
    d_c_bpack_mult("N", xb, yb, &nloc, &nloc, &nvec, &bmat, &option, &stats, &ptree);
    for (int i = 0; i < n; i++) y[perm[i] - 1] = yb[i];
    double e = kernel_matvec_error(n, P, x, y);
    d_c_bpack_delete(&bmat);
    d_c_bpack_deletekernelquant(&ker);
    d_c_bpack_deletemesh(&msh);
    d_c_bpack_deletestats(&stats);
    d_c_bpack_deleteproctree(&ptree);
    d_c_bpack_deleteoption(&option);
    free(perm), free(x), free(y), free(xb), free(yb), free(P);
    return report("bpack", "HODLR matvec rel. error (tol 1e-6)", e, 1e-4);
}
EOF

cat > "$SMOKE/smoke_h2opus.cpp" <<'EOF'
// H2Opus: H2 matrix of the kernel on 2,000 points by Chebyshev interpolation (6^3 points per box),
// matvec in H2Opus's ordering mapped back through index_map, against the dense product.
#include <h2opus.h>
#include <h2opus/util/boxentrygen.h>
#include <cmath>
#include <vector>
#include "smoke_common.h"

struct Points : public H2OpusDataSet<H2Opus_Real> {
    const double *p; int n;
    Points(const double *q, int m) : p(q), n(m) {}
    int getDimension() const override { return 3; }
    size_t getDataSetSize() const override { return n; }
    H2Opus_Real getDataPoint(size_t i, int d) const override { return p[3 * i + d]; }
};
struct Kern {
    H2Opus_Real operator()(H2Opus_Real *x, H2Opus_Real *y) {
        double dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];
        return 1.0 / (0.1 + std::sqrt(dx * dx + dy * dy + dz * dz));
    }
};
int main() {
    const int n = 2000;
    std::vector<double> p(3 * n), x(n), y(n), xi(n), yi(n);
    sphere_points(n, p.data());
    for (int i = 0; i < n; i++) x[i] = std::sin(0.37 * i);
    h2opusHandle_t h;
    h2opusCreateHandle(&h);
    HMatrix H(n, true);
    Points pts(p.data(), n);
    H2OpusBoxCenterAdmissibility adm(1.0);
    Kern kf;
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, Kern> gen(kf);
    buildHMatrix(H, &pts, adm, gen, 64, 6);
    const int *map = vec_ptr(H.u_basis_tree.index_map);
    for (int i = 0; i < n; i++) xi[i] = x[map[i]];
    hgemv(H2Opus_NoTrans, 1.0, H, xi.data(), n, 0.0, yi.data(), n, 1, h);
    for (int i = 0; i < n; i++) y[map[i]] = yi[i];
    double e = kernel_matvec_error(n, p.data(), x.data(), y.data());
    h2opusDestroyHandle(h);
    return report("h2opus", "H2 (Chebyshev 6^3) matvec rel. error", e, 1e-3);
}
EOF

cat > "$SMOKE/smoke_starsh.c" <<'EOF'
/* STARS-H: tile low-rank approximation (tiles of 250, tolerance 1e-3) of its spatial-statistics
   exponential kernel on 2,500 points in 2D (one of the kernels generated at configure time),
   relative Frobenius error against the dense matrix (CBLAS + LAPACKE inside). */
#include <starsh.h>
#include <starsh-spatial.h>
#include "smoke_common.h"
int main(void) {
    int N = 2500, ndim = 2, maxrank = 100, onfly = 0, block_size = 250;
    char symm = 'N', dtype = 'd';
    STARSH_int shape[2] = {N, N};
    double tol = 1e-3;
    srand(0);
    STARSH_ssdata *data;
    STARSH_kernel *kernel_fn;
    STARSH_problem *problem;
    STARSH_cluster *cluster;
    STARSH_blrf *format;
    STARSH_blrm *matrix;
    if (starsh_init() ||
        starsh_application((void **)&data, &kernel_fn, N, dtype, STARSH_SPATIAL, STARSH_SPATIAL_EXP_SIMD,
                           STARSH_SPATIAL_NDIM, ndim, STARSH_SPATIAL_BETA, 0.1, STARSH_SPATIAL_NOISE, 0.0,
                           STARSH_SPATIAL_PLACE, STARSH_PARTICLES_UNIFORM, 0) ||
        starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data, kernel_fn, "smoke") ||
        starsh_cluster_new_plain(&cluster, data, N, block_size) ||
        starsh_blrf_new_tlr(&format, problem, symm, cluster, cluster) ||
        starsh_blrm_approximate(&matrix, format, maxrank, tol, onfly)) {
        printf("smoke starsh: setup FAILED\n");
        return 1;
    }
    double e = starsh_blrm__dfe_omp(matrix);
    starsh_blrm_free(matrix);
    starsh_blrf_free(format);
    starsh_cluster_free(cluster);
    starsh_problem_free(problem);
    starsh_ssdata_free(data);
    return report("starsh", "TLR rel. Frobenius error (tol 1e-3)", e, 1e-2);
}
EOF

cat > "$SMOKE/smoke_h2pack.c" <<'EOF'
/* H2Pack: H2 matrix of the 3D Coulomb kernel 1/r on 16,000 random points in a cube (proxy points,
   QR_REL_NRM 1e-6, leaves of at most 300 points so that the far-field blocks are compressed), one
   matvec in the original ordering, checked on 400 rows against the kernel. */
#include "H2Pack.h"
#include "H2Pack_kernels.h"
#include "smoke_common.h"
int main(void) {
    int n = 16000, pt_dim = 3, krnl_dim = 1, nchk = 400;
    double tol = 1e-6, L = cbrt((double)n);
    double *coord = malloc(sizeof(double) * n * pt_dim), *x = malloc(sizeof(double) * n);
    double *y = malloc(sizeof(double) * n), *A = malloc(sizeof(double) * nchk * n);
    srand48(1);
    for (int i = 0; i < n * pt_dim; i++) coord[i] = L * drand48(); /* coordinate-major: x(1:n), y(1:n), z(1:n) */
    for (int i = 0; i < n; i++) x[i] = sin(0.37 * i);
    H2Pack_p h;
    H2P_init(&h, pt_dim, krnl_dim, QR_REL_NRM, &tol);
    H2P_calc_enclosing_box(pt_dim, n, coord, NULL, &h->root_enbox);
    H2P_partition_points(h, n, coord, 300, 0.0);
    H2P_dense_mat_p *pp;
    H2P_generate_proxy_point_ID_file(h, NULL, Coulomb_3D_eval_intrin_t, NULL, &pp);
    H2P_build(h, pp, 0, NULL, Coulomb_3D_eval_intrin_t, Coulomb_3D_krnl_bimv_intrin_t, Coulomb_3D_krnl_bimv_flop);
    H2P_matvec(h, x, y);
    /* rows 0..nchk-1 of the dense matrix, row-major */
    Coulomb_3D_eval_intrin_t(coord, n, nchk, coord, n, n, NULL, A, n);
    double e = 0, r = 0;
    for (int i = 0; i < nchk; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += A[(size_t)i * n + j] * x[j];
        e += (y[i] - s) * (y[i] - s), r += s * s;
    }
    int levels = h->max_level, adm_pairs = h->n_r_adm_pair;
    H2P_destroy(&h);
    for (int l = 0; l <= levels; l++) H2P_dense_mat_destroy(&pp[l]); /* not owned by h */
    free(pp), free(coord), free(x), free(y), free(A);
    printf("smoke h2pack: %d levels, %d admissible pairs\n", levels, adm_pairs);
    if (adm_pairs < 1) return 1;
    return report("h2pack", "H2 matvec rel. error (QR_REL_NRM 1e-6)", sqrt(e / r), 1e-4);
}
EOF

say "checking config.mk: one program per library (log: $SMOKE_LOG)"
SMOKE_FAILED=()
for id in "${LIBS_ALL[@]}"; do
    installed "$id" || continue
    status=FAILED detail=
    {
        echo "== smoke_$id"
        make -C "$SMOKE" -f Makefile CONFIG="$CONFIG" "smoke_$id" && {
            ldd "$SMOKE/smoke_$id" | grep -iE 'libmpi|libmpich|libopen-pal' && echo "smoke_$id links an MPI library"
            OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout 300 "$SMOKE/smoke_$id" > "$SMOKE/$id.out" 2>&1
            echo "exit status $?"
            cat "$SMOKE/$id.out"
        }
    } >> "$SMOKE_LOG" 2>&1 || true
    if [ -f "$SMOKE/$id.out" ]; then
        detail=$(grep '^smoke ' "$SMOKE/$id.out" | tail -n1 | cut -d' ' -f3-)
        grep -q "^SMOKE_OK $id\$" "$SMOKE/$id.out" && ! grep -q "smoke_$id links an MPI library" "$SMOKE_LOG" && status=ok
    else
        detail="does not compile or link"
    fi
    [ "$status" = ok ] || SMOKE_FAILED+=("$id")
    printf '%-7s %-8s %-28s %8s  %s\n' "$id" check "$status" - "${detail:-no result (see $SMOKE_LOG)}"
done

# ---- summary --------------------------------------------------------------------------------
say "disk use (MB): sources, build trees, install"
for id in "${LIBS_ALL[@]}"; do
    extra_src=() extra_inst=()
    [ "$id" != h2opus ] || { extra_src=("$SRC_DIR/thrust"); extra_inst=("$PREFIX/thrust"); }
    printf '  %-7s %6s %6s %6s\n' "$id" "$(du_mb "$SRC_DIR/$id" "${extra_src[@]}")" \
        "$(du_mb "$BUILD_DIR/$id")" "$(du_mb "$PREFIX/$id" "${extra_inst[@]}")"
done
printf '  %-7s %6s %6s %6s\n' total "$(du_mb "$SRC_DIR")" "$(du_mb "$BUILD_DIR")" "$(du_mb "$PREFIX")"
if [ ${#FAILED[@]} -gt 0 ] || [ ${#SMOKE_FAILED[@]} -gt 0 ]; then
    say "build_deps.sh: FAILED: ${FAILED[*]:-}${SMOKE_FAILED:+ (check: ${SMOKE_FAILED[*]})}"
    exit 1
fi
say "build_deps.sh: done. Build the package with: make   (or make CONFIG=$CONFIG)"
