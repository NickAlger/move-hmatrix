# H2Pack file bridge

Moves real symmetric H² matrices into [H2Pack](https://github.com/scalable-matrix/H2Pack)
(tested at commit 2aa58a6) through H2Pack's own file format.

| File | What it is |
|---|---|
| `h2pack_write.py` | Python module and command-line tool: writes an H² matrix given as plain numpy arrays in H2Pack's format (`<prefix>_meta.json`, `<prefix>_aux.json`, `<prefix>.bin`), after checking everything H2Pack's reader assumes; computes reference products in numpy. |
| `h2pack_read_check.c` | C program: reads a file set with `H2P_read_from_file`, multiplies with `H2P_matmul` and `H2P_matvec`, compares with reference vectors. |
| `h2pack_reader_fix.c`, `.h` | Link-time fix for a memory bug in `H2P_read_from_file`, for any program that reads these files. |

Built by `mk/20-h2pack.mk`; tested by `tests/test_h2pack_file.sh` (with
`tests/test_h2pack_writer.py`).

## Why a file bridge, outside `hmb_hmatrix`

The package's neutral form (`include/hmb/neutral.h`) stores each low-rank leaf with its own
factors; it has no nested bases, so it cannot carry an H² matrix. H2Pack, in turn, builds H²
matrices only from a kernel and points; its one entry point for a matrix made elsewhere is its
file reader. So this bridge writes H2Pack's files directly from the H² arrays (leaf bases,
transfer matrices, coupling and dense blocks) and stands apart from `hmb_hmatrix` and the
adapters.

## Input format

An `H2Matrix` (see the docstring of `h2pack_write.py` for the exact rules):

- `n`; a cluster tree: `children[i]` (in index order), `start[i]`, `size[i]` (node i covers
  the permuted positions `[start, start + size)`), one root covering `[0, n)`;
- `perm[k]` = ORIGINAL index at permuted position k (`None` = identity);
- `U[leaf]` (`size x k`), `E[c]` (`k_c x k_parent`, so that the basis of a node is
  `blockdiag(children's bases) @ vstack(E_c)`), `B[(i, j)]` (`k_i x k_j`, admissible pairs on the
  same level), `D[(i, j)]` (`size_i x size_j`, pairs of leaves, every leaf's diagonal block,
  symmetric);
- one block of each symmetric pair; the blocks must tile the matrix exactly once;
- optional `coords` (`n x d`, original ordering).

The same arrays in an `.npz` file (`save_npz` / `load_npz`): `n`, `start`, `size`,
`child_ptr`/`child_idx` (children in CSR form), optional `perm` and `coords`, and one array per
block: `U_<leaf>`, `E_<node>`, `B_<i>_<j>`, `D_<i>_<j>`.

## Usage

```sh
# convert an .npz file
python3 tools/h2pack/h2pack_write.py write matrix.npz out/A
# demo: kernel 1/(0.1+r) on 2000 points of a sphere, H^2 built in numpy, plus references
python3 tools/h2pack/h2pack_write.py demo out/S --n 2000 --dim 3 --eta 2 --tol 1e-9 --nrhs 3
# read with H2Pack and compare with the numpy H^2 product (out/S_ydense.bin: dense kernel)
build/bin/h2pack_read_check --meta out/S_meta.json --aux out/S_aux.json --bin out/S.bin \
    --x out/S_x.bin --y out/S_y.bin --nrhs 3 --tol 1e-12 \
    --converter <H2Pack>/examples/meta_json_to_txt.py --python python3
```

```python
import h2pack_write as hw
h = hw.H2Matrix(n=n, children=children, start=start, size=size, U=U, E=E, B=B, D=D, perm=perm)
info = hw.write_h2pack(h, "out/A")      # raises hw.H2PackFormatError with the reason
Y = hw.h2_matvec(h, X)                  # reference product, ORIGINAL ordering
```

Vectors for the checker are raw little-endian float64, `n x k` column-major
(`write_vectors` / `read_vectors`).

## H2Pack's reader: what the files must satisfy, and its bugs

The writer guarantees (GOTCHA ids as in the package):

- node records in index order (the reader files the children of the i-th record under node i);
  the leaves' diagonal blocks first among the dense blocks; row-major float64 data;
- at most 2^d children per node with d = `dim_point` (1 to 3); admissible pairs on equal levels
  (H2Pack's "partially admissible" pairs are not written); real (`scalar-type`) and
  symmetric (`symmetric-storage`) data only;
- every node on a level >= `min_adm_level` has a basis with at least one column: nodes without
  one get a zero column, rank-0 bases are padded (an exact rewrite). With zero columns H2Pack
  calls DGEMV with a leading dimension of 0: OpenBLAS prints "parameter number 6 had an illegal
  value", reference BLAS stops the program;
- H2 files only (`is_HSS = 0`), AOT (B and D stored; no skeleton points, so H2Pack's
  just-in-time mode is not available for these files). A matrix without admissible blocks is
  written with `min_adm_level` = number of levels (no basis sweeps).

What the writer cannot avoid (`h2pack-reader-bugs`): `H2P_read_from_file` computes one bounding
box for all points, then `H2P_calc_node_inadm_lists` reads one box per node from it (a heap
over-read for every file with more than one node) and writes the nodes it finds
"H2-inadmissible but HSS-admissible" from that garbage into `node_inadm_lists`, past the end of
the array when there are more than 3^d. The result is heap corruption that shows up or not
depending on the heap layout: with H2Pack's own reader, the study's 1-D file had 2 to 29 nodes
over the limit in every run, and the program aborted with "double free or corruption (out)" or
finished normally depending on the OpenMP thread count and on whether stdout was a terminal, a
pipe or a file. The lists matter only for HSS matrices, whose products then crash or are wrong. **Every program that reads these files must link `h2pack_reader_fix.o` with
`-Wl,--wrap=H2P_calc_node_inadm_lists`** (a GNU-compatible linker and a static `libH2Pack.a`;
calls inside a shared library are not redirected): the reader then gets zeroed lists and the
per-node admissible-pair counts it computed itself, which is what H2 and HSS products need.
`hmb_h2pack_reader_fix_calls()` tells whether the fix ran; `h2pack_read_check` refuses to report
results otherwise.

Other properties of the reader (`h2pack-cwd-scripts`): it runs `./meta_json_to_txt.py` from
the current directory through `system()` (Python from `PATH`, via the script's shebang), writes
`tmp_metadata.txt` there under that fixed name (two readers in one directory collide), and
returns without setting its output when the script is missing. `h2pack_read_check` runs the
read in a private temporary directory containing a wrapper that runs H2Pack's converter with a
chosen interpreter, and passes absolute file names. H2Pack's construction routines also set the
BLAS thread count (`h2pack-blas-threads`); the checker saves and restores it (reading and
multiplying do not change it).

## Tested

`tests/test_h2pack_file.sh`: 3-D points on a sphere (n = 2000, nested bases on two levels,
8 padded bases), 1-D points (n = 1024, nested bases on four levels), a rank-0 basis, and the
`.npz` route (identical files); H2Pack's products equal the numpy H² products to
0 (`H2P_matmul`) and 5e-16 (`H2P_matvec`), and the dense kernel products to 7e-10 and 1e-9
(tolerance 1e-9); refusals of invalid inputs. Python 3.13, numpy 2.5.3, H2Pack 2aa58a6 with
OpenBLAS 0.3.26.

## Limits

Real symmetric matrices only (H2Pack's format and products); square; one point dimension of
1 to 3 (at most 8 children per node); H2 only (HSS files are not written: the reader's HSS
support needs the fix above and HSS-specific pairs); no partially admissible pairs;
`int`-sized indices (H2Pack's). H2Pack's `DTYPE` must be double (the checker checks it at
compile time).
