# Logs of the original study

Raw output of the three prototypes this package was built from, kept for comparison with
the package's own runs (`results/logs/package/`). They ran on the same kind of machine (a
2-core VM, OpenBLAS 0.3.26, GCC 13.3) with the same library commits, but through the
prototypes' own drivers, so option names and output formats differ from the examples'.

| Directory | Prototype | Package example |
|---|---|---|
| `copy_htool_h2lib/` | E1: Htool-DDM -> H2Lib block copy, then H -> H² in H2Lib | `examples/copy_htool_h2lib.cpp` |
| `entries_hmatoss_htool/` | E3: hmat-oss -> Htool-DDM rebuild from entries | `examples/entries_hmatoss_htool.cpp` |
| `matvec_htool_bpack_h2opus/` | E2: Htool-DDM -> ButterflyPACK and H2Opus rebuild from matvecs | `examples/matvec_htool_bpack.cpp`, `examples/matvec_htool_h2opus.cpp` |

Notable runs: `copy_htool_h2lib/leaf10_4000.log` shows H2Lib's H -> H² on a non-strict tree
(57% error, GOTCHA(strict-tree)); `entries_hmatoss_htool/pitfall_index.log` the 1-based
`get_values` (GOTCHA(index-base)); `matvec_htool_bpack_h2opus/sens_N4000_HODLR_bpdefaults_exitcode.log`
ButterflyPACK's STOP with exit status 0 (GOTCHA(fortran-stop)) and `test_errsol0.log` the
segfault without ErrSol (GOTCHA(bpack-errsol)); `summary.txt` E2's table.
