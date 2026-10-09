# Notes of each part's authors

The reports written by whoever built each part of the package, when it was built: what
they made, how they tested it, what they measured and which library quirks they found. They
are a record, not documentation: file names, line numbers and some APIs have moved since
(for example, the LeakSanitizer suppressions are now merged in `tests/lsan.supp`). The
current state is in README.md, GOTCHAS.md, RESULTS.md, LAYOUT.md and the header comments.

| File | Part |
|---|---|
| `deps.md` | `deps/build_deps.sh`: pinned builds of the seven libraries |
| `copy.md` | neutral core tests, Htool-DDM block copy, H2Lib adapter, `copy_htool_h2lib` |
| `entries.md` | hmat-oss adapter, `entries_hmatoss_htool` |
| `matvec.md` | ButterflyPACK and H2Opus HARA adapters, `matvec_htool_*`, `chain_four_libraries` |
| `tlr_h2pack.md` | STARS-H and H2Opus TLR adapters, H2Pack file bridge, `copy_tlr_starsh_h2opus` |
