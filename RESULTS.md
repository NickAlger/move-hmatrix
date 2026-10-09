# Results

What each transfer costs and loses, measured with the package's examples. The raw output is
in `results/logs/package/` (re-run everything with `results/run_reference.sh`); the study's
prototype runs are in `results/logs/study/` for comparison.

## Conditions

- **Machine.** 2-core VM (Intel Xeon @ 2.10 GHz), 7 GB, Linux; GCC and gfortran 13.3,
  OpenBLAS 0.3.26 with one BLAS thread. The machine was otherwise idle.
- **Libraries.** The pinned builds of `deps/build_deps.sh` (README.md lists the commits).
- **Problems.** N points on the unit sphere.
  - Copy and matvec routes, and the chain: K(x, y) = 1 / (0.1 + |x - y|), random points; the
    copy route and the chain add 10 on the diagonal, as the study did.
  - Entry route: K(x, y) = (1 + 0.5 (x_z - y_z)) / (4π sqrt(|x - y|² + h²)), h² = 4π/N, a
    non-symmetric kernel on a shuffled Fibonacci lattice.
- **Tolerances.** 1e-6 for every library unless stated (Htool-DDM's ACA, H2Lib's truncation,
  hmat-oss's ACA+, ButterflyPACK's and HARA's construction, STARS-H's TLR).
- **Unit.** Costs are times divided by the time of one matvec of the source library with one
  vector, sequential (the study's unit). A ratio carries the noise of two timings: repeated
  runs moved the ratios by 10-30%.
- **Errors** are ||y - y_ref|| / ||y_ref|| over 1-4 random vectors.
- **Threads.** One for everything timed, except where a table says otherwise.

## Summary

| Route | From -> to | N | Cost, source matvecs | Error vs source | What moved |
|---|---|---|---|---|---|
| Block copy | Htool-DDM -> H2Lib, copy / borrow / strict | 4,000 | 3.3 / 1.4 / 3.0 | 1.1e-15 | 5.6 M / 1.7 M / 5.6 M values copied |
| | | 20,000 | 2.2 / 2.1 / 4.5 | 1.1e-15 | 39.3 M / 16.2 M / 39.3 M |
| Block copy | STARS-H TLR -> H2Opus TLR, full / one triangle | 4,000 | 11.2 / 9.4 | 7.7e-17 | 4.6 M / 2.9 M |
| | | 16,000 | 2.6 / 2.6 | 6e-17 | 30.6 M / 17.8 M |
| Block copy | each of 8 sources into each of 6 destinations (48 pairs) | 1,600 | not timed | at most 1.0e-15 | `tests/test_cross_copy.cpp` |
| Conversion | H -> H² inside H2Lib, after the block copy | 4,000 / 20,000 | 68 / 60 | 3.8e-7 / 3.3e-7 | |
| Rebuild from entries | hmat-oss -> Htool-DDM | 5,000 / 20,000 | 67 / 76 | 9.4e-7 / 1.4e-6 | 1.26 / 1.33 x the source's stored values |
| Rebuild from matvecs | Htool-DDM -> ButterflyPACK HODLR | 4,000 / 16,000 | 1,097 / 1,407 | 4.3e-6 / 5.6e-6 | 4,060 / 8,414 vectors |
| | Htool-DDM -> ButterflyPACK H, tol 1e-6 | 4,000 / 16,000 | 1,522 / 1,377 | 5.3e-6 / 1.3e-5 | 7,062 / 8,566 vectors |
| | Htool-DDM -> ButterflyPACK H, tol 1e-7 | 4,000 / 16,000 | 1,450 / 1,706 | 9.9e-7 / 1.5e-6 | 6,136 / 10,987 vectors |
| | Htool-DDM -> H2Opus H² (HARA) | 4,000 / 16,000 | 2,717 / 3,321 | 2.1e-6 / 4.6e-6 | 2,065 / 4,173 vectors |
| Chain | Htool -> H2Lib -> ButterflyPACK -> H2Opus | 4,000 | 24,031 | residual of A⁻¹: 1.2e-5 | see below |

The block copy is lossless and costs a few source matvecs; the two rebuild routes cost tens and
thousands. Every rebuild here also took longer than building the destination directly from the
kernel (below), so a rebuild pays off only when the kernel is not available to the destination.

## Block copy

### Htool-DDM -> H2Lib, then H -> H² (`copy_htool_h2lib`)

Htool's export is zero-copy (dense blocks, U, V and permutations aliased). H2Lib's copy
import copies every value; the borrow import aliases dense blocks and U and copies only the
transposed V; the strict import refines the block tree for H2Lib's H -> H² (GOTCHA(strict-tree)).
Leaf size 64, eta 10, one thread (H2Lib with pardepth 0).

| Run | One source matvec | Export | + copy import | + borrow | + strict | H -> H² | Lossless | H² vs Htool / kernel |
|---|---|---|---|---|---|---|---|---|
| N = 4,000 | 3.7 ms | 0.06 | 3.27 | 1.41 | 3.00 | 68.0 | 1.05e-15 | 3.8e-7 / 8.3e-7 |
| study (E1) | 3.3 ms | 0.04 | 3.6 | 1.25 | 3.9 | 73.5 | 1.01e-15 | 4.2e-7 / - |
| N = 4,000, leaf 10 (*) | 8.0 ms | 0.53 | 3.23 | 2.76 | 5.18 | 75.3 (strict) | 1.09e-15 | 4.4e-7 / 9.1e-7 |
| study (E1) | 5.1 ms | 0.64 | 4.2 | 2.3 | 5.8 | 108 (strict) | 1.01e-15 | 4.9e-7 / - |
| N = 4,000, consistency off (**) | 3.6 ms | 0.07 | 3.52 | 1.44 | 2.91 | 65.1 (strict) | 1.05e-15 | 3.8e-7 / 8.3e-7 |
| study (E1) | 3.2 ms | 0.05 | 3.6 | 1.34 | 3.33 | 77.0 | 1.01e-15 | 4.2e-7 / - |
| N = 20,000 | 41.4 ms | 0.09 | 2.22 | 2.11 | 4.54 | 60.0 | 1.06e-15 | 3.3e-7 / 8.1e-7 |
| study (E1) | 40.6 ms | 0.09 | 2.30 | 1.16 | 2.52 | 64.4 | 1.41e-15 | 5.4e-7 / - |

Costs in source matvecs; import columns include the export. (*) Leaf size 10: Htool's ACA
falls back to dense blocks above the leaf level, 2,356 dense leaves on non-leaf clusters. The
package refuses H -> H² on the copy import and converts the strict import; the study converted
that tree and got a 57% error. (**) One-sided splits (3 super blocks keep a non-leaf cluster):
refused likewise.

Memory at N = 4,000: Htool stores 5.6 M values (0.35 N²); the copy import adds 42.7 MB
(1.006 x Htool's leaf bytes), the borrow import 13.2 MB, the H² takes 26.2 MB (0.58 x).
At N = 20,000: 39.3 M values (0.10 N²), copy import +300 MB, borrow +127 MB, H² 115 MB
(0.36 x). These, the H² ranks (sums 6,697 and 45,808, at most 31 and 34) and the values
copied are identical to the study's.

### STARS-H TLR -> H2Opus TLR (`copy_tlr_starsh_h2opus`)

STARS-H's tile low-rank approximation (tiles of 300, maxrank 100, over a recursive bisection
ordering) is exported zero-copy and imported into H2Opus's `TTLR_Matrix`, whose last tile is
padded. Unit: one STARS-H matvec.

| N | Storage | Export | Import | Cost | Values copied | H2Opus vs STARS-H | vs kernel |
|---|---|---|---|---|---|---|---|
| 4,000 | full | 0.06 ms | 18.1 ms | 11.2 | 4,572,000 | 7.7e-17 | 7.1e-7 |
| 4,000 | one triangle | 0.05 ms | 12.0 ms | 9.4 | 2,881,000 | 7.7e-17 | 7.1e-7 |
| 16,000 | full | 0.65 ms | 57.5 ms | 2.6 | 30,599,200 | 5.9e-17 | 8.6e-7 |
| 16,000 | one triangle | 0.73 ms | 35.4 ms | 2.6 | 17,784,600 | 6.0e-17 | 8.6e-7 |

The import's time is mostly first-touch page faults of H2Opus's new storage. For comparison,
H2Opus rebuilding its TLR matrix from the kernel's entries (`h2opus_tlr_from_entries`,
16 M and 256 M entries) took 698 and 728 STARS-H matvecs, with error 3.1e-7 and 3.6e-7.

### Every pair (`test_cross_copy`)

Eight sources (Htool-DDM in full and symmetric storage, H2Lib, hmat-oss in full and
lower-symmetric storage, STARS-H TLR in full and symmetric storage, H2Opus TLR) into six
destinations (Htool-DDM, H2Lib copy and strict, hmat-oss, STARS-H, H2Opus TLR), N = 1,600,
non-symmetric kernel where storage allows: every product difference is at most 1.0e-15 for
'N' and 'T', and so is every re-export. H2Opus TLR refuses the five hierarchical sources
("tile (0, 0) is subdivided") and accepts the three TLR ones.

hmat-oss's own block copy goes through its serialization stream: export 1.8-3.3 and import
3.0-7.5 hmat-oss matvecs, bit-identical round trip (measured once, on a loaded machine).

## Rebuild from entries

### hmat-oss -> Htool-DDM (`entries_hmatoss_htool`)

hmat-oss (ACA+, eta 2) is the source; Htool-DDM (ACA, eta 10) compresses again from
`hmatoss_entries`. "Direct" is Htool-DDM's own build from the kernel.

| N | Threads | Source matvec | Direct build | Transfer | Transfer / direct | Entries requested | Stored, destination / source |
|---|---|---|---|---|---|---|---|
| 5,000 | 1 | 6.8 ms | 17.5 | 67.3 | 3.85 x | 7.97 M (1.26 x source's stored values) | 1.25 |
| 5,000 | 2 | 6.6 ms | 9.4 | 36.4 | 3.86 x | | |
| study (E3) | 1 | 6.1 ms | 15.5 | 70.1 | 4.52 x | 7.97 M (1.26 x) | 1.25 |
| 20,000 | 1 | 35.7 ms | 18.2 | 75.5 | 4.15 x | 46.2 M (1.33 x) | 1.33 |
| 20,000 | 2 | 35.2 ms | 10.1 | 39.3 | 3.88 x | | |
| study (E3) | 1 | 37.9 ms | 14.9 | 69.0 | 4.49 x | 46.2 M (1.33 x) | 1.33 |

Accuracy at N = 20,000: transfer vs source 1.38e-6, vs kernel 1.53e-6; source vs kernel
8.6e-7; direct build vs kernel 8.5e-7. Entry counts, ranks (at most 30) and memory (+317 MB)
match the study exactly. Each entry costs 52 ns through hmat-oss's `get_values` against 4.3 ns
from the kernel.

Tolerances (GOTCHA(tolerance-stacking)), N = 5,000, error against the kernel:

| Source -> destination tolerance | Error | Entries (x source's stored) | Build |
|---|---|---|---|
| 1e-6 -> 1e-6 | 1.09e-6 | 1.26 x | 0.26 s |
| 1e-8 -> 1e-6 (source 100x tighter) | 5.53e-7 | 1.19 x | 0.26 s |
| 1e-6 -> 1e-8 (destination 100x tighter) | 7.03e-7 | 3.37 x | 0.63 s |
| kernel -> 1e-6 (direct) | 5.57e-7 | 1.19 x | 0.07 s |

With equal tolerances the errors add; a source 100x tighter than the destination gives the
accuracy of a direct build at the same cost; a tighter destination buys little.

## Rebuild from matvecs

### Htool-DDM -> ButterflyPACK (`matvec_htool_bpack`)

Source: Htool-DDM in symmetric storage (so every A^T request is answered with A x), eta 10,
leaf 64. ButterflyPACK: leaf 64, near_para 2, rank0 64, rankrate 2. "Entries" rows are
ButterflyPACK's own construction from the kernel's entries, the direct baseline. Cost: time /
one single-vector source matvec (3.3 ms at N = 4,000, 31.2 ms at 16,000).

| N | Construction | Tol | Calls (A^T) | Vectors or entries | Seconds | Cost | Error vs source | vs kernel | Max rank | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 4,000 | HODLR from matvecs | 1e-6 | 52 (17) | 4,060 | 3.6 | 1,097 | 4.3e-6 | 4.4e-6 | 232 | 111 |
| | H from matvecs | 1e-6 | 260 (116) | 7,062 | 5.0 | 1,522 | 5.3e-6 | 5.3e-6 | 17 | 76 |
| | H from matvecs | 1e-7 | 196 (84) | 6,136 | 4.8 | 1,450 | 9.9e-7 | 1.3e-6 | 25 | 88 |
| | HODLR from entries | 1e-6 | | 15.6 M (0.97 N²) | 1.8 | 561 | 2.5e-6 | 2.4e-6 | 198 | 87 |
| | H from entries | 1e-6 | | 40.5 M (2.53 N²) | 1.8 | 560 | 9.1e-7 | 4.3e-7 | 18 | 73 |
| 16,000 | HODLR from matvecs | 1e-6 | 70 (23) | 8,414 | 43.9 | 1,407 | 5.6e-6 | 5.7e-6 | 453 | 862 |
| | H from matvecs | 1e-6 | 358 (165) | 8,566 | 43.0 | 1,377 | 1.3e-5 | 1.3e-5 | 18 | 389 |
| | H from matvecs | 1e-7 | 361 (167) | 10,987 | 53.2 | 1,706 | 1.5e-6 | 1.8e-6 | 25 | 473 |
| | HODLR from entries | 1e-6 | | 106 M (0.41 N²) | 23.7 | 760 | 3.7e-6 | 3.6e-6 | 349 | 480 |
| | H from entries | 1e-6 | | 260 M (1.02 N²) | 12.1 | 389 | 1.2e-6 | 5.0e-7 | 21 | 391 |

The study (E2) measured, at N = 4,000 / 16,000: HODLR 934 / 1,364 source matvecs (error
4.1e-6 / 7.4e-6), H at 1e-6 1,497 / 1,380 (6.1e-6 / 7.4e-6), H at 1e-7 1,514 / 2,019
(1.1e-6 / 3.2e-6). The source's products take 55-75% of the construction time.

ButterflyPACK's matvec construction misses its tolerance, H by far at times
(GOTCHA(tolerance-stacking)): over 164 runs at N = 1,500-2,000 and tolerance 1e-6, H's error
against the source had median 4x, 90th percentile 8x, and 4 runs above 15x (worst 56x);
HODLR stayed within 0.6-1.6x at those sizes and 4-7x at 4,000-16,000. Results vary between
runs because ButterflyPACK reseeds its random generator from the clock (GOTCHA(bpack-global-state)).
Asking 10x tighter (1e-7) brought H to 1.0-1.5e-6. The adapter reports the achieved error
(`BpackStats::check_error`) after every matvec construction.

### Htool-DDM -> H2Opus H² (`matvec_htool_h2opus`)

HARA on a KD-tree of the same points (leaf 64, box-center admissibility 1), at most 512 samples
per level, absolute tolerance = 1e-6 x a sampled norm estimate. HARA asks only for A x.

| N | Calls | Vectors | Seconds | Cost | Error vs source | vs kernel | Max rank | H2Opus MB | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| 4,000 | 75 (8 for the norm) | 2,065 (0.52 N) | 8.5 | 2,717 | 2.1e-6 | 2.2e-6 | 33 | 27.3 | 241 |
| 16,000 | 141 (10) | 4,173 (0.26 N) | 106.2 | 3,321 | 4.6e-6 | 4.7e-6 | 54 | 136.7 | 988 |
| study (E2), 4,000 | 75 | 2,062 | 8.7 | 2,910 | 2.0e-6 | | | 27.3 | 240 |
| study (E2), 16,000 | 141 | 4,174 | 105.2 | 4,782 | 4.6e-6 | | | 137 | 977 |

Peak MB is the growth of the process's peak resident memory during the first construction in
the process (later ones in the same process peaked lower, 165-168 and 609-622 MB in the study).

The source's products take 12% of the time; the rest is H2Opus's own recompression. The
study's direct Chebyshev build of the same matrix in H2Opus took 2.5 s and 11.0 s and reached
only 6-9e-5. After HARA, H2Opus kept 203 MB for this 27 MB matrix; the adapter compacts it
(GOTCHA(h2opus-workspace)).

## The four-library chain (`chain_four_libraries`)

The study's "extreme example" with the libraries bridged here: construct in one library,
convert format in a second, do algebra in a third, multiply in a fourth. N = 4,000, symmetric
kernel plus 10 on the diagonal, tolerances 1e-6, one thread. Unit: one Htool-DDM matvec
(2.7 ms).

| Hop | Library and route | Seconds | Cost | What moved | Error vs previous hop | vs kernel |
|---|---|---|---|---|---|---|
| 1 | Htool-DDM builds H from the kernel's entries (ACA) | 0.09 | 33 | 5.6 M entries | - | 7.9e-7 |
| 2 | block copy Htool-DDM -> H2Lib (strict import) | 0.02 | 9 | 5.6 M values | 9.5e-16 | 7.9e-7 |
| 3 | H -> H² inside H2Lib | 0.25 | 93 | - | 4.1e-7 | 8.8e-7 |
| 4 | H2Lib's H² products -> ButterflyPACK HODLR | 8.4 | 3,103 | 4,016 vectors (1,968 A^T) | 4.8e-6 | 4.8e-6 |
| 5 | ButterflyPACK factorization | 0.19 | 68 | - | 2.0e-15 | residual 1.7e-5 |
| 6 | ButterflyPACK solves -> H2Opus H² of A⁻¹ (HARA) | 56.3 | 20,725 | 2,941 vectors | 1.0e-5 | residual 1.2e-5 |

Total 65 s, 24,031 source matvecs. The two lossless hops cost 9 and 93 source matvecs; the
two rebuilds cost 3,103 and 20,725, so the chain's cost is set by its rebuilds. Hop 6 samples
(S + Sᵀ)/2 rather than ButterflyPACK's solve S itself: S is not symmetric
(||(S - Sᵀ)x|| / ||Sx|| = 2.1e-5), and HARA builds only symmetric matrices
(GOTCHA(inverse-asymmetry)); sampling S directly gave a 4.7x larger residual. "Residual" is
||K (A⁻¹ x) - x|| / ||x|| with the exact kernel. At this size the last hop does not pay off:
applying the H2Opus inverse costs 3.5 source matvecs per vector against 1.1 for ButterflyPACK's
own solve.
