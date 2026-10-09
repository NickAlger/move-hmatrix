#!/usr/bin/env bash
# Re-runs the measurements behind RESULTS.md: every example at the study's sizes.
#
#   results/run_reference.sh [BIN_DIR] [LOG_DIR]
#
# BIN_DIR defaults to build/bin (build first with `make`), LOG_DIR to results/logs/package.
# Run on a quiet machine: the costs are ratios of wall-clock times. About 20 minutes on a
# 2-core VM; the N = 16,000-20,000 runs need about 1 GB of memory.
#
# Thread settings follow the study: one BLAS thread everywhere; one OpenMP thread for the
# matvec and TLR routes (E2 ran single-threaded); two for the entry route (as E3); the copy
# route times only single-threaded code (Htool's sequential build and product, H2Lib with
# pardepth 0), its extra thread only speeds up the exact reference products.
set -eu
BIN=${1:-build/bin}
LOG=${2:-results/logs/package}
mkdir -p "$LOG"
export OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=passive

run() { # run <threads> <log name> <program> [args...]
  local threads=$1 name=$2 prog=$3; shift 3
  if [ ! -x "$BIN/$prog" ]; then echo "skip $name ($BIN/$prog not built)"; return; fi
  printf "%-36s" "$name"
  local t0=$(date +%s)
  if OMP_NUM_THREADS=$threads nice "$BIN/$prog" "$@" > "$LOG/$name.log" 2>&1; then r=ok; else r="FAILED (status $?)"; fi
  echo "$r, $(( $(date +%s) - t0 )) s"
}

echo "machine: $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs), load $(cut -d' ' -f1-3 /proc/loadavg)" | tee "$LOG/machine.txt"

# block copy, Htool-DDM -> H2Lib, then H -> H2 (E1)
run 2 copy_htool_h2lib_n4000             copy_htool_h2lib --n 4000
run 2 copy_htool_h2lib_n4000_leaf10      copy_htool_h2lib --n 4000 --leaf 10
run 2 copy_htool_h2lib_n4000_inconsistent copy_htool_h2lib --n 4000 --inconsistent
run 2 copy_htool_h2lib_n20000            copy_htool_h2lib --n 20000
# every block-copy exporter into every importer, and the entry route from each view
run 2 test_cross_copy_n1600              test_cross_copy --n 1600
# block copy, STARS-H TLR -> H2Opus TLR
run 1 copy_tlr_starsh_h2opus_n4000       copy_tlr_starsh_h2opus --n 4000
run 1 copy_tlr_starsh_h2opus_n16000      copy_tlr_starsh_h2opus --n 16000
# rebuild from entries, hmat-oss -> Htool-DDM (E3)
run 2 entries_hmatoss_htool_n5000        entries_hmatoss_htool --n 5000
run 2 entries_hmatoss_htool_n20000       entries_hmatoss_htool --n 20000 --no-tolerance
# rebuild from matvecs, Htool-DDM -> ButterflyPACK and H2Opus (E2)
run 1 matvec_htool_bpack_n4000           matvec_htool_bpack --n 4000
run 1 matvec_htool_h2opus_n4000          matvec_htool_h2opus --n 4000
run 1 matvec_htool_bpack_n16000          matvec_htool_bpack --n 16000
run 1 matvec_htool_h2opus_n16000         matvec_htool_h2opus --n 16000
# the four-library chain: construct (Htool), convert (H2Lib), factor (ButterflyPACK), apply (H2Opus)
run 1 chain_four_libraries_n4000         chain_four_libraries --n 4000
echo "logs in $LOG/"
