#!/usr/bin/env bash
# hmatrix-bridges test runner: ./tests/run_tests.sh <executable or script> ...
#
# A check passes when it exits with status 0 AND prints a line starting with
# HMB_TEST_PASSED. Both are required: ButterflyPACK ends the process with a Fortran STOP
# (exit status 0) when its randomized construction does not converge, so status alone
# would report that failure as a pass.
#
# Output of each check goes to $HMB_LOGDIR/<name>.log (default build/test-logs).
# Environment: HMB_TEST_TIMEOUT (seconds per check, default 900), OMP_NUM_THREADS
# (default: min(nproc, 4)), HMB_TEST_FILTER (only run checks whose name matches this
# extended regular expression).
set -u
LOGDIR=${HMB_LOGDIR:-build/test-logs}
TIMEOUT=${HMB_TEST_TIMEOUT:-900}
NP=$(nproc 2>/dev/null || echo 2)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( NP < 4 ? NP : 4 ))}
# keep threaded BLAS from oversubscribing on top of OpenMP
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
mkdir -p "$LOGDIR"

if [ $# -eq 0 ]; then
  echo "run_tests.sh: nothing to run (no library configured? try 'make info')"
  exit 0
fi

pass=0; fail=0; skipped=0; failed=()
printf "%-44s %-6s %8s  %s\n" "check" "result" "seconds" "detail"
for t in "$@"; do
  name=$(basename "$t"); name=${name%.sh}; name=${name%.py}
  if [ -n "${HMB_TEST_FILTER:-}" ] && ! [[ $name =~ $HMB_TEST_FILTER ]]; then
    skipped=$((skipped + 1)); continue
  fi
  log="$LOGDIR/$name.log"
  t0=$(date +%s.%N)
  timeout "$TIMEOUT" "$t" > "$log" 2>&1
  rc=$?
  dt=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
  if [ $rc -eq 0 ] && grep -q '^HMB_TEST_PASSED' "$log"; then
    detail=$(grep '^HMB_TEST_PASSED' "$log" | tail -1 | cut -c17- | cut -c1-70)
    printf "%-44s %-6s %8s  %s\n" "$name" PASS "$dt" "$detail"
    pass=$((pass + 1))
  else
    if [ $rc -eq 124 ]; then why="timeout after ${TIMEOUT}s"
    elif [ $rc -ne 0 ]; then why="exit status $rc"
    else why="exit status 0 but no HMB_TEST_PASSED line (silent STOP/abort?)"; fi
    printf "%-44s %-6s %8s  %s\n" "$name" FAIL "$dt" "$why"
    tail -n 8 "$log" | sed 's/^/      | /'
    fail=$((fail + 1)); failed+=("$name")
  fi
done
echo
echo "passed $pass, failed $fail$( [ $skipped -gt 0 ] && echo ", filtered out $skipped" ); logs in $LOGDIR/"
[ $fail -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
