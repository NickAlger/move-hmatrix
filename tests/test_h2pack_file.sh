#!/usr/bin/env bash
# H2Pack file bridge. Symmetric H^2 matrices built in numpy (tools/h2pack/h2pack_write.py) are
# written in H2Pack's file format, read by H2Pack (H2P_read_from_file) and multiplied with
# H2P_matmul and H2P_matvec in tools/h2pack/h2pack_read_check, then compared with
#   - the numpy H^2 product of the same arrays: lossless transfer, <= 1e-12;
#   - the dense kernel product: approximation, <= 10 tol.
# Cases: points on a sphere (nested bases on two levels, padded bases), points on a line
# (nested bases on four levels), a rank-0 basis, the same sphere matrix through the .npz input
# format (files must be identical), and writer refusals (tests/test_h2pack_writer.py). Every
# H2Pack read must print no BLAS parameter error (zero-rank bases) and run the reader fix.
#
# Environment (exported by `make check`, mk/20-h2pack.mk): HMB_PYTHON (python3 with numpy),
# HMB_H2PACK_READ_CHECK (the checker), HMB_H2PACK_META_SCRIPT (H2Pack's
# examples/meta_json_to_txt.py). Prints HMB_TEST_PASSED only if every check passed.
set -u
here=$(cd "$(dirname "$0")" && pwd)
py=${HMB_PYTHON:-python3}
checker=${HMB_H2PACK_READ_CHECK:-$here/../build/bin/h2pack_read_check}
conv=${HMB_H2PACK_META_SCRIPT:-}
writer=$here/../tools/h2pack/h2pack_write.py
checks=0
failed=0
ok() { checks=$((checks + 1)); printf 'check %-56s ok%s\n' "$1" "${2:+  $2}"; }
bad() { checks=$((checks + 1)); failed=$((failed + 1)); printf 'check %-56s FAILED%s\n' "$1" "${2:+  $2}"; }

if [ -z "$conv" ] || [ ! -f "$conv" ]; then
  echo "test_h2pack_file: H2Pack's meta_json_to_txt.py not found (HMB_H2PACK_META_SCRIPT='$conv');"
  echo "  set H2PACK_META_SCRIPT in the configuration to <H2Pack>/examples/meta_json_to_txt.py"
  echo "HMB_TEST_FAILED test_h2pack_file: no converter"
  exit 1
fi
if [ ! -x "$checker" ]; then
  echo "test_h2pack_file: checker $checker not built"
  echo "HMB_TEST_FAILED test_h2pack_file: no checker"
  exit 1
fi
work=$(mktemp -d "${TMPDIR:-/tmp}/test_h2pack_file.XXXXXX")
trap 'rm -rf "$work"' EXIT

# read_case NAME PREFIX NRHS [TOL]: H2Pack's products vs the numpy H^2 product (lossless), and
# vs the dense kernel product when TOL is given (<= 10 TOL)
read_case() {
  local name=$1 p=$2 k=$3 tol=${4:-} out rc
  out=$("$checker" --meta "${p}_meta.json" --aux "${p}_aux.json" --bin "$p.bin" --x "${p}_x.bin" --y "${p}_y.bin" \
    --nrhs "$k" --tol 1e-12 --converter "$conv" --python "$py" 2> "$work/err.txt")
  rc=$?
  printf '%s\n' "$out" | sed 's/^/  /'
  if [ $rc -eq 0 ]; then
    ok "$name: H2Pack = numpy H^2 product" "$(printf '%s' "$out" | grep -o 'matmul.*' | cut -d' ' -f6-8)"
  else
    bad "$name: H2Pack = numpy H^2 product" "exit $rc: $(tail -n 3 "$work/err.txt" | tr '\n' ' ')"
  fi
  if grep -q "illegal value" "$work/err.txt"; then
    bad "$name: no BLAS parameter errors" "$(grep -c 'illegal value' "$work/err.txt") DGEMV errors"
  else
    ok "$name: no BLAS parameter errors"
  fi
  if printf '%s' "$out" | grep -q "reader fix applied"; then ok "$name: reader fix ran"; else bad "$name: reader fix ran"; fi
  if [ -n "$tol" ]; then
    local lim
    lim=$("$py" -c "print(10 * $tol)")
    out=$("$checker" --meta "${p}_meta.json" --aux "${p}_aux.json" --bin "$p.bin" --x "${p}_x.bin" \
      --y "${p}_ydense.bin" --nrhs "$k" --tol "$lim" --converter "$conv" --python "$py" 2>> "$work/err.txt")
    rc=$?
    if [ $rc -eq 0 ]; then
      ok "$name: vs dense kernel (<= 10 tol)" "$(printf '%s' "$out" | grep -o 'matmul.*' | cut -d' ' -f6-8)"
    else
      bad "$name: vs dense kernel (<= 10 tol)" "exit $rc: $(printf '%s' "$out" | grep matmul)"
    fi
  fi
}

# writer: summary must show nested bases and padding where expected
demo() { # name prefix args...
  local name=$1 p=$2
  shift 2
  if "$py" "$writer" demo "$p" "$@" > "$work/info.json" 2> "$work/err.txt"; then
    ok "$name: written" "$(cat "$work/info.json" | cut -c1-110)"
  else
    bad "$name: written" "$(tail -n 2 "$work/err.txt" | tr '\n' ' ')"
  fi
}
field() { "$py" -c "import json,sys; print(json.load(open('$work/info.json'))['$1'])"; }

demo "sphere n=2000" "$work/s" --n 2000 --dim 3 --leaf 64 --eta 2 --tol 1e-9 --nrhs 3 --save-npz "$work/s.npz"
lev=$(field levels) mal=$(field min_adm_level) pad=$(field padded_bases)
if [ "$mal" -le $((lev - 2)) ] && [ "$pad" -gt 0 ]; then
  ok "sphere: nested bases and padded bases" "levels $lev, min_adm_level $mal, $pad padded"
else
  bad "sphere: nested bases and padded bases" "levels $lev, min_adm_level $mal, $pad padded"
fi
read_case "sphere" "$work/s" 3 1e-9

demo "line n=1024" "$work/l" --n 1024 --dim 1 --leaf 32 --tol 1e-9 --nrhs 2
read_case "line" "$work/l" 2 1e-9

if "$py" "$writer" write "$work/s.npz" "$work/z" --qr-stop-tol 1e-9 > /dev/null 2> "$work/err.txt" &&
  cmp -s "$work/s.bin" "$work/z.bin" && cmp -s "$work/s_meta.json" "$work/z_meta.json" &&
  cmp -s "$work/s_aux.json" "$work/z_aux.json"; then
  ok "npz input writes the same files"
else
  bad "npz input writes the same files" "$(tail -n 2 "$work/err.txt" | tr '\n' ' ')"
fi

mkdir -p "$work/py"
"$py" -B "$here/test_h2pack_writer.py" "$work/py" # -B: no __pycache__ in tools/h2pack
rc=$?
checks=$((checks + 1))
[ $rc -eq 0 ] || { failed=$((failed + 1)); echo "check writer checks: $rc failed"; }
read_case "rank-0 basis" "$work/py/rank0" 2

if [ $failed -eq 0 ]; then
  echo "HMB_TEST_PASSED test_h2pack_file: $checks checks; sphere n=2000 eta=2, line n=1024, tol 1e-9"
  exit 0
fi
echo "HMB_TEST_FAILED test_h2pack_file: $failed of $checks checks failed"
exit 1
