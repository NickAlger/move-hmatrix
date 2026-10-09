#!/usr/bin/env python3
"""Checks of tools/h2pack/h2pack_write.py that need no H2Pack (run by test_h2pack_file.sh):
the numpy H^2 product against the dense expansion, the .npz round trip, the padding of
zero-rank bases, and the inputs the writer must refuse (each with its reason).
Prints one line per check; the exit status is the number of failed checks.
Usage: test_h2pack_writer.py <work directory>"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "h2pack"))
import h2pack_write as hw  # noqa: E402

failures = 0


def check(ok, what, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"check {what:<56} {'ok' if ok else 'FAILED'}{'  ' + detail if detail else ''}")


def refused(h, want, what, **kw):
    try:
        hw.write_h2pack(h, os.path.join(work, "refused"), **kw)
        check(False, what, "accepted")
    except hw.H2PackFormatError as e:
        check(want in str(e), what, str(e))


work = sys.argv[1]
pts = hw.demo_points(300, 1, 3)
base = hw.kernel_h2(pts, hw.sphere_kernel, leaf_size=20, eta=1.0, tol=1e-10)
dense = hw.sphere_kernel(pts, pts)
A = hw.h2_dense(base)
X = np.random.default_rng(1).standard_normal((300, 2))
check(np.allclose(hw.h2_matvec(base, X), A @ X, rtol=0, atol=1e-12 * np.abs(A @ X).max()), "h2_matvec = dense expansion")
err = np.linalg.norm(A - dense) / np.linalg.norm(dense)
check(err <= 1e-9, "H^2 approximation of the kernel (<= 10 tol)", f"{err:.2e}")
check(np.allclose(A, A.T, rtol=0, atol=1e-14 * np.abs(A).max()), "represented matrix symmetric")

npz = os.path.join(work, "base.npz")
hw.save_npz(base, npz)
back = hw.load_npz(npz)
check(np.array_equal(hw.h2_dense(back), A), "save_npz / load_npz round trip")

# a node without basis and a rank-0 basis are padded with zero columns, exactly
info = hw.write_h2pack(base, os.path.join(work, "base"))
check(info["padded_bases"] > 0, "nodes swept by H2P_matvec without a basis padded", f"{info['padded_bases']} padded")
z = copy.deepcopy(base)
r0 = next(i for i in z.U if z.U[i].shape[1] > 0)  # a leaf whose basis becomes rank 0
z.U[r0] = z.U[r0][:, :0]
if r0 in z.E:
    z.E[r0] = z.E[r0][:0, :]
for (i, j) in list(z.B):
    z.B[(i, j)] = z.B[(i, j)][:0 if i == r0 else None, :0 if j == r0 else None]
info0 = hw.write_h2pack(z, os.path.join(work, "rank0"))
check(info0["padded_bases"] == info["padded_bases"] + 1, "rank-0 basis padded", f"{info0['padded_bases']} padded")
hw.write_vectors(os.path.join(work, "rank0_x.bin"), X)  # read back by H2Pack in test_h2pack_file.sh
hw.write_vectors(os.path.join(work, "rank0_y.bin"), hw.h2_matvec(z, X))

# refusals
h = copy.deepcopy(base)
i = next(iter(h.D))
h.D[i] = h.D[i] + 0j
refused(h, "complex", "complex block refused (scalar-type)")
leaves = [k for k, ch in enumerate(base.children) if not ch]
h = copy.deepcopy(base)
h.D[(leaves[0], leaves[0])] = h.D[(leaves[0], leaves[0])] + np.triu(np.ones_like(h.D[(leaves[0], leaves[0])]), 1)
refused(h, "not symmetric", "non-symmetric diagonal block refused")
h = copy.deepcopy(base)
(a, b), blk = next(iter(h.B.items()))
h.B[(b, a)] = blk.T
refused(h, "both given", "both blocks of a symmetric pair refused")
h = copy.deepcopy(base)
del h.D[(leaves[1], leaves[1])]
refused(h, "no diagonal block", "missing diagonal block refused")
h = copy.deepcopy(base)
(a, b), blk = next(iter(h.B.items()))
del h.B[(a, b)]
refused(h, "covered by no block", "missing block refused")
h = copy.deepcopy(base)
p = next(k for k, ch in enumerate(h.children) if ch and all(not h.children[c] for c in ch))  # parent of two leaves
h.D[(p, p)] = np.zeros((h.size[p], h.size[p]))
refused(h, "pair two leaves", "dense block on a non-leaf refused")
h = copy.deepcopy(base)
(a, b), blk = next((k, v) for k, v in h.B.items() if h.children[k[0]])
c = h.children[a][0]
h.B[(c, b)] = np.zeros((blk.shape[0], blk.shape[1]))
refused(h, "equal levels", "admissible pair on two levels refused")
# a root with 3 children needs dim_point >= 2; 9 children are too many for H2Pack
three = hw.H2Matrix(n=3, children=[[1, 2, 3], [], [], []], start=[0, 0, 1, 2], size=[3, 1, 1, 1], U={}, E={},
                    B={}, D={(x, y): np.ones((1, 1)) for x in (1, 2, 3) for y in (1, 2, 3) if x <= y})
refused(three, "2^dim_point", "3 children with dim_point 1 refused", dim_point=1)
nine = hw.H2Matrix(n=9, children=[list(range(1, 10))] + [[] for _ in range(9)], start=[0] + list(range(9)),
                   size=[9] + [1] * 9, U={}, E={}, B={},
                   D={(x, y): np.ones((1, 1)) for x in range(1, 10) for y in range(1, 10) if x <= y})
refused(nine, "at most", "9 children refused")
h = copy.deepcopy(base)
h.perm = np.zeros(300, dtype=int)
refused(h, "not a permutation", "bad permutation refused")
sys.exit(failures)
