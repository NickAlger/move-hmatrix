#!/usr/bin/env python3
"""h2pack_write: write a real symmetric H^2 matrix, given as plain arrays, in H2Pack's file format.

H2Pack (https://github.com/scalable-matrix/H2Pack, tested at commit 2aa58a6) stores an H^2
matrix as three files, read by its C function H2P_read_from_file:

    <prefix>_meta.json   structure: tree, block lists, matrix sizes
    <prefix>_aux.json    H2Pack's settings: point dimension, permutation, coordinates, ...
    <prefix>.bin         the numbers: bases, coupling blocks B, dense blocks D (float64)

This module builds those files from an H2Matrix (below), checking everything H2Pack's reader
assumes, and computes reference products in numpy. It is independent of hmb_hmatrix: the
neutral form has no nested bases, so H^2 matrices cross into H2Pack through files (see
README.md). Run it as a script for a demo or to convert an .npz file (``--help``).

Input format (class H2Matrix)
-----------------------------
All indices are 0-based. Arrays are numpy arrays of real numbers (converted to float64).

n
    Matrix order (rows = columns).
children
    ``children[i]`` = list of the child node ids of node i, in index order ([] for a leaf).
    The root is the one node that is nobody's child. At most 2**dim_point children per
    node (H2Pack's trees are 2^d-ary), and at most 8.
start, size
    Node i covers the permuted positions ``[start[i], start[i] + size[i])``; the children of
    a node partition its range in order; the root covers ``[0, n)``; no node is empty.
perm
    ``perm[k]`` = ORIGINAL index of the row/column at permuted position k (None = identity).
    H2Pack calls it the permutation array (coord_idx) and uses the same direction.
U
    ``U[leaf]``: leaf basis, ``size[leaf] x k_leaf``.
E
    ``E[c]``: transfer matrix of a child c to its parent p, ``k_c x k_p``. The full basis of a
    non-leaf node p is ``blockdiag(Ufull_c for c in children[p]) @ vstack(E_c ...)``; a
    non-leaf node has a basis when every child has an E (and then its rank k_p is the number
    of columns of those E).
B
    ``B[(i, j)]``: coupling matrix ``k_i x k_j`` of the admissible pair (i, j), i != j, i and
    j on the same tree level (H2Pack's "partially admissible" pairs are not written).
D
    ``D[(i, j)]``: dense block ``size[i] x size[j]`` of the inadmissible pair of LEAVES
    (i, j); every leaf has its diagonal block ``D[(i, i)]``, which must be symmetric.
coords
    Optional ``n x d`` point coordinates in the ORIGINAL ordering (H2Pack uses them only to
    recompute blocks just-in-time, which these files do not ask for); zeros if None.

Symmetric storage: give ONE block of each symmetric pair, (i, j) or (j, i), not both;
the other is its transpose (GOTCHA(symmetric-storage)). The represented matrix, in the
permuted ordering, is ``Ap[I_i, I_j] = Ufull_i @ B_ij @ Ufull_j.T`` (admissible) or
``D_ij`` (dense), and ``Ap[I_j, I_i]`` the transpose; ``A[perm[a], perm[b]] = Ap[a, b]``.
The blocks must tile the matrix exactly once. Real only: complex input is refused
(GOTCHA(scalar-type)); H2Pack is real-valued (double unless built with DTYPE_SIZE=4).

What H2Pack's reader needs, and how the writer stays clear of its bugs
(GOTCHA(h2pack-reader-bugs))
--------------------------------------------------------------------------------
* Node records are written in index order: the reader stores node i's children in the slot
  of the i-th record, whatever index the record names.
* The first ``num_leaves`` dense blocks are the leaves' diagonal blocks, then the
  off-diagonal ones; ``num_inadmissible_blocks`` counts both (the converter subtracts the
  leaves).
* Every node on a level >= min_adm_level is swept by H2P_matvec, so it gets a basis: nodes
  without one get a single zero column, and rank-0 bases are padded to one zero column (B
  blocks padded with zeros to match). An exact rewrite: with 0 columns, H2Pack calls DGEMV
  with a leading dimension of 0, which OpenBLAS reports as "parameter number 6 had an
  illegal value" (and reference BLAS's XERBLA stops the program): GOTCHA(h2pack-zero-rank).
* Admissible pairs on equal levels only, real symmetric data only, at most 2^dim_point
  children per node: H2Pack's assumptions, checked.
* The file is written as H2 (is_HSS = 0): HSS matvecs on a file-read matrix crash or give
  wrong results because the reader computes the per-node counts they use from garbage.
* What the writer CANNOT avoid: H2P_read_from_file computes one bounding box for all points
  and then reads one box per node (H2P_calc_node_inadm_lists), a heap over-read that also
  writes past node_inadm_lists when the garbage boxes look "inadmissible" (heap corruption,
  seen as "double free or corruption" at exit). Every file with more than one node triggers
  it. Programs that read these files must link the fix in h2pack_reader_fix.c (see
  README.md), as h2pack_read_check does.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["H2Matrix", "H2PackFormatError", "write_h2pack", "h2_matvec", "h2_dense", "kernel_h2",
           "save_npz", "load_npz", "write_vectors", "read_vectors"]

H2PACK_COMMIT = "2aa58a6"


class H2PackFormatError(ValueError):
    """The matrix cannot be written in H2Pack's format; the message says why."""


@dataclass
class H2Matrix:
    """A real symmetric H^2 matrix as plain arrays (see the module docstring)."""
    n: int
    children: list
    start: list
    size: list
    U: dict
    E: dict
    B: dict
    D: dict
    perm: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None


# ------------------------------------------------------------------------------------------
# structure checks
# ------------------------------------------------------------------------------------------
class _Tree:
    """Root, parents, levels, breadth-first order, leaves and ranks of an H2Matrix."""

    def __init__(self, h: H2Matrix):
        nn = len(h.children)
        if len(h.start) != nn or len(h.size) != nn:
            raise H2PackFormatError(f"children, start and size describe {nn}, {len(h.start)} and {len(h.size)} nodes")
        self.nn, self.n = nn, int(h.n)
        self.children = [[int(c) for c in ch] for ch in h.children]
        self.start = [int(s) for s in h.start]
        self.size = [int(s) for s in h.size]
        self.parent = [-1] * nn
        for i, ch in enumerate(self.children):
            for c in ch:
                if not 0 <= c < nn:
                    raise H2PackFormatError(f"node {i} has child {c}, not a node id")
                if self.parent[c] != -1 or c == i:
                    raise H2PackFormatError(f"node {c} has more than one parent")
                self.parent[c] = i
        roots = [i for i in range(nn) if self.parent[i] == -1]
        if len(roots) != 1:
            raise H2PackFormatError(f"the tree needs exactly one root, found {len(roots)}")
        self.root = roots[0]
        self.level = [-1] * nn
        self.level[self.root] = 0
        self.order = [self.root]  # breadth-first: parents before children
        for i in self.order:
            for c in self.children[i]:
                self.level[c] = self.level[i] + 1
                self.order.append(c)
        if len(self.order) != nn:
            raise H2PackFormatError("some nodes are not reachable from the root")
        if self.start[self.root] != 0 or self.size[self.root] != self.n:
            raise H2PackFormatError(f"the root covers [{self.start[self.root]}, "
                                    f"{self.start[self.root] + self.size[self.root]}), not [0, {self.n})")
        for i in range(nn):
            if self.size[i] <= 0:
                raise H2PackFormatError(f"node {i} is empty")
            nxt = self.start[i]
            for c in self.children[i]:
                if self.start[c] != nxt:
                    raise H2PackFormatError(f"the children of node {i} do not partition its range in order")
                nxt += self.size[c]
            if self.children[i] and nxt != self.start[i] + self.size[i]:
                raise H2PackFormatError(f"the children of node {i} do not cover its range")
        self.leaves = [i for i in range(nn) if not self.children[i]]
        self.max_level = max(self.level)
        self.max_children = max(len(ch) for ch in self.children)
        # ranks: a leaf with U, a non-leaf whose children all have E
        self.rank = {}
        for i in reversed(self.order):
            if not self.children[i]:
                if i in h.U:
                    self.rank[i] = h.U[i].shape[1]
            elif all(c in h.E for c in self.children[i]):
                cols = {h.E[c].shape[1] for c in self.children[i]}
                if len(cols) != 1:
                    raise H2PackFormatError(f"the transfer matrices of node {i}'s children have different numbers "
                                            f"of columns {sorted(cols)}")
                self.rank[i] = cols.pop()

    def rng(self, i):
        return slice(self.start[i], self.start[i] + self.size[i])


def _real(a, what):
    a = np.asarray(a)
    if np.iscomplexobj(a):  # GOTCHA(scalar-type): H2Pack is real-valued
        raise H2PackFormatError(f"{what} is complex; H2Pack stores real matrices only")
    if a.ndim != 2:
        raise H2PackFormatError(f"{what} must be a 2-D array (has shape {a.shape})")
    return np.ascontiguousarray(a, dtype=np.float64)


def _check(h: H2Matrix) -> _Tree:
    """Validates h for H2Pack and returns its tree; raises H2PackFormatError."""
    h.U = {int(i): _real(a, f"U[{i}]") for i, a in h.U.items()}
    h.E = {int(i): _real(a, f"E[{i}]") for i, a in h.E.items()}
    h.B = {(int(i), int(j)): _real(a, f"B[{i}, {j}]") for (i, j), a in h.B.items()}
    h.D = {(int(i), int(j)): _real(a, f"D[{i}, {j}]") for (i, j), a in h.D.items()}
    t = _Tree(h)
    if h.perm is not None:
        p = np.asarray(h.perm)
        if p.shape != (t.n,) or not np.array_equal(np.sort(p), np.arange(t.n)):
            raise H2PackFormatError("perm is not a permutation of 0 .. n-1")
    if h.coords is not None:
        c = np.asarray(h.coords)
        if c.ndim != 2 or c.shape[0] != t.n:
            raise H2PackFormatError(f"coords must be n x d (n = {t.n}), has shape {c.shape}")
    for i, u in h.U.items():
        if i not in t.leaves:
            raise H2PackFormatError(f"U[{i}] given for a node that is not a leaf (non-leaf bases come from E)")
        if u.shape[0] != t.size[i]:
            raise H2PackFormatError(f"U[{i}] has {u.shape[0]} rows, node {i} has {t.size[i]}")
    for c, e in h.E.items():
        p = t.parent[c] if 0 <= c < t.nn else -1
        if p < 0:
            raise H2PackFormatError(f"E[{c}] given for the root or for no node")
        if c not in t.rank or p not in t.rank:
            raise H2PackFormatError(f"E[{c}]: node {c} or its parent {p} has no basis")
        if e.shape != (t.rank[c], t.rank[p]):
            raise H2PackFormatError(f"E[{c}] is {e.shape[0]} x {e.shape[1]}, expected {t.rank[c]} x {t.rank[p]}")
    seen = set()
    for (i, j), b in h.B.items():
        if i == j or not (0 <= i < t.nn and 0 <= j < t.nn):
            raise H2PackFormatError(f"B[{i}, {j}]: an admissible pair needs two different nodes")
        if (min(i, j), max(i, j)) in seen:
            raise H2PackFormatError(f"B[{i}, {j}] and B[{j}, {i}] both given (symmetric storage: give one)")
        seen.add((min(i, j), max(i, j)))
        if t.level[i] != t.level[j]:
            raise H2PackFormatError(f"B[{i}, {j}] pairs levels {t.level[i]} and {t.level[j]}; H2Pack's files hold "
                                    "admissible pairs on equal levels only")
        for a in (i, j):  # the pair's nodes and their descendants need bases
            stack = [a]
            while stack:
                x = stack.pop()
                if x not in t.rank:
                    raise H2PackFormatError(f"node {x} needs a basis (it is, or is under, node {a} of the admissible "
                                            f"pair ({i}, {j}))")
                stack.extend(t.children[x])
        if b.shape != (t.rank[i], t.rank[j]):
            raise H2PackFormatError(f"B[{i}, {j}] is {b.shape[0]} x {b.shape[1]}, expected {t.rank[i]} x {t.rank[j]}")
    leafset = set(t.leaves)
    for (i, j), d in h.D.items():
        if i not in leafset or j not in leafset:
            raise H2PackFormatError(f"D[{i}, {j}]: dense blocks pair two leaves")
        if (min(i, j), max(i, j)) in seen:
            raise H2PackFormatError(f"block ({i}, {j}) given twice (B and D, or D[{i}, {j}] and D[{j}, {i}])")
        seen.add((min(i, j), max(i, j)))
        if d.shape != (t.size[i], t.size[j]):
            raise H2PackFormatError(f"D[{i}, {j}] is {d.shape[0]} x {d.shape[1]}, expected {t.size[i]} x {t.size[j]}")
        if i == j and not np.allclose(d, d.T, rtol=0, atol=1e-13 * max(1.0, float(np.abs(d).max()))):
            raise H2PackFormatError(f"D[{i}, {i}] is not symmetric (H2Pack stores symmetric matrices only)")
    missing = [i for i in t.leaves if (i, i) not in h.D]
    if missing:
        raise H2PackFormatError(f"leaves {missing[:5]}{'...' if len(missing) > 5 else ''} have no diagonal block D")
    if t.max_children > 8:
        raise H2PackFormatError(f"a node has {t.max_children} children; H2Pack's trees have at most 2^3 = 8")
    _check_tiling(t, list(seen) + [(j, i) for (i, j) in seen if i != j])
    return t


def _check_tiling(t: _Tree, blocks) -> None:
    """The blocks (both orientations) tile the matrix exactly once: descend from (root, root),
    splitting a side whenever no block spans it entirely."""
    def son_towards(node, anc):  # the child of anc on the path to its strict descendant node
        while t.parent[node] != anc:
            node = t.parent[node]
        return node

    def where(r, c):
        return (f"rows [{t.start[r]}, {t.start[r] + t.size[r]}) x columns "
                f"[{t.start[c]}, {t.start[c] + t.size[c]}) (nodes {r}, {c})")

    stack = [(t.root, t.root, blocks)]
    while stack:
        r, c, inside = stack.pop()
        if not inside:
            raise H2PackFormatError(f"{where(r, c)} are covered by no block")
        if (r, c) in inside:
            if len(inside) > 1:
                raise H2PackFormatError(f"block {where(r, c)} overlaps {len(inside) - 1} other block(s)")
            continue
        split_r = bool(t.children[r]) and all(b[0] != r for b in inside)
        split_c = bool(t.children[c]) and all(b[1] != c for b in inside)
        if not split_r and not split_c:
            raise H2PackFormatError(f"blocks inside {where(r, c)} overlap")
        buckets = {}
        for b in inside:
            key = (son_towards(b[0], r) if split_r else r, son_towards(b[1], c) if split_c else c)
            buckets.setdefault(key, []).append(b)
        for rr in (t.children[r] if split_r else [r]):
            for cc in (t.children[c] if split_c else [c]):
                stack.append((rr, cc, buckets.get((rr, cc), [])))


# ------------------------------------------------------------------------------------------
# writer
# ------------------------------------------------------------------------------------------
def _pad(a, rows, cols):
    out = np.zeros((rows, cols))
    out[:a.shape[0], :a.shape[1]] = a
    return out


def write_h2pack(h: H2Matrix, prefix: str, dim_point: Optional[int] = None, qr_stop_tol: float = 0.0) -> dict:
    """Writes h as <prefix>_meta.json, <prefix>_aux.json and <prefix>.bin.

    dim_point: H2Pack's point dimension (its trees have at most 2**dim_point children per
    node): the coordinates' dimension if h.coords is given, otherwise the smallest d >= 1 with
    2**d >= the largest number of children. qr_stop_tol is recorded in the aux file
    (informational). Returns a summary (sizes, levels, padded bases, values written).
    """
    t = _check(h)
    n, nn = t.n, t.nn
    if dim_point is None:
        dim_point = h.coords.shape[1] if h.coords is not None else max(1, math.ceil(math.log2(max(2, t.max_children))))
    if not 1 <= dim_point <= 3 or 2 ** dim_point < t.max_children:
        raise H2PackFormatError(f"dim_point = {dim_point}: H2Pack needs 1 to 3 and 2^dim_point >= {t.max_children} "
                                "children")
    if h.coords is not None and h.coords.shape[1] != dim_point:
        raise H2PackFormatError(f"coords have dimension {h.coords.shape[1]}, dim_point is {dim_point}")
    min_adm = min((t.level[i] for (i, _) in h.B), default=t.max_level + 1)

    # GOTCHA(h2pack-zero-rank): every node H2P_matvec sweeps (levels >= min_adm_level) gets a
    # basis of >= 1 column (zeros where there is none): H2Pack calls DGEMV with lda = 0 otherwise
    swept = [i for i in range(nn) if t.level[i] >= min_adm]
    k = {i: max(1, t.rank.get(i, 0)) for i in swept}
    padded = sum(1 for i in swept if t.rank.get(i, 0) == 0)
    Uw = {}
    for i in swept:
        if not t.children[i]:
            Uw[i] = _pad(h.U[i], t.size[i], k[i]) if i in t.rank else np.zeros((t.size[i], 1))
        else:
            rows = sum(k[c] for c in t.children[i])
            Uw[i] = (np.vstack([_pad(h.E[c], k[c], k[i]) for c in t.children[i]]) if i in t.rank
                     else np.zeros((rows, 1)))
    Bw = [((i, j), _pad(b, k[i], k[j])) for (i, j), b in h.B.items()]
    Dw = [((i, i), h.D[(i, i)]) for i in t.leaves] + [((i, j), d) for (i, j), d in h.D.items() if i != j]

    perm = np.arange(n) if h.perm is None else np.asarray(h.perm, dtype=np.int64)
    coords = np.zeros((n, dim_point)) if h.coords is None else np.asarray(h.coords, dtype=np.float64)
    shape = lambda i: Uw[i].shape if i in Uw else (0, 0)
    meta = dict(
        nrow_matrix=n, ncol_matrix=n, is_symmetric=1, num_node_row=nn, num_node_col=nn,
        root_node_row=t.root, root_node_col=t.root, num_level_row=t.max_level + 1, num_level_col=t.max_level + 1,
        # GOTCHA(h2pack-reader-bugs): in index order, the reader files node i's children under
        # the i-th record whatever index the record names
        nodes_row=[dict(index=i, level=t.level[i], cluster_head=t.start[i], cluster_tail=t.start[i] + t.size[i] - 1,
                        num_children=len(t.children[i]), children=t.children[i]) for i in range(nn)],
        basis_matrices_row=[dict(node=i, num_row=shape(i)[0], num_col=shape(i)[1]) for i in range(nn)],
        num_admissible_blocks=len(Bw), num_inadmissible_blocks=len(Dw), has_partial_adm_blocks=0,
        B_matrices=[dict(node_row=i, node_col=j, num_row=b.shape[0], num_col=b.shape[1], is_part_adm=0)
                    for (i, j), b in Bw],
        D_matrices=[dict(node_row=i, node_col=j, num_row=d.shape[0], num_col=d.shape[1]) for (i, j), d in Dw])
    aux = dict(dim_point=dim_point, dim_kernel=1, num_point=n, is_HSS=0, min_adm_level=min_adm,
               max_leaf_points=max(t.size[i] for i in t.leaves), QR_stop_tol=float(qr_stop_tol),
               has_skeleton_points=0, point_coordinate=[float(v) for v in coords.reshape(-1)],
               permutation_array=[int(v) for v in perm])
    values = 0
    with open(prefix + ".bin", "wb") as f:
        # row-major float64: bases in node order, then B, then D (diagonal blocks first)
        for i in range(nn):
            if i in Uw:
                f.write(np.ascontiguousarray(Uw[i], dtype="<f8").tobytes())
                values += Uw[i].size
        for _, b in Bw:
            f.write(np.ascontiguousarray(b, dtype="<f8").tobytes())
            values += b.size
        for _, d in Dw:
            f.write(np.ascontiguousarray(d, dtype="<f8").tobytes())
            values += d.size
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f)
    with open(prefix + "_aux.json", "w") as f:
        json.dump(aux, f)
    return dict(n=n, nodes=nn, levels=t.max_level + 1, leaves=len(t.leaves), min_adm_level=min_adm,
                admissible_blocks=len(Bw), dense_blocks=len(Dw), padded_bases=padded,
                max_rank=max((b.shape[0] for _, b in Bw), default=0), values=values, dim_point=dim_point,
                files=[prefix + "_meta.json", prefix + "_aux.json", prefix + ".bin"])


# ------------------------------------------------------------------------------------------
# reference products
# ------------------------------------------------------------------------------------------
def h2_matvec(h: H2Matrix, X) -> np.ndarray:
    """Y = A X in the ORIGINAL ordering (X: n or n x k), straight from the H^2 arrays."""
    t = _check(h)
    X = np.asarray(X, dtype=np.float64)
    X2 = X.reshape(t.n, -1)
    perm = np.arange(t.n) if h.perm is None else np.asarray(h.perm)
    Xp, Yp = X2[perm], np.zeros_like(X2)
    y0 = {}
    for i in reversed(t.order):  # upward: y0_i = Ufull_i^T x_i
        if i not in t.rank:
            continue
        y0[i] = (h.U[i].T @ Xp[t.rng(i)] if not t.children[i]
                 else sum(h.E[c].T @ y0[c] for c in t.children[i]))
    y1 = {}
    for (i, j), b in h.B.items():
        y1[i] = y1.get(i, 0) + b @ y0[j]
        y1[j] = y1.get(j, 0) + b.T @ y0[i]
    for i in t.order:  # downward: y_i += Ufull_i y1_i
        if i not in y1:
            continue
        if not t.children[i]:
            Yp[t.rng(i)] += h.U[i] @ y1[i]
        else:
            for c in t.children[i]:
                y1[c] = y1.get(c, 0) + h.E[c] @ y1[i]
    for (i, j), d in h.D.items():
        Yp[t.rng(i)] += d @ Xp[t.rng(j)]
        if i != j:
            Yp[t.rng(j)] += d.T @ Xp[t.rng(i)]
    Y = np.empty_like(Yp)
    Y[perm] = Yp
    return Y.reshape(X.shape)


def h2_dense(h: H2Matrix) -> np.ndarray:
    """The represented matrix, n x n, ORIGINAL ordering (for small n)."""
    return h2_matvec(h, np.eye(int(h.n)))


# ------------------------------------------------------------------------------------------
# a kernel H^2 matrix built in numpy (demo and tests)
# ------------------------------------------------------------------------------------------
def kernel_h2(points, kernel, leaf_size: int = 64, eta: float = 1.0, tol: float = 1e-8) -> H2Matrix:
    """H^2 approximation of K(points, points), nothing H2Pack-specific.

    A balanced binary tree (median bisection along the widest coordinate, every leaf on the
    same level, breadth-first numbering), admissibility max(diam) <= eta * dist of bounding
    boxes, and orthonormal nested bases from truncated SVDs of the far-field blocks (relative
    tolerance tol) computed bottom-up; nodes with an empty far field get no basis (the writer
    pads them). kernel(a, b) returns the |a| x |b| block for point arrays a and b.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    n = pts.shape[0]
    depth = max(0, math.ceil(math.log2(n / leaf_size))) if n > leaf_size else 0
    perm = np.arange(n)
    start, size, children, level = [0], [n], [[]], [0]
    for i in range(1 << (depth + 1)):  # breadth-first: node i's sons get the next two ids
        if i >= len(start) or level[i] >= depth:
            continue
        s, e = start[i], start[i] + size[i]
        sub = pts[perm[s:e]]
        d = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        perm[s:e] = perm[s:e][np.argsort(sub[:, d], kind="stable")]
        m = s + (e - s) // 2
        for a, b in ((s, m), (m, e)):
            children[i].append(len(start))
            start.append(a), size.append(b - a), children.append([]), level.append(level[i] + 1)
    nn = len(start)
    lo = [pts[perm[start[i]:start[i] + size[i]]].min(axis=0) for i in range(nn)]
    hi = [pts[perm[start[i]:start[i] + size[i]]].max(axis=0) for i in range(nn)]

    def adm(i, j):
        diam = max(np.linalg.norm(hi[i] - lo[i]), np.linalg.norm(hi[j] - lo[j]))
        gap = np.maximum(0.0, np.maximum(lo[j] - hi[i], lo[i] - hi[j]))
        dist = np.linalg.norm(gap)
        return dist > 0 and diam <= eta * dist

    far, near = [], []

    def traverse(i, j):  # i <= j; both on the same level (balanced tree)
        if i != j and adm(i, j):
            far.append((i, j))
        elif not children[i]:
            near.append((i, j))
        elif i == j:
            for a in children[i]:
                for b in children[i]:
                    if a <= b:
                        traverse(a, b)
        else:
            for a in children[i]:
                for b in children[j]:
                    traverse(min(a, b), max(a, b))

    traverse(0, 0)
    partners = {i: [] for i in range(nn)}
    for i, j in far:
        partners[i].append(j)
        partners[j].append(i)
    parent = {c: i for i in range(nn) for c in children[i]}

    def farfield(i):
        cols, x = [], i
        while x is not None:
            for j in partners[x]:
                cols.append(np.arange(start[j], start[j] + size[j]))
            x = parent.get(x)
        return np.unique(np.concatenate(cols)) if cols else np.zeros(0, dtype=int)

    P = pts[perm]  # points in the permuted order
    Ufull, U, E = {}, {}, {}
    for i in sorted(range(nn), key=lambda x: -level[x]):  # children before parents
        F = farfield(i)
        if F.size == 0:
            continue
        rows = np.arange(start[i], start[i] + size[i])
        if children[i]:
            C = np.vstack([Ufull[c].T @ kernel(P[np.arange(start[c], start[c] + size[c])], P[F]) for c in children[i]])
        else:
            C = kernel(P[rows], P[F])
        Q, s, _ = np.linalg.svd(C, full_matrices=False)
        r = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
        Qr = Q[:, :r]
        if children[i]:
            off, blocks = 0, []
            for c in children[i]:
                kc = Ufull[c].shape[1]
                E[c] = Qr[off:off + kc]
                blocks.append(Ufull[c] @ E[c])
                off += kc
            Ufull[i] = np.vstack(blocks)
        else:
            U[i] = Qr
            Ufull[i] = Qr
    rng = lambda i: slice(start[i], start[i] + size[i])
    B = {(i, j): Ufull[i].T @ kernel(P[rng(i)], P[rng(j)]) @ Ufull[j] for (i, j) in far}
    D = {(i, j): kernel(P[rng(i)], P[rng(j)]) for (i, j) in near}
    for i in range(nn):  # symmetric diagonal blocks, as H2Pack requires
        if (i, i) in D:
            D[(i, i)] = 0.5 * (D[(i, i)] + D[(i, i)].T)
    return H2Matrix(n=n, children=children, start=start, size=size, U=U, E=E, B=B, D=D, perm=perm,
                    coords=pts if pts.shape[1] <= 3 else None)


def sphere_kernel(a, b):
    """K(x, y) = 1 / (0.1 + |x - y|): the Sym kernel of the C++ tests."""
    d = np.sqrt(np.maximum(0.0, (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T))
    return 1.0 / (0.1 + d)


def demo_points(n: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    if dim == 3:  # on the unit sphere
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None]
    return rng.random((n, dim))


# ------------------------------------------------------------------------------------------
# plain-array files
# ------------------------------------------------------------------------------------------
def save_npz(h: H2Matrix, path: str) -> None:
    """Stores h as plain arrays: n, start, size, child_ptr / child_idx (children in CSR form),
    perm and coords (if any), U_<leaf>, E_<node>, B_<i>_<j>, D_<i>_<j>."""
    ptr = np.cumsum([0] + [len(c) for c in h.children])
    arrays = dict(n=np.array(h.n), start=np.asarray(h.start), size=np.asarray(h.size), child_ptr=ptr,
                  child_idx=np.array([c for ch in h.children for c in ch], dtype=np.int64))
    if h.perm is not None:
        arrays["perm"] = np.asarray(h.perm)
    if h.coords is not None:
        arrays["coords"] = np.asarray(h.coords)
    arrays.update({f"U_{i}": a for i, a in h.U.items()})
    arrays.update({f"E_{i}": a for i, a in h.E.items()})
    arrays.update({f"B_{i}_{j}": a for (i, j), a in h.B.items()})
    arrays.update({f"D_{i}_{j}": a for (i, j), a in h.D.items()})
    np.savez(path, **arrays)


def load_npz(path: str) -> H2Matrix:
    """Reads the arrays written by save_npz (the input format of `h2pack_write.py write`)."""
    z = np.load(path)
    ptr, idx = z["child_ptr"], z["child_idx"]
    children = [[int(c) for c in idx[ptr[i]:ptr[i + 1]]] for i in range(len(ptr) - 1)]
    U, E, B, D = {}, {}, {}, {}
    for key in z.files:
        kind, _, rest = key.partition("_")
        if kind in ("U", "E") and rest.isdigit():
            (U if kind == "U" else E)[int(rest)] = z[key]
        elif kind in ("B", "D") and rest.count("_") == 1:
            i, j = rest.split("_")
            (B if kind == "B" else D)[(int(i), int(j))] = z[key]
    return H2Matrix(n=int(z["n"]), children=children, start=list(z["start"]), size=list(z["size"]), U=U, E=E, B=B,
                    D=D, perm=z["perm"] if "perm" in z.files else None,
                    coords=z["coords"] if "coords" in z.files else None)


def write_vectors(path: str, X) -> None:
    """n x k vectors as raw little-endian float64, column-major (what h2pack_read_check reads)."""
    np.asarray(X, dtype="<f8").reshape(X.shape[0], -1).T.tofile(path)


def read_vectors(path: str, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(-1, n).T


# ------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="build a kernel H^2 matrix in numpy, write it and reference products")
    d.add_argument("prefix")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--dim", type=int, default=3, choices=(1, 2, 3), help="points: 1-D/2-D random, 3-D on a sphere")
    d.add_argument("--leaf", type=int, default=64)
    d.add_argument("--eta", type=float, default=1.0)
    d.add_argument("--tol", type=float, default=1e-8)
    d.add_argument("--nrhs", type=int, default=3)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--save-npz", help="also store the H^2 arrays in this .npz file")
    w = sub.add_parser("write", help="write an H^2 matrix stored as plain arrays (see save_npz)")
    w.add_argument("npz")
    w.add_argument("prefix")
    w.add_argument("--dim-point", type=int)
    w.add_argument("--qr-stop-tol", type=float, default=0.0, help="recorded in the aux file (informational)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "demo":
            pts = demo_points(a.n, a.dim, a.seed)
            h = kernel_h2(pts, sphere_kernel, leaf_size=a.leaf, eta=a.eta, tol=a.tol)
            info = write_h2pack(h, a.prefix, qr_stop_tol=a.tol)
            X = np.random.default_rng(a.seed + 1).standard_normal((a.n, a.nrhs))
            write_vectors(a.prefix + "_x.bin", X)
            write_vectors(a.prefix + "_y.bin", h2_matvec(h, X))                   # the H^2 product
            write_vectors(a.prefix + "_ydense.bin", sphere_kernel(pts, pts) @ X)  # the kernel's product
            if a.save_npz:
                save_npz(h, a.save_npz)
        else:
            info = write_h2pack(load_npz(a.npz), a.prefix, dim_point=a.dim_point, qr_stop_tol=a.qr_stop_tol)
    except H2PackFormatError as e:
        print(f"h2pack_write: {e}", file=sys.stderr)
        return 2
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
