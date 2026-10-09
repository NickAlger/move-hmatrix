// move-hmatrix: hmat-oss serialization stream, structure part (internal to the adapter).
//
// hmat-oss writes a matrix as two streams through user callbacks (hmat.h write_struct /
// write_data; format in hmat-oss src/serialization.cpp, which documents nothing):
//
// structure:  int scalar type (0 S, 1 D, 2 C, 3 Z), int factorization (-1 none, 0 LU, 1 LDLt,
//             2 LLt, 3 HODLR; HODLRSYM makes write_struct throw), row cluster tree,
//             column cluster tree, block tree.
//   cluster tree: int n, int dim, double coords[n * dim] (point-major, ORIGINAL ordering),
//             int has_group_index [+ int group[n]], int perm[n] (hmat position -> original
//             index), then the nodes in pre-order: int offset, int size, int nchild
//             (a NULL node is a single int -1).
//   block node (pre-order): uchar bits (1 upper-symmetric, 2 lower-symmetric, 4/8 triangular,
//             16 keep same rows, 32 keep same columns; 128 alone = NULL child), int
//             approximate rank, int rank (>= 0 low-rank, -1 dense, -2 non-leaf,
//             -3 not assembled), double low-rank epsilon, int nchild. Child (i, j) is number
//             i + j * nr with nr = 1 if "keep same rows" else the row cluster's son count.
// data:       the leaves in pre-order, skipping NULL children and blocks with an empty row
//             or column cluster. Low-rank: int rank [, A rows x rank, B cols x rank
//             (block = A * B^T), int ortho_a, int ortho_b]. Dense: int flags (1 null,
//             2 pivots follow, 4 diagonal follows) [, D rows x cols]. Arrays are contiguous,
//             column-major.
//
// The stream writer only reads the matrix, so parsing it is the one way to see hmat-oss's
// block tree, factor layout and factorization state through the public C API.
#pragma once
#include "hmatoss_native.hpp"

#include <string>
#include <vector>

namespace hmb {
namespace hmatoss_detail {

// "who: <printf-formatted message>"
std::string fmt(const char *who, const char *f, ...) __attribute__((format(printf, 2, 3)));
// Wall-clock seconds.
double wall();

enum { UNINITIALIZED_BLOCK = -3, NONLEAF_BLOCK = -2, FULL_BLOCK = -1 };
enum : unsigned { BIT_UPPER = 1, BIT_LOWER = 2, BIT_TRIUPPER = 4, BIT_TRILOWER = 8, BIT_SAMEROWS = 16,
                  BIT_SAMECOLS = 32, BIT_NULL = 128 };
enum { STREAM_TYPE_DOUBLE = 1, STREAM_FACT_NONE = -1 };

struct SCluster {
    int offset = 0, size = 0;
    std::vector<int> kids; // indices into STree::cl
};

struct STree {
    int n = 0, dim = 0;
    std::vector<double> coords; // point-major, ORIGINAL ordering
    std::vector<int> perm;      // perm[k] = original index at hmat position k
    std::vector<SCluster> cl;   // pre-order; cl[0] is the root
};

struct SBlock {
    unsigned bits = 0;
    int rank = 0;          // >= 0 low-rank, FULL_BLOCK, NONLEAF_BLOCK or UNINITIALIZED_BLOCK
    int rc = 0, cc = 0;    // cluster indices in SStruct::rows / SStruct::cols
    int nr = 1, nc = 1;    // child grid of a non-leaf
    std::vector<int> kids; // nr * nc block indices, child (i, j) at i + j * nr; -1 = NULL; empty for a leaf
    bool leaf() const { return kids.empty(); }
};

struct SStruct {
    int type = -1, factorization = STREAM_FACT_NONE;
    STree rows, cols;
    std::vector<SBlock> bl; // pre-order; bl[0] is the root
    bool same_trees = false;
    char storage = 'N';     // root flags: 'L' lower-symmetric, 'U' upper-symmetric, 'N' full
    bool triangular = false;
};

// Runs write_struct on M and parses the stream. Throws std::runtime_error (message prefixed
// with who) if hmat-oss cannot serialize the matrix, if the stream is not double precision
// (GOTCHA(scalar-type)) or if it does not parse.
SStruct read_structure(const HmatossMatrix &M, const char *who);

// Throws std::invalid_argument unless the matrix is unfactorized (GOTCHA(factorized-state)).
void require_unfactorized(const SStruct &s, const char *who);

// Checks the handle (non-null, double-precision interface); returns the matrix.
const HmatossMatrix &checked(const HmatossHandle &A, const char *who);

} // namespace hmatoss_detail
} // namespace hmb
