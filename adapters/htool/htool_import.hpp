// move-hmatrix: Htool-DDM import from the neutral view, native side (includes Htool).
//
// hmb::htool_import (hmb_htool_copy.hpp) returns the trees and the matrix in an HtoolHandle,
// whose fields htool_native.hpp exposes. This header adds the cluster-tree half on its own,
// for code that wants another library's ordering and tree in Htool, e.g. to build an Htool
// matrix with Htool's own compressor on the tree of the matrix it replaces. Do not include
// this header in a translation unit that includes H2Lib or H2Opus (see htool_native.hpp).
#pragma once
#include "hmb_htool_copy.hpp"
#include "htool_native.hpp"

#include <memory>

namespace hmb {

// The cluster tree of one side of a (valid) view: side 0 = rows, 1 = columns. The returned
// root holds the side's permutation (Htool's perm[k] = user index at tree position k, the
// neutral convention) and the clusters' structure, with the conventions of htool_import:
// the root's sons are the partition clusters, empty clusters are left out, a leaf root gets
// one partition cluster below it with the same range, and balls enclose the view's boxes.
// The tree owns its data; the view may be freed afterwards. Throws std::invalid_argument
// ("hmb::htool_import_cluster_tree: ...") for an invalid view, side, or an empty side.
std::unique_ptr<htool::Cluster<double>> htool_import_cluster_tree(const hmb_hmatrix &A, int side);

} // namespace hmb
