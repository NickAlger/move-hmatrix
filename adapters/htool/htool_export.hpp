// hmatrix-bridges: Htool-DDM export to the neutral view, native side (includes Htool).
//
// For code that holds htool::HMatrix objects not wrapped in an hmb::HtoolHandle (built by its
// own code, or reached through Htool's DistributedOperator on one process). Conventions,
// copies and refusals are those of hmb::htool_export in hmb_htool_copy.hpp. Do not include
// this header in a translation unit that includes H2Lib or H2Opus (see htool_native.hpp).
#pragma once
#include "hmb/neutral_store.hpp"
#include "hmb_htool_copy.hpp"
#include "htool_native.hpp"

namespace hmb {

// Fills `out` (which must be empty) with the full matrix A; the view is out.A. It reads A's
// leaves and the permutation arrays of its trees in place: A and both cluster trees must
// outlive `out` and stay unmodified. A's target and source clusters must each cover their
// whole tree (no MPI-local block row). Throws std::invalid_argument("hmb::htool_export: ...").
void htool_export(const htool::HMatrix<double> &A, NeutralStore &out, HtoolExportInfo *info = nullptr);

} // namespace hmb
