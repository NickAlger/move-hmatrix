// move-hmatrix: Htool-DDM adapter (build from entries, matvec, statistics).
#include "htool_native.hpp"

#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace hmb {

static double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

HtoolHandle htool_build(const EntrySource &entries, int dim, const double *coords, const HtoolOptions &opt,
                        HtoolStats *stats) {
    if (entries.n_rows != entries.n_cols)
        throw std::invalid_argument("hmb::htool_build: square matrices only (rows and columns share one tree)");
    if (opt.symmetry != 'N' && opt.symmetry != 'S')
        throw std::invalid_argument("hmb::htool_build: symmetry must be 'N' or 'S'");
    if (opt.symmetry == 'S' && !opt.block_tree_consistency)
        // GOTCHA(htool-sym-inconsistent): Htool only logs an error, and its products then omit
        // the missing triangle (67% error at N = 2,000)
        throw std::invalid_argument("hmb::htool_build: symmetry 'S' needs block_tree_consistency");
    const int n = entries.n_rows;
    auto h = std::make_shared<HtoolMatrix>();
    h->symmetry = opt.symmetry;
    const double t0 = wall();

    htool::ClusterTreeBuilder<double> ctb;
    ctb.set_maximal_leaf_size(opt.leaf_size);
    // binary tree; partition size 2 as in the measurements of RESULTS.md (sequential use)
    h->row_tree = std::make_unique<htool::Cluster<double>>(ctb.create_cluster_tree(n, dim, coords, 2, 2));

    HtoolEntryGenerator gen(entries);
    htool::HMatrixTreeBuilder<double> tb(opt.epsilon, opt.eta, opt.symmetry, opt.symmetry == 'S' ? 'L' : 'N');
    tb.set_block_tree_consistency(opt.block_tree_consistency);
    if (opt.parallel)
        h->A = std::make_unique<htool::HMatrix<double>>(tb.build(exec_compat::par, gen, *h->row_tree, *h->row_tree));
    else
        h->A = std::make_unique<htool::HMatrix<double>>(tb.build(exec_compat::seq, gen, *h->row_tree, *h->row_tree));
    h->build_seconds = wall() - t0;
    if (stats) *stats = htool_stats(h);
    return h;
}

BlackBox htool_blackbox(const HtoolHandle &A) {
    return blackbox_from_htool(*A->A, A->symmetry == 'S', A);
}

HtoolStats htool_stats(const HtoolHandle &A) {
    HtoolStats s;
    s.build_seconds = A->build_seconds;
    std::vector<const htool::HMatrix<double> *> leaves, sym_leaves;
    std::tie(leaves, sym_leaves) = htool::get_leaves_from(*A->A);
    for (const auto *l : leaves) {
        const long long m = l->get_target_cluster().get_size(), n = l->get_source_cluster().get_size();
        if (l->is_dense()) {
            s.stored_values += m * n;
            s.n_dense_leaves++;
        } else {
            const int r = l->get_low_rank_data()->rank_of();
            s.stored_values += (long long)r * (m + n);
            s.n_lowrank_leaves++;
            s.max_rank = std::max(s.max_rank, r);
        }
    }
    return s;
}

int htool_size(const HtoolHandle &A) { return A->A->nb_rows(); }

} // namespace hmb
