// hmatrix-bridges: Htool-DDM block copy (export to and import from the neutral view).
// Conventions and limits: hmb_htool_copy.hpp.
#include "htool_export.hpp"
#include "htool_import.hpp"

#include <htool/hmatrix/interfaces/virtual_lrmat_generator.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hmb {
namespace {

using Cl = htool::Cluster<double>;
using HM = htool::HMatrix<double>;

[[noreturn]] void fail(const char *fn, const std::string &why) { throw std::invalid_argument(std::string(fn) + ": " + why); }

std::string range(int offset, int size) {
    return "[" + std::to_string(offset) + ", " + std::to_string(offset + size) + ")";
}

// ---------------------------------------------------------------------------------------
// Export: a walk of Htool's block tree. Every block of the view is an Htool node or, under
// symmetric storage, the mirror image of one; where Htool's children skip a cluster level,
// the view gets intermediate blocks.
// ---------------------------------------------------------------------------------------
class Exporter {
    static constexpr const char *kFn = "hmb::htool_export";
    struct Node {
        const HM *h;
        bool tr;    // the block is the transpose of h (a mirror image under symmetric storage)
        int rc, cc; // clusters of the block in the view
    };
    NeutralStore &S;
    HtoolExportInfo &info;
    int dim = 0;
    bool mirror = false, one_tree = false;
    std::unordered_map<const Cl *, int> id_; // Htool cluster -> cluster id in the view
    std::vector<const Cl *> node_;           // cluster id -> Htool cluster

  public:
    Exporter(NeutralStore &s, HtoolExportInfo &i) : S(s), info(i) {}

    void run(const HM &A) {
        const Cl &rt = A.get_target_cluster(), &ct = A.get_source_cluster();
        if (A.get_symmetry() != 'N' && A.get_symmetry_for_leaves() == 'N')
            // GOTCHA(htool-sym-inconsistent): Htool flags the root for mirroring only when one of
            // its children is symmetric; with block_tree_consistency off the root splits one-sided
            // into non-symmetric children, and Htool's own products then skip the missing triangle.
            fail(kFn, "symmetric storage whose root is not flagged for mirroring (built with "
                      "block_tree_consistency off?): Htool's own products ignore the missing triangle; "
                      "rebuild with block_tree_consistency on");
        dim = rt.get_center().size() <= 3 ? (int)rt.get_center().size() : 0;
        const int rroot = add_tree(rt), croot = add_tree(ct);
        // Htool's products apply transposed every leaf that has a symmetric ancestor and
        // different row and column offsets, if the root's symmetry_for_leaves is set
        // (add_hmatrix_vector_product.hpp, L88-103); here the children of each symmetric
        // node with different offsets are mirrored as a whole, which is the same set of leaves.
        mirror = A.get_symmetry_for_leaves() != 'N';
        one_tree = &rt == &ct;
        emit(Node{&A, false, rroot, croot}, false);
        S.A.dim = dim;
        S.A.n_rows = rt.get_size();
        S.A.n_cols = ct.get_size();
        S.A.row_root = rroot;
        S.A.col_root = croot;
        S.finalize();
        S.A.row_perm = rt.get_permutation().data();
        S.A.col_perm = ct.get_permutation().data();
    }

  private:
    std::string region(int rc, int cc) const {
        return "rows " + range(S.cl[rc].offset, S.cl[rc].size) + " x columns " + range(S.cl[cc].offset, S.cl[cc].size);
    }

    int new_cluster(const Cl &c) {
        hmb_cluster n{};
        n.offset = c.get_offset();
        n.size = c.get_size();
        const auto &ctr = c.get_center();
        for (int d = 0; d < dim && d < (int)ctr.size(); d++) {
            n.bmin[d] = ctr[d] - c.get_radius();
            n.bmax[d] = ctr[d] + c.get_radius();
        }
        id_[&c] = (int)S.cl.size();
        node_.push_back(&c);
        S.cl.push_back(n);
        return (int)S.cl.size() - 1;
    }

    // Numbers a tree breadth-first, so that the sons of each cluster get consecutive ids.
    int add_tree(const Cl &root) {
        auto it = id_.find(&root);
        if (it != id_.end()) return it->second; // rows and columns share the tree
        if (root.get_offset() != 0 || root.get_size() != (int)root.get_permutation().size())
            fail(kFn, "the matrix covers " + range(root.get_offset(), root.get_size()) + " of a cluster tree of size " +
                          std::to_string(root.get_permutation().size()) + " (an MPI-local block row?)");
        const int rid = new_cluster(root);
        std::deque<const Cl *> queue{&root};
        while (!queue.empty()) {
            const Cl *c = queue.front();
            queue.pop_front();
            const int cid = id_.at(c), s0 = (int)S.cl.size();
            int next = c->get_offset();
            for (const auto &s : c->get_children()) {
                if (s->get_offset() != next)
                    fail(kFn, "the sons of cluster " + range(c->get_offset(), c->get_size()) + " are not contiguous");
                next += s->get_size();
                new_cluster(*s);
                queue.push_back(s.get());
            }
            if (!c->get_children().empty() && next != c->get_offset() + c->get_size())
                fail(kFn, "the sons of cluster " + range(c->get_offset(), c->get_size()) + " do not cover it");
            S.cl[cid].nsons = (int)c->get_children().size();
            S.cl[cid].son0 = S.cl[cid].nsons ? s0 : 0;
        }
        return rid;
    }

    // Id of cluster c, a son of cluster p, or p itself, or (Htool skipped a level) deeper.
    int cluster_of(int p, const Cl &c) const {
        if (node_[p] == &c) return p;
        const hmb_cluster &q = S.cl[p];
        for (int s = 0; s < q.nsons; s++)
            if (node_[q.son0 + s] == &c) return q.son0 + s;
        auto it = id_.find(&c);
        if (it == id_.end()) fail(kFn, "a block uses a cluster outside the matrix's cluster trees");
        return it->second;
    }

    // Index of the son of cluster p that contains cluster x, or -1 (x == p, or no such son).
    int son_index(int p, int x) const {
        if (x == p) return -1;
        const hmb_cluster &q = S.cl[x];
        for (int s = 0; s < S.cl[p].nsons; s++) {
            const hmb_cluster &c = S.cl[S.cl[p].son0 + s];
            if (c.offset <= q.offset && q.offset + q.size <= c.offset + c.size) return s;
        }
        return -1;
    }

    int new_block(int rc, int cc) {
        S.bl.push_back(hmb_block{});
        S.bl.back().rc = rc;
        S.bl.back().cc = cc;
        return (int)S.bl.size() - 1;
    }

    // Block for node n (sym_anc: n.h has a symmetric ancestor); returns its id.
    int emit(const Node &n, bool sym_anc) {
        const int id = new_block(n.rc, n.cc);
        const HM &h = *n.h;
        if (h.is_leaf()) {
            if (!h.is_dense() && !h.is_low_rank()) fail(kFn, "a leaf without data");
            leaf(id, h, n.tr);
            return id;
        }
        const bool sym_node = mirror && h.get_symmetry() != 'N', child_anc = sym_anc || h.get_symmetry() != 'N';
        std::vector<Node> parts;
        parts.reserve(h.get_children().size() * (sym_node ? 2 : 1));
        for (const auto &c : h.get_children()) {
            const Cl &t = c->get_target_cluster(), &s = c->get_source_cluster();
            const int it = cluster_of(n.tr ? n.cc : n.rc, t), is = cluster_of(n.tr ? n.rc : n.cc, s);
            parts.push_back(n.tr ? Node{c.get(), true, is, it} : Node{c.get(), false, it, is});
            if (mirror && child_anc && t.get_offset() == s.get_offset() && t.get_size() != s.get_size())
                // Htool would mirror some of its leaves and not others: not a block tree
                fail(kFn, "a block straddling the diagonal under symmetric storage: " + region(it, is));
            if (sym_node && t.get_offset() != s.get_offset()) {
                if (!one_tree) fail(kFn, "symmetric storage over two different cluster trees");
                parts.push_back(Node{c.get(), true, is, it}); // GOTCHA(symmetric-storage): the missing triangle
            }
        }
        place(id, parts, child_anc);
        return id;
    }

    // Makes block id a super block over `parts`, which must tile it. A side is split when a
    // part does not span it; parts that land in one son block together (Htool skipped a
    // level) get an intermediate block.
    void place(int id, const std::vector<Node> &parts, bool sym_anc) {
        const int rc = S.bl[id].rc, cc = S.bl[id].cc;
        bool rsplit = false, csplit = false;
        for (const Node &p : parts) {
            rsplit = rsplit || p.rc != rc;
            csplit = csplit || p.cc != cc;
        }
        if (!rsplit && !csplit) fail(kFn, "blocks overlap in " + region(rc, cc));
        const hmb_cluster r = S.cl[rc], c = S.cl[cc];
        const int rsons = rsplit ? r.nsons : 1, csons = csplit ? c.nsons : 1;
        std::vector<int> cell(parts.size());
        for (size_t k = 0; k < parts.size(); k++) {
            const int i = rsplit ? son_index(rc, parts[k].rc) : 0, j = csplit ? son_index(cc, parts[k].cc) : 0;
            if (i < 0 || j < 0) fail(kFn, "blocks overlap in " + region(rc, cc));
            cell[k] = i + j * rsons;
        }
        const int s0 = (int)S.sons.size();
        S.sons.resize(s0 + size_t(rsons) * csons, -1);
        S.bl[id].kind = HMB_SUPER;
        S.bl[id].rsons = rsons;
        S.bl[id].csons = csons;
        S.bl[id].son0 = s0;
        for (int q = 0; q < rsons * csons; q++) {
            const int crc = rsplit ? r.son0 + q % rsons : rc, ccc = csplit ? c.son0 + q / rsons : cc;
            int count = 0, last = -1;
            for (size_t k = 0; k < parts.size(); k++)
                if (cell[k] == q) count++, last = (int)k;
            if (count == 0) fail(kFn, "no block covers " + region(crc, ccc));
            int ch;
            if (count == 1 && parts[last].rc == crc && parts[last].cc == ccc) {
                ch = emit(parts[last], sym_anc);
            } else {
                std::vector<Node> group;
                for (size_t k = 0; k < parts.size(); k++)
                    if (cell[k] == q) group.push_back(parts[k]);
                ch = new_block(crc, ccc);
                info.added_blocks++;
                place(ch, group, sym_anc);
            }
            S.sons[s0 + q] = ch;
        }
    }

    void leaf(int id, const HM &h, bool tr) {
        hmb_block &b = S.bl[id]; // S.alloc below leaves S.bl alone
        const long long m = h.nb_rows(), n = h.nb_cols();
        if (tr) info.mirrored_leaves++;
        if (h.is_dense()) {
            const htool::Matrix<double> &D = *h.get_dense_data();
            if (D.nb_rows() != m || D.nb_cols() != n) fail(kFn, "a dense leaf whose size differs from its clusters'");
            b.kind = HMB_DENSE;
            if (h.get_symmetry() != 'N') {
                // GOTCHA(symmetric-storage): Htool multiplies a symmetric diagonal leaf with
                // symv/symm and never reads the triangle opposite to UPLO: rebuild it.
                const bool lower = h.get_UPLO() != 'U';
                double *F = S.alloc(size_t(m * n));
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++) F[i + j * m] = (lower ? i >= j : i <= j) ? D(i, j) : D(j, i);
                b.D = F;
                b.ldD = std::max(1, (int)m);
                info.copied_values += m * n;
            } else if (!tr) {
                b.D = D.data();
                b.ldD = std::max(1, (int)m);
                info.aliased_values += m * n;
            } else {
                double *F = S.alloc(size_t(m * n)); // the n x m transpose
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++) F[j + i * n] = D(i, j);
                b.D = F;
                b.ldD = std::max(1, (int)n);
                info.copied_values += m * n;
            }
            return;
        }
        const htool::LowRankMatrix<double> &lr = *h.get_low_rank_data();
        const int k = lr.rank_of();
        const htool::Matrix<double> &U = lr.get_U(), &V = lr.get_V();
        if (U.nb_rows() != m || V.nb_cols() != n || V.nb_rows() != k)
            fail(kFn, "a low-rank leaf whose factors do not match its clusters");
        b.kind = HMB_LOWRANK;
        b.rank = k;
        if (!tr) {
            // GOTCHA(lowrank-convention): Htool's block is U*V with V rank x cols: vtrans = 0
            b.U = U.data();
            b.ldU = std::max(1, (int)m);
            b.V = V.data();
            b.ldV = std::max(1, k);
            b.vtrans = 0;
            info.aliased_values += k * (m + n);
        } else {
            // mirror (U V)^T = V^T U^T: the n x k factor V^T is copied, U serves as the
            // (block columns) x k factor of vtrans = 1
            double *Vt = S.alloc(size_t(n) * k);
            for (int l = 0; l < k; l++)
                for (int j = 0; j < n; j++) Vt[j + size_t(l) * n] = V(l, j);
            b.U = Vt;
            b.ldU = std::max(1, (int)n);
            b.V = U.data();
            b.ldV = std::max(1, (int)m);
            b.vtrans = 1;
            info.copied_values += k * n;
            info.aliased_values += k * m;
        }
    }
};

// ---------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------
struct ImportedTree {
    std::unique_ptr<Cl> root;
    std::vector<const Cl *> node; // view cluster id -> Htool cluster (nullptr: empty, or not in this tree)
};

// The ball around cluster id's bounding box.
void ball(const hmb_hmatrix &A, int id, std::vector<double> &center, double &radius) {
    const hmb_cluster &c = A.cl[id];
    center.assign(A.dim, 0.0);
    double r2 = 0;
    for (int d = 0; d < A.dim; d++) {
        center[d] = 0.5 * (c.bmin[d] + c.bmax[d]);
        r2 += 0.25 * (c.bmax[d] - c.bmin[d]) * (c.bmax[d] - c.bmin[d]);
    }
    radius = std::sqrt(r2);
}

class TreeImporter {
    const hmb_hmatrix &A;
    ImportedTree &T;
    int counter = 0; // unique per cluster: Htool's Cluster::operator== compares counters
    int max_leaf = 0;
    unsigned min_depth = ~0u, max_depth = 0;

    void add_sons(int id, Cl *c, int rank) {
        const hmb_cluster &p = A.cl[id];
        if (p.nsons == 0) {
            max_leaf = std::max(max_leaf, p.size);
            min_depth = std::min(min_depth, c->get_depth());
            max_depth = std::max(max_depth, c->get_depth());
            return;
        }
        std::vector<double> ctr;
        double rad;
        for (int s = 0; s < p.nsons; s++) {
            const int sid = p.son0 + s;
            if (A.cl[sid].size == 0) continue; // GOTCHA(htool-empty-cluster)
            ball(A, sid, ctr, rad);
            Cl *ch = c->add_child(rad, ctr, rank, A.cl[sid].offset, A.cl[sid].size, ++counter, false);
            T.node[sid] = ch;
            add_sons(sid, ch, rank);
        }
    }

  public:
    TreeImporter(const hmb_hmatrix &a, ImportedTree &t) : A(a), T(t) {}

    void run(const char *fn, int side) {
        const int root = side ? A.col_root : A.row_root, n = side ? A.n_cols : A.n_rows;
        const int *perm = side ? A.col_perm : A.row_perm;
        if (n == 0) fail(fn, std::string("the matrix has no ") + (side ? "columns" : "rows"));
        T.node.assign(A.ncl, nullptr);
        std::vector<double> ctr;
        double rad;
        ball(A, root, ctr, rad);
        T.root = std::make_unique<Cl>(rad, ctr, -1, 0, n);
        if (perm) std::copy(perm, perm + n, T.root->get_permutation().begin());
        const hmb_cluster &rc = A.cl[root];
        if (rc.nsons == 0) {
            // GOTCHA(htool-partition): user_to_cluster / cluster_to_user read the first
            // partition cluster unchecked, so a leaf root gets one, and the matrix lives on it.
            Cl *part = T.root->add_child(rad, ctr, 0, 0, n, ++counter, true);
            T.root->set_is_permutation_local(true);
            T.node[root] = part;
            add_sons(root, part, 0);
        } else {
            // the root's sons form the partition, ranks 0, 1, ... as Htool's ClusterTreeBuilder makes them
            T.node[root] = T.root.get();
            int rank = 0;
            for (int s = 0; s < rc.nsons; s++) {
                const int sid = rc.son0 + s;
                if (A.cl[sid].size == 0) continue; // GOTCHA(htool-empty-cluster)
                ball(A, sid, ctr, rad);
                Cl *ch = T.root->add_child(rad, ctr, rank, A.cl[sid].offset, A.cl[sid].size, ++counter, true);
                T.node[sid] = ch;
                add_sons(sid, ch, rank);
                rank++;
            }
        }
        T.root->set_maximal_leaf_size(max_leaf);
        T.root->set_minimal_depth(min_depth);
        T.root->set_maximal_depth(max_depth);
    }
};

ImportedTree import_tree(const char *fn, const hmb_hmatrix &A, int side) {
    ImportedTree T;
    TreeImporter(A, T).run(fn, side);
    return T;
}

void validate_or_throw(const char *fn, const hmb_hmatrix &A) {
    char msg[512];
    if (hmb_validate(&A, msg, sizeof msg)) fail(fn, std::string("invalid view: ") + msg);
}

// Hands one leaf's factors to HMatrix::compute_low_rank_data, the only public way to give an
// Htool block low-rank data (the survey's "injection" route): no recompression.
class InjectFactors : public htool::VirtualInternalLowRankGenerator<double> {
    const hmb_block &b;

  public:
    explicit InjectFactors(const hmb_block &blk) : b(blk) {}
    bool copy_low_rank_approximation(int M, int N, int, int, htool::LowRankMatrix<double> &lr) const override {
        const int k = b.rank;
        if (k == 0) return true; // the LowRankMatrix starts as (M x 0) * (0 x N)
        htool::Matrix<double> &U = lr.get_U(), &V = lr.get_V();
        U.resize(M, k);
        V.resize(k, N);
        for (int l = 0; l < k; l++)
            for (int i = 0; i < M; i++) U(i, l) = b.U[i + size_t(l) * b.ldU];
        // GOTCHA(lowrank-convention): Htool wants V as rank x cols; transpose a cols x rank V
        for (int j = 0; j < N; j++)
            for (int l = 0; l < k; l++) V(l, j) = b.vtrans ? b.V[j + size_t(l) * b.ldV] : b.V[l + size_t(j) * b.ldV];
        return true;
    }
    bool copy_low_rank_approximation(int M, int N, int ro, int co, int, htool::LowRankMatrix<double> &lr) const override {
        return copy_low_rank_approximation(M, N, ro, co, lr);
    }
};

class BlockImporter {
    const hmb_hmatrix &A;
    const ImportedTree &R, &C;
    double eps;
    HtoolImportInfo &info;

  public:
    BlockImporter(const hmb_hmatrix &a, const ImportedTree &r, const ImportedTree &c, double e, HtoolImportInfo &i)
        : A(a), R(r), C(c), eps(e), info(i) {}

    void fill(HM &h, int id) {
        const hmb_block &b = A.bl[id];
        const long long m = A.cl[b.rc].size, n = A.cl[b.cc].size;
        if (b.kind == HMB_SUPER) {
            for (int i = 0; i < b.rsons; i++) // row by row, the order of Htool's own builder
                for (int j = 0; j < b.csons; j++) {
                    const int ch = A.sons[b.son0 + i + j * b.rsons];
                    const Cl *t = R.node[A.bl[ch].rc], *s = C.node[A.bl[ch].cc];
                    if (t && s) fill(*h.add_child(t, s), ch); // else an empty block
                }
            return;
        }
        if (b.kind == HMB_DENSE) {
            auto D = std::make_unique<htool::Matrix<double>>((int)m, (int)n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++) (*D)(i, j) = b.D[i + size_t(j) * b.ldD];
            h.set_dense_data(std::move(D));
            info.copied_values += m * n;
            return;
        }
        InjectFactors inject(b);
        h.compute_low_rank_data(inject, -1, eps);
        info.copied_values += b.rank * (m + n);
        if (b.vtrans) info.transposed_values += b.rank * n;
    }
};

bool same_permutation(const hmb_hmatrix &A) {
    if (A.row_perm == A.col_perm) return true;
    if (A.n_rows != A.n_cols) return false;
    for (int k = 0; k < A.n_rows; k++)
        if ((A.row_perm ? A.row_perm[k] : k) != (A.col_perm ? A.col_perm[k] : k)) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------
void htool_export(const htool::HMatrix<double> &A, NeutralStore &out, HtoolExportInfo *info) {
    if (!out.cl.empty() || !out.bl.empty() || !out.sons.empty() || !out.data.empty())
        fail("hmb::htool_export", "the output store must be empty");
    HtoolExportInfo local;
    Exporter(out, local).run(A);
    if (info) *info = local;
}

std::shared_ptr<const hmb_hmatrix> htool_export(const HtoolHandle &A, HtoolExportInfo *info) {
    if (!A || !A->A) fail("hmb::htool_export", "null handle");
    struct Held {
        HtoolHandle keep; // declared first, destroyed last
        NeutralStore store;
    };
    auto held = std::make_shared<Held>();
    held->keep = A;
    htool_export(*A->A, held->store, info);
    return std::shared_ptr<const hmb_hmatrix>(held, &held->store.A); // aliasing: owns `held`
}

std::unique_ptr<htool::Cluster<double>> htool_import_cluster_tree(const hmb_hmatrix &A, int side) {
    const char *fn = "hmb::htool_import_cluster_tree";
    if (side != 0 && side != 1) fail(fn, "side must be 0 (rows) or 1 (columns)");
    validate_or_throw(fn, A);
    return import_tree(fn, A, side).root;
}

HtoolHandle htool_import(const hmb_hmatrix &A, const HtoolImportOptions &opt, HtoolImportInfo *info) {
    const char *fn = "hmb::htool_import";
    validate_or_throw(fn, A);
    ImportedTree R = import_tree(fn, A, 0), C;
    const bool shared = A.row_root == A.col_root && same_permutation(A);
    if (!shared) C = import_tree(fn, A, 1);
    const ImportedTree &CC = shared ? R : C;
    auto H = std::make_unique<HM>(*R.node[A.row_root], *CC.node[A.col_root]);
    H->set_epsilon(opt.epsilon);
    H->set_eta(opt.eta);
    HtoolImportInfo local;
    BlockImporter(A, R, CC, opt.epsilon, local).fill(*H, 0);
    if (info) *info = local;
    return htool_wrap(std::move(R.root), shared ? nullptr : std::move(C.root), std::move(H), 'N');
}

} // namespace hmb
