// move-hmatrix: hmat-oss block copy through its serialization stream (export to and import
// from the neutral view). Stream format: hmatoss_stream.hpp.
#include "hmatoss_stream.hpp"
#include "hmb/neutral_store.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hmb {

using namespace hmatoss_detail;

namespace {

// ==========================================================================================
// Export
// ==========================================================================================

// Payload of one hmat-oss leaf, stored in the NeutralStore.
struct Leaf {
    enum Kind { NONE, ZERO, LOWRANK, DENSE } kind = NONE;
    int rank = 0;
    double *a = nullptr, *b = nullptr; // block = a * b^T, a rows x rank, b cols x rank, contiguous
    double *d = nullptr;               // rows x cols, contiguous
};

bool is_void(const SStruct &s, const SBlock &b) { return s.rows.cl[b.rc].size == 0 || s.cols.cl[b.cc].size == 0; }

// Leaves in the order write_data emits them: pre-order, skipping NULL children and blocks with
// an empty row or column cluster.
void data_order(const SStruct &s, int b, std::vector<int> &out) {
    const SBlock &B = s.bl[b];
    if (is_void(s, B)) return;
    if (B.leaf()) {
        out.push_back(b);
        return;
    }
    for (int k : B.kids)
        if (k >= 0) data_order(s, k, out);
}

// Consumes hmat-oss's data stream (write_data callback) as a byte stream, whatever the call
// granularity, and copies each leaf array straight into the store.
class DataReader {
  public:
    DataReader(const SStruct &s, NeutralStore &st, std::vector<Leaf> &leaf) : s_(s), st_(st), leaf_(leaf) {
        data_order(s, 0, order_);
        if (order_.empty()) item_ = DONE;
        else start(HEADER, &word_, sizeof word_);
    }
    static void write_cb(void *buf, size_t n, void *self) {
        static_cast<DataReader *>(self)->feed(static_cast<const char *>(buf), n);
    }
    void finish(const char *who) const {
        if (!error_.empty()) throw std::runtime_error(fmt(who, "%s", error_.c_str()));
        if (item_ != DONE) throw std::runtime_error(fmt(who, "hmat-oss data stream ended before the last leaf"));
    }
    long long values = 0;

  private:
    enum Item { HEADER, FACTOR_A, FACTOR_B, ORTHO_A, ORTHO_B, DENSE, DONE };
    const SStruct &s_;
    NeutralStore &st_;
    std::vector<Leaf> &leaf_;
    std::vector<int> order_;
    size_t li_ = 0;
    Item item_ = HEADER;
    char *dst_ = nullptr;
    size_t need_ = 0, got_ = 0;
    int word_ = 0;
    std::string error_;

    void start(Item it, void *dst, size_t bytes) {
        item_ = it;
        dst_ = static_cast<char *>(dst);
        need_ = bytes;
        got_ = 0;
    }
    void feed(const char *p, size_t n) {
        while (n > 0 && error_.empty()) {
            if (item_ == DONE) {
                error_ = "hmat-oss wrote more data than its structure describes";
                return;
            }
            const size_t take = std::min(n, need_ - got_);
            std::memcpy(dst_ + got_, p, take);
            got_ += take;
            p += take;
            n -= take;
            if (got_ == need_) next();
        }
    }
    void next_leaf() {
        if (++li_ == order_.size()) item_ = DONE;
        else start(HEADER, &word_, sizeof word_);
    }
    void next() {
        const int b = order_[li_];
        Leaf &L = leaf_[b];
        const size_t m = s_.rows.cl[s_.bl[b].rc].size, n = s_.cols.cl[s_.bl[b].cc].size;
        switch (item_) {
        case HEADER: header(b, m, n); break;
        case FACTOR_A:
            values += (long long)(m * L.rank);
            start(FACTOR_B, L.b, sizeof(double) * n * L.rank);
            break;
        case FACTOR_B:
            values += (long long)(n * L.rank);
            start(ORTHO_A, &word_, sizeof word_);
            break;
        case ORTHO_A: start(ORTHO_B, &word_, sizeof word_); break; // orthogonality flags: not needed
        case ORTHO_B: next_leaf(); break;
        case DENSE:
            values += (long long)(m * n);
            next_leaf();
            break;
        case DONE: break;
        }
    }
    void header(int b, size_t m, size_t n) {
        const SBlock &B = s_.bl[b];
        Leaf &L = leaf_[b];
        const int h = word_;
        if (B.rank >= 0) { // low-rank leaf: h is its rank
            if (h != B.rank) {
                error_ = "leaf rank " + std::to_string(h) + " in the data stream, " + std::to_string(B.rank) + " in the structure";
                return;
            }
            if (h == 0) {
                L.kind = Leaf::ZERO;
                next_leaf();
                return;
            }
            L.kind = Leaf::LOWRANK;
            L.rank = h;
            L.a = st_.alloc(m * h);
            L.b = st_.alloc(n * h);
            start(FACTOR_A, L.a, sizeof(double) * m * h);
        } else if (B.rank == FULL_BLOCK) { // dense leaf: h holds flags
            if (h & 1) {
                L.kind = Leaf::ZERO;
                next_leaf();
            } else if (h & 6) {
                error_ = "dense leaf with LU pivots or an LDLt diagonal (factorized matrix)";
            } else {
                L.kind = Leaf::DENSE;
                L.d = st_.alloc(m * n);
                start(DENSE, L.d, sizeof(double) * m * n);
            }
        } else {
            error_ = "leaf block not assembled (rank code " + std::to_string(B.rank) + ")";
        }
    }
};

// Builds the neutral view from the parsed structure and the leaf payloads.
class ViewBuilder {
  public:
    ViewBuilder(const SStruct &s, const std::vector<Leaf> &leaf, NeutralStore &st) : s_(s), leaf_(leaf), st_(st) {}
    long long transposed = 0, mirrored = 0;

    void build() {
        rmap_.assign(s_.rows.cl.size(), -1);
        const int rroot = alloc_clusters(1);
        emit(s_.rows, rmap_, 0, rroot);
        int croot = rroot;
        if (s_.same_trees) {
            cmap_ = rmap_;
        } else {
            cmap_.assign(s_.cols.cl.size(), -1);
            croot = alloc_clusters(1);
            emit(s_.cols, cmap_, 0, croot);
        }
        if (block(0) != 0) throw std::runtime_error("hmb::hmatoss_export: internal error, root block is not block 0");
        hmb_hmatrix &A = st_.A;
        A.dim = (s_.rows.dim >= 1 && s_.rows.dim <= 3 && s_.rows.dim == s_.cols.dim) ? s_.rows.dim : 0;
        A.n_rows = s_.rows.n;
        A.n_cols = s_.cols.n;
        A.row_root = rroot;
        A.col_root = croot;
        // hmat-oss "indices" map hmat position -> original index: the neutral direction.
        st_.row_perm = s_.rows.perm;
        st_.col_perm = s_.cols.perm;
        if (A.dim == 0)
            for (hmb_cluster &c : st_.cl) std::fill(c.bmin, c.bmin + 3, 0.0), std::fill(c.bmax, c.bmax + 3, 0.0);
        st_.finalize();
    }

  private:
    const SStruct &s_;
    const std::vector<Leaf> &leaf_;
    NeutralStore &st_;
    std::vector<int> rmap_, cmap_; // stream cluster index -> neutral cluster id

    int alloc_clusters(int k) {
        const int id = (int)st_.cl.size();
        st_.cl.resize(st_.cl.size() + k, hmb_cluster{});
        return id;
    }
    // Neutral cluster `id` for stream node `node`. A chain of single sons covers the same range:
    // it collapses onto one neutral cluster (blocks split on such a side keep the cluster).
    void emit(const STree &t, std::vector<int> &map, int node, int id) {
        map[node] = id;
        while (t.cl[node].kids.size() == 1) {
            node = t.cl[node].kids[0];
            map[node] = id;
        }
        const SCluster &c = t.cl[node];
        st_.cl[id].offset = c.offset;
        st_.cl[id].size = c.size;
        if (t.dim >= 1 && t.dim <= 3 && c.size > 0) {
            hmb_cluster &h = st_.cl[id];
            for (int d = 0; d < t.dim; d++) h.bmin[d] = h.bmax[d] = t.coords[(size_t)t.dim * t.perm[c.offset] + d];
            for (int k = c.offset; k < c.offset + c.size; k++)
                for (int d = 0; d < t.dim; d++) {
                    const double x = t.coords[(size_t)t.dim * t.perm[k] + d];
                    h.bmin[d] = std::min(h.bmin[d], x);
                    h.bmax[d] = std::max(h.bmax[d], x);
                }
        }
        const int k = (int)c.kids.size();
        if (k == 0) return;
        const int s0 = alloc_clusters(k);
        st_.cl[id].son0 = s0;
        st_.cl[id].nsons = k;
        for (int q = 0; q < k; q++) emit(t, map, c.kids[q], s0 + q);
    }

    int add(const hmb_block &b) {
        st_.bl.push_back(b);
        return (int)st_.bl.size() - 1;
    }
    int zero(int rc, int cc) {
        hmb_block b{};
        b.kind = HMB_LOWRANK;
        b.rc = rc;
        b.cc = cc;
        b.rank = 0;
        b.ldU = b.ldV = 1;
        b.vtrans = 1;
        return add(b);
    }
    int child_cluster(int c, int nsplit, int i) const { return nsplit > 1 ? st_.cl[c].son0 + i : c; }
    int size_of(int c) const { return st_.cl[c].size; }

    // Neutral block for stream block b.
    int block(int b) {
        const SBlock &B = s_.bl[b];
        const int rc = rmap_[B.rc], cc = cmap_[B.cc];
        if (B.leaf()) return leaf_block(b, rc, cc, false);
        if (B.nr == 1 && B.nc == 1) { // a 1 x 1 split: the child is the same block
            const int k = B.kids[0];
            return k >= 0 ? block(k) : zero(rc, cc);
        }
        hmb_block nb{};
        nb.kind = HMB_SUPER;
        nb.rc = rc;
        nb.cc = cc;
        nb.rsons = B.nr;
        nb.csons = B.nc;
        nb.son0 = (int)st_.sons.size();
        st_.sons.resize(st_.sons.size() + (size_t)B.nr * B.nc, -1);
        const int id = add(nb);
        const unsigned sym = B.bits & (BIT_LOWER | BIT_UPPER);
        if (sym && B.nr != B.nc) throw std::runtime_error("hmb::hmatoss_export: symmetric hmat-oss block with a non-square split");
        for (int j = 0; j < B.nc; j++)
            for (int i = 0; i < B.nr; i++) {
                const int k = B.kids[i + j * B.nr];
                int child;
                if (k >= 0) {
                    child = block(k);
                } else if ((sym & BIT_LOWER && j > i) || (sym & BIT_UPPER && i > j)) {
                    // GOTCHA(symmetric-storage): hmat-oss stores only one triangle; the view is
                    // the full matrix, so block (i, j) is the transpose of the stored (j, i).
                    const int src = B.kids[j + i * B.nr];
                    child = src >= 0 ? mirror(src) : zero(child_cluster(rc, B.nr, i), child_cluster(cc, B.nc, j));
                } else {
                    child = zero(child_cluster(rc, B.nr, i), child_cluster(cc, B.nc, j)); // NULL ("inert") block
                }
                st_.sons[nb.son0 + i + j * B.nr] = child;
            }
        return id;
    }

    // Neutral block for the transpose of stream block b (rows and columns share one tree).
    int mirror(int b) {
        const SBlock &B = s_.bl[b];
        const int rc = cmap_[B.cc], cc = rmap_[B.rc];
        if (B.leaf()) return leaf_block(b, rc, cc, true);
        if (B.bits & (BIT_LOWER | BIT_UPPER))
            throw std::runtime_error("hmb::hmatoss_export: symmetric hmat-oss block outside the diagonal");
        if (B.nr == 1 && B.nc == 1) {
            const int k = B.kids[0];
            return k >= 0 ? mirror(k) : zero(rc, cc);
        }
        hmb_block nb{};
        nb.kind = HMB_SUPER;
        nb.rc = rc;
        nb.cc = cc;
        nb.rsons = B.nc;
        nb.csons = B.nr;
        nb.son0 = (int)st_.sons.size();
        st_.sons.resize(st_.sons.size() + (size_t)B.nr * B.nc, -1);
        const int id = add(nb);
        for (int j = 0; j < B.nr; j++)     // new column j = old row j
            for (int i = 0; i < B.nc; i++) { // new row i = old column i
                const int k = B.kids[j + i * B.nr];
                st_.sons[nb.son0 + i + j * B.nc] = k >= 0 ? mirror(k) : zero(child_cluster(rc, B.nc, i), child_cluster(cc, B.nr, j));
            }
        return id;
    }

    // Leaf at neutral clusters (rc, cc); transposed: the leaf is the transpose of stream leaf b.
    int leaf_block(int b, int rc, int cc, bool transposed_leaf) {
        const SBlock &B = s_.bl[b];
        const Leaf &L = leaf_[b];
        const int m = s_.rows.cl[B.rc].size, n = s_.cols.cl[B.cc].size; // of the stored block
        if (is_void(s_, B) || L.kind == Leaf::ZERO) return zero(rc, cc);
        if (L.kind == Leaf::NONE) throw std::runtime_error("hmb::hmatoss_export: internal error, leaf without data");
        if (transposed_leaf) mirrored++;
        hmb_block nb{};
        nb.rc = rc;
        nb.cc = cc;
        if (L.kind == Leaf::LOWRANK) {
            // GOTCHA(lowrank-convention): hmat-oss stores A * B^T with B cols x rank: U = A,
            // V = B, vtrans = 1, aliased. The transpose B * A^T swaps the two (no copy).
            nb.kind = HMB_LOWRANK;
            nb.rank = L.rank;
            nb.vtrans = 1;
            nb.U = transposed_leaf ? L.b : L.a;
            nb.ldU = transposed_leaf ? n : m;
            nb.V = transposed_leaf ? L.a : L.b;
            nb.ldV = transposed_leaf ? m : n;
            return add(nb);
        }
        nb.kind = HMB_DENSE;
        if (!transposed_leaf) {
            nb.D = L.d;
            nb.ldD = m;
            return add(nb);
        }
        double *t = st_.alloc((size_t)m * n); // n x m
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) t[j + (size_t)n * i] = L.d[i + (size_t)m * j];
        transposed += (long long)m * n;
        nb.D = t;
        nb.ldD = n;
        return add(nb);
    }
};

// ==========================================================================================
// Import
// ==========================================================================================

class Bytes {
  public:
    std::vector<char> b;
    template <class T> void put(T v) {
        const char *p = reinterpret_cast<const char *>(&v);
        b.insert(b.end(), p, p + sizeof v);
    }
    void put_raw(const void *p, size_t n) {
        const char *c = static_cast<const char *>(p);
        b.insert(b.end(), c, c + n);
    }
};

// Serves a prepared byte buffer to read_struct.
struct ByteSource {
    const std::vector<char> &b;
    size_t pos = 0;
    bool overrun = false;
    static void read_cb(void *buf, size_t n, void *self) {
        auto *s = static_cast<ByteSource *>(self);
        if (s->b.size() - s->pos < n) {
            s->overrun = true;
            std::memset(buf, 0, n);
            return;
        }
        std::memcpy(buf, s->b.data() + s->pos, n);
        s->pos += n;
    }
};

// A chain of single sons covers one range: it becomes one hmat-oss cluster, the bottom's sons
// being its sons. Over such a cluster a block with rsons == 1 may keep the cluster or descend
// into its single son (neutral.h); either way its children get the same hmat-oss cluster, as
// hmat-oss's "keep same rows" gives them, and a split (rsons > 1) uses the bottom's sons.
void put_cluster(Bytes &w, const hmb_hmatrix *A, int c) {
    w.put<int>(A->cl[c].offset);
    w.put<int>(A->cl[c].size);
    while (A->cl[c].nsons == 1) c = A->cl[c].son0;
    const hmb_cluster &cl = A->cl[c];
    w.put<int>(cl.nsons);
    for (int s = 0; s < cl.nsons; s++) put_cluster(w, A, cl.son0 + s);
}

void put_tree(Bytes &w, const hmb_hmatrix *A, int root, int n, const int *perm, int dim, const double *coords) {
    if (coords && dim > 0) {
        w.put<int>(n);
        w.put<int>(dim);
        w.put_raw(coords, sizeof(double) * (size_t)n * dim);
    } else { // placeholder coordinates (see HmatossImportOptions)
        w.put<int>(n);
        w.put<int>(1);
        const std::vector<double> z(n, 0.0);
        w.put_raw(z.data(), sizeof(double) * (size_t)n);
    }
    w.put<int>(0); // no group index
    if (perm) {
        w.put_raw(perm, sizeof(int) * (size_t)n);
    } else {
        for (int k = 0; k < n; k++) w.put<int>(k);
    }
    put_cluster(w, A, root);
}

void put_node(Bytes &w, unsigned bits, int rank, double eps, int nchild) {
    w.put<unsigned char>((unsigned char)bits);
    w.put<int>(UNINITIALIZED_BLOCK); // approximate rank: only used by hmat-oss's JSON dumps
    w.put<int>(rank);
    w.put<double>(eps);
    w.put<int>(nchild);
}

// Writes block id (pre-order) and collects the leaves that carry data, in read_data's order.
void put_block(Bytes &w, const hmb_hmatrix *A, int id, double eps, std::vector<int> &order) {
    const hmb_block *b = A->bl + id;
    // a 1 x 1 split: the child has the same hmat-oss clusters (same ones, or single sons)
    while (b->kind == HMB_SUPER && b->rsons == 1 && b->csons == 1) b = A->bl + A->sons[b->son0];
    const int m = A->cl[b->rc].size, n = A->cl[b->cc].size;
    if (m == 0 || n == 0) { // empty block: hmat-oss skips its data and its subtree
        put_node(w, BIT_SAMEROWS | BIT_SAMECOLS, UNINITIALIZED_BLOCK, eps, 0);
        return;
    }
    if (b->kind == HMB_SUPER) {
        // hmat-oss child (i, j) is number i + j * nrChildRow, nrChildRow = 1 when the rows are
        // kept, else all row sons: the neutral layout (hmb_validate checked all pairings). With
        // single-son chains collapsed (put_cluster), rsons == 1 always means "keep".
        put_node(w, (b->rsons == 1 ? BIT_SAMEROWS : 0) | (b->csons == 1 ? BIT_SAMECOLS : 0), NONLEAF_BLOCK, eps,
                 b->rsons * b->csons);
        for (int k = 0; k < b->rsons * b->csons; k++) put_block(w, A, A->sons[b->son0 + k], eps, order);
        return;
    }
    put_node(w, BIT_SAMEROWS | BIT_SAMECOLS, b->kind == HMB_DENSE ? FULL_BLOCK : b->rank, eps, 0);
    order.push_back((int)(b - A->bl));
}

// Produces hmat-oss's data stream for read_data lazily: each item (an int, or one leaf array
// in hmat-oss's layout) is staged in a buffer the size of one leaf, then served as bytes, so
// any read granularity works and the whole stream never exists at once.
class DataWriter {
  public:
    DataWriter(const hmb_hmatrix *A, const std::vector<int> &order) : A_(A), order_(order) {
        select();
        if (kind_ != END) stage();
    }
    static void read_cb(void *buf, size_t n, void *self) { static_cast<DataWriter *>(self)->read(static_cast<char *>(buf), n); }
    bool overrun = false;
    bool complete() const { return kind_ == END; }

  private:
    enum Kind { INT, MAT_U, MAT_B, MAT_D, END };
    const hmb_hmatrix *A_;
    const std::vector<int> &order_;
    size_t li_ = 0;
    int part_ = 0;
    Kind kind_ = END;
    int ival_ = 0;
    size_t bytes_ = 0, pos_ = 0;
    std::vector<double> stage_;

    const hmb_block &blk() const { return A_->bl[order_[li_]]; }
    // Selects item part_ of leaf li_, moving to the next leaf when this one is complete.
    // Low-rank leaf: rank, A (rows x rank), B (cols x rank), two orthogonality flags (0:
    // unknown); a rank-0 leaf has the rank only. Dense leaf: flags (0: not null, no pivots, no
    // diagonal), D.
    void select() {
        pos_ = 0;
        for (; li_ < order_.size(); li_++, part_ = 0) {
            const hmb_block &b = blk();
            const size_t m = A_->cl[b.rc].size, n = A_->cl[b.cc].size;
            if (b.kind == HMB_LOWRANK) {
                const size_t k = b.rank;
                if (part_ == 0) return set_int(b.rank);
                if (k == 0) continue;
                if (part_ == 1) return set_mat(MAT_U, m * k);
                if (part_ == 2) return set_mat(MAT_B, n * k);
                if (part_ <= 4) return set_int(0);
            } else {
                if (part_ == 0) return set_int(0);
                if (part_ == 1) return set_mat(MAT_D, m * n);
            }
        }
        kind_ = END;
    }
    void set_int(int v) {
        kind_ = INT;
        ival_ = v;
        bytes_ = sizeof(int);
    }
    void set_mat(Kind k, size_t count) {
        kind_ = k;
        bytes_ = sizeof(double) * count;
    }
    void advance() {
        part_++;
        select();
        if (kind_ != END) stage();
    }
    // Writes the current item to stage_.
    void stage() {
        stage_.resize((bytes_ + sizeof(double) - 1) / sizeof(double));
        if (kind_ == INT) {
            std::memcpy(stage_.data(), &ival_, sizeof ival_);
            return;
        }
        const hmb_block &b = blk();
        const int m = A_->cl[b.rc].size, n = A_->cl[b.cc].size, k = b.rank;
        double *o = stage_.data();
        if (kind_ == MAT_D) {
            for (int j = 0; j < n; j++) std::memcpy(o + (size_t)m * j, b.D + (size_t)b.ldD * j, sizeof(double) * m);
        } else if (kind_ == MAT_U) {
            for (int l = 0; l < k; l++) std::memcpy(o + (size_t)m * l, b.U + (size_t)b.ldU * l, sizeof(double) * m);
        } else if (b.vtrans) { // V is cols x rank: B = V
            for (int l = 0; l < k; l++) std::memcpy(o + (size_t)n * l, b.V + (size_t)b.ldV * l, sizeof(double) * n);
        } else { // GOTCHA(lowrank-convention): V is rank x cols (block = U V): B = V^T
            for (int l = 0; l < k; l++)
                for (int j = 0; j < n; j++) o[j + (size_t)n * l] = b.V[l + (size_t)b.ldV * j];
        }
    }
    void read(char *out, size_t n) {
        while (n > 0) {
            if (kind_ == END) { // hmat-oss expects more leaves than the view has
                overrun = true;
                std::memset(out, 0, n);
                return;
            }
            const size_t take = std::min(n, bytes_ - pos_);
            std::memcpy(out, reinterpret_cast<const char *>(stage_.data()) + pos_, take);
            pos_ += take;
            out += take;
            n -= take;
            if (pos_ == bytes_) advance();
        }
    }
};

} // namespace

std::shared_ptr<const hmb_hmatrix> hmatoss_export(const HmatossHandle &A, HmatossExportStats *stats) {
    const char *who = "hmb::hmatoss_export";
    const HmatossMatrix &M = checked(A, who);
    const double t0 = wall();
    const SStruct s = read_structure(M, who);
    require_unfactorized(s, who);
    if (s.rows.n == 0 || s.cols.n == 0) throw std::invalid_argument(fmt(who, "empty matrix"));

    auto store = std::make_shared<NeutralStore>();
    std::vector<Leaf> leaf(s.bl.size());
    DataReader rd(s, *store, leaf);
    M.hi.write_data(M.A, DataReader::write_cb, &rd);
    rd.finish(who);

    ViewBuilder vb(s, leaf, *store);
    vb.build();
    char msg[256];
    if (hmb_validate(&store->A, msg, sizeof msg) != 0)
        throw std::runtime_error(fmt(who, "unexpected hmat-oss structure, the exported view is invalid: %s", msg));
    if (stats) {
        stats->seconds = wall() - t0;
        stats->stream_values = rd.values;
        stats->transposed_values = vb.transposed;
        stats->mirrored_leaves = vb.mirrored;
        stats->storage = s.storage;
    }
    return std::shared_ptr<const hmb_hmatrix>(store, &store->A); // aliasing: owns the store
}

HmatossHandle hmatoss_import(const hmb_hmatrix &V, const HmatossImportOptions &opt) {
    const char *who = "hmb::hmatoss_import";
    const hmb_hmatrix *A = &V;
    char msg[256];
    if (hmb_validate(A, msg, sizeof msg) != 0) throw std::invalid_argument(fmt(who, "invalid view: %s", msg));
    if (A->n_rows == 0 || A->n_cols == 0) throw std::invalid_argument(fmt(who, "empty matrix"));
    if (!(opt.epsilon > 0)) throw std::invalid_argument(fmt(who, "epsilon must be positive"));
    const double *ccoords = opt.col_coords ? opt.col_coords : (A->n_rows == A->n_cols ? opt.row_coords : nullptr);

    Bytes w;
    w.put<int>(STREAM_TYPE_DOUBLE); // GOTCHA(scalar-type): read by a double-precision interface only
    w.put<int>(STREAM_FACT_NONE);
    put_tree(w, A, A->row_root, A->n_rows, A->row_perm, opt.dim, opt.row_coords);
    put_tree(w, A, A->col_root, A->n_cols, A->col_perm, opt.dim, ccoords);
    std::vector<int> order;
    put_block(w, A, 0, opt.epsilon, order);

    auto h = std::make_shared<HmatossMatrix>();
    hmat_init_default_interface(&h->hi, HMAT_DOUBLE_PRECISION);
    h->hi.init();
    ByteSource src{w.b};
    try {
        h->A = h->hi.read_struct(ByteSource::read_cb, &src);
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt(who, "hmat-oss read_struct failed: %s", e.what()));
    }
    if (!h->A || src.overrun || src.pos != w.b.size())
        throw std::runtime_error(fmt(who, "hmat-oss read %zu of %zu structure bytes%s", src.pos, w.b.size(),
                                     src.overrun ? " and asked for more" : ""));
    DataWriter dw(A, order);
    h->hi.read_data(h->A, DataWriter::read_cb, &dw);
    if (dw.overrun || !dw.complete())
        throw std::runtime_error(fmt(who, "hmat-oss read %s data than the view holds", dw.overrun ? "more" : "less"));
    return h;
}

} // namespace hmb
