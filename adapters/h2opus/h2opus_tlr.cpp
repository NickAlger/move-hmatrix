// move-hmatrix: H2Opus TLR adapter (see hmb_h2opus_tlr.hpp): import, construction from
// entries, export and products of TTLR_Matrix<double, CPU>.
#include "h2opus_tlr_native.hpp"
#include "hmb/neutral_store.hpp"

#include <h2opus/core/tlr/tlr_construct.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace hmb {
namespace {

double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

[[noreturn]] void refuse(const char *who, const std::string &why) {
    throw std::invalid_argument(std::string(who) + ": " + why);
}

int perm_at(const int *p, int k) { return p ? p[k] : k; }

// rows x cols block of src (leading dimension ld) into dst (leading dimension bs), rows
// rows .. bs - 1 of dst zeroed (H2Opus pads every tile to bs rows)
void copy_padded(double *dst, int bs, const double *src, int ld, int rows, int cols) {
    for (int c = 0; c < cols; c++) {
        std::memcpy(dst + size_t(c) * bs, src + size_t(c) * ld, sizeof(double) * size_t(rows));
        std::fill(dst + size_t(c) * bs + rows, dst + size_t(c + 1) * bs, 0.0);
    }
}

// dst (bs x k, padded) = transpose of src (k x cols, leading dimension ld)
void copy_transposed_padded(double *dst, int bs, const double *src, int ld, int cols, int k) {
    for (int l = 0; l < k; l++) {
        double *d = dst + size_t(l) * bs;
        for (int c = 0; c < cols; c++) d[c] = src[l + size_t(c) * ld];
        std::fill(d + cols, d + bs, 0.0);
    }
}

// D (rows x cols, leading dimension ldd) = U * op(V) of a low-rank leaf
void expand(const hmb_block &b, int rows, int cols, double *D, int ldd) {
    if (b.rank == 0) {
        for (int c = 0; c < cols; c++) std::fill(D + size_t(c) * ldd, D + size_t(c) * ldd + rows, 0.0);
        return;
    }
    h2opus_fbl_gemm(H2OpusFBLNoTrans, b.vtrans ? H2OpusFBLTrans : H2OpusFBLNoTrans, rows, cols, b.rank, 1.0, b.U,
                    b.ldU, b.V, b.ldV, 0.0, D, ldd);
}

// Block ids of the p x p tiles of a uniform tile view (tile (i, j) at i + j * p).
struct Tiles {
    int p = 1, bs = 0;
    std::vector<int> leaf;
};

Tiles tile_structure(const hmb_hmatrix *A, const char *who) {
    char msg[256];
    if (hmb_validate(A, msg, sizeof msg)) refuse(who, std::string("invalid view: ") + msg);
    const int n = A->n_rows;
    if (A->n_rows != A->n_cols)
        refuse(who, "H2Opus TLR matrices are square; the view is " + std::to_string(A->n_rows) + " x " +
                        std::to_string(A->n_cols));
    if (n == 0) refuse(who, "empty matrix");
    for (int k = 0; k < n; k++)
        if (perm_at(A->row_perm, k) != perm_at(A->col_perm, k))
            refuse(who, "the row and column permutations differ (H2Opus TLR has a single index map)");
    Tiles t;
    const hmb_block &root = A->bl[0];
    if (root.kind != HMB_SUPER) { // the whole matrix as one diagonal tile
        t.bs = n;
        t.leaf.assign(1, 0);
        return t;
    }
    const hmb_cluster &R = A->cl[A->row_root], &C = A->cl[A->col_root];
    if (root.rsons != root.csons || root.rsons != R.nsons || root.csons != C.nsons || R.nsons == 0)
        refuse(who, "not a tile low-rank matrix: the root block must split the rows and the columns into the same "
                    "number of tiles (it splits into " +
                        std::to_string(root.rsons) + " x " + std::to_string(root.csons) + ")");
    t.p = root.rsons;
    t.bs = A->cl[R.son0].size;
    for (int k = 0; k < t.p; k++) {
        const int want = k < t.p - 1 ? t.bs : n - (t.p - 1) * t.bs;
        const int rs = A->cl[R.son0 + k].size, cs = A->cl[C.son0 + k].size;
        if (rs != want || cs != want || want < 1 || want > t.bs)
            refuse(who, "not a uniform tile partition: tile " + std::to_string(k) + " is " + std::to_string(rs) + " x " +
                            std::to_string(cs) + ", H2Opus needs " + std::to_string(t.bs) +
                            " rows and columns per tile except the last (1 to " + std::to_string(t.bs) + ")");
    }
    t.leaf.resize(size_t(t.p) * t.p);
    for (int j = 0; j < t.p; j++)
        for (int i = 0; i < t.p; i++) {
            const int id = A->sons[root.son0 + i + j * t.p];
            if (A->bl[id].kind == HMB_SUPER)
                refuse(who, "not a tile low-rank matrix: tile (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is subdivided");
            t.leaf[i + size_t(j) * t.p] = id;
        }
    return t;
}

int tile_dim(const H2opusTlr &M, int k) { return H2OPUS_TLR_BLOCK_DIM(k, M.block_size, M.n_block, M.n); }

void count_storage(const H2opusTlr &M, H2opusTlrStats &st) {
    const int nb = M.n_block, bs = M.block_size;
    st.n = M.n;
    st.block_size = bs;
    st.n_block = nb;
    st.padded_n = nb * bs;
    st.symmetric = M.type == H2OpusTLR_Symmetric;
    st.stored_values = (long long)nb * bs * bs;
    st.max_rank = 0;
    for (int j = 0; j < nb; j++)
        for (int i = 0; i < nb; i++) {
            if (i == j || (st.symmetric && i < j)) continue; // upper tiles of the symmetric type alias the lower ones
            const int r = M.block_ranks[i + size_t(j) * nb];
            st.stored_values += 2LL * bs * r;
            st.max_rank = std::max(st.max_rank, r);
        }
}

// GOTCHA(h2opus-tlr-diag-rank): for H2OpusTLR_NonSymmetric, construct_tlr_matrix compresses
// every tile of a block column, the diagonal one included, but its generator skips the
// diagonal tile (tlr_batch_cpu.h: "if (block == block_column) continue"), so a stale buffer
// is compressed into it; tlr_gemv then adds that low-rank term to the dense diagonal tile.
int clear_diagonal_ranks(H2opusTlr &M) {
    if (M.type != H2OpusTLR_NonSymmetric) return 0;
    int k = 0;
    for (int j = 0; j < M.n_block; j++) {
        int &r = M.block_ranks[j + size_t(j) * M.n_block];
        if (r != 0) {
            r = 0;
            k++;
        }
    }
    return k;
}

// One entry at a time in TLR order (H2Opus's construct_tlr_matrix interface), mapped to the
// ORIGINAL ordering. H2Opus calls it inside OpenMP loops (generateDenseBlocks), which an
// exception must not leave: the first one is kept, the remaining entries read as 0 without
// calling the source, and h2opus_tlr_from_entries rethrows it once the construction returns.
struct EntryGen {
    const EntrySource *src;
    const int *map;
    bool serialize;             // the source is not thread-safe
    std::mutex lock;            // serializes the source's calls (if serialize) and guards error
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    double operator()(int i, int j) {
        if (failed.load(std::memory_order_relaxed)) return 0.0;
        int r = map[i], c = map[j];
        double v = 0.0;
        try {
            if (serialize) {
                std::lock_guard<std::mutex> g(lock);
                src->get(1, 1, &r, &c, &v);
            } else {
                src->get(1, 1, &r, &c, &v);
            }
        } catch (...) {
            std::lock_guard<std::mutex> g(lock); // not held here: the guard above has unwound
            if (!error) error = std::current_exception();
            failed = true;
            return 0.0;
        }
        return v;
    }
};

struct TlrStore : NeutralStore {
    H2opusTlrHandle source; // the view points into its tiles
};

} // namespace

// ------------------------------------------------------------------------------------------
H2opusTlrHandle h2opus_tlr_import(const hmb_hmatrix &view, const H2opusTlrImportOptions &opt, H2opusTlrStats *stats) {
    const char *who = "hmb::h2opus_tlr_import";
    const double t0 = wall();
    const hmb_hmatrix *A = &view;
    const Tiles T = tile_structure(A, who);
    const int n = A->n_rows, p = T.p, bs = T.bs;
    std::vector<int> imap(n);
    for (int k = 0; k < n; k++) imap[k] = perm_at(A->row_perm, k);

    auto h = std::make_shared<H2opusTlrMatrix>();
    h2opusComputeStream_t stream = h->handle->getMainStream();
    h->A.reset(new H2opusTlr(n, opt.symmetric ? H2OpusTLR_Symmetric : H2OpusTLR_NonSymmetric, bs, imap.data(),
                             H2OpusTLRTile, stream));
    H2opusTlr &M = *h->A;
    if (M.n_block != p)
        throw std::runtime_error(std::string(who) + ": internal error: H2Opus made " + std::to_string(M.n_block) +
                                 " tiles for " + std::to_string(p));
    H2opusTlrStats st;
    auto leaf = [&](int i, int j) -> const hmb_block & { return A->bl[T.leaf[i + size_t(j) * p]]; };
    // a tile is copied as U, V when it is low-rank of rank <= bs, otherwise stored as D * I
    auto as_factors = [&](const hmb_block &b) { return b.kind == HMB_LOWRANK && b.rank <= bs; };

    std::vector<int> ranks(p);
    std::vector<double> prod;
    for (int j = 0; j < p; j++) {
        const int r0 = opt.symmetric ? j + 1 : 0; // the symmetric type stores the tiles below the diagonal
        if (r0 >= p) continue;
        for (int i = r0; i < p; i++)
            ranks[i - r0] = i == j ? 0 : (as_factors(leaf(i, j)) ? leaf(i, j).rank : tile_dim(M, j));
        M.allocateBlockColumn(j, ranks.data(), stream);
        for (int i = r0; i < p; i++) {
            if (i == j) continue;
            const hmb_block &b = leaf(i, j);
            const int rows = tile_dim(M, i), cols = tile_dim(M, j), k = M.block_ranks[i + size_t(j) * p];
            double *U = M.block_U_ptrs[i + size_t(j) * p], *V = M.block_V_ptrs[i + size_t(j) * p];
            if (as_factors(b)) {
                copy_padded(U, bs, b.U, b.ldU, rows, k);
                // GOTCHA(lowrank-convention): H2Opus stores U * V^T with V cols x rank
                if (b.vtrans) copy_padded(V, bs, b.V, b.ldV, cols, k);
                else copy_transposed_padded(V, bs, b.V, b.ldV, cols, k);
                st.copied_values += (long long)k * (rows + cols);
                continue;
            }
            // dense tile (or rank > bs): U = D, V = I, exact
            const double *D = b.D;
            int ldd = b.ldD;
            if (b.kind == HMB_LOWRANK) {
                prod.assign(size_t(rows) * cols, 0.0);
                expand(b, rows, cols, prod.data(), rows);
                D = prod.data();
                ldd = rows;
                st.copied_values += (long long)b.rank * (rows + cols);
            } else {
                st.copied_values += (long long)rows * cols;
            }
            copy_padded(U, bs, D, ldd, rows, cols);
            for (int l = 0; l < cols; l++)
                for (int c = 0; c < bs; c++) V[c + size_t(l) * bs] = c == l ? 1.0 : 0.0;
            st.dense_offdiagonal_tiles++;
        }
    }
    for (int i = 0; i < p; i++) {
        const hmb_block &b = leaf(i, i);
        const int rows = tile_dim(M, i);
        double *D = M.getDiagonalBlockHostPtr(i);
        std::fill(D, D + size_t(bs) * bs, 0.0);
        if (b.kind == HMB_DENSE) {
            for (int c = 0; c < rows; c++)
                std::memcpy(D + size_t(c) * bs, b.D + size_t(c) * b.ldD, sizeof(double) * size_t(rows));
            st.copied_values += (long long)rows * rows;
        } else {
            expand(b, rows, rows, D, bs);
            st.lowrank_diagonal_tiles++;
            st.copied_values += 2LL * b.rank * rows;
        }
        for (int k = rows; k < bs; k++) D[k + size_t(k) * bs] = 1.0; // padding: H2Opus's generator's convention
    }
    count_storage(M, st);
    st.seconds = wall() - t0;
    h->stats = st;
    if (stats) *stats = st;
    return h;
}

// ------------------------------------------------------------------------------------------
H2opusTlrHandle h2opus_tlr_from_entries(const EntrySource &entries, const int *perm, const H2opusTlrOptions &opt,
                                        H2opusTlrStats *stats) {
    const char *who = "hmb::h2opus_tlr_from_entries";
    const double t0 = wall();
    const int n = entries.n_rows;
    if (n <= 0 || entries.n_cols != n) refuse(who, "need a square EntrySource");
    if (opt.block_size <= 0 || !(opt.eps >= 0)) refuse(who, "need block_size > 0 and eps >= 0");
    if (!entries.get) refuse(who, "the EntrySource has no get function");
    std::vector<int> imap(n);
    std::vector<char> seen(n, 0);
    for (int k = 0; k < n; k++) {
        imap[k] = perm_at(perm, k);
        if (imap[k] < 0 || imap[k] >= n || seen[imap[k]]++) refuse(who, "perm is not a permutation");
    }
    const int bs = std::min(opt.block_size, n);
    auto h = std::make_shared<H2opusTlrMatrix>();
    h2opusComputeStream_t stream = h->handle->getMainStream();
    h->A.reset(new H2opusTlr(n, opt.symmetric ? H2OpusTLR_Symmetric : H2OpusTLR_NonSymmetric, bs, imap.data(),
                             H2OpusTLRTile, stream));
    EntryGen gen{&entries, imap.data(), !entries.thread_safe};
    construct_tlr_matrix<double, EntryGen, H2OPUS_HWTYPE_CPU>(*h->A, gen, opt.eps, stream);
    if (gen.error) {
        try {
            std::rethrow_exception(gen.error);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string(who) + ": the EntrySource threw: " + e.what());
        } catch (...) {
            throw std::runtime_error(std::string(who) + ": the EntrySource threw a non-standard exception");
        }
    }
    H2opusTlrStats st;
    st.cleared_diagonal_ranks = clear_diagonal_ranks(*h->A);
    count_storage(*h->A, st);
    for (int j = 0; j < st.n_block; j++) // entries requested: diagonal tiles and the stored off-diagonal tiles
        for (int i = opt.symmetric ? j : 0; i < st.n_block; i++)
            st.copied_values += (long long)tile_dim(*h->A, i) * tile_dim(*h->A, j);
    st.seconds = wall() - t0;
    h->stats = st;
    if (stats) *stats = st;
    return h;
}

H2opusTlrHandle h2opus_tlr_wrap(std::unique_ptr<H2opusTlr> A) {
    const char *who = "hmb::h2opus_tlr_wrap";
    if (!A) refuse(who, "null matrix");
    if (A->alloc != H2OpusTLRTile) refuse(who, "only tile allocation (H2OpusTLRTile) is supported");
    if (A->type != H2OpusTLR_NonSymmetric && A->type != H2OpusTLR_Symmetric)
        refuse(who, "triangular TLR types are not supported");
    auto h = std::make_shared<H2opusTlrMatrix>();
    h->A = std::move(A);
    H2opusTlrStats st;
    st.cleared_diagonal_ranks = clear_diagonal_ranks(*h->A);
    count_storage(*h->A, st);
    h->stats = st;
    return h;
}

// ------------------------------------------------------------------------------------------
std::shared_ptr<const hmb_hmatrix> h2opus_tlr_export(const H2opusTlrHandle &h) {
    const char *who = "hmb::h2opus_tlr_export";
    if (!h || !h->A) refuse(who, "null matrix");
    const H2opusTlr &M = *h->A;
    const int n = M.n, bs = M.block_size, p = M.n_block;
    auto s = std::make_shared<TlrStore>();
    s->source = h;
    hmb_cluster root{};
    root.offset = 0, root.size = n, root.son0 = 1, root.nsons = p;
    s->cl.push_back(root);
    for (int k = 0; k < p; k++) {
        hmb_cluster c{};
        c.offset = k * bs, c.size = tile_dim(M, k);
        s->cl.push_back(c);
    }
    hmb_block top{};
    top.rc = top.cc = 0, top.kind = HMB_SUPER, top.rsons = top.csons = p, top.son0 = 0;
    s->bl.push_back(top);
    s->sons.resize(size_t(p) * p);
    for (int j = 0; j < p; j++)
        for (int i = 0; i < p; i++) {
            const size_t t = i + size_t(j) * p;
            hmb_block b{};
            b.rc = 1 + i, b.cc = 1 + j;
            if (i == j) { // diagonal ranks are ignored: see clear_diagonal_ranks
                b.kind = HMB_DENSE, b.D = M.diagonal_block_ptrs[i], b.ldD = bs;
            } else {
                b.kind = HMB_LOWRANK, b.rank = M.block_ranks[t], b.vtrans = 1;
                b.U = M.block_U_ptrs[t], b.V = M.block_V_ptrs[t], b.ldU = b.ldV = bs;
            }
            s->sons[t] = (int)s->bl.size();
            s->bl.push_back(b);
        }
    s->finalize();
    s->A.n_rows = s->A.n_cols = n, s->A.row_root = s->A.col_root = 0;
    const int *map = vec_ptr(M.index_map);
    bool ident = true;
    for (int k = 0; k < n && ident; k++) ident = map[k] == k;
    s->A.row_perm = s->A.col_perm = ident ? nullptr : map; // aliased, kept alive by source
    char msg[256];
    if (hmb_validate(&s->A, msg, sizeof msg))
        throw std::runtime_error(std::string(who) + ": internal error, invalid view: " + msg);
    return std::shared_ptr<const hmb_hmatrix>(s, &s->A); // shares s's control block
}

// ------------------------------------------------------------------------------------------
void h2opus_tlr_matvec(const H2opusTlrHandle &h, char trans, int nrhs, const double *X, int ldx, double *Y, int ldy) {
    const char *who = "hmb::h2opus_tlr_matvec";
    if (!h || !h->A) refuse(who, "null matrix");
    const bool t = trans == 'T' || trans == 't';
    if (!t && trans != 'N' && trans != 'n') refuse(who, "trans must be 'N' or 'T'");
    H2opusTlr &M = *h->A;
    const int n = M.n, pn = M.getPaddedDim();
    if (nrhs < 0 || ldx < n || ldy < n) refuse(who, "bad nrhs or leading dimension");
    if (nrhs == 0) return;
    const int *map = vec_ptr(M.index_map);
    // tlr_gemv works in TLR order on vectors padded to n_block * block_size (zero padding)
    std::vector<double> Xp(size_t(pn) * nrhs, 0.0), Yp(size_t(pn) * nrhs, 0.0);
    for (int c = 0; c < nrhs; c++)
        for (int k = 0; k < n; k++) Xp[k + size_t(c) * pn] = X[map[k] + size_t(c) * ldx];
    // GOTCHA(h2opus-tlr-gemv-multi): tlr_gemv accumulates each of its npc parallel block
    // columns in a buffer at offset (j - j_start) * n (tlr_gemv_marshal.h), which holds one
    // vector only: with several vectors and npc > 1 the buffers overlap and the product is
    // wrong. Its reduction also assumes ldy = n_block * block_size (we pass exactly that).
    const int npc = nrhs == 1 ? 2 : 1; // 2: H2Opus's examples' value
    {
        std::lock_guard<std::mutex> g(h->lock);
        tlr_gemv<double, H2OPUS_HWTYPE_CPU>(t ? H2Opus_Trans : H2Opus_NoTrans, 1.0, M, Xp.data(), pn, 0.0, Yp.data(),
                                            pn, nrhs, npc, h->handle);
    }
    for (int c = 0; c < nrhs; c++)
        for (int k = 0; k < n; k++) Y[map[k] + size_t(c) * ldy] = Yp[k + size_t(c) * pn];
}

BlackBox h2opus_tlr_blackbox(const H2opusTlrHandle &h) {
    if (!h || !h->A) refuse("hmb::h2opus_tlr_blackbox", "null matrix");
    BlackBox b;
    b.n = h->A->n;
    b.symmetric = h->A->type == H2OpusTLR_Symmetric;
    b.apply = [h](char trans, int k, const double *X, double *Y) {
        h2opus_tlr_matvec(h, trans, k, X, h->A->n, Y, h->A->n);
    };
    return b;
}

H2opusTlrStats h2opus_tlr_stats(const H2opusTlrHandle &h) {
    if (!h || !h->A) refuse("hmb::h2opus_tlr_stats", "null matrix");
    return h->stats;
}

} // namespace hmb
