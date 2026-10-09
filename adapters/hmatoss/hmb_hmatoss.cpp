// move-hmatrix: hmat-oss adapter (build from entries, wrap, entries, matvec, statistics,
// structure stream parser). Block copy is in hmb_hmatoss_copy.cpp.
#include "hmatoss_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hmb {
namespace hmatoss_detail {

std::string fmt(const char *who, const char *f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(who) + ": " + buf;
}

double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static const char *factorization_name(int f) {
    switch (f) {
    case 0: return "LU";
    case 1: return "LDLt";
    case 2: return "LLt";
    case 3: return "HODLR";
    default: return "unknown";
    }
}

static const char *type_name(int t) {
    switch (t) {
    case 0: return "single precision (HMAT_SIMPLE_PRECISION)";
    case 1: return "double precision";
    case 2: return "single complex (HMAT_SIMPLE_COMPLEX)";
    case 3: return "double complex (HMAT_DOUBLE_COMPLEX)";
    default: return "unknown";
    }
}

const HmatossMatrix &checked(const HmatossHandle &A, const char *who) {
    if (!A || !A->A) throw std::invalid_argument(fmt(who, "null hmat-oss handle"));
    // GOTCHA(scalar-type): hmat-oss writes values through void* in the interface's type.
    if (A->hi.value_type != HMAT_DOUBLE_PRECISION)
        throw std::invalid_argument(fmt(who, "the hmat-oss interface is %s; only double precision is supported",
                                        type_name(A->hi.value_type)));
    return *A;
}

namespace {

void sink(void *buf, size_t n, void *user) {
    auto *v = static_cast<std::vector<char> *>(user);
    const char *p = static_cast<const char *>(buf);
    v->insert(v->end(), p, p + n);
}

class Cursor {
    const std::vector<char> &b_;
    size_t p_ = 0;
    const char *who_;

  public:
    Cursor(const std::vector<char> &b, const char *who) : b_(b), who_(who) {}
    void take(void *dst, size_t n) {
        if (b_.size() - p_ < n) throw std::runtime_error(fmt(who_, "truncated hmat-oss structure stream"));
        std::memcpy(dst, b_.data() + p_, n);
        p_ += n;
    }
    template <class T> T get() {
        T v;
        take(&v, sizeof v);
        return v;
    }
    void skip(size_t n) {
        if (b_.size() - p_ < n) throw std::runtime_error(fmt(who_, "truncated hmat-oss structure stream"));
        p_ += n;
    }
    bool done() const { return p_ == b_.size(); }
};

int parse_cluster(Cursor &c, STree &t, const char *who) {
    const int offset = c.get<int>();
    if (offset == -1) throw std::runtime_error(fmt(who, "hmat-oss cluster tree with a NULL son: not supported"));
    const int size = c.get<int>(), nk = c.get<int>();
    if (offset < 0 || size < 0 || offset + size > t.n || nk < 0)
        throw std::runtime_error(fmt(who, "bad cluster [%d, +%d) with %d sons in a tree of size %d", offset, size, nk, t.n));
    const int id = (int)t.cl.size();
    t.cl.push_back(SCluster{offset, size, {}});
    std::vector<int> kids(nk);
    for (int k = 0; k < nk; k++) kids[k] = parse_cluster(c, t, who);
    t.cl[id].kids = std::move(kids);
    return id;
}

void parse_tree(Cursor &c, STree &t, const char *who) {
    t.n = c.get<int>();
    t.dim = c.get<int>();
    if (t.n < 0 || t.dim < 0) throw std::runtime_error(fmt(who, "bad cluster tree header (n %d, dim %d)", t.n, t.dim));
    t.coords.resize((size_t)t.n * t.dim);
    c.take(t.coords.data(), sizeof(double) * t.coords.size());
    if (c.get<int>()) c.skip(sizeof(int) * (size_t)t.n); // group index: not needed
    t.perm.resize(t.n);
    c.take(t.perm.data(), sizeof(int) * (size_t)t.n);
    std::vector<char> seen(t.n, 0);
    for (int k = 0; k < t.n; k++) {
        if (t.perm[k] < 0 || t.perm[k] >= t.n || seen[t.perm[k]])
            throw std::runtime_error(fmt(who, "hmat-oss cluster permutation is not a permutation"));
        seen[t.perm[k]] = 1;
    }
    parse_cluster(c, t, who);
    if (t.cl[0].offset != 0 || t.cl[0].size != t.n)
        throw std::runtime_error(fmt(who, "hmat-oss root cluster does not cover [0, %d)", t.n));
}

int parse_block(Cursor &c, SStruct &s, int rc, int cc, const char *who) {
    const unsigned bits = c.get<unsigned char>();
    if (bits & BIT_NULL) return -1;
    SBlock b;
    b.bits = bits;
    c.get<int>(); // approximate rank: used by hmat-oss's JSON dumps only
    b.rank = c.get<int>();
    c.get<double>(); // low-rank epsilon of the block
    const int nchild = c.get<int>();
    b.rc = rc;
    b.cc = cc;
    const int id = (int)s.bl.size();
    s.bl.push_back(b);
    if (nchild <= 0) {
        if (b.rank == NONLEAF_BLOCK) throw std::runtime_error(fmt(who, "hmat-oss non-leaf block without children"));
        return id;
    }
    const SCluster &R = s.rows.cl[rc], &C = s.cols.cl[cc];
    const int nr = (bits & BIT_SAMEROWS) ? 1 : (int)R.kids.size(), nc = (bits & BIT_SAMECOLS) ? 1 : (int)C.kids.size();
    if (nchild > nr * nc)
        throw std::runtime_error(fmt(who, "hmat-oss block with %d children on a %d x %d split", nchild, nr, nc));
    std::vector<int> kids(nr * nc, -1); // absent trailing children are NULL
    for (int t = 0; t < nchild; t++) {
        const int i = t % nr, j = t / nr;
        kids[t] = parse_block(c, s, (bits & BIT_SAMEROWS) ? rc : R.kids[i], (bits & BIT_SAMECOLS) ? cc : C.kids[j], who);
    }
    s.bl[id].nr = nr;
    s.bl[id].nc = nc;
    s.bl[id].kids = std::move(kids);
    return id;
}

bool same_tree(const STree &a, const STree &b) {
    if (a.n != b.n || a.perm != b.perm || a.cl.size() != b.cl.size()) return false;
    for (size_t k = 0; k < a.cl.size(); k++)
        if (a.cl[k].offset != b.cl[k].offset || a.cl[k].size != b.cl[k].size || a.cl[k].kids != b.cl[k].kids) return false;
    return true;
}

} // namespace

SStruct read_structure(const HmatossMatrix &M, const char *who) {
    std::vector<char> buf;
    try {
        M.hi.write_struct(M.A, sink, &buf);
    } catch (const std::exception &e) {
        // HODLRSYM has no code in hmat-oss's stream format: write_struct throws (after printing
        // a backtrace) instead of returning an error.
        std::string w = e.what();
        std::replace(w.begin(), w.end(), '\n', ' ');
        w.erase(0, w.find_first_not_of(' '));
        throw std::runtime_error(fmt(who, "hmat-oss cannot serialize this matrix (a hodlrsym-factorized matrix cannot be): %s",
                                     w.c_str()));
    }
    Cursor c(buf, who);
    SStruct s;
    s.type = c.get<int>();
    // GOTCHA(scalar-type): the stream records the scalar type the matrix was created with.
    if (s.type != STREAM_TYPE_DOUBLE)
        throw std::runtime_error(fmt(who, "hmat-oss matrix is %s; only double precision is supported", type_name(s.type)));
    s.factorization = c.get<int>();
    parse_tree(c, s.rows, who);
    parse_tree(c, s.cols, who);
    if (parse_block(c, s, 0, 0, who) != 0) throw std::runtime_error(fmt(who, "hmat-oss root block is NULL"));
    if (!c.done()) throw std::runtime_error(fmt(who, "unexpected data after the hmat-oss block tree"));
    s.same_trees = same_tree(s.rows, s.cols);
    s.storage = (s.bl[0].bits & BIT_LOWER) ? 'L' : (s.bl[0].bits & BIT_UPPER) ? 'U' : 'N';
    for (const SBlock &b : s.bl)
        if (b.bits & (BIT_TRIUPPER | BIT_TRILOWER)) s.triangular = true;
    if (s.storage != 'N' && !s.same_trees)
        throw std::runtime_error(fmt(who, "hmat-oss symmetric storage with different row and column trees"));
    return s;
}

void require_unfactorized(const SStruct &s, const char *who) {
    // GOTCHA(factorized-state): after LU, get_values silently returns factor entries; after LLt,
    // gemv applies L. The state is only visible in the stream.
    if (s.factorization != STREAM_FACT_NONE)
        throw std::invalid_argument(fmt(who, "the hmat-oss matrix is %s-factorized: its leaves hold the factors, not the "
                                             "matrix (extract, multiply or export before factorizing)",
                                        factorization_name(s.factorization)));
    if (s.triangular) throw std::invalid_argument(fmt(who, "the hmat-oss matrix has triangular (factorized) blocks"));
}

} // namespace hmatoss_detail

using namespace hmatoss_detail;

// ------------------------------------------------------------------------------------------
// Build from entries: hmat-oss's block assembly interface (prepare + block_compute) calls the
// source for whole blocks and single rows or columns, with 0-based original indices
// (GOTCHA(index-base): unlike get_values, the assembly callbacks are 0-based).
// ------------------------------------------------------------------------------------------
namespace {

struct BuildContext {
    const EntrySource *src;
    std::string error; // first exception thrown by the source
};

struct BlockData {
    BuildContext *ctx;
    const int *rows, *cols; // original indices of the block's rows / columns
};

void release_block(void *p) { delete static_cast<BlockData *>(p); }

void prepare_block(int row_start, int, int col_start, int, int *row_hmat2client, int *, int *col_hmat2client, int *,
                   void *context, hmat_block_info_t *info) {
    info->user_data = new BlockData{static_cast<BuildContext *>(context), row_hmat2client + row_start, col_hmat2client + col_start};
    info->release_user_data = release_block;
}

// out: row_count x col_count, column-major, leading dimension row_count (1 for a row).
void compute_block(void *data, int r0, int nr, int c0, int nc, void *out) {
    auto *b = static_cast<BlockData *>(data);
    double *o = static_cast<double *>(out);
    if (b->ctx->error.empty()) {
        // An exception must not unwind through hmat-oss: record it, fail after assembly.
        try {
            b->ctx->src->get(nr, nc, b->rows + r0, b->cols + c0, o);
            return;
        } catch (const std::exception &e) {
            b->ctx->error = e.what();
        }
    }
    std::fill(o, o + (size_t)nr * nc, 0.0);
}

} // namespace

HmatossHandle hmatoss_build(const EntrySource &entries, int dim, const double *coords, const HmatossOptions &opt,
                            HmatossStats *stats) {
    const char *who = "hmb::hmatoss_build";
    if (!entries.get) throw std::invalid_argument(fmt(who, "entry source without get()"));
    if (entries.n_rows != entries.n_cols || entries.n_rows <= 0)
        throw std::invalid_argument(fmt(who, "square non-empty matrices only (got %d x %d)", entries.n_rows, entries.n_cols));
    if (dim <= 0 || !coords) throw std::invalid_argument(fmt(who, "coordinates are required (dim %d)", dim));
    if (!(opt.epsilon > 0) || !(opt.eta > 0) || opt.leaf_size < 1)
        throw std::invalid_argument(fmt(who, "bad options (epsilon %g, eta %g, leaf_size %d)", opt.epsilon, opt.eta, opt.leaf_size));
    const int n = entries.n_rows;
    auto h = std::make_shared<HmatossMatrix>();
    hmat_init_default_interface(&h->hi, HMAT_DOUBLE_PRECISION);
    h->hi.init();
    const double t0 = wall();

    std::vector<double> xyz(coords, coords + (size_t)dim * n); // hmat-oss takes a non-const pointer
    hmat_clustering_algorithm_t *median = hmat_create_clustering_median();
    hmat_clustering_algorithm_t *algo = hmat_create_clustering_max_dof(median, opt.leaf_size);
    h->tree = hmat_create_cluster_tree(xyz.data(), dim, n, algo);
    hmat_delete_clustering(algo);
    hmat_delete_clustering(median);
    if (!h->tree) throw std::runtime_error(fmt(who, "hmat-oss could not build the cluster tree"));

    hmat_admissibility_t *adm = hmat_create_admissibility_standard(opt.eta);
    h->A = h->hi.create_empty_hmatrix_admissibility(h->tree, h->tree, opt.lower_symmetric ? 1 : 0, adm);
    if (!h->A) {
        hmat_delete_admissibility(adm);
        throw std::runtime_error(fmt(who, "hmat-oss could not create the block tree"));
    }
    // GOTCHA(hmat-epsilon): every compressed block is truncated at the matrix's low-rank
    // epsilon (default 1e-4), whatever the compressor's epsilon: set both.
    h->hi.set_low_rank_epsilon(h->A, opt.epsilon);

    BuildContext ctx{&entries, {}};
    hmat_assemble_context_t ac;
    hmat_assemble_context_init(&ac);
    hmat_compression_algorithm_t *comp = hmat_create_compression_aca_plus(opt.epsilon);
    ac.compression = comp;
    ac.prepare = prepare_block;
    ac.block_compute = compute_block;
    ac.user_context = &ctx;
    ac.lower_symmetric = opt.lower_symmetric ? 1 : 0;
    ac.progress = nullptr; // no progress bar on stdout
    const int rc = h->hi.assemble_generic(h->A, &ac);
    hmat_delete_compression(comp);
    hmat_delete_admissibility(adm);
    if (!ctx.error.empty()) throw std::runtime_error(fmt(who, "the entry source failed: %s", ctx.error.c_str()));
    if (rc != 0) throw std::runtime_error(fmt(who, "hmat-oss assemble_generic failed (its message is on stderr)"));
    h->build_seconds = wall() - t0;
    if (stats) *stats = hmatoss_stats(h);
    return h;
}

HmatossHandle hmatoss_wrap(const hmat_interface_t &iface, hmat_matrix_t *A, bool take_ownership) {
    const char *who = "hmb::hmatoss_wrap";
    if (!A) throw std::invalid_argument(fmt(who, "null matrix"));
    // GOTCHA(scalar-type): a float or complex matrix read into double buffers gives garbage, silently.
    if (iface.value_type != HMAT_DOUBLE_PRECISION)
        throw std::invalid_argument(fmt(who, "the hmat-oss interface is %s; only double precision (HMAT_DOUBLE_PRECISION) is "
                                             "supported",
                                        type_name(iface.value_type)));
    auto h = std::make_shared<HmatossMatrix>();
    h->hi = iface;
    h->A = A;
    h->owns_matrix = take_ownership;
    return h;
}

// ------------------------------------------------------------------------------------------
// Entries over get_values
// ------------------------------------------------------------------------------------------
namespace {

// rows / cols: 1-based original indices (get_values takes non-const pointers but only reads).
void get_values(const HmatossMatrix &h, int m, int *rows, int n, int *cols, double *out) {
    hmat_get_values_context_t ctx;
    std::memset(&ctx, 0, sizeof ctx);
    ctx.matrix = h.A;
    ctx.values = out;
    ctx.row_indices = rows;
    ctx.row_size = m;
    ctx.col_indices = cols;
    ctx.col_size = n;
    if (h.hi.get_values(&ctx) != 0)
        throw std::runtime_error("hmb::hmatoss_entries: hmat-oss get_values failed (its message is on stderr)");
}

} // namespace

EntrySource hmatoss_entries(const HmatossHandle &A) {
    const char *who = "hmb::hmatoss_entries";
    checked(A, who);
    const SStruct s = read_structure(*A, who);
    require_unfactorized(s, who);
    const int nr = s.rows.n, nc = s.cols.n;
    const char storage = s.storage;
    // Position of each original index in hmat-oss's ordering, to find the stored triangle.
    std::shared_ptr<std::vector<int>> pos;
    if (storage != 'N') {
        pos = std::make_shared<std::vector<int>>(nr);
        for (int k = 0; k < nr; k++) (*pos)[s.rows.perm[k]] = k;
    }
    EntrySource e;
    e.n_rows = nr;
    e.n_cols = nc;
    // get_values only reads the matrix and keeps its state per call (c_wrapping.hpp get_values,
    // uncompressed_values.hpp); the study's Helgrind/DRD runs and test_hmatoss_entries agree.
    e.thread_safe = true;
    HmatossHandle h = A;
    e.get = [h, storage, pos, nr, nc](int M, int N, const int *rows, const int *cols, double *out) {
        if (M <= 0 || N <= 0) return;
        std::vector<int> idx((size_t)M + N);
        for (int i = 0; i < M; i++) {
            if (rows[i] < 0 || rows[i] >= nr)
                throw std::invalid_argument(fmt("hmb::hmatoss_entries", "row index %d outside [0, %d)", rows[i], nr));
            idx[i] = rows[i] + 1; // GOTCHA(index-base): get_values takes 1-based indices (0-based: 92% error, rc 0)
        }
        for (int j = 0; j < N; j++) {
            if (cols[j] < 0 || cols[j] >= nc)
                throw std::invalid_argument(fmt("hmb::hmatoss_entries", "column index %d outside [0, %d)", cols[j], nc));
            idx[(size_t)M + j] = cols[j] + 1;
        }
        // get_values writes only the entries of stored blocks; anything else must not be garbage.
        std::fill(out, out + (size_t)M * N, 0.0);
        get_values(*h, M, idx.data(), N, idx.data() + M, out);
        if (storage == 'N') return;
        // GOTCHA(symmetric-storage): get_values leaves the unstored triangle unwritten (rc 0).
        // Entry (r, c) is stored if pos[r] >= pos[c] ('L') or pos[r] <= pos[c] ('U'); the
        // others are read as A(c, r) from a transposed request.
        const std::vector<int> &p = *pos;
        auto missing = [&](int r, int c) { return storage == 'L' ? p[r] < p[c] : p[r] > p[c]; };
        int rlo = nr, rhi = -1, clo = nc, chi = -1;
        for (int i = 0; i < M; i++) rlo = std::min(rlo, p[rows[i]]), rhi = std::max(rhi, p[rows[i]]);
        for (int j = 0; j < N; j++) clo = std::min(clo, p[cols[j]]), chi = std::max(chi, p[cols[j]]);
        if (storage == 'L' ? rlo >= chi : rhi <= clo) return; // everything requested is stored
        std::vector<double> t((size_t)N * M, 0.0);           // A(cols, rows)
        get_values(*h, N, idx.data() + M, M, idx.data(), t.data());
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                if (missing(rows[i], cols[j])) out[i + (size_t)M * j] = t[j + (size_t)N * i];
    };
    return e;
}

// ------------------------------------------------------------------------------------------
// Matvec over gemv (original ordering)
// ------------------------------------------------------------------------------------------
BlackBox hmatoss_blackbox(const HmatossHandle &A) {
    const char *who = "hmb::hmatoss_blackbox";
    checked(A, who);
    const SStruct s = read_structure(*A, who);
    require_unfactorized(s, who);
    if (s.rows.n != s.cols.n) throw std::invalid_argument(fmt(who, "square matrices only (%d x %d)", s.rows.n, s.cols.n));
    const int n = s.rows.n;
    BlackBox b;
    b.n = n;
    b.symmetric = s.storage != 'N';
    HmatossHandle h = A;
    b.apply = [h, n](char trans, int k, const double *X, double *Y) {
        if (k <= 0) return;
        const bool t = (trans == 'T' || trans == 't');
        // GOTCHA(gemv-mutates-input): hmat-oss's gemv permutes its input vector in place (and
        // back), so it gets a copy: X is const and may be shared with other threads.
        std::vector<double> x(X, X + (size_t)n * k);
        double one = 1, zero = 0;
        if (h->hi.gemv(t ? 'T' : 'N', &one, h->A, x.data(), &zero, Y, k) != 0)
            throw std::runtime_error("hmb::hmatoss_blackbox: hmat-oss gemv failed (its message is on stderr)");
    };
    return b;
}

HmatossStats hmatoss_stats(const HmatossHandle &A) {
    const char *who = "hmb::hmatoss_stats";
    const HmatossMatrix &M = checked(A, who);
    const SStruct s = read_structure(M, who);
    hmat_info_t info;
    std::memset(&info, 0, sizeof info);
    M.hi.get_info(M.A, &info);
    HmatossStats st;
    st.n_rows = s.rows.n;
    st.n_cols = s.cols.n;
    st.build_seconds = M.build_seconds;
    st.stored_values = (long long)info.compressed_size;
    st.dense_values = (long long)info.full_size;
    st.n_dense_leaves = (long long)info.full_count;
    st.n_lowrank_leaves = (long long)info.rk_count;
    st.storage = s.storage;
    return st;
}

} // namespace hmb
