// Htool-DDM -> H2Lib block copy (route E1): Htool's export, H2Lib's import in each mode, and
// H2Lib's H -> H2 conversion, on Htool trees that are H2Lib-strict (leaf 64), non-strict
// through ACA fallbacks (leaf 10: dense blocks above the leaf level) and non-strict through
// one-sided splits (block_tree_consistency off). Every import multiplies like Htool at
// rounding level ('N' and 'T'); borrowing copies exactly Htool's V factors (transposed);
// the conversion refuses the non-strict trees and, after a strict import, is within 10 eps
// of Htool and of the kernel.
#include "hmb/neutral_bridge.hpp"
#include "hmb_h2lib.hpp"
#include "hmb_htool.hpp"
#include "hmb_htool_copy.hpp"
#include "hmb_test.hpp"

#include <cfloat>

using namespace hmb;
using namespace hmb::test;

namespace {

const double kRound = 512 * DBL_EPSILON; // lossless transfers agree at rounding level

double diff(const BlackBox &A, const BlackBox &B, char t) { return blackbox_relerr(A, B, t, 3, 11); }

long long lowrank_cols(const hmb_hmatrix &V) { // what BORROW copies of an Htool view: V^T
    long long s = 0;
    for (int b = 0; b < V.nbl; b++)
        if (V.bl[b].kind == HMB_LOWRANK) s += (long long)V.bl[b].rank * V.cl[V.bl[b].cc].size;
    return s;
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_h2lib_htool");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double eps = 1e-6;
    hmb_h2lib_init(0);
    Problem P(n, Kernel::Sym, Geometry::RandomSphere, 10.0);
    const BlackBox exact = P.exact_blackbox();
    struct Case {
        const char *tag;
        int leaf;
        bool consistent;
    };
    for (Case c : {Case{"[strict]", 64, true}, Case{"[leaf 10]", 10, true}, Case{"[inconsistent]", 64, false}}) {
        HtoolOptions o;
        o.epsilon = eps, o.leaf_size = c.leaf, o.block_tree_consistency = c.consistent;
        HtoolHandle A = htool_build(P.entries(), 3, P.xyz.data(), o);
        const BlackBox src = htool_blackbox(A);
        auto V = htool_export(A); // std::shared_ptr<const hmb_hmatrix>, keeps A alive
        long viol[2];
        hmb_strict_violations(V.get(), viol);
        const bool strict = viol[0] + viol[1] == 0;
        char what[160];
        std::printf("  %s: strict violations (%ld, %ld)\n", c.tag, viol[0], viol[1]);
        if (c.leaf == 10 || !c.consistent) {
            std::snprintf(what, sizeof what, "%s Htool's tree is not strict", c.tag);
            C.check(!strict, what, "(%ld, %ld)", viol[0], viol[1]);
        }
        const long long stored = (long long)hmb_stored_values(V.get());
        for (int mode : {HMB_H2LIB_COPY, HMB_H2LIB_BORROW, HMB_H2LIB_STRICT}) {
            const char *mn = mode == HMB_H2LIB_COPY ? "copy" : mode == HMB_H2LIB_BORROW ? "borrow" : "strict";
            hmb_h2lib_import_info info;
            H2libHandle M = h2lib_import(*V, mode, &info, V); // V kept alive for borrowed leaves
            const BlackBox dst = h2lib_blackbox(M);
            for (char t : {'N', 'T'}) {
                std::snprintf(what, sizeof what, "%s %s: H2Lib %c vs Htool", c.tag, mn, t);
                C.check_le(diff(dst, src, t), kRound, what);
            }
            if (mode == HMB_H2LIB_BORROW) {
                std::snprintf(what, sizeof what, "%s borrow: only Htool's V is copied (transposed)", c.tag);
                C.check((long long)info.copied == lowrank_cols(*V) && (long long)(info.copied + info.borrowed) == stored,
                        what, "copied %zu of %lld", info.copied, stored);
            }
            if (mode == HMB_H2LIB_COPY) {
                std::snprintf(what, sizeof what, "%s copy: every value copied", c.tag);
                C.check((long long)info.copied == stored, what, "%zu of %lld", info.copied, stored);
                char msg[512] = "";
                struct _h2matrix *H2 = nullptr;
                int rc = hmb_h2lib_to_h2(hmb_h2lib_hmatrix(M.get()), eps, &H2, nullptr, msg, sizeof msg);
                std::snprintf(what, sizeof what, "%s copy: H -> H2 %s", c.tag, strict ? "succeeds" : "is refused");
                C.check(strict ? rc == 0 : rc == HMB_H2LIB_ENOTSTRICT, what, "%s", msg);
                hmb_h2lib_free_h2(H2);
            }
            if (mode == HMB_H2LIB_STRICT) {
                long v[2];
                hmb_h2lib_strict_violations(hmb_h2lib_hmatrix(M.get()), v);
                std::snprintf(what, sizeof what, "%s strict: no strict violation", c.tag);
                C.check(v[0] == 0 && v[1] == 0, what, "(%ld, %ld)", v[0], v[1]);
                const BlackBox h2 = h2lib_h2_blackbox(h2lib_to_h2(M, eps));
                for (char t : {'N', 'T'}) {
                    std::snprintf(what, sizeof what, "%s strict: H2 %c vs Htool", c.tag, t);
                    C.check_le(diff(h2, src, t), 10 * eps, what);
                }
                std::snprintf(what, sizeof what, "%s strict: H2 vs kernel", c.tag);
                C.check_le(diff(h2, exact, 'N'), 10 * eps, what);
            }
        }
    }
    return C.finish("n=%d", n);
}
