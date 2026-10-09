// Htool adapter: build from an EntrySource (sequential and parallel, general and symmetric),
// and the BlackBox matvec in the original ordering, against the exact kernel.
#include "hmb_htool.hpp"
#include "hmb_test.hpp"

using namespace hmb;
using namespace hmb::test;

int main(int argc, char **argv) {
    Checker C("test_htool_build");
    const int n = arg_int(argc, argv, "--n", 2000);
    for (Kernel k : {Kernel::Sym, Kernel::Nonsym}) {
        Problem P(n, k, k == Kernel::Sym ? Geometry::RandomSphere : Geometry::ShuffledFibonacci);
        BlackBox exact = P.exact_blackbox();
        for (char sym : {'N', 'S'}) {
            if (sym == 'S' && k != Kernel::Sym) continue;
            for (bool par : {false, true}) {
                HtoolOptions o;
                o.epsilon = 1e-6;
                o.symmetry = sym;
                o.parallel = par;
                HtoolStats st;
                HtoolHandle A = htool_build(P.entries(), 3, P.xyz.data(), o, &st);
                BlackBox bb = htool_blackbox(A);
                double eN = blackbox_relerr(bb, exact, 'N'), eT = blackbox_relerr(bb, exact, 'T');
                char what[96];
                std::snprintf(what, sizeof what, "%s sym=%c par=%d matvec N", k == Kernel::Sym ? "sym-kernel" : "nonsym-kernel", sym, par);
                C.check_le(eN, 1e-5, what);
                what[std::strlen(what) - 1] = 'T';
                C.check_le(eT, 1e-5, what);
                std::printf("  stored %lld (%.1f%% of dense), max rank %d, %d dense + %d low-rank leaves, %.2fs\n",
                            st.stored_values, 100.0 * st.stored_values / (double(n) * n), st.max_rank,
                            st.n_dense_leaves, st.n_lowrank_leaves, st.build_seconds);
                C.check(htool_size(A) == n, "size");
                C.check(bb.symmetric == (sym == 'S'), "symmetric flag");
            }
        }
    }
    return C.finish("n=%d", n);
}
