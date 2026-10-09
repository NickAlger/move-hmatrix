// Matvec route with an Htool-DDM source (experiment E2's route): Htool H-matrix -> BlackBox
// -> ButterflyPACK HODLR and H. Two sources: symmetric storage of the symmetric kernel ('S',
// a symmetric BlackBox: ButterflyPACK's A^T requests are answered with Htool's 'N' product)
// and full storage of the nonsymmetric kernel ('N': the A^T requests reach Htool's
// transposed product, GOTCHA(bpack-transpose)). Accuracy of 'N' and 'T' against the source
// and the kernel.
#include "hmb_bpack.hpp"
#include "hmb_htool.hpp"
#include "test_bpack_h2opus_common.hpp"

using namespace hmb;
using namespace hmb::test;

int main(int argc, char **argv) {
    Checker C("test_bpack_htool");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6); // ButterflyPACK's tolerance
    const double eps = 1e-6;                                // Htool's (the source's)
    // GOTCHA(tolerance-stacking): HODLR within 1-2x here, H with a heavy tail (median 4x,
    // worst seen 56x; see test_bpack_matvec)
    const double hodlr_bound = 20 * tol, h_bound = 100 * tol;
    for (Kernel kern : {Kernel::Sym, Kernel::Nonsym}) {
        const bool sym = kern == Kernel::Sym;
        Problem P(n, kern, sym ? Geometry::RandomSphere : Geometry::ShuffledFibonacci);
        HtoolOptions ho;
        ho.epsilon = eps;
        ho.symmetry = sym ? 'S' : 'N';
        HtoolHandle src = htool_build(P.entries(), 3, P.xyz.data(), ho);
        const BlackBox S = htool_blackbox(src), K = P.exact_blackbox();
        const double src_err = blackbox_relerr(S, K);
        const char *kname = sym ? "sym/Htool 'S'" : "nonsym/Htool 'N'";
        std::printf("%s source: %s BlackBox, error against the kernel %.2e\n", kname,
                    S.symmetric ? "symmetric" : "nonsymmetric", src_err);
        C.check(S.symmetric == sym, "source BlackBox symmetric flag follows Htool's storage");
        for (BpackFormat f : {BpackFormat::HODLR, BpackFormat::H}) {
            const char *fname = bpack_format_name(f);
            const double err_bound = f == BpackFormat::H ? h_bound : hodlr_bound;
            char what[128];
            BpackOptions o;
            o.format = f;
            o.tolerance = tol;
            auto cnt = std::make_shared<CallCounts>();
            BpackStats st;
            BpackHandle h = bpack_from_matvec(counted(S, cnt), 3, P.xyz.data(), o, &st);
            BlackBox B = bpack_blackbox(h);
            std::printf("%s -> %s: %ld calls (%ld for A^T), %lld vectors, rank %d, %.2fs (source %.2fs), check_error %.2e\n",
                        kname, fname, st.calls, st.calls_transposed, st.vectors, st.rank_max, st.seconds_build,
                        st.seconds_in_source, st.check_error);
            std::snprintf(what, sizeof what, "%s -> %s matvec 'N' against the source", kname, fname);
            C.check_le(blackbox_relerr(B, S, 'N'), err_bound, what);
            std::snprintf(what, sizeof what, "%s -> %s matvec 'T' against the source", kname, fname);
            C.check_le(blackbox_relerr(B, S, 'T'), err_bound, what);
            std::snprintf(what, sizeof what, "%s -> %s matvec 'N' against the kernel", kname, fname);
            C.check_le(blackbox_relerr(B, K, 'N'), err_bound + 2 * src_err, what);
            std::snprintf(what, sizeof what, "%s -> %s A^T requests reach the source as %s", kname, fname,
                          sym ? "'N'" : "'T'");
            C.check(st.calls_transposed > 0 && cnt->calls_transposed == (sym ? 0 : st.calls_transposed), what,
                    "%ld requests, %ld 'T' calls to Htool", st.calls_transposed, cnt->calls_transposed.load());
        }
    }
    return C.finish("n=%d tol=%g", n, tol);
}
