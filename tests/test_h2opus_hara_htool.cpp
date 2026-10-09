// Matvec route with an Htool-DDM source into H2Opus (experiment E2's HARA route): Htool
// H-matrix -> BlackBox -> H2Opus H^2. Htool and H2Opus meet only through the BlackBox (their
// headers clash, GOTCHA(header-clash)): this file includes neutral headers only. Symmetric
// storage ('S') gives a symmetric BlackBox; full storage ('N') of the same symmetric kernel is
// refused unless assume_symmetric is set (GOTCHA(symmetric-storage)). Accuracy against the
// source and the kernel.
#include "hmb_h2opus.hpp"
#include "hmb_htool.hpp"
#include "test_bpack_h2opus_common.hpp"

using namespace hmb;
using namespace hmb::test;

int main(int argc, char **argv) {
    Checker C("test_h2opus_hara_htool");
    const int n = arg_int(argc, argv, "--n", 2000);
    const double tol = arg_dbl(argc, argv, "--tol", 1e-6); // HARA's (relative) tolerance
    const double err_bound = 20 * tol;                      // E2: 2.0-4.6e-6 at 1e-6
    Problem P(n);
    const BlackBox K = P.exact_blackbox();
    for (char storage : {'S', 'N'}) {
        HtoolOptions ho;
        ho.epsilon = 1e-6;
        ho.symmetry = storage;
        HtoolHandle src = htool_build(P.entries(), 3, P.xyz.data(), ho);
        const BlackBox S = htool_blackbox(src);
        const double src_err = blackbox_relerr(S, K);
        char what[128];
        H2opusOptions o;
        o.tolerance = tol;
        if (storage == 'N') {
            bool refused = false;
            try {
                h2opus_from_matvec(S, 3, P.xyz.data(), o);
            } catch (const std::invalid_argument &) {
                refused = true;
            }
            C.check(refused, "Htool 'N' source (not flagged symmetric) refused");
            o.assume_symmetric = true;
        }
        auto cnt = std::make_shared<CallCounts>();
        H2opusStats st;
        H2opusHandle h = h2opus_from_matvec(counted(S, cnt), 3, P.xyz.data(), o, &st);
        BlackBox B = h2opus_blackbox(h);
        std::printf("Htool '%c' (error against the kernel %.2e) -> H2Opus: %ld calls, %lld vectors, max rank %d, "
                    "%.2fs (hara %.2fs, source %.2fs)\n",
                    storage, src_err, st.calls, st.vectors, st.max_rank, st.seconds_structure + st.seconds_norm + st.seconds_hara,
                    st.seconds_hara, st.seconds_in_source);
        std::snprintf(what, sizeof what, "Htool '%c' -> H2Opus matvec against the source", storage);
        C.check_le(blackbox_relerr(B, S), err_bound, what);
        std::snprintf(what, sizeof what, "Htool '%c' -> H2Opus matvec against the kernel", storage);
        C.check_le(blackbox_relerr(B, K), err_bound + 2 * src_err, what);
        std::snprintf(what, sizeof what, "Htool '%c' source never asked for A^T", storage);
        C.check(cnt->calls_transposed == 0, what, "%ld", cnt->calls_transposed.load());
    }
    return C.finish("n=%d tol=%g", n, tol);
}
