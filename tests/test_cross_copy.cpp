// Every block-copy exporter into every block-copy importer, through the neutral form.
//
// Sources (each exported to a neutral view): Htool-DDM (general and symmetric storage),
// H2Lib (its own export of an imported matrix), hmat-oss (general and lower-symmetric
// storage), STARS-H TLR (general and symmetric) and H2Opus TLR. Destinations: Htool-DDM,
// H2Lib (copy and strict modes), hmat-oss, STARS-H and H2Opus TLR. For every pair the
// destination's own product must equal the source view's product to rounding ('N' and 'T',
// a non-symmetric kernel where storage allows), and the destination's re-export must too.
// H2Opus TLR takes uniform tile partitions only: it must refuse the hierarchical sources
// with an exception, and accept the TLR ones.
//
// Then the entry route from a view: entries_from_neutral serves the entries of every
// non-symmetric source (including libraries that have no entry access of their own) to every
// builder that compresses from entries (Htool-DDM, hmat-oss, STARS-H TLR, H2Opus TLR); the
// rebuilt matrix must match the view to the builders' tolerance.
//
// Built by mk/60-cross.mk with -DHMB_WITH_<LIB> for each configured library (at least two).
#include "hmb/neutral.h"
#include "hmb/neutral_bridge.hpp"
#include "hmb_test.hpp"
#ifdef HMB_WITH_HTOOL
#include "hmb_htool.hpp"
#include "hmb_htool_copy.hpp"
#endif
#ifdef HMB_WITH_H2LIB
#include "hmb_h2lib.hpp"
#endif
#ifdef HMB_WITH_HMAT
#include "hmb_hmatoss.hpp"
#endif
#ifdef HMB_WITH_STARSH
#include "hmb_starsh.hpp"
#endif
#ifdef HMB_WITH_H2OPUS
#include "hmb_h2opus_tlr.hpp"
#endif

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hmb;
using namespace hmb::test;

namespace {

using View = std::shared_ptr<const hmb_hmatrix>;

struct Source {
    std::string name;
    View view;
    bool uniform_tiles; // a TLR partition: H2Opus TLR must accept it
    bool from_P;        // built from the non-symmetric problem P (else from S)
};

// A builder that compresses from entries, as a BlackBox of its result.
struct Builder {
    std::string name;
    std::function<BlackBox(const EntrySource &, const Problem &, const std::vector<int> &perm)> build;
};

// An imported matrix: its product, and its re-export (null if the library has no exporter).
struct Imported {
    BlackBox product;
    std::function<View()> reexport;
};

struct Destination {
    std::string name;
    bool tiles_only; // accepts only uniform tile partitions (refusal expected otherwise)
    std::function<Imported(const hmb_hmatrix &)> import;
};

// relative difference of op(B) X and op(R) X on k random vectors
double product_diff(const BlackBox &B, const BlackBox &R, char trans) {
    return blackbox_relerr(B, R, trans, 3, trans == 'N' ? 11 : 13);
}

} // namespace

int main(int argc, char **argv) {
    Checker C("test_cross_copy");
    const int n = arg_int(argc, argv, "--n", 1600), tile = arg_int(argc, argv, "--tile", 300);
    const double eps = 1e-6, lossless = 1e-13;
    Problem P(n, Kernel::Nonsym, Geometry::ShuffledFibonacci); // non-symmetric: transposes show
    Problem S(n, Kernel::Sym, Geometry::RandomSphere);         // for one-triangle storage
    const std::vector<int> permP = bisection_perm(P, tile / 2), permS = bisection_perm(S, tile / 2);
    (void)permS;

    // ---- sources ----------------------------------------------------------------------
    std::vector<Source> sources;
#ifdef HMB_WITH_HTOOL
    {
        HtoolOptions o;
        o.epsilon = eps;
        sources.push_back({"Htool", htool_export(htool_build(P.entries(), 3, P.xyz.data(), o)), false, true});
        o.symmetry = 'S';
        sources.push_back({"Htool sym", htool_export(htool_build(S.entries(), 3, S.xyz.data(), o)), false, false});
    }
#endif
#if defined(HMB_WITH_H2LIB) && defined(HMB_WITH_HTOOL)
    // H2Lib's own export (U = A, V = B, vtrans 1; permutations from its clusters)
    sources.push_back({"H2Lib", h2lib_export(h2lib_import(*sources[0].view, HMB_H2LIB_COPY)), false, true});
#endif
#ifdef HMB_WITH_HMAT
    {
        HmatossOptions o;
        o.epsilon = eps;
        sources.push_back({"hmat-oss", hmatoss_export(hmatoss_build(P.entries(), 3, P.xyz.data(), o)), false, true});
        o.lower_symmetric = true;
        sources.push_back({"hmat-oss sym", hmatoss_export(hmatoss_build(S.entries(), 3, S.xyz.data(), o)), false, false});
    }
#endif
#ifdef HMB_WITH_STARSH
    {
        hmb_starsh_tlr_opts o = {tile, eps, tile / 3, 'N', 1};
        sources.push_back({"STARS-H TLR", starsh_export(starsh_from_entries(P.entries(), permP.data(), o)), true, true});
        o.symm = 'S';
        sources.push_back({"STARS-H TLR sym", starsh_export(starsh_from_entries(S.entries(), permS.data(), o)), true, false});
    }
#endif
#ifdef HMB_WITH_H2OPUS
    {
        H2opusTlrOptions o;
        o.block_size = tile;
        o.eps = 1e-4; // absolute, per tile
        sources.push_back({"H2Opus TLR", h2opus_tlr_export(h2opus_tlr_from_entries(P.entries(), permP.data(), o)), true, true});
    }
#endif

    // ---- destinations -----------------------------------------------------------------
    std::vector<Destination> dests;
#ifdef HMB_WITH_HTOOL
    dests.push_back({"Htool", false, [](const hmb_hmatrix &A) {
                         HtoolHandle H = htool_import(A);
                         return Imported{htool_blackbox(H), [H] { return htool_export(H); }};
                     }});
#endif
#ifdef HMB_WITH_H2LIB
    for (int mode : {HMB_H2LIB_COPY, HMB_H2LIB_STRICT})
        dests.push_back({mode == HMB_H2LIB_COPY ? "H2Lib copy" : "H2Lib strict", false, [mode](const hmb_hmatrix &A) {
                             H2libHandle H = h2lib_import(A, mode);
                             return Imported{h2lib_blackbox(H), [H] { return h2lib_export(H); }};
                         }});
#endif
#ifdef HMB_WITH_HMAT
    dests.push_back({"hmat-oss", false, [](const hmb_hmatrix &A) {
                         HmatossHandle H = hmatoss_import(A);
                         return Imported{hmatoss_blackbox(H), [H] { return hmatoss_export(H); }};
                     }});
#endif
#ifdef HMB_WITH_STARSH
    dests.push_back({"STARS-H", false, [](const hmb_hmatrix &A) {
                         StarshHandle H = starsh_import(A);
                         return Imported{starsh_blackbox(H), [H] { return starsh_export(H); }};
                     }});
#endif
#ifdef HMB_WITH_H2OPUS
    dests.push_back({"H2Opus TLR", true, [](const hmb_hmatrix &A) {
                         H2opusTlrHandle H = h2opus_tlr_import(A);
                         return Imported{h2opus_tlr_blackbox(H), [H] { return h2opus_tlr_export(H); }};
                     }});
#endif

    std::printf("n = %d, %zu sources x %zu destinations; lossless means a relative product difference <= %.0e\n", n,
                sources.size(), dests.size(), lossless);
    char what[160], msg[256];
    for (const Source &s : sources) {
        C.check(hmb_validate(s.view.get(), msg, sizeof msg) == 0, (s.name + " export validates").c_str(), "%s", msg);
        const BlackBox ref = blackbox_from_neutral(s.view);
        std::printf("source %-16s %6zu stored values, %5d blocks\n", s.name.c_str(), hmb_stored_values(s.view.get()),
                    s.view->nbl);
        for (const Destination &d : dests) {
            std::snprintf(what, sizeof what, "%s -> %s", s.name.c_str(), d.name.c_str());
            if (d.tiles_only && !s.uniform_tiles) {
                bool refused = false;
                std::string why;
                try {
                    d.import(*s.view);
                } catch (const std::exception &e) {
                    refused = true;
                    why = e.what();
                }
                C.check(refused, (std::string(what) + " refused (not uniform tiles)").c_str(), "%.90s", why.c_str());
                continue;
            }
            try {
                Imported I = d.import(*s.view);
                const double eN = product_diff(I.product, ref, 'N'), eT = product_diff(I.product, ref, 'T');
                C.check_le(std::max(eN, eT), lossless, (std::string(what) + " product (N, T)").c_str());
                View back = I.reexport();
                const BlackBox again = blackbox_from_neutral(back);
                const double eR = std::max(product_diff(again, ref, 'N'), product_diff(again, ref, 'T'));
                C.check(hmb_validate(back.get(), msg, sizeof msg) == 0 && eR <= lossless,
                        (std::string(what) + " re-export").c_str(), "%.3e <= %.0e %s", eR, lossless, msg);
            } catch (const std::exception &e) {
                C.check(false, what, "threw: %.120s", e.what());
            }
        }
    }
    // ---- the entry route from a view --------------------------------------------------
    std::vector<Builder> builders;
#ifdef HMB_WITH_HTOOL
    builders.push_back({"Htool build", [eps](const EntrySource &E, const Problem &Q, const std::vector<int> &) {
                            HtoolOptions o;
                            o.epsilon = eps;
                            return htool_blackbox(htool_build(E, 3, Q.xyz.data(), o));
                        }});
#endif
#ifdef HMB_WITH_HMAT
    builders.push_back({"hmat-oss build", [eps](const EntrySource &E, const Problem &Q, const std::vector<int> &) {
                            HmatossOptions o;
                            o.epsilon = eps;
                            return hmatoss_blackbox(hmatoss_build(E, 3, Q.xyz.data(), o));
                        }});
#endif
#ifdef HMB_WITH_STARSH
    builders.push_back({"STARS-H TLR", [eps, tile](const EntrySource &E, const Problem &, const std::vector<int> &perm) {
                            hmb_starsh_tlr_opts o = {tile, eps, tile / 3, 'N', 1};
                            return starsh_blackbox(starsh_from_entries(E, perm.data(), o));
                        }});
#endif
#ifdef HMB_WITH_H2OPUS
    builders.push_back({"H2Opus TLR", [tile](const EntrySource &E, const Problem &, const std::vector<int> &perm) {
                            H2opusTlrOptions o;
                            o.block_size = tile;
                            o.eps = 1e-4; // absolute, per tile
                            return h2opus_tlr_blackbox(h2opus_tlr_from_entries(E, perm.data(), o));
                        }});
#endif
    const double rebuilt = 10 * eps; // the builders' tolerances, relative to the view
    for (const Source &s : sources) {
        if (!s.from_P) continue;
        const BlackBox ref = blackbox_from_neutral(s.view);
        for (const Builder &b : builders) {
            std::snprintf(what, sizeof what, "entries of %s -> %s", s.name.c_str(), b.name.c_str());
            try {
                const double t0 = now();
                BlackBox R = b.build(entries_from_neutral(s.view), P, permP);
                const double secs = now() - t0;
                const double e = std::max(product_diff(R, ref, 'N'), product_diff(R, ref, 'T'));
                C.check(e <= rebuilt, what, "%.3e <= %.0e (%.2f s)", e, rebuilt, secs);
            } catch (const std::exception &ex) {
                C.check(false, what, "threw: %.120s", ex.what());
            }
        }
    }
    return C.finish("n=%d, %zu x %zu pairs, %zu entry builders", n, sources.size(), dests.size(), builders.size());
}
