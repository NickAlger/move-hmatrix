// Block copy STARS-H TLR -> H2Opus TLR (lossless), with the cost of the transfer.
//
// STARS-H builds the tile low-rank approximation of the Sym kernel on N points of the sphere
// (starsh_blrm_approximate on tiles of a clustering order), the adapter exports it (zero-copy)
// and H2Opus imports it into a TTLR_Matrix with the same tiles: STARS-H's tiles are uniform
// except the last, which is exactly H2Opus's layout, with the last tile padded. Checks: the
// H2Opus product equals STARS-H's (lossless) for A and A^T, both approximate the kernel to the
// tolerance. Reports the transfer time (export + import, best of --reps runs) in units of one
// STARS-H matvec (one vector, best of --reps), the values copied, the memory the import adds,
// and, as the alternative route, H2Opus rebuilding its TLR matrix from the kernel's entries.
//
//   copy_tlr_starsh_h2opus [--n 4000] [--tile 300] [--eps 1e-6] [--maxrank tile/3]
//                          [--mode N|S|both] [--reps 5] [--no-rebuild]
// --mode S stores one triangle on both sides (STARS-H symm 'S', H2OpusTLR_Symmetric).
#include "hmb_h2opus_tlr.hpp"
#include "hmb_starsh.hpp"
#include "hmb_test.hpp"

#include <memory>
#include <string>

using namespace hmb;
using namespace hmb::test;

int main(int argc, char **argv) {
    Checker C("copy_tlr_starsh_h2opus");
    const int n = arg_int(argc, argv, "--n", 4000), tile = arg_int(argc, argv, "--tile", 300);
    const int maxrank = arg_int(argc, argv, "--maxrank", tile / 3), reps = arg_int(argc, argv, "--reps", 5);
    const double eps = arg_dbl(argc, argv, "--eps", 1e-6);
    const std::string mode = arg_str(argc, argv, "--mode", "both");
    const bool rebuild = !arg_flag(argc, argv, "--no-rebuild");
    std::printf("N=%d tile=%d (%d tiles, last %d) eps=%.0e maxrank=%d threads=%s\n", n, tile, (n + tile - 1) / tile,
                n - (n - 1) / tile * tile, eps, maxrank, std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "default");

    Problem P(n, Kernel::Sym, Geometry::RandomSphere);
    const std::vector<int> perm = bisection_perm(P, tile / 2);
    const int k = 4;
    const std::vector<double> X = randvec(size_t(n) * k, 17);
    std::vector<double> Yexact(size_t(n) * k);
    P.exact_apply('N', k, X.data(), Yexact.data());

    for (char symm : {'N', 'S'}) {
        if (mode != "both" && mode[0] != symm) continue;
        const std::string tag = std::string("symm ") + symm;
        // ---- source: STARS-H's TLR approximation ------------------------------------------
        const hmb_starsh_tlr_opts o{tile, eps, maxrank, symm, 1};
        StarshHandle S;
        double t0 = now();
        try {
            S = starsh_from_entries(P.entries(), perm.data(), o);
        } catch (const std::exception &e) {
            C.check(false, (tag + ": STARS-H build").c_str(), "%s", e.what());
            continue;
        }
        const double t_build = now() - t0;
        const BlackBox SB = starsh_blackbox(S); // STARS-H's product, ORIGINAL ordering
        // Timings are the best of reps runs (the machine may be shared; OpenMP threads that
        // wait for a busy core inflate single runs). One source matvec: one vector, ORIGINAL
        // ordering.
        std::vector<double> x1(X.begin(), X.begin() + n), y1(n);
        double t_mv = 1e300;
        for (int r = 0; r < reps; r++) {
            t0 = now();
            SB.apply('N', 1, x1.data(), y1.data());
            t_mv = std::min(t_mv, now() - t0);
        }

        // ---- transfer: export (zero-copy) + import (copy) ---------------------------------
        hmb_starsh_export_stats es{};
        H2opusTlrStats ts;
        H2opusTlrImportOptions io;
        io.symmetric = symm == 'S';
        std::shared_ptr<const hmb_hmatrix> V;
        H2opusTlrHandle H;
        double t_export = 1e300, t_import = 1e300;
        long import_kb = 0;
        std::string error;
        try {
            for (int r = 0; r < reps; r++) {
                V.reset();
                t0 = now();
                V = starsh_export(S, &es);
                t_export = std::min(t_export, now() - t0);
                H.reset();
                const long rss0 = rss_kb();
                t0 = now();
                H = h2opus_tlr_import(*V, io, &ts);
                t_import = std::min(t_import, now() - t0);
                if (r == 0) import_kb = rss_kb() - rss0;
            }
        } catch (const std::exception &e) {
            error = e.what();
        }
        if (!C.check(error.empty(), (tag + ": export and import").c_str(), "%s",
                     error.empty() ? "STARS-H -> neutral view -> H2Opus TLR" : error.c_str()))
            continue;
        const double cost = (t_export + t_import) / t_mv;

        // ---- checks -------------------------------------------------------------------------
        std::vector<double> Ys(size_t(n) * k), Yh(Ys.size()), Yst(Ys.size()), Yht(Ys.size());
        SB.apply('N', k, X.data(), Ys.data());
        h2opus_tlr_matvec(H, 'N', k, X.data(), n, Yh.data(), n);
        const double lossless = relerr(Yh, Ys);
        C.check_le(lossless, 1e-13, (tag + ": H2Opus product = STARS-H product").c_str());
        // A^T: STARS-H's BlackBox answers it through the exported leaves ('N') or with its own
        // product (symmetric format, its own transpose)
        SB.apply('T', k, X.data(), Yst.data());
        h2opus_tlr_matvec(H, 'T', k, X.data(), n, Yht.data(), n);
        C.check_le(relerr(Yht, Yst), 1e-13, (tag + ": H2Opus A^T = STARS-H BlackBox A^T").c_str());
        const double err_s = relerr(Ys, Yexact), err_h = relerr(Yh, Yexact);
        C.check_le(err_s, 10 * eps, (tag + ": STARS-H vs kernel (<= 10 eps)").c_str());
        C.check_le(err_h, 10 * eps, (tag + ": H2Opus vs kernel (<= 10 eps)").c_str());

        double t_mv_h = 1e300;
        for (int r = 0; r < reps; r++) {
            t0 = now();
            h2opus_tlr_matvec(H, 'N', 1, x1.data(), n, y1.data(), n);
            t_mv_h = std::min(t_mv_h, now() - t0);
        }
        std::printf("  %s: STARS-H build %.2fs (%lld far + %lld near tiles, max rank %d), one STARS-H matvec %.4fs, "
                    "one H2Opus matvec %.4fs\n",
                    tag.c_str(), t_build, es.far_blocks, es.near_blocks, es.max_rank, t_mv, t_mv_h);
        std::printf("  %s: export %.5fs (%lld values aliased, %lld copied, %lld mirrored tiles), import %.4fs (%lld values "
                    "copied, %d low-rank diagonal tiles expanded, %d dense off-diagonal tiles as D*I)\n",
                    tag.c_str(), t_export, es.aliased_values, es.copied_values, es.mirrored_blocks, t_import,
                    ts.copied_values, ts.lowrank_diagonal_tiles, ts.dense_offdiagonal_tiles);
        std::printf("  %s: H2Opus stores %lld values (%d tiles of %d, padded to %d), +%ld kB resident; transfer cost "
                    "%.2f source matvecs; accuracy STARS-H %.2e, H2Opus %.2e, STARS-H vs H2Opus %.2e\n",
                    tag.c_str(), ts.stored_values, ts.n_block, ts.block_size, ts.padded_n, import_kb, cost, err_s,
                    err_h, lossless);
        std::printf("RESULT copy_tlr_starsh_h2opus symm=%c N=%d tile=%d eps=%.0e cost_matvecs=%.2f values_copied=%lld "
                    "lossless=%.1e err=%.2e\n",
                    symm, n, tile, eps, cost, ts.copied_values + es.copied_values, lossless, err_h);

        // ---- alternative: H2Opus rebuilds its TLR matrix from the kernel's entries ----------
        if (rebuild && symm == 'N') {
            double normF = 0; // H2Opus's tolerance is absolute per tile: aim at eps * |A|_F overall
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) normF += P.K(i, j) * P.K(i, j);
            H2opusTlrOptions ro;
            ro.block_size = tile, ro.eps = eps * std::sqrt(normF) / ts.n_block, ro.symmetric = false;
            H2opusTlrStats rs;
            H2opusTlrHandle R = h2opus_tlr_from_entries(P.entries(), perm.data(), ro, &rs);
            std::vector<double> Yr(size_t(n) * k);
            h2opus_tlr_matvec(R, 'N', k, X.data(), n, Yr.data(), n);
            const double rebuild_err = relerr(Yr, Yexact);
            std::printf("  rebuild from entries in H2Opus (construct_tlr_matrix, eps_abs %.2e): %.2fs = %.0f source "
                        "matvecs, %lld entries, max rank %d, accuracy %.2e\n",
                        ro.eps, rs.seconds, rs.seconds / t_mv, rs.copied_values, rs.max_rank, rebuild_err);
            C.check_le(rebuild_err, 10 * eps, "rebuild from entries vs kernel (<= 10 eps)");
        }
    }
    return C.finish("N=%d tile=%d eps=%.0e", n, tile, eps);
}
