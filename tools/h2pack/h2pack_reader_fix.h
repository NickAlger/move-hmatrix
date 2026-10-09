/* hmatrix-bridges: link-time fix for H2Pack's file reader (GOTCHA(h2pack-reader-bugs)).
 *
 * The bug (H2Pack 2aa58a6, H2Pack_file_IO.c:744-745): H2P_read_from_file computes ONE
 * bounding box for all points (H2P_calc_enclosing_box) and then calls
 * H2P_calc_node_inadm_lists, which reads one box per node (enbox + node * 2 * pt_dim): a heap
 * over-read for every tree with more than one node. From the garbage boxes it collects the
 * nodes that look "H2-inadmissible but HSS-admissible", storing up to max_neighbor = 3^pt_dim of
 * them per node in node_inadm_lists, and writes past that array when the garbage yields more:
 * heap corruption, seen as "double free or corruption (out)" at exit or a crash later,
 * depending on the heap layout (it changed with the thread count and with stdout being a
 * pipe in our runs). The lists matter only for HSS matrices: H2P_matvec of an HSS matrix reads
 * node_n_r_inadm as each node's number of admissible pairs, so a file-read HSS matrix crashes
 * or multiplies wrongly.
 *
 * The fix: link the program that calls H2P_read_from_file with h2pack_reader_fix.o and
 *     -Wl,--wrap=H2P_calc_node_inadm_lists
 * The reader's call then runs __wrap_H2P_calc_node_inadm_lists (below) instead: it allocates
 * node_inadm_lists zeroed (only H2Pack's HSS construction reads it, never after a file read)
 * and sets node_n_r_inadm to the per-node admissible-pair counts the reader computed from the
 * file (what HSS products need). H2Pack's own HSS construction calls the original function from
 * inside its own object file and is not redirected. Needs a GNU-compatible linker (bfd, gold,
 * lld) and a STATIC libH2Pack.a: calls inside a shared libH2Pack.so are not redirected;
 * hmb_h2pack_reader_fix_calls() tells whether the fix ran.
 */
#ifndef HMB_H2PACK_READER_FIX_H
#define HMB_H2PACK_READER_FIX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of H2P_read_from_file calls the fix has handled in this process (0: not in effect). */
int hmb_h2pack_reader_fix_calls(void);

#ifdef __cplusplus
}
#endif
#endif
