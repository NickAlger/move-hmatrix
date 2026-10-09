/* hmatrix-bridges: link-time fix for H2Pack's file reader; see h2pack_reader_fix.h.
 * Link with -Wl,--wrap=H2P_calc_node_inadm_lists. */
#include "h2pack_reader_fix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* before H2Pack.h, whose H2Pack_aux_structs.h uses memcpy without it */

#include "H2Pack.h"

static int fix_calls;

void __wrap_H2P_calc_node_inadm_lists(H2Pack_p h2pack);

/* GOTCHA(h2pack-reader-bugs): replaces the reader's call, which reads one bounding box per
 * node from a one-box array and may write past node_inadm_lists. */
void __wrap_H2P_calc_node_inadm_lists(H2Pack_p h) {
  const size_t nn = (size_t)h->n_node;
  h->node_inadm_lists = (int *)calloc(nn * (size_t)h->max_neighbor + 1, sizeof(int));
  h->node_n_r_inadm = (int *)malloc(sizeof(int) * (nn + 1));
  if (!h->node_inadm_lists || !h->node_n_r_inadm) {
    fprintf(stderr, "h2pack_reader_fix: out of memory\n");
    abort();
  }
  /* computed by H2P_read_from_file from the file's admissible pairs before this call */
  memcpy(h->node_n_r_inadm, h->node_n_r_adm, sizeof(int) * nn);
  fix_calls++;
}

int hmb_h2pack_reader_fix_calls(void) { return fix_calls; }
