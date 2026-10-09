# H2Pack file bridge (tools/h2pack/): h2pack_write.py writes H^2 matrices in H2Pack's file
# format; h2pack_read_check reads them with H2P_read_from_file (linked with the reader fix)
# and multiplies; tests/test_h2pack_file.sh drives both.
# Variables: H2PACK_CPPFLAGS (H2Pack's include directory and the -D it was built with),
# H2PACK_LIBS (a STATIC libH2Pack.a, which the reader fix needs, and its BLAS/LAPACKE),
# PYTHON (with numpy), and H2PACK_META_SCRIPT: H2Pack's examples/meta_json_to_txt.py, which
# H2P_read_from_file runs; by default found next to the include directory of H2PACK_CPPFLAGS.
ifeq ($(HAVE_H2PACK),1)

H2PACK_META_SCRIPT ?= $(firstword $(wildcard $(addsuffix /../examples/meta_json_to_txt.py,$(patsubst -I%,%,$(filter -I%,$(H2PACK_CPPFLAGS))))))

H2PACK_TOOL_OBJS := $(OBJ)/tools/h2pack/h2pack_read_check.o $(OBJ)/tools/h2pack/h2pack_reader_fix.o
# -Wno-cpp: H2Pack's SIMD header warns (#warning) when not compiled with its -march=native
$(H2PACK_TOOL_OBJS): EXTRA_CFLAGS += $(H2PACK_CPPFLAGS) -Wno-cpp

# GOTCHA(h2pack-reader-bugs): the reader's call to H2P_calc_node_inadm_lists goes to the fix
$(BIN)/h2pack_read_check: $(H2PACK_TOOL_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) -Wl,--wrap=H2P_calc_node_inadm_lists $^ $(H2PACK_LIBS) $(BLAS_LIBS) -o $@

TOOLS += $(BIN)/h2pack_read_check

SCRIPT_CHECKS += tests/test_h2pack_file.sh
# what the test script needs (it also runs stand-alone with these set by hand)
check: export HMB_PYTHON := $(PYTHON)
check: export HMB_H2PACK_READ_CHECK := $(abspath $(BIN)/h2pack_read_check)
check: export HMB_H2PACK_META_SCRIPT := $(abspath $(H2PACK_META_SCRIPT))

endif
