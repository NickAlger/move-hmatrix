# H2Opus adapter, HARA part: adapters/h2opus/{hmb_h2opus.hpp, h2opus_native.hpp, h2opus_hara.cpp}.
# (The TLR part of adapters/h2opus/ has its own fragment, mk/22-h2opus-tlr.mk.)
# Variables: H2OPUS_CPPFLAGS (include directories: the generated h2opusconf.h, H2Opus's
# include/, thrust and cub), H2OPUS_LIBS (the library or its objects).
ifeq ($(HAVE_H2OPUS),1)

H2OPUS_HARA_OBJS := $(OBJ)/adapters/h2opus/h2opus_hara.o

$(OBJ)/adapters/h2opus/h2opus_hara.o: EXTRA_CXXFLAGS += $(H2OPUS_CPPFLAGS)
# this test uses the native header (the H^2 matrix and its permutation)
$(OBJ)/tests/test_h2opus_hara.o: EXTRA_CXXFLAGS += $(H2OPUS_CPPFLAGS)

# Single-library test: the sources are the test kernels (dense, from test_bpack_h2opus_common.o).
$(BIN)/test_h2opus_hara: $(OBJ)/tests/test_h2opus_hara.o $(OBJ)/tests/test_bpack_h2opus_common.o $(H2OPUS_HARA_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_h2opus_hara

endif
