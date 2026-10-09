# H2Opus TLR adapter: adapters/h2opus/{hmb_h2opus_tlr.hpp, h2opus_tlr_native.hpp,
# h2opus_tlr.cpp}. The rest of adapters/h2opus/ (HARA) has its own fragment (20-h2opus.mk).
# Variables: H2OPUS_CPPFLAGS (H2Opus include directories with the generated h2opusconf.h,
# thrust, cub), H2OPUS_LIBS (the library or its objects).
ifeq ($(HAVE_H2OPUS),1)

H2OPUS_TLR_OBJS := $(OBJ)/adapters/h2opus/h2opus_tlr.o

$(OBJ)/adapters/h2opus/h2opus_tlr.o: EXTRA_CXXFLAGS += $(H2OPUS_CPPFLAGS)

$(BIN)/test_h2opus_tlr: $(OBJ)/tests/test_h2opus_tlr.o $(H2OPUS_TLR_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

# H2Opus's own objects through the native header (reproduces the library's TLR quirks)
$(OBJ)/tests/test_h2opus_tlr_native.o: EXTRA_CXXFLAGS += $(H2OPUS_CPPFLAGS)

$(BIN)/test_h2opus_tlr_native: $(OBJ)/tests/test_h2opus_tlr_native.o $(H2OPUS_TLR_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_h2opus_tlr $(BIN)/test_h2opus_tlr_native

endif
