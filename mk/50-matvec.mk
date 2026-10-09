# Matvec route across libraries: Htool-DDM matrices as BlackBox sources for ButterflyPACK
# (bpack_from_matvec) and H2Opus (h2opus_from_matvec, HARA). All four programs include only
# neutral-facing headers (Htool and H2Opus headers must not meet), so they need no extra flags.
#   tests:    test_bpack_htool, test_h2opus_hara_htool
#   examples: matvec_htool_bpack, matvec_htool_h2opus (reproduce E2's runs; see their headers)
ifeq ($(HAVE_HTOOL),1)

ifeq ($(HAVE_BPACK),1)
$(BIN)/test_bpack_htool: $(OBJ)/tests/test_bpack_htool.o $(OBJ)/tests/test_bpack_h2opus_common.o $(BPACK_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BPACK_LIBS) $(BLAS_LIBS) -o $@

$(BIN)/matvec_htool_bpack: $(OBJ)/examples/matvec_htool_bpack.o $(BPACK_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BPACK_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_bpack_htool
EXAMPLES += $(BIN)/matvec_htool_bpack
endif

ifeq ($(HAVE_H2OPUS),1)
$(BIN)/test_h2opus_hara_htool: $(OBJ)/tests/test_h2opus_hara_htool.o $(OBJ)/tests/test_bpack_h2opus_common.o $(H2OPUS_HARA_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

$(BIN)/matvec_htool_h2opus: $(OBJ)/examples/matvec_htool_h2opus.o $(H2OPUS_HARA_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_h2opus_hara_htool
EXAMPLES += $(BIN)/matvec_htool_h2opus
endif

endif
