# Every block-copy exporter into every block-copy importer: tests/test_cross_copy.cpp, built
# with the block-copy libraries that are configured (at least two). The test sees only
# neutral-facing headers; -DHMB_WITH_<LIB> tells it which ones exist.
CROSS_LIBS := $(strip $(if $(HAVE_HTOOL),HTOOL) $(if $(HAVE_H2LIB),H2LIB) $(if $(HAVE_HMAT),HMAT) \
                $(if $(HAVE_STARSH),STARSH) $(if $(HAVE_H2OPUS),H2OPUS))

ifneq ($(word 2,$(CROSS_LIBS)),)

CROSS_OBJS := $(if $(HAVE_HTOOL),$(HTOOL_COPY_OBJS) $(HTOOL_OBJS)) $(if $(HAVE_H2LIB),$(H2LIB_OBJS)) \
              $(if $(HAVE_HMAT),$(HMATOSS_OBJS)) $(if $(HAVE_STARSH),$(STARSH_OBJS)) \
              $(if $(HAVE_H2OPUS),$(H2OPUS_TLR_OBJS))
CROSS_LDLIBS := $(if $(HAVE_H2LIB),$(H2LIB_LIBS)) $(if $(HAVE_HMAT),$(HMAT_LIBS)) $(if $(HAVE_STARSH),$(STARSH_LIBS)) \
                $(if $(HAVE_H2OPUS),$(H2OPUS_LIBS))

$(OBJ)/tests/test_cross_copy.o: EXTRA_CXXFLAGS += $(addprefix -DHMB_WITH_,$(CROSS_LIBS))

$(BIN)/test_cross_copy: $(OBJ)/tests/test_cross_copy.o $(CROSS_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(CROSS_LDLIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_cross_copy

endif
