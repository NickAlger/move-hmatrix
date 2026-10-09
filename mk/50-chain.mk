# The four-library chain, examples/chain_four_libraries.cpp: Htool-DDM -> block copy -> H2Lib
# (H -> H2) -> matvecs -> ButterflyPACK (HODLR, factorization) -> solves -> H2Opus (HARA).
# Needs all four libraries. The example includes only neutral-facing headers, so it needs no
# extra flags.
ifeq ($(HAVE_HTOOL)$(HAVE_H2LIB)$(HAVE_BPACK)$(HAVE_H2OPUS),1111)

$(BIN)/chain_four_libraries: $(OBJ)/examples/chain_four_libraries.o $(HTOOL_COPY_OBJS) $(HTOOL_OBJS) $(H2LIB_OBJS) \
                             $(BPACK_OBJS) $(H2OPUS_HARA_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2LIB_LIBS) $(BPACK_LIBS) $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

EXAMPLES += $(BIN)/chain_four_libraries

endif
