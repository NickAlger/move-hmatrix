# Routes between the tile low-rank libraries: examples/copy_tlr_starsh_h2opus.cpp (block copy
# STARS-H -> H2Opus TLR). Needs the STARS-H (20-starsh.mk) and H2Opus TLR (22-h2opus-tlr.mk)
# adapters.
ifeq ($(HAVE_STARSH)$(HAVE_H2OPUS),11)

$(BIN)/copy_tlr_starsh_h2opus: $(OBJ)/examples/copy_tlr_starsh_h2opus.o $(STARSH_OBJS) $(H2OPUS_TLR_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(STARSH_LIBS) $(H2OPUS_LIBS) $(BLAS_LIBS) -o $@

EXAMPLES += $(BIN)/copy_tlr_starsh_h2opus

endif
