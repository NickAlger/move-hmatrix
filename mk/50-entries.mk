# Rebuild from entries: hmat-oss -> Htool-DDM (examples/entries_hmatoss_htool.cpp, the study's
# experiment E3). Needs the hmat-oss and Htool-DDM adapters (mk/20-hmatoss.mk, mk/20-htool.mk).
ifeq ($(HAVE_HMAT),1)
ifeq ($(HAVE_HTOOL),1)

$(BIN)/entries_hmatoss_htool: $(OBJ)/examples/entries_hmatoss_htool.o $(HMATOSS_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(HMAT_LIBS) $(BLAS_LIBS) -o $@

EXAMPLES += $(BIN)/entries_hmatoss_htool

endif
endif
