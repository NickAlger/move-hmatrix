# Htool-DDM adapter (header-only library): adapters/htool/hmb_htool.*, htool_native.hpp,
# htool_from_entries.hpp. Variables: HTOOL_CPPFLAGS (-I of the directory containing htool/).
ifeq ($(HAVE_HTOOL),1)

HTOOL_OBJS := $(OBJ)/adapters/htool/hmb_htool.o

# Any object under adapters/htool/ sees the Htool headers.
$(OBJ)/adapters/htool/%.o: EXTRA_CXXFLAGS += $(HTOOL_CPPFLAGS) -Wno-unused-variable -Wno-unused-but-set-variable

$(BIN)/test_htool_build: $(OBJ)/tests/test_htool_build.o $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_htool_build

endif
