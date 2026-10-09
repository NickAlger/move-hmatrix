# Htool-DDM block copy: adapters/htool/hmb_htool_copy.{hpp,cpp}, htool_export.hpp,
# htool_import.hpp (compile flags: the adapters/htool/ pattern rule of mk/20-htool.mk).
ifeq ($(HAVE_HTOOL),1)

HTOOL_COPY_OBJS := $(OBJ)/adapters/htool/hmb_htool_copy.o

# The test uses the native headers (Htool's own products and trees) besides the adapter API.
$(OBJ)/tests/test_htool_copy.o: EXTRA_CXXFLAGS += $(HTOOL_CPPFLAGS) -Wno-unused-variable -Wno-unused-but-set-variable

$(BIN)/test_htool_copy: $(OBJ)/tests/test_htool_copy.o $(HTOOL_COPY_OBJS) $(HTOOL_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_htool_copy

endif
