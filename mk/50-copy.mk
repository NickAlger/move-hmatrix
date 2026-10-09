# Block copy Htool-DDM -> H2Lib, then H -> H2 inside H2Lib (the study's route E1): the example
# examples/copy_htool_h2lib.cpp and the combined test tests/test_h2lib_htool.cpp.
ifeq ($(HAVE_HTOOL),1)
ifeq ($(HAVE_H2LIB),1)

# The example times Htool's own product (the cost unit), so it sees Htool's headers; it uses
# H2Lib only through hmb_h2lib.h / hmb_h2lib.hpp (GOTCHA(header-clash)).
$(OBJ)/examples/copy_htool_h2lib.o: EXTRA_CXXFLAGS += $(HTOOL_CPPFLAGS) -Wno-unused-variable -Wno-unused-but-set-variable

COPY_ROUTE_OBJS := $(HTOOL_COPY_OBJS) $(HTOOL_OBJS) $(H2LIB_OBJS)

$(BIN)/copy_htool_h2lib: $(OBJ)/examples/copy_htool_h2lib.o $(COPY_ROUTE_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2LIB_LIBS) $(BLAS_LIBS) -o $@

$(BIN)/test_h2lib_htool: $(OBJ)/tests/test_h2lib_htool.o $(COPY_ROUTE_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2LIB_LIBS) $(BLAS_LIBS) -o $@

EXAMPLES += $(BIN)/copy_htool_h2lib
TESTS += $(BIN)/test_h2lib_htool

endif
endif
