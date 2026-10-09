# hmat-oss adapter: adapters/hmatoss/hmb_hmatoss.hpp (neutral-facing), hmatoss_native.hpp
# (native), hmatoss_stream.hpp (internal), hmb_hmatoss.cpp (build, wrap, entries, matvec),
# hmb_hmatoss_copy.cpp (block copy through hmat-oss's serialization stream).
# Variables: HMAT_CPPFLAGS (hmat-oss source include/ and the build directory holding
# hmat/config.h), HMAT_LIBS (-L<build> -lhmat with an rpath; libhmat brings its own BLAS).
ifeq ($(HAVE_HMAT),1)

HMATOSS_OBJS := $(OBJ)/adapters/hmatoss/hmb_hmatoss.o $(OBJ)/adapters/hmatoss/hmb_hmatoss_copy.o

# The adapter and its tests see hmat.h (the tests drive hmat-oss directly to show the gotchas).
$(OBJ)/adapters/hmatoss/%.o: EXTRA_CXXFLAGS += $(HMAT_CPPFLAGS)
$(OBJ)/tests/test_hmatoss%.o: EXTRA_CXXFLAGS += $(HMAT_CPPFLAGS)

$(BIN)/test_hmatoss_entries: $(OBJ)/tests/test_hmatoss_entries.o $(HMATOSS_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(HMAT_LIBS) $(BLAS_LIBS) -o $@

$(BIN)/test_hmatoss_copy: $(OBJ)/tests/test_hmatoss_copy.o $(HMATOSS_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(HMAT_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_hmatoss_entries $(BIN)/test_hmatoss_copy

endif
