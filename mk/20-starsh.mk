# STARS-H adapter: adapters/starsh/hmb_starsh.{h,c} (C, the only adapter file that includes
# starsh.h) and hmb_starsh.hpp (C++ conveniences over the C interface, no STARS-H header).
# Variables: STARSH_CPPFLAGS (STARS-H include directory, plus the directory of the
# CBLAS/LAPACKE headers it includes), STARSH_LIBS (libstarsh.a, LAPACKE, CBLAS/BLAS).
ifeq ($(HAVE_STARSH),1)

STARSH_OBJS := $(OBJ)/adapters/starsh/hmb_starsh.o

$(OBJ)/adapters/starsh/hmb_starsh.o: EXTRA_CFLAGS += $(STARSH_CPPFLAGS)
# test helper that builds matrices with STARS-H's API directly
$(OBJ)/tests/test_starsh_native.o: EXTRA_CFLAGS += $(STARSH_CPPFLAGS)

$(BIN)/test_starsh: $(OBJ)/tests/test_starsh.o $(OBJ)/tests/test_starsh_native.o $(STARSH_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(STARSH_LIBS) $(BLAS_LIBS) -o $@

# the C++ interface, compiled without STARSH_CPPFLAGS (hmb_starsh.hpp needs no STARS-H header)
$(BIN)/test_starsh_cpp: $(OBJ)/tests/test_starsh_cpp.o $(STARSH_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(STARSH_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_starsh $(BIN)/test_starsh_cpp

endif
