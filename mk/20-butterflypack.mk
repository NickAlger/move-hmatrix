# ButterflyPACK adapter: adapters/butterflypack/{hmb_bpack.hpp, bpack_native.hpp, hmb_bpack.cpp,
# bpack_flush.f90}. Variables: BPACK_CPPFLAGS (-I of the directory holding dBPACK_wrapper.h),
# BPACK_LIBS (the double-precision library, its MPI stub, -lgfortran, -lgomp), and FC, which
# must be the Fortran compiler ButterflyPACK was built with (bpack_flush.f90 flushes the
# Fortran runtime's output units; one runtime per program).
ifeq ($(HAVE_BPACK),1)

BPACK_OBJS := $(OBJ)/adapters/butterflypack/hmb_bpack.o $(OBJ)/adapters/butterflypack/bpack_flush.o

$(OBJ)/adapters/butterflypack/hmb_bpack.o: EXTRA_CXXFLAGS += $(BPACK_CPPFLAGS)
# this test uses the native header (raw ButterflyPACK calls, a native solve)
$(OBJ)/tests/test_bpack_safety.o: EXTRA_CXXFLAGS += $(BPACK_CPPFLAGS)

# The only Fortran source of the package: a module-free subroutine, no include files.
$(OBJ)/adapters/butterflypack/%.o: adapters/butterflypack/%.f90
	@mkdir -p $(@D)
	$(FC) $(OPT) -fPIC -c $< -o $@

# Single-library tests: the sources are the test kernels (dense or entries).
# test_bpack_h2opus_common.o: the dense kernel operator shared with the H2Opus tests.
BPACK_TESTS := $(BIN)/test_bpack_matvec $(BIN)/test_bpack_entries $(BIN)/test_bpack_solve $(BIN)/test_bpack_safety \
               $(BIN)/test_bpack_stop

$(BPACK_TESTS): $(BIN)/%: $(OBJ)/tests/%.o $(OBJ)/tests/test_bpack_h2opus_common.o $(BPACK_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BPACK_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BPACK_TESTS)

endif
