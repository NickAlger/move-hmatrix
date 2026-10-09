# Core: libhmb.a is built by the top-level Makefile from src/*.c and src/*.cpp.
# Tests of the core utilities (neutral.c, cstore.c, neutral_store.hpp, neutral_bridge.hpp)
# register here. They need no third-party library beyond BLAS.

CORE_TESTS := $(BIN)/test_neutral

$(BIN)/test_neutral: $(OBJ)/tests/test_neutral.o $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(BLAS_LIBS) -o $@

TESTS += $(CORE_TESTS)
