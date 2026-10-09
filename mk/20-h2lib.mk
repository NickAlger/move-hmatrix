# H2Lib adapter (C): adapters/h2lib/hmb_h2lib.{h,c}, hmb_h2lib.hpp. Variables: H2LIB_CPPFLAGS
# (-I.../Library plus the -D flags libh2.a was built with: struct layouts depend on them),
# H2LIB_LIBS (libh2.a and its link dependencies).
ifeq ($(HAVE_H2LIB),1)

H2LIB_OBJS := $(OBJ)/adapters/h2lib/hmb_h2lib.o

# H2Lib's headers are only ever seen by C translation units (GOTCHA(header-clash)): the
# adapter and the tests' C helpers tests/test_h2lib_*.c.
$(OBJ)/adapters/h2lib/%.o: EXTRA_CFLAGS += $(H2LIB_CPPFLAGS)
$(OBJ)/tests/test_h2lib_%.o: EXTRA_CFLAGS += $(H2LIB_CPPFLAGS)

$(BIN)/test_h2lib: $(OBJ)/tests/test_h2lib.o $(OBJ)/tests/test_h2lib_native.o $(H2LIB_OBJS) $(LIBHMB)
	@mkdir -p $(@D)
	$(CXX) $(HMB_LDFLAGS) $^ $(H2LIB_LIBS) $(BLAS_LIBS) -o $@

TESTS += $(BIN)/test_h2lib

endif
