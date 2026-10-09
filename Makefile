# hmatrix-bridges: top-level Makefile (GNU make).
#
#   make            build libhmb.a and every adapter, example and test whose libraries
#                   are configured, and one archive per adapter (build/lib/libhmb_<lib>.a)
#   make check      build, then run all tests and examples (tests/run_tests.sh)
#   make info       show which libraries are enabled, what will be built, and the link
#                   flags for using the archives from your own program
#   make clean
#   make B=<dir>    any of the above in build directory <dir> instead of build/ (for
#                   example a sanitizer build next to the normal one; see LAYOUT.md)
#
# Library locations come from a configuration file: ./config.mk if it exists (hand-written,
# start from config.mk.example), otherwise deps/install/config.mk (written by
# deps/build_deps.sh). Override with `make CONFIG=path/to/file.mk`.
#
# A library is enabled when its <LIB>_CPPFLAGS variable is non-empty, so a partial
# configuration builds the routes it can and skips the rest. See LAYOUT.md for the
# conventions the mk/*.mk fragments follow.

ifeq ($(wildcard config.mk),)
CONFIG ?= deps/install/config.mk
else
CONFIG ?= config.mk
endif
-include $(CONFIG)

# ---- toolchain (the configuration may override any of these) ---------------------------
ifeq ($(origin CC),default)
CC := gcc
endif
ifeq ($(origin CXX),default)
CXX := g++
endif
ifeq ($(origin FC),default)
FC := gfortran
endif
OPT       ?= -O2
OPENMP    ?= -fopenmp
BLAS_LIBS ?= -llapack -lblas
PYTHON    ?= python3

# ---- rebuild when the configuration changes ---------------------------------------------
# $(B)/config.stamp records every configuration value and the time stamps of the static
# libraries the configuration names. It is rewritten only when that record changes, and
# every object depends on it: editing config.mk, overriding a variable on the command line,
# or rebuilding a static library in place recompiles and relinks what depends on it.
CONFIG_VARS   := CC CXX FC OPT OPENMP BLAS_LIBS PYTHON HTOOL_CPPFLAGS H2LIB_CPPFLAGS H2LIB_LIBS \
                 HMAT_CPPFLAGS HMAT_LIBS BPACK_CPPFLAGS BPACK_LIBS H2OPUS_CPPFLAGS H2OPUS_LIBS \
                 STARSH_CPPFLAGS STARSH_LIBS H2PACK_CPPFLAGS H2PACK_LIBS
CONFIG_STATIC := $(wildcard $(filter %.a %.o,$(H2LIB_LIBS) $(BPACK_LIBS) $(H2OPUS_LIBS) $(STARSH_LIBS) $(H2PACK_LIBS)))
CONFIG_STAMP   = $(B)/config.stamp

# ---- library switches -------------------------------------------------------------------
HAVE_HTOOL  := $(if $(strip $(HTOOL_CPPFLAGS)),1)
HAVE_H2LIB  := $(if $(strip $(H2LIB_CPPFLAGS)),1)
HAVE_HMAT   := $(if $(strip $(HMAT_CPPFLAGS)),1)
HAVE_BPACK  := $(if $(strip $(BPACK_CPPFLAGS)),1)
HAVE_H2OPUS := $(if $(strip $(H2OPUS_CPPFLAGS)),1)
HAVE_STARSH := $(if $(strip $(STARSH_CPPFLAGS)),1)
HAVE_H2PACK := $(if $(strip $(H2PACK_CPPFLAGS)),1)
LIBRARIES   := htool h2lib hmat bpack h2opus starsh h2pack
ENABLED     := $(strip $(if $(HAVE_HTOOL),htool) $(if $(HAVE_H2LIB),h2lib) $(if $(HAVE_HMAT),hmat) \
                 $(if $(HAVE_BPACK),bpack) $(if $(HAVE_H2OPUS),h2opus) $(if $(HAVE_STARSH),starsh) \
                 $(if $(HAVE_H2PACK),h2pack))

# ---- layout and flags -------------------------------------------------------------------
B   := build
OBJ := $(B)/obj
BIN := $(B)/bin
LIB := $(B)/lib

# Every adapter directory is on the include path: adapter file names carry an hmb_ or
# <library>_ prefix so they never shadow a library's own headers.
HMB_CPPFLAGS := -Iinclude -Itests/common $(addprefix -I,$(sort $(wildcard adapters/*)))
HMB_CFLAGS   := -std=gnu11 $(OPT) $(OPENMP) -fPIC -Wall
HMB_CXXFLAGS := -std=c++17 $(OPT) $(OPENMP) -fPIC -Wall -Wno-sign-compare
HMB_LDFLAGS  := $(OPENMP)

# Core library: every C/C++ file in src/ (no third-party dependency).
CORE_SRCS := $(wildcard src/*.c) $(wildcard src/*.cpp)
CORE_OBJS := $(patsubst %,$(OBJ)/%.o,$(basename $(CORE_SRCS)))
LIBHMB    := $(LIB)/libhmb.a

# Filled in by the fragments. EXAMPLES and TESTS are executables in $(BIN); SCRIPT_CHECKS are
# test scripts in the source tree. `make check` runs all three; each must finish in a few
# minutes with no arguments and print a line starting with HMB_TEST_PASSED on success.
EXAMPLES      :=
TESTS         :=
SCRIPT_CHECKS :=
# TOOLS: programs that `make` builds and `make check` does not run itself.
TOOLS         :=
# INFO_EXTRA: quoted lines that `make info` prints after its own.
INFO_EXTRA    :=

.DEFAULT_GOAL := all

define NEWLINE


endef
# (not for `make clean`, `make info` or a dry run `make -n`)
ifneq ($(filter-out clean info,$(MAKECMDGOALS)),$(if $(MAKECMDGOALS),,x))
ifeq ($(findstring n,$(firstword -$(MAKEFLAGS))),)
$(shell mkdir -p $(B))
$(file >$(CONFIG_STAMP).new,$(foreach v,$(CONFIG_VARS),$(v)=$($(v))$(NEWLINE))$(if $(CONFIG_STATIC),$(shell stat -c '%n %Y' $(CONFIG_STATIC))))
$(shell cmp -s $(CONFIG_STAMP).new $(CONFIG_STAMP) && rm -f $(CONFIG_STAMP).new || mv -f $(CONFIG_STAMP).new $(CONFIG_STAMP))
endif
endif
# (re)created by `make clean all` in one invocation; the next run records the real values
$(CONFIG_STAMP):
	@mkdir -p $(@D)
	@echo "created after clean" > $@

include $(sort $(wildcard mk/*.mk))

all: $(LIBHMB) $(EXAMPLES) $(TESTS) $(TOOLS)

$(LIBHMB): $(CORE_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

# ---- generic compile rules (fragments add flags through EXTRA_CFLAGS / EXTRA_CXXFLAGS,
#      set as pattern-specific variables on object files, never on executables) -----------
$(OBJ)/%.o: %.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(HMB_CPPFLAGS) $(EXTRA_CFLAGS) $(HMB_CFLAGS) -MMD -MP -c $< -o $@

$(OBJ)/%.o: %.cpp $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CXX) $(HMB_CPPFLAGS) $(EXTRA_CXXFLAGS) $(HMB_CXXFLAGS) -MMD -MP -c $< -o $@

check: all
	@HMB_LOGDIR=$(B)/test-logs ./tests/run_tests.sh $(TESTS) $(EXAMPLES) $(SCRIPT_CHECKS)

info:
	@echo "config:    $(if $(wildcard $(CONFIG)),$(CONFIG),$(CONFIG) (missing))"
	@echo "enabled:   $(ENABLED)"
	@echo "disabled:  $(filter-out $(ENABLED),$(LIBRARIES))"
	@echo "examples:  $(notdir $(EXAMPLES))"
	@echo "tests:     $(notdir $(TESTS)) $(SCRIPT_CHECKS)"
	@echo "tools:     $(notdir $(TOOLS))"
	@printf '%s\n' $(INFO_EXTRA)

clean:
	rm -rf $(B)

.PHONY: all check info clean

-include $(shell find $(OBJ) -name '*.d' 2>/dev/null)
