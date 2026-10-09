# One static archive per configured adapter, for programs outside the package:
# build/lib/libhmb_<lib>.a. `make info` prints what to link with each one.
#
#   g++ -std=c++17 -fopenmp -I<pkg>/include -I<pkg>/adapters/<lib> ... my.cpp \
#       <pkg>/build/lib/libhmb_<lib>.a ... <pkg>/build/lib/libhmb.a <LIB>_LIBS ... BLAS_LIBS

ARCHIVE_LINK :=

define hmb_archive # $(1) name, $(2) objects, $(3) name of the library's link variable
$(LIB)/libhmb_$(1).a: $(2)
	@mkdir -p $$(@D)
	rm -f $$@
	$$(AR) rcs $$@ $$^
ARCHIVES += $(LIB)/libhmb_$(1).a
ARCHIVE_LINK += "  libhmb_$(1).a:  $(if $(3),$(3) = $(strip $($(3))),nothing (header-only library))"
endef

ARCHIVES :=
$(if $(HAVE_HTOOL),$(eval $(call hmb_archive,htool,$(HTOOL_OBJS) $(HTOOL_COPY_OBJS),)))
$(if $(HAVE_H2LIB),$(eval $(call hmb_archive,h2lib,$(H2LIB_OBJS),H2LIB_LIBS)))
$(if $(HAVE_HMAT),$(eval $(call hmb_archive,hmatoss,$(HMATOSS_OBJS),HMAT_LIBS)))
$(if $(HAVE_BPACK),$(eval $(call hmb_archive,bpack,$(BPACK_OBJS),BPACK_LIBS)))
$(if $(HAVE_H2OPUS),$(eval $(call hmb_archive,h2opus,$(H2OPUS_HARA_OBJS) $(H2OPUS_TLR_OBJS),H2OPUS_LIBS)))
$(if $(HAVE_STARSH),$(eval $(call hmb_archive,starsh,$(STARSH_OBJS),STARSH_LIBS)))

all: $(ARCHIVES)

INFO_EXTRA += "archives:  $(notdir $(ARCHIVES)) (in $(LIB)/)" \
              "to use them in your own program, link: your objects, the archives you need, $(LIB)/libhmb.a," \
              "then each archive's library flags below, then BLAS_LIBS = $(strip $(BLAS_LIBS))" $(ARCHIVE_LINK)
