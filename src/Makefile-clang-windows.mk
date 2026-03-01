#
# Makefile to Compile Edax for Windows (Clang)
#
# Original Work:
# Copyright 1998 - 2024
# Richard Delorme
# Version 4.6
#
# Modified by Yuichiro Okashita, 2026
# - Simplified and optimized exclusively for Windows + Clang
#

# Default settings
BIN = ../bin
LIBS = -lws2_32
CC = clang

ifeq ($(BUILD),)
    BUILD = optimize
endif

# Clang settings
CFLAGS = -std=c17 -pedantic -W -Wall -Wno-invalid-source-encoding -D_GNU_SOURCE=1
PGO_GEN = -fprofile-instr-generate
PGO_USE = -fprofile-instr-use=edax.profdata
PGO = llvm-profdata merge -output=edax.profdata $(BIN)/*.profraw

ifeq ($(BUILD),optimize)
    CFLAGS += -O3 -fuse-ld=lld -flto -ffast-math -fomit-frame-pointer -DNDEBUG
endif
ifeq ($(BUILD),profile)
    CFLAGS += -O2 -g -DNDEBUG -fno-inline-functions
endif   
ifeq ($(BUILD),coverage)
    CFLAGS += -O2 -fprofile-instr-generate -fcoverage-mapping -mllvm -runtime-counter-relocation -DNDEBUG
endif   
ifeq ($(BUILD),debug)
    CFLAGS += -O0 -g -DDEBUG
endif

ifneq ($(ARCH),)
    CFLAGS += -march=$(ARCH)
endif

# EXE
EXE = wEdax-$(ARCH).exe

# SRC
SRC= bit.c board.c move.c crc32c.c hash.c ybwc.c eval.c endgame.c midgame.c root.c search.c \
book.c opening.c game.c base.c perft.c obftest.c util.c event.c histogram.c \
stats.c options.c play.c ui.c edax.c cassio.c gtp.c ggs.c nboard.c xboard.c main.c   

# RULES
help:
	@echo ""
	@echo "To compile Edax for Windows using Clang:"
	@echo ""
	@echo " make -f Makefile-windows target [ARCH=cpu]"
	@echo ""
	@echo "Targets:"
	@echo "   build      Build optimized version"
	@echo "   pgo-build  Build PGO-optimized version"
	@echo "   debug      Build debug version."
	@echo "   clean      Clean up."
	@echo "   help* Print this message"
	@echo ""
	@echo "Archs:"
	@echo " x86-64-v4       x64 with sse2, avx, sse4 & popcount & avx2 & avx512 support"
	@echo " x86-64-v3       x64 with sse2, avx, sse4 & popcount & avx2 support"
	@echo " x86-64-v2       with sse2, avx, sse4 & popcount support"
	@echo " x86-64          x64 with sse2 support"
	@echo " native          Your cpu"
	@echo ""
	@echo "* default setting"

build:
	@echo "building edax..."
	$(CC) $(CFLAGS) $(DFLAGS) all.c -o $(BIN)/$(EXE) $(LIBS)

pgo-build:
	@echo "building edax with pgo..."
	$(MAKE) -f Makefile-windows clean
	$(CC) $(CFLAGS) $(PGO_GEN) $(DFLAGS) all.c -o $(BIN)/$(EXE) $(LIBS)
	cd $(BIN) && $(EXE) -l 60 -bench 10 -n 1
	if exist $(BIN)\book.pgo del $(BIN)\book.pgo
	if exist $(BIN)\book.pgo.store del $(BIN)\book.pgo.store
	$(PGO)  
	$(CC) $(CFLAGS) $(PGO_USE) $(DFLAGS) all.c -o $(BIN)/$(EXE) $(LIBS)

pgo-rebuild:
	@echo "rebuilding edax with pgo..."
	$(CC) $(CFLAGS) $(PGO_USE) all.c -o $(BIN)/$(EXE) $(LIBS)

prof:
	@echo "building edax for profiling..."
	$(MAKE) -f Makefile-windows ARCH=$(ARCH) BUILD=profile build

cov:
	@echo "building edax for coverage..."
	$(MAKE) -f Makefile-windows ARCH=$(ARCH) BUILD=coverage build

debug:
	$(MAKE) -f Makefile-windows ARCH=$(ARCH) BUILD=debug build

clean:
	del /Q /F pgopti* *.dyn *.gc* *~ *.o *.prof*

default:
	help