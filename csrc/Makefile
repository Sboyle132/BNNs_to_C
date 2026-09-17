# Desktop:        make                     -> bnn_infer (verified engine)
# Per-op profile: make profile             -> bnn_infer_prof (adds per-op timing)
# Cross-compile:  make CC=arm-linux-gnueabihf-gcc ARCH="-march=armv7-a -mfpu=neon"
#
# bnn_infer already prints coarse timing (blob load / forward / write) to stderr.
# bnn_infer_prof additionally prints the per-op breakdown and roll-up by op-type.
# The two use separate object files, so the verified binary is never rebuilt
# with the profiling flag.
CC   ?= gcc
ARCH ?= -march=native
CFLAGS ?= -O3 -Wall $(ARCH)
OBJ = bnn_conv.o bnn_conv_realin.o bnn_maxpool.o bnn_fold.o bnn_head.o bnn_model.o bnn_prof.o

bnn_infer: main.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ main.c $(OBJ) -lm

%.o: %.c bnn_model.h bnn_prof.h
	$(CC) $(CFLAGS) -c $< -o $@

# --- profiling build: same sources, -DBNN_PROFILE, separate .p.o objects ---
PROBJ = $(OBJ:.o=.p.o)

profile: bnn_infer_prof

bnn_infer_prof: main.c $(PROBJ)
	$(CC) $(CFLAGS) -DBNN_PROFILE -o $@ main.c $(PROBJ) -lm

%.p.o: %.c bnn_model.h bnn_prof.h
	$(CC) $(CFLAGS) -DBNN_PROFILE -c $< -o $@

clean:
	rm -f *.o *.p.o bnn_infer bnn_infer_prof
.PHONY: clean profile
