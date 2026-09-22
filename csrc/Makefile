# Desktop:        make                     -> bnn_infer (verified engine)
# Per-op profile: make profile             -> bnn_infer_prof (adds per-op timing)
# Cross-compile:  make CC=arm-linux-gnueabihf-gcc ARCH="-march=armv7-a -mfpu=neon"
#
# -fopenmp parallelises the binary-conv kernels over output rows. It degrades
# safely to single-thread if the toolchain lacks OpenMP (pragmas ignored).
# Control threads at runtime with OMP_NUM_THREADS.
CC   ?= gcc
ARCH ?= -march=native
CFLAGS ?= -O3 -Wall $(ARCH) -fopenmp
OBJ = bnn_conv.o bnn_conv_realin.o bnn_maxpool.o bnn_fold.o bnn_head.o bnn_model.o bnn_prof.o

bnn_infer: main.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ main.c $(OBJ) -lm

%.o: %.c bnn_model.h bnn_prof.h
	$(CC) $(CFLAGS) -c $< -o $@

PROBJ = $(OBJ:.o=.p.o)
profile: bnn_infer_prof
bnn_infer_prof: main.c $(PROBJ)
	$(CC) $(CFLAGS) -DBNN_PROFILE -o $@ main.c $(PROBJ) -lm
%.p.o: %.c bnn_model.h bnn_prof.h
	$(CC) $(CFLAGS) -DBNN_PROFILE -c $< -o $@

clean:
	rm -f *.o *.p.o bnn_infer bnn_infer_prof
.PHONY: clean profile