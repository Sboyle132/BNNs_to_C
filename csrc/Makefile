# Desktop:      make
# Cross-compile: make CC=arm-linux-gnueabihf-gcc ARCH="-march=armv7-a -mfpu=neon"
CC   ?= gcc
ARCH ?= -march=native
CFLAGS ?= -O3 -Wall $(ARCH)
OBJ = bnn_conv.o bnn_conv_realin.o bnn_maxpool.o bnn_fold.o bnn_head.o bnn_model.o

bnn_infer: main.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ main.c $(OBJ) -lm

%.o: %.c bnn_model.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o bnn_infer
.PHONY: clean
