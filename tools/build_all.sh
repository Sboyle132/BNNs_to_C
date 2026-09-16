#!/usr/bin/env bash
# build the shared libs the Python verification harnesses load via ctypes
set -euo pipefail
cd "$(dirname "$0")/../csrc"
CC=${CC:-gcc}; ARCH=${ARCH:--march=native}
FLAGS="-O3 -fPIC -shared $ARCH"
$CC $FLAGS -o libbnnconv.so        bnn_conv.c
$CC $FLAGS -o libbnnconv_realin.so bnn_conv_realin.c
$CC $FLAGS -o libbnnmaxpool.so     bnn_maxpool.c
$CC $FLAGS -o libbnnhead.so        bnn_head.c
$CC $FLAGS -o libbnnfold.so        bnn_fold.c
echo "built .so libs into csrc/"
