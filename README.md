# BNNs_to_C

From-scratch C inference engine for the HYPSO Bi-Real binary UNet
(`bnn_unet_cpba_a2_dualskip_bireal`), ported from the trained PyTorch model
and verified bit-exact against it. No LARQ / daBNN / ONNX.

The port reproduces the model with integer XNOR-popcount convs and folded
integer-threshold activations; the only real-valued compute is the first
conv (raw float spectra) and the FP32 output head, by design.

## Layout

    csrc/          pure C, no Python. cross-compiles to the board.
      bnn_conv.c            XNOR-popcount binary conv (+ zero-pad validity mask)
      bnn_conv_realin.c     real-input binary-weight conv (first layer)
      bnn_maxpool.c         integer max-pool on the accumulator P
      bnn_fold.c            activation folds: integer-threshold + real
      bnn_head.c            FP32 1x1 output head
      bnn_model.{c,h}       blob loader + assembled forward
      main.c                CLI: blob + input.bin -> output.bin
      Makefile
    python/
      bnn2c/         exporter (imports the training framework)
        blob.py              THE blob layout, writer + reader (contract)
        export.py            checkpoint -> weights.blob
      verify/        dev-time verification (imports the framework + live model)
        forward_driver.py    whole-net C-vs-model golden diff
        dump_model_spec.py   per-layer type/shape/domain + accumulator check
        derive_thresholds.py alpha+BN+RPReLU+RSign -> per-channel fold rule
        *_harness.py         per-kernel self-tests + model checks
        dump_patch.py        save a real normalised HYPSO patch
    tools/build_all.sh       build the .so libs the harnesses load
    tests/test_selftest.sh   every harness self-test (no data needed)
    artifacts/               weights.blob, patch0.npy, *.so  (gitignored)

## Dependency direction

BNNs_to_C imports the training repo (`models.*`, `data.*`); the training repo
never imports this. Make the framework importable one of two ways:

    export HYPSO_REPO=/path/to/hypso_slc_segmentation      # bnn2c/__init__ adds it to sys.path
    # or: pip install -e /path/to/hypso_slc_segmentation

## Export a trained model

    export HYPSO_REPO=~/Coding/ESA_BNNs/hypso_slc_segmentation
    python3 python/bnn2c/export.py \
        --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
        --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
        --checkpoint   $HYPSO_REPO/runs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch/best_model.pt \
        --upsampler    nearest \
        --out          artifacts/weights.blob

## Build + run the engine

    make -C csrc
    # input.bin: n_bands*H*W float32, layout [band,row,col]
    csrc/bnn_infer artifacts/weights.blob input.bin output.bin 32 32
    # output.bin: n_classes*H*W float32 logits

## Verify (the whole point)

    tools/build_all.sh                         # build .so for harnesses
    tests/test_selftest.sh                     # kernels + full net, random weights
    # against the trained model + a real patch:
    cd python/verify
    python3 forward_driver.py --model-module ... --model-class ... \
        --checkpoint ... --input-npy patch0.npy

The standalone binary is verified by matching forward_driver (which is
verified against the model): on the trained checkpoint + a real patch, all 22
activations are bit-exact, logits match to ~2e-7, argmax 100%.

## Cross-compile for the ARM board

    make -C csrc CC=arm-linux-gnueabihf-gcc ARCH="-march=armv7-a -mfpu=neon"

Blob format is little-endian; both x86 and the ARM target are LE. Binary
stages must match desktop exactly; the FP32 head may differ within float
tolerance (libm/rounding).

## Variants

The blob header carries an upsampler tag (nearest / pixshuf / nnrefine).
`nearest` is implemented in bnn_model.c. pixshuf/nnrefine add binary upsample
convs; when one wins the accuracy A/B, its upsample op is added to the C
forward (the kernels it needs already exist).
