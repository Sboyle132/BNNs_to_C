#!/usr/bin/env bash
# Source this from the BNNs_to_C repo root:  source env.sh
# Sets HYPSO_REPO (so the port scripts find the training framework) and a few
# convenience vars. Safe to source repeatedly.

# resolve the training repo relative to this script's location, override by
# exporting HYPSO_REPO before sourcing if your layout differs.
_bnn_root="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export HYPSO_REPO="${HYPSO_REPO:-$( cd "$_bnn_root/../hypso_slc_segmentation" && pwd )}"

# handy shorthands for the usual args
export BNN_CKPT="$HYPSO_REPO/runs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch/best_model.pt"
export BNN_CFG="$HYPSO_REPO/configs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch.yaml"
export BNN_MODULE="models.bnn_unet_cpba_a2_dualskip_bireal_student"
export BNN_CLASS="BNN_UNet_CPBA_A2_DualSkip_BiReal_Student"

echo "HYPSO_REPO = $HYPSO_REPO"
if [ ! -d "$HYPSO_REPO" ]; then
    echo "  WARNING: that directory does not exist; set HYPSO_REPO manually."
fi
