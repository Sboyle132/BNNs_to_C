#!/usr/bin/env bash
# CI-able: every kernel harness self-test + the local full-forward self-test.
# Needs the training repo importable (HYPSO_REPO or pip install), plus numpy/torch.
set -euo pipefail
cd "$(dirname "$0")/../python/verify"
for h in bconv_harness realin_conv_harness maxpool_harness head_harness; do
    echo "=== $h --self-test ==="; python3 $h.py --self-test
done
echo "=== forward_driver --self-test (full net, random weights) ==="
python3 forward_driver.py --self-test
