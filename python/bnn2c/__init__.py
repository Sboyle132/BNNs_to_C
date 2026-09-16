"""bnn2c - export a trained HYPSO BNN to a C-loadable blob.

The exporter and verifiers import the training framework (models.*, data.*).
Point HYPSO_REPO at the training repo root so those imports resolve without
installing it. If the training repo is pip-installed (editable), this is a
no-op.
"""
import os
import sys

_repo = os.environ.get("HYPSO_REPO")
if _repo and _repo not in sys.path:
    sys.path.insert(0, _repo)
