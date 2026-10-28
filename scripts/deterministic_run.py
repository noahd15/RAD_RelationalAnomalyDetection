#!/usr/bin/env python3
"""Run a script unchanged, with PyTorch's deterministic algorithms on.

    python scripts/deterministic_run.py SCRIPT [ARGS ...]

The RAD and RDL scripts only call `seed_everything(seed)`, so nondeterministic GPU kernels
change their results from run to run at the same seed (docs/paper_errata.md, item 13). This
wrapper sets the switches below and then runs SCRIPT as __main__. Any op without a
deterministic implementation raises an error instead of silently running nondeterministically.

Settings:
- CUBLAS_WORKSPACE_CONFIG=:4096:8, set before torch is imported. A different preset value
  is refused.
- torch.use_deterministic_algorithms(True)
- torch.backends.cudnn.deterministic = True, torch.backends.cudnn.benchmark = False

It prints one `deterministic_mode: true ...` line first. The runners check for that line.
"""

import os
import runpy
import sys

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MODE_LINE_PREFIX = "deterministic_mode: true"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    preset = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if preset not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG is already {preset!r}; expected {CUBLAS_WORKSPACE_CONFIG!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"{MODE_LINE_PREFIX} | CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} "
          f"use_deterministic_algorithms=True cudnn.deterministic=True cudnn.benchmark=False "
          f"torch={torch.__version__}", flush=True)

    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
