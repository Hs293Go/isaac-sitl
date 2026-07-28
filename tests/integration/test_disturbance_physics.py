"""External-wrench physics through a live Isaac Sim (the F/m check).

Spawned as a SUBPROCESS, not booted in-process: Kit segfaults/hangs under
pytest's interpreter (it parses argv and can't re-init), so the smoke script
boots once and hard-exits with an honest code; the test asserts exit 0.

Run explicitly (GPU + the isaaclab group; excluded from the default suite):

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        uv run --group isaaclab pytest -m integration -q
"""

import os
from pathlib import Path
import subprocess  # noqa: S404
import sys

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_TIMEOUT_S = 900  # first boot compiles shaders; be generous


def test_external_wrench_physics_smoke():
    """F/m, tau/Izz, and the clean-path restore, on the live articulation."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_disturbance.py", "--headless"],
        cwd=_REPO,
        env=env,
        timeout=_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, f"smoke_disturbance failed (exit {proc.returncode})"
