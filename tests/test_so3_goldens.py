"""so3 vs the hs293math golden record -- the 'faithful port' claim as a test.

hs293math publishes tests/golden/rotation_goldens.safetensors (seeded inputs +
its own outputs for the five shared ops; generator: tests/src/
gen_rotation_goldens.cpp). This test fetches that record at a PINNED commit
from raw.githubusercontent.com and holds `so3` to it elementwise -- no C++
toolchain on this side; the file is the parity contract.

Both sides are fp32 on identical inputs, so agreement is ~1e-7; tol 5e-6.
Quats compare up to the q/-q double cover.

Offline behavior: the download caches beside this file (gitignored); with no
cache and no network the test SKIPS loudly (CI always has network, so the
gate is enforced there). Override the source with HS293MATH_GOLDENS_PATH to
check against a local build before pushing upstream.
"""

import os
from pathlib import Path
import urllib.request

import pytest
from safetensors.torch import load_file
import torch

from isaacsitl import so3

# Bump the pin when upstream's rotation semantics (and thus the record) move.
_PIN = "c4950264aa96f71ec899e0be315551c0f0166ea4"
_URL = (
    "https://raw.githubusercontent.com/Hs293Go/hs293math/"
    f"{_PIN}/tests/golden/rotation_goldens.safetensors"
)
_CACHE = Path(__file__).parent / ".golden_cache" / f"rotation_{_PIN[:12]}.safetensors"
_TOL = 5e-6


def _goldens() -> dict[str, torch.Tensor]:
    override = os.environ.get("HS293MATH_GOLDENS_PATH")
    if override:
        return load_file(override)
    if not _CACHE.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(_URL, timeout=20) as r:  # noqa: S310
                _CACHE.write_bytes(r.read())
        except OSError as e:
            pytest.skip(f"golden record unreachable ({e}); no cache at {_CACHE}")
    return load_file(_CACHE)


def _quat_dev(got: torch.Tensor, want: torch.Tensor) -> float:
    """Max elementwise deviation up to the quaternion double cover."""
    direct = (got - want).abs().amax(dim=-1)
    flipped = (got + want).abs().amax(dim=-1)
    return float(torch.minimum(direct, flipped).max())


def test_so3_matches_the_hs293math_golden_record():
    g = _goldens()
    aa, quat, euler = g["angle_axis"], g["quat_xyzw"], g["euler_rpy"]
    devs = {
        "angle_axis_to_quat": _quat_dev(
            so3.angle_axis_to_quat(aa), g["quat_from_angle_axis_xyzw"]
        ),
        "quat_to_angle_axis": float(
            (so3.quat_to_angle_axis(quat) - g["angle_axis_from_quat"]).abs().max()
        ),
        "angle_axis_to_matrix": float(
            (
                so3.angle_axis_to_matrix(aa).reshape(-1, 9)
                - g["matrix_from_angle_axis_rowmajor"]
            )
            .abs()
            .max()
        ),
        "euler_to_quat": _quat_dev(so3.euler_to_quat(euler), g["quat_from_euler_xyzw"]),
        "quat_to_euler": float(
            (so3.quat_to_euler(quat) - g["euler_from_quat_rpy"]).abs().max()
        ),
    }
    bad = {op: dev for op, dev in devs.items() if dev > _TOL}
    assert not bad, f"so3 drifted from rotation.hpp: {bad} (tol {_TOL})"
