"""Vector/quaternion conversions between ENU-FLU Isaac and NED-FRD aerospace frames.

torch-native and batch-aware (..., 3/4). Isaac Lab returns torch tensors throughout, so
the kernel speaks torch — there is no NumPy backend and no dispatch. NED appears only at
the autopilot/SITL boundary. (The NumPy analytic twin that motivated a dual backend in
isaac-drone-racing lives in the `isaacrace` MWE, not here — see README.)

The euler/rotate helpers delegate to `so3` — the two were verified to be the SAME
convention to machine epsilon (scipy extrinsic-'xyz' ≡ intrinsic aerospace ZYX), so one
implementation serves both names and the pair cannot drift apart.
"""

import torch

from isaacsitl.so3 import euler_to_quat, quat_to_euler

SQRT_1_2 = 0.70710678118654757


def vec_enu_ned(v: torch.Tensor) -> torch.Tensor:
    """ENU (East-North-Up) -> NED (North-East-Down). Batch-aware (..., 3)."""
    if v.shape[-1] != 3:
        raise ValueError(f"3D vector expected, got {v.shape[-1]} components")
    return torch.stack([v[..., 1], v[..., 0], -v[..., 2]], dim=-1)


def vec_flu_frd(v: torch.Tensor) -> torch.Tensor:
    """FLU (Forward-Left-Up) -> FRD (Forward-Right-Down). Batch-aware (..., 3)."""
    if v.shape[-1] != 3:
        raise ValueError(f"3D vector expected, got {v.shape[-1]} components")
    return torch.stack([v[..., 0], -v[..., 1], -v[..., 2]], dim=-1)


def quat_aero_isaac(q: torch.Tensor) -> torch.Tensor:
    """Swap a quat between aero (FRD->NED) and Isaac (FLU->ENU). (..., 4).

    This function is self-inverse.
    """
    if q.shape[-1] != 4:
        raise ValueError(
            f"Quaternion (4 components) expected, got {q.shape[-1]} components"
        )

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            -(SQRT_1_2 * (x + y)),
            -(SQRT_1_2 * (x - y)),
            (SQRT_1_2 * (z - w)),
            -(SQRT_1_2 * (z + w)),
        ],
        dim=-1,
    )


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Isaac articulation quats are (w, x, y, z); the kernel uses (x, y, z, w)."""
    return q[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Kernel (x, y, z, w) -> Isaac articulation (w, x, y, z); inverse of the above."""
    return q[..., [3, 0, 1, 2]]


def quat_xyzw_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """Intrinsic XYZ (roll, pitch, yaw) from an (x, y, z, w) quat; matches scipy 'xyz'.

    Delegates to `so3.quat_to_euler` (same convention, one implementation).
    """
    return quat_to_euler(q)


def euler_xyz_to_quat_xyzw(e: torch.Tensor) -> torch.Tensor:
    """Inverse of quat_xyzw_to_euler_xyz; matches scipy from_euler('xyz').as_quat().

    Delegates to `so3.euler_to_quat` (same convention, one implementation).
    """
    return euler_to_quat(e)
