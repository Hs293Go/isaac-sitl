"""so3: a torch-native, batch-aware SO(3) library (ported from hs293go/rotation.hpp).

Conventions:
  - quaternions are (..., 4) in (x, y, z, w) order (scipy / Isaac-after-conversion);
  - rotation matrices are (..., 3, 3), active (R @ v rotates v);
  - angle-axis / rotation vectors are (..., 3) with magnitude == the rotation angle;
  - Euler triples are (..., 3) = (roll, pitch, yaw), aerospace Z-Y-X: the returned quat
    is the active body->world rotation, so `quat_rotate(q, v_body) == v_world`.

Every function is batch-aware over leading dims and BRANCHLESS -- the per-element
FP-zero / gimbal-lock / near-pi cases are resolved with `torch.where`, not Python `if`,
so the library vmaps and runs on GPU. The quaternion<->angle-axis maps are Ceres's
(linear in the angle, wrapped to (-pi, pi]) so large errors are not half-angle
compressed); the Euler<->quaternion maps are gimbal-lock safe; the sign ops resolve the
q/-q double cover. Ported from the C++ Eigen library hs293go/rotation.hpp.
"""

from __future__ import annotations

import math
from typing import cast

import torch


def _unbind4(q: torch.Tensor):
    """Components x, y, z, w of an (..., 4) quaternion."""
    return q[..., 0], q[..., 1], q[..., 2], q[..., 3]


# --- skew / unskew ----------------------------------------------------------


def hat(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric cross-product matrix: K @ u == cross(v, u)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([o, -z, y], dim=-1),
            torch.stack([z, o, -x], dim=-1),
            torch.stack([-y, x, o], dim=-1),
        ],
        dim=-2,
    )


def vee(m: torch.Tensor) -> torch.Tensor:
    """Inverse hat: (..., 3, 3) skew-symmetric in so(3) -> its (..., 3) axis."""
    return torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], dim=-1)


# --- principal-axis rotation matrices ---------------------------------------


def rotation_matrix_x(angle: torch.Tensor) -> torch.Tensor:
    """Active rotation by `angle` about +X. (...,) -> (..., 3, 3)."""
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(angle), torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def rotation_matrix_y(angle: torch.Tensor) -> torch.Tensor:
    """Active rotation by `angle` about +Y. (...,) -> (..., 3, 3)."""
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(angle), torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
    """Active rotation by `angle` about +Z. (...,) -> (..., 3, 3)."""
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(angle), torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


# --- quaternion algebra -----------------------------------------------------


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse rotation for a unit quat): negate the vector part."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a*b of two (x, y, z, w) quaternions. Batch-aware."""
    ax, ay, az, aw = _unbind4(a)
    bx, by, bz, bw = _unbind4(b)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Active rotation of v by quaternion q=(x, y, z, w): v' = R(q) v. Batch-aware."""
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


# --- exp / log: angle-axis <-> quaternion (Ceres) ---------------------------


def angle_axis_to_quat(angle_axis: torch.Tensor) -> torch.Tensor:
    """SO(3) exp: rotation vector (angle*axis) -> unit quaternion (x, y, z, w)."""
    angle = torch.linalg.vector_norm(angle_axis, dim=-1)
    zero = angle == 0
    angle_safe = torch.where(zero, torch.ones_like(angle), angle)
    # vec scale sin(angle/2)/angle, angle->0 limit 1/2 (the FP-zero branch)
    scale = torch.where(
        zero, torch.full_like(angle, 0.5), torch.sin(angle / 2) / angle_safe
    )
    w = torch.where(zero, torch.ones_like(angle), torch.cos(angle / 2))
    return torch.cat([scale[..., None] * angle_axis, w[..., None]], dim=-1)


def quat_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    """SO(3) log: unit quaternion (x, y, z, w) -> rotation vector, wrapped to (-pi, pi].

    Linear in the angle (Ceres QuaternionToAngleAxis), so large attitude errors are not
    compressed by the half-angle sine the way 2*vec(q) is.
    """
    vec, w = q[..., :3], q[..., 3]
    n = torch.linalg.vector_norm(vec, dim=-1)
    zero = n == 0
    n_safe = torch.where(zero, torch.ones_like(n), n)
    # fold the |theta| <= pi wrap into atan2 via the sign of w (w<0 <=> theta>pi)
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    atan_nbyw = torch.atan2(sign * n, sign * w)
    # angle->0 limit is 2 (the FP_ZERO branch returns 2*vec)
    factor = torch.where(zero, torch.full_like(n, 2.0), 2.0 * atan_nbyw / n_safe)
    return factor[..., None] * vec


# --- angle-axis <-> rotation matrix (Rodrigues) -----------------------------


def angle_axis_to_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """SO(3) exp to a matrix: rotation vector -> (..., 3, 3) via Rodrigues."""
    theta = torch.linalg.vector_norm(angle_axis, dim=-1)
    zero = theta == 0
    theta_safe = torch.where(zero, torch.ones_like(theta), theta)
    axis = angle_axis / theta_safe[..., None]
    eye = torch.eye(3, dtype=angle_axis.dtype, device=angle_axis.device).expand((
        *angle_axis.shape[:-1],
        3,
        3,
    ))
    c, s = torch.cos(theta), torch.sin(theta)
    outer = axis[..., :, None] * axis[..., None, :]
    general = (
        c[..., None, None] * eye
        + (1.0 - c)[..., None, None] * outer
        + s[..., None, None] * hat(axis)
    )
    near_zero = eye + hat(angle_axis)  # first-order limit, avoids 0/0
    return torch.where(zero[..., None, None], near_zero, general)


def matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """SO(3) log from a matrix: (..., 3, 3) -> rotation vector (via the quaternion)."""
    return quat_to_angle_axis(matrix_to_quat(rotation_matrix))


# --- quaternion <-> rotation matrix -----------------------------------------


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (x, y, z, w) -> active rotation matrix (..., 3, 3)."""
    x, y, z, w = _unbind4(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_quat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Active rotation matrix (..., 3, 3) -> unit quaternion (x, y, z, w).

    Shoemake's method (Ken Shoemake, "Quaternion Calculus and Fast Animation",
    1987 SIGGRAPH course notes) -- the algorithm Ceres uses in
    RotationMatrixToQuaternion: the trace branch when trace >= 0, else the branch
    keyed on the largest diagonal entry as a cyclic (i, j, k) permutation.
    Vectorized branchlessly; the sqrt in every lane is guarded so the unselected
    lanes never produce NaN.
    """
    m = rotation_matrix
    eps = torch.finfo(m.dtype).eps

    def diag_branch(i: int) -> torch.Tensor:
        j, k = (i + 1) % 3, (i + 2) % 3
        rad = m[..., i, i] - m[..., j, j] - m[..., k, k] + 1
        t = torch.sqrt(rad.clamp_min(eps))
        inv = 0.5 / t
        vec = cast(list[torch.Tensor], [None, None, None])
        vec[i] = 0.5 * t
        vec[j] = (m[..., j, i] + m[..., i, j]) * inv
        vec[k] = (m[..., k, i] + m[..., i, k]) * inv
        w = (m[..., k, j] - m[..., j, k]) * inv
        return torch.stack([vec[0], vec[1], vec[2], w], dim=-1)

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    t = torch.sqrt((trace + 1).clamp_min(eps))
    inv = 0.5 / t
    q_trace = torch.stack(
        [
            (m[..., 2, 1] - m[..., 1, 2]) * inv,
            (m[..., 0, 2] - m[..., 2, 0]) * inv,
            (m[..., 1, 0] - m[..., 0, 1]) * inv,
            0.5 * t,
        ],
        dim=-1,
    )

    diag = torch.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], dim=-1)
    i = diag.argmax(dim=-1)
    q_diag = torch.where(
        (i == 0)[..., None],
        diag_branch(0),
        torch.where((i == 1)[..., None], diag_branch(1), diag_branch(2)),
    )
    return torch.where((trace >= 0)[..., None], q_trace, q_diag)


# --- aerospace Euler (Z-Y-X) <-> quaternion ---------------------------------


def euler_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """Aerospace (roll, pitch, yaw) Z-Y-X -> body->world quaternion (x, y, z, w).

    q = qz(yaw) * qy(pitch) * qx(roll), so `quat_rotate(q, v_body)` gives v_world.
    """
    half = rpy * 0.5
    cr, sr = torch.cos(half[..., 0]), torch.sin(half[..., 0])
    cp, sp = torch.cos(half[..., 1]), torch.sin(half[..., 1])
    cy, sy = torch.cos(half[..., 2]), torch.sin(half[..., 2])
    return torch.stack(
        [
            cy * cp * sr - sy * sp * cr,  # x
            cy * sp * cr + sy * cp * sr,  # y
            sy * cp * cr - cy * sp * sr,  # z
            cy * cp * cr + sy * sp * sr,  # w
        ],
        dim=-1,
    )


def quat_to_euler(q: torch.Tensor) -> torch.Tensor:
    """Body->world quaternion (x, y, z, w) -> aerospace (roll, pitch, yaw) Z-Y-X.

    Gimbal-lock safe: at pitch = +/-pi/2 the yaw/roll DOF collapse; that DOF is assigned
    to yaw and roll is pinned to 0 (matching the C++ QuaternionToEuler).
    """
    x, y, z, w = _unbind4(q)
    sin_pitch = 2 * (w * y - x * z)
    r21 = 2 * (w * x + y * z)
    r22 = 1 - 2 * (x * x + y * y)
    cos_pitch = torch.hypot(r21, r22)
    lock = cos_pitch < math.sqrt(torch.finfo(q.dtype).eps)

    half_pi = torch.full_like(sin_pitch, math.pi / 2)
    roll = torch.where(lock, torch.zeros_like(r21), torch.atan2(r21, r22))
    pitch = torch.where(
        lock, torch.copysign(half_pi, sin_pitch), torch.atan2(sin_pitch, cos_pitch)
    )
    yaw_lock = torch.atan2(2 * (w * z - x * y), 1 - 2 * (x * x + z * z))
    yaw_gen = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = torch.where(lock, yaw_lock, yaw_gen)
    return torch.stack([roll, pitch, yaw], dim=-1)


# --- sign / double-cover canonicalization -----------------------------------


def negated(q: torch.Tensor) -> torch.Tensor:
    """The other representative of the same rotation: all four coefficients negated."""
    return -q


def is_sign_flipped(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """True where q is on the opposite hemisphere from `reference` (dot < 0)."""
    return (q * reference).sum(dim=-1) < 0


def match_sign(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return +/-q, whichever shares `reference`'s hemisphere (convention-free)."""
    return torch.where(is_sign_flipped(q, reference)[..., None], -q, q)


def canonicalize_positive_w(q: torch.Tensor) -> torch.Tensor:
    """Return +/-q with non-negative w; ties at w==0 broken by first nonzero of xyz."""
    x, y, z, w = _unbind4(q)
    lead = torch.where(w != 0, w, torch.where(x != 0, x, torch.where(y != 0, y, z)))
    return torch.where((lead < 0)[..., None], -q, q)
