"""Tests for isaacsitl.so3 -- ported (and slimmed) from test_rotation.cpp.

Judgment on the C++ suite's heaviness: kept the rigor (scipy as the independent oracle
that Eigen was; round-trips; principal/known values; gimbal lock; near-pi; sign/double
cover), but each "kNumTrials" loop becomes ONE batched tensor op (free in torch), tested
in float64 only (no float/long-double triple), and the FP-underflow micro-probes are
dropped (they probe representations float32 cannot hold). Run:
    env -u PYTHONPATH uv run --directory . pytest -q tests/test_so3.py
"""

import math

from scipy.spatial.transform import Rotation as Rot
import torch

from isaacsitl import so3

N = 2000
PI = math.pi


def _rng():
    return torch.Generator().manual_seed(0)


def _rand_angle_axis(n, gen, lo=-PI, hi=PI):
    axis = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    axis /= axis.norm(dim=-1, keepdim=True)
    angle = torch.empty(n, dtype=torch.float64).uniform_(lo, hi, generator=gen)
    return axis * angle[:, None]


def _rand_quat(n, gen):
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _quat_close(a, b, atol=1e-10):
    """Equal up to an overall sign (the q/-q double cover)."""
    d = torch.minimum((a - b).norm(dim=-1), (a + b).norm(dim=-1))
    return bool((d <= atol).all())


# --- hat / vee --------------------------------------------------------------


def test_quat_component_order_is_xyzw():
    # The kernel convention as a FACT, not a docstring: scalar part LAST (the scipy
    # as_quat() order). A pure yaw uses distinct z/w magnitudes so a wxyz layout
    # cannot pass by symmetry.
    q = so3.euler_to_quat(torch.tensor([0.0, 0.0, PI / 3], dtype=torch.float64))
    want = torch.tensor(
        [0.0, 0.0, math.sin(PI / 6), math.cos(PI / 6)], dtype=torch.float64
    )
    assert torch.allclose(q, want, atol=1e-12)
    assert torch.allclose(
        torch.as_tensor(Rot.from_euler("z", PI / 3).as_quat()), q, atol=1e-12
    )


def test_hat_is_cross_product_and_vee_inverts():
    gen = _rng()
    v = torch.randn(N, 3, generator=gen, dtype=torch.float64)
    u = torch.randn(N, 3, generator=gen, dtype=torch.float64)
    hv = so3.hat(v)
    assert torch.allclose((hv @ u[..., None]).squeeze(-1), torch.cross(v, u, dim=-1))
    assert torch.allclose(so3.vee(hv), v)


# --- principal rotations ----------------------------------------------------


def test_principal_known_value_and_composition():
    rz = so3.rotation_matrix_z(torch.tensor(PI / 2, dtype=torch.float64))
    expected = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    assert torch.allclose(rz, expected, atol=1e-12)
    # Rz(yaw) Ry(pitch) Rx(roll) == quat_to_matrix(euler_to_quat(rpy))
    gen = _rng()
    rpy = torch.empty(N, 3, dtype=torch.float64).uniform_(-PI, PI, generator=gen)
    composed = (
        so3.rotation_matrix_z(rpy[..., 2])
        @ so3.rotation_matrix_y(rpy[..., 1])
        @ so3.rotation_matrix_x(rpy[..., 0])
    )
    assert torch.allclose(
        composed, so3.quat_to_matrix(so3.euler_to_quat(rpy)), atol=1e-10
    )


# --- exp / log vs scipy oracle ----------------------------------------------


def test_angle_axis_to_quat_matches_scipy():
    gen = _rng()
    aa = _rand_angle_axis(N, gen)
    ours = so3.angle_axis_to_quat(aa)
    oracle = torch.from_numpy(Rot.from_rotvec(aa.numpy()).as_quat())  # xyzw
    assert _quat_close(ours, oracle, atol=1e-12)
    assert torch.allclose(ours.norm(dim=-1), torch.ones(N, dtype=torch.float64))


def test_angle_axis_to_matrix_matches_scipy_and_orthonormal():
    gen = _rng()
    aa = _rand_angle_axis(N, gen)
    ours = so3.angle_axis_to_matrix(aa)
    oracle = torch.from_numpy(Rot.from_rotvec(aa.numpy()).as_matrix())
    assert torch.allclose(ours, oracle, atol=1e-10)
    eye = torch.eye(3, dtype=torch.float64).expand(N, 3, 3)
    assert torch.allclose(ours.transpose(-1, -2) @ ours, eye, atol=1e-10)


def test_quat_to_angle_axis_round_trip_and_wrapped():
    gen = _rng()
    aa = _rand_angle_axis(N, gen)
    rt = so3.quat_to_angle_axis(so3.angle_axis_to_quat(aa))
    # away from the pi seam the rotation vector is recovered exactly
    assert torch.allclose(rt, aa, atol=1e-9)
    # identity -> 0
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(
        so3.quat_to_angle_axis(ident), torch.zeros(1, 3, dtype=torch.float64)
    )
    # theta > pi (w < 0) must wrap into [-pi, pi]
    half = 0.75 * PI  # -> raw angle 1.5*pi
    q = torch.tensor([[math.sin(half), 0.0, 0.0, math.cos(half)]], dtype=torch.float64)
    assert float(so3.quat_to_angle_axis(q).norm()) <= PI + 1e-12


# --- quaternion <-> matrix --------------------------------------------------


def test_quat_matrix_round_trips_and_oracle():
    gen = _rng()
    q = _rand_quat(N, gen)
    R = so3.quat_to_matrix(q)
    oracle = torch.from_numpy(Rot.from_quat(q.numpy()).as_matrix())
    assert torch.allclose(R, oracle, atol=1e-10)
    assert _quat_close(so3.matrix_to_quat(R), q, atol=1e-9)  # Shepperd inverts


def test_matrix_to_angle_axis_near_pi():
    gen = _rng()
    axis = torch.randn(N, 3, generator=gen, dtype=torch.float64)
    axis /= axis.norm(dim=-1, keepdim=True)
    angle = torch.empty(N, dtype=torch.float64).uniform_(
        PI * (1 - 1e-5), PI, generator=gen
    )
    aa = axis * angle[:, None]
    rt = so3.matrix_to_angle_axis(so3.angle_axis_to_matrix(aa))
    # near pi the axis sign is ambiguous: compare up to sign, relative to the angle
    d = torch.minimum((rt - aa).norm(dim=-1), (rt + aa).norm(dim=-1)) / angle
    assert float(d.max()) < 1e-6


# --- aerospace Euler --------------------------------------------------------


def test_euler_to_quat_matches_scipy():
    gen = _rng()
    rpy = torch.empty(N, 3, dtype=torch.float64).uniform_(-PI, PI, generator=gen)
    ours = so3.euler_to_quat(rpy)
    # scipy intrinsic ZYX with [yaw, pitch, roll] == qz(yaw) qy(pitch) qx(roll)
    yaw_pitch_roll = rpy.flip(-1).numpy()
    oracle = torch.from_numpy(Rot.from_euler("ZYX", yaw_pitch_roll).as_quat())
    assert _quat_close(ours, oracle, atol=1e-12)


def test_euler_round_trip_away_from_lock():
    gen = _rng()
    rpy = torch.empty(N, 3, dtype=torch.float64)
    rpy[:, 0].uniform_(-3.0, 3.0, generator=gen)
    rpy[:, 1].uniform_(-1.4, 1.4, generator=gen)  # |pitch| < pi/2
    rpy[:, 2].uniform_(-3.0, 3.0, generator=gen)
    rt = so3.quat_to_euler(so3.euler_to_quat(rpy))
    assert torch.allclose(rt, rpy, atol=1e-10)


def test_euler_gimbal_lock():
    rows = []
    for pitch in (PI / 2, -PI / 2):
        for yaw in (-2.0, -0.5, 0.5, 2.0):
            rows.extend([roll, pitch, yaw] for roll in (-1.0, 0.0, 1.0))
    rpy = torch.tensor(rows, dtype=torch.float64)
    q = so3.euler_to_quat(rpy)
    rec = so3.quat_to_euler(q)
    assert torch.allclose(rec[:, 0], torch.zeros_like(rec[:, 0]), atol=1e-9)
    half_pi = torch.full_like(rec[:, 1], PI / 2)
    assert torch.allclose(rec[:, 1].abs(), half_pi, atol=1e-7)
    assert _quat_close(so3.euler_to_quat(rec), q, atol=1e-9)  # same rotation


# --- sign / double cover ----------------------------------------------------


def test_negated_is_same_rotation():
    gen = _rng()
    q = _rand_quat(N, gen)
    v = torch.randn(N, 3, generator=gen, dtype=torch.float64)
    assert torch.allclose(so3.negated(q), -q)
    assert torch.allclose(
        so3.quat_rotate(so3.negated(q), v), so3.quat_rotate(q, v), atol=1e-10
    )


def test_match_sign_aligns_to_reference():
    gen = _rng()
    ref = _rand_quat(N, gen)
    q = _rand_quat(N, gen)
    assert not bool(so3.is_sign_flipped(so3.match_sign(q, ref), ref).any())
    assert torch.allclose(so3.match_sign(so3.negated(q), ref), so3.match_sign(q, ref))


def test_canonicalize_positive_w_unique_and_tiebreak():
    gen = _rng()
    q = _rand_quat(N, gen)
    c = so3.canonicalize_positive_w(q)
    assert bool((c[:, 3] >= 0).all())
    assert torch.allclose(so3.canonicalize_positive_w(so3.negated(q)), c)
    assert torch.allclose(so3.canonicalize_positive_w(c), c)  # idempotent
    # w == 0 tie-break (180 deg): first nonzero of (x, y, z) made positive
    half_turn = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)  # xyzw
    canon = so3.canonicalize_positive_w(half_turn)
    assert float(canon[0, 0]) > 0
    assert torch.allclose(so3.canonicalize_positive_w(-half_turn), canon)


def test_quat_multiply_identity_and_rotate_consistency():
    gen = _rng()
    a = _rand_quat(N, gen)
    b = _rand_quat(N, gen)
    v = torch.randn(N, 3, generator=gen, dtype=torch.float64)
    # rotating by a*b == rotating by b then a
    lhs = so3.quat_rotate(so3.quat_multiply(a, b), v)
    rhs = so3.quat_rotate(a, so3.quat_rotate(b, v))
    assert torch.allclose(lhs, rhs, atol=1e-10)
    # q * conj(q) == identity
    ident = so3.quat_multiply(a, so3.quat_conjugate(a))
    eye_q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64).expand(N, 4)
    assert _quat_close(ident, eye_q)
