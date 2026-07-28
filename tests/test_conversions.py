"""Frame conversions: known answers, involutions, and a scipy ground-truth check."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from isaacsitl.conversions import (
    quat_aero_isaac,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    vec_enu_ned,
    vec_flu_frd,
)


def as_2dtensor(v):
    return torch.atleast_2d(torch.as_tensor(v, dtype=torch.float32))


def test_vec_enu_ned_basis_vectors():
    # East -> ENU x == NED y; North -> ENU y == NED x; Up -> -Down
    torch.testing.assert_close(
        vec_enu_ned(as_2dtensor([1.0, 0, 0])), as_2dtensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        vec_enu_ned(as_2dtensor([0, 1.0, 0])), as_2dtensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        vec_enu_ned(as_2dtensor([0, 0, 1.0])), as_2dtensor([0.0, 0.0, -1.0])
    )


def test_vec_flu_frd_basis_vectors():
    # Forward unchanged; Left -> -Right; Up -> -Down
    torch.testing.assert_close(
        vec_flu_frd(as_2dtensor([1.0, 0, 0])), as_2dtensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        vec_flu_frd(as_2dtensor([0, 1.0, 0])), as_2dtensor([0.0, -1.0, 0.0])
    )
    torch.testing.assert_close(
        vec_flu_frd(as_2dtensor([0, 0, 1.0])), as_2dtensor([0.0, 0.0, -1.0])
    )


@pytest.mark.parametrize("conv", [vec_enu_ned, vec_flu_frd])
def test_vec_conversions_are_involutions(conv):
    # Both maps are their own inverse: applying twice returns the input (batched).
    v = as_2dtensor(np.random.default_rng(0).standard_normal((32, 3)))
    torch.testing.assert_close(conv(conv(v)), v)


def test_vec_enu_ned_batched():
    vectors = as_2dtensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    torch.testing.assert_close(
        vec_enu_ned(vectors), as_2dtensor([[2, 1, -3], [5, 4, -6]])
    )


@pytest.mark.parametrize("conv", [vec_enu_ned, vec_flu_frd])
def test_vec_conversions_reject_non_3d(conv):
    with pytest.raises(ValueError, match="3D vector"):
        conv(as_2dtensor([1.0, 2.0]))


def test_quat_aero_isaac_involution():
    # Two applications return the original rotation (up to quaternion sign).
    rng = np.random.default_rng(1)
    q = as_2dtensor(Rotation.random(16, random_state=rng).as_quat())
    q2 = quat_aero_isaac(quat_aero_isaac(q))
    sign = torch.sign(torch.sum(q * q2, dim=-1, keepdim=True))
    # Using atol/rtol explicitly inside torch.testing.assert_close
    torch.testing.assert_close(sign * q2, q, atol=1e-7, rtol=1e-7)


def test_quat_aero_isaac_matches_vector_conversions():
    # Ground truth: rotating in one convention then converting the vector must equal
    # converting the quaternion and rotating the converted vector.
    rng = np.random.default_rng(2)
    q_isaac = as_2dtensor(Rotation.random(32, random_state=rng).as_quat())
    v_flu = as_2dtensor(rng.standard_normal((32, 3)))

    world_enu = as_2dtensor(Rotation.from_quat(q_isaac.numpy()).apply(v_flu.numpy()))

    # FIX: Extract underlying numpy arrays for SciPy's Rotation class
    q_aero_np = quat_aero_isaac(q_isaac).numpy()
    v_frd_np = vec_flu_frd(v_flu).numpy()
    world_ned = torch.as_tensor(
        Rotation.from_quat(q_aero_np).apply(v_frd_np), dtype=torch.float32
    )

    torch.testing.assert_close(vec_enu_ned(world_enu), world_ned, atol=1e-5, rtol=1e-5)


def test_quat_aero_isaac_level_yaw_identity():
    q_isaac = as_2dtensor(Rotation.from_euler("z", np.pi / 2).as_quat())

    # FIX: Convert the output tensor to numpy before passing to SciPy
    q_aero_tensor = quat_aero_isaac(q_isaac)
    angle = Rotation.from_quat(q_aero_tensor.numpy()).magnitude()

    assert angle == pytest.approx(0.0, abs=1e-12)


def test_quat_aero_isaac_rejects_non_quaternion():
    with pytest.raises(ValueError, match="4 components"):
        quat_aero_isaac(as_2dtensor([0.0, 0.0, 1.0]))


def test_quat_wxyz_xyzw_reorder_known_answer_and_round_trip():
    # The Isaac boundary reorder as a fact: articulation identity (w,x,y,z)=(1,0,0,0)
    # -> kernel identity (x,y,z,w)=(0,0,0,1); labeled components map one-to-one.
    torch.testing.assert_close(
        quat_wxyz_to_xyzw(as_2dtensor([1.0, 0.0, 0.0, 0.0])),
        as_2dtensor([0.0, 0.0, 0.0, 1.0]),
    )
    q_wxyz = as_2dtensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    torch.testing.assert_close(
        q_xyzw, as_2dtensor([[2.0, 3.0, 4.0, 1.0], [6.0, 7.0, 8.0, 5.0]])
    )
    # the two reorders are exact inverses (batched)
    torch.testing.assert_close(quat_xyzw_to_wxyz(q_xyzw), q_wxyz)
    torch.testing.assert_close(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q_wxyz)), q_wxyz)
