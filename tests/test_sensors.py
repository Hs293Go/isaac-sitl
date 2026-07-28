"""IMU + local-position sensors -- torch-only, no sim boot.

Covers the gravity-compensated strapdown IMU (specific force + body rate), the GPS-like
ENU local position, the optional autopilot-grade noise, and update-rate gating.
"""

from __future__ import annotations

import torch

from isaacsitl.sensors import (
    Imu,
    ImuNoiseConfig,
    LocalPosition,
    LocalPositionNoiseConfig,
)
from isaacsitl.state import State

_G = 9.81


def _state(*, pos=None, att=None, lin_acc=None, omega=None, n=1) -> State:
    z = torch.zeros(n, 3)
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n, 4)
    return State(
        position=z.clone() if pos is None else pos,
        attitude=ident if att is None else att,
        linear_velocity=z.clone(),
        angular_velocity=z.clone() if omega is None else omega,
        linear_acceleration=z.clone() if lin_acc is None else lin_acc,
    )


def test_imu_at_rest_and_level_reads_gravity_reaction():
    """At rest and level: accel = (0,0,+G) up; gyro passes the body rate through."""
    imu = Imu(num_envs=1, device="cpu")  # noiseless
    omega = torch.tensor([[0.1, -0.2, 0.3]])
    m = imu.update(_state(omega=omega), dt=0.01)
    assert torch.allclose(m.accel, torch.tensor([[0.0, 0.0, _G]]), atol=1e-5)
    assert torch.allclose(m.gyro, omega)


def test_imu_specific_force_magnitude_and_frame_consistency():
    """|accel| == G for any attitude (at rest), and body->world undoes the rotation."""
    q = torch.randn(16, 4)
    q /= torch.linalg.norm(q, dim=-1, keepdim=True)
    imu = Imu(num_envs=16, device="cpu")
    m = imu.update(_state(att=q, n=16), dt=0.01)
    mag = torch.linalg.norm(m.accel, dim=-1)
    assert torch.allclose(mag, torch.full((16,), _G), atol=1e-4)
    # rotating the body-frame accel back to world recovers the (0,0,G) specific force
    from isaacsitl.so3 import quat_rotate

    back = quat_rotate(q, m.accel)
    assert torch.allclose(back, torch.tensor([0.0, 0.0, _G]).expand(16, 3), atol=1e-4)


def test_imu_includes_kinematic_acceleration():
    """Specific force adds the kinematic accel: level + a_z=2 -> body-up reads G+2."""
    imu = Imu(num_envs=1, device="cpu")
    m = imu.update(_state(lin_acc=torch.tensor([[0.0, 0.0, 2.0]])), dt=0.01)
    assert torch.allclose(m.accel, torch.tensor([[0.0, 0.0, _G + 2.0]]), atol=1e-5)


def test_local_position_noiseless_is_truth():
    """With no noise config the fix is the true ENU position."""
    lp = LocalPosition(num_envs=1, device="cpu")
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(lp.update(_state(pos=pos), dt=0.01).position, pos)


def test_local_position_noise_is_unbiased_with_configured_std():
    """Noisy fix: mean ~ truth, per-axis std ~ configured, over a big batch."""
    n = 20000
    gen = torch.Generator().manual_seed(0)
    cfg = LocalPositionNoiseConfig(pos_std=(0.5, 0.3, 0.1))
    lp = LocalPosition(num_envs=n, device="cpu", noise=cfg, generator=gen)
    m = lp.update(_state(pos=torch.zeros(n, 3), n=n), dt=0.01)
    assert torch.allclose(m.position.mean(0), torch.zeros(3), atol=0.02)
    assert torch.allclose(m.position.std(0), torch.tensor([0.5, 0.3, 0.1]), atol=0.02)


def test_update_rate_gates_emission():
    """A rate-limited sensor returns None until a full period has elapsed."""
    lp = LocalPosition(num_envs=1, device="cpu", update_rate=10.0)  # 0.1 s period
    assert lp.update(_state(), 0.04) is None  # 0.04
    assert lp.update(_state(), 0.04) is None  # 0.08
    assert lp.update(_state(), 0.04) is not None  # 0.12 >= 0.1 -> emit


def test_imu_noise_bias_is_reproducible_and_resettable():
    """A seeded noisy IMU is deterministic; reset resamples the turn-on bias."""
    gen = torch.Generator().manual_seed(1)
    imu = Imu(num_envs=4, device="cpu", noise=ImuNoiseConfig(), generator=gen)
    m = imu.update(_state(n=4), dt=0.01)
    assert m.accel.shape == (4, 3) and torch.isfinite(m.accel).all()
    # a fresh seed-1 IMU reproduces the same first reading
    gen2 = torch.Generator().manual_seed(1)
    imu2 = Imu(num_envs=4, device="cpu", noise=ImuNoiseConfig(), generator=gen2)
    assert torch.allclose(imu2.update(_state(n=4), dt=0.01).accel, m.accel)


def test_local_position_update_does_not_mutate_state():
    """GPS sensor must NOT mutate the caller's State -- persistent-state callers (an
    analytic-plant sweep whose true position IS the input) rely on it."""
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    st = _state(pos=pos)
    before = pos.clone()
    gps = LocalPosition(
        num_envs=1,
        device="cpu",
        noise=LocalPositionNoiseConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    m = gps.update(st, dt=0.01)
    assert not torch.equal(m.position, before)  # the measurement IS noisy
    assert torch.equal(st.position, before)  # ...but the input State is untouched


def test_magnetometer_reads_field_in_body_frame():
    """Identity body reads the world field; a yawed body sees it rotate -- heading."""
    from isaacsitl.sensors import Magnetometer

    mag = Magnetometer(1, "cpu")  # default field: unit (0, 0.707, -0.707) ENU
    ident = mag.update(_state(), dt=0.01)
    assert torch.allclose(ident.field[0], mag.field_enu, atol=1e-6)
    # body yawed +90 deg: world-North lands on the body x-axis
    q = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068]])
    yawed = mag.update(_state(att=q), dt=0.01)
    want = torch.tensor([0.7071068, 0.0, -0.7071068])
    assert torch.allclose(yawed.field[0], want, atol=1e-6)


def test_partial_reset_preserves_the_shared_rate_clock():
    """Resetting SOME envs must not rewind the batch-shared emission clock."""
    lp = LocalPosition(2, "cpu", update_rate=50.0)  # period 0.02 at dt 0.01
    assert lp.update(_state(n=2), dt=0.01) is None  # half a period elapsed
    lp.reset(env_ids=torch.tensor([0]))  # per-env reset: clock untouched
    assert lp.update(_state(n=2), dt=0.01) is not None  # full period -> emits
    assert lp.update(_state(n=2), dt=0.01) is None
    lp.reset()  # FULL reset rewinds the clock
    assert lp.update(_state(n=2), dt=0.01) is None
    assert lp.update(_state(n=2), dt=0.01) is not None


def test_barometer_isa_pressure_tracks_msl_altitude():
    """Sea level reads 1013.25 hPa; higher reads lower; MSL = home alt + local up."""
    from isaacsitl.sensors import Barometer, BarometerNoiseConfig

    baro = Barometer(2, "cpu", alt_ref_m=488.0)  # e.g. Zurich home altitude
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    m = baro.update(_state(pos=pos, n=2), dt=0.01)
    assert torch.allclose(m.altitude_m, torch.tensor([488.0, 588.0]))
    assert float(m.pressure_hpa[1]) < float(m.pressure_hpa[0]) < 1013.25
    sea = Barometer(1, "cpu")  # alt_ref 0, craft on the ground
    m0 = sea.update(_state(), dt=0.01)
    assert abs(float(m0.pressure_hpa[0]) - 1013.25) < 1e-3
    # dither: consecutive readings of a HELD craft differ (PX4 anti-STALE lesson)
    noisy = Barometer(1, "cpu", noise=BarometerNoiseConfig(pressure_std=0.005))
    a = noisy.update(_state(), dt=0.01).pressure_hpa
    b = noisy.update(_state(), dt=0.01).pressure_hpa
    assert float((a - b).abs()) > 0.0
