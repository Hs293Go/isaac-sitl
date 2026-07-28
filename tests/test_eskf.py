"""Basic ESKF -- torch-only, no sim boot.

Validates that IMU prediction + position correction track a known truth: a static
hover stays put, and a constant-velocity truth has its (unobserved-at-init) velocity
recovered from the position fixes alone.
"""

from __future__ import annotations

import torch

from isaacsitl.gnc.eskf import Eskf, EskfConfig

_G = 9.81
_IDENT = torch.tensor([0.0, 0.0, 0.0, 1.0])


def test_static_hover_estimate_stays_at_truth():
    """At rest and level (accel=(0,0,g), gyro=0), the estimate does not drift."""
    p0 = torch.tensor([0.0, 0.0, 1.5])
    eskf = Eskf(position=p0, attitude=_IDENT, device="cpu")
    accel = torch.tensor([0.0, 0.0, _G])
    gyro = torch.zeros(3)
    for _ in range(300):
        eskf.predict(accel, gyro, dt=0.01)
        eskf.update_position(p0)
    s = eskf.state()
    assert torch.allclose(s.position[0], p0, atol=1e-2)
    assert torch.allclose(s.linear_velocity[0], torch.zeros(3), atol=1e-2)
    assert torch.allclose(s.attitude[0], _IDENT, atol=1e-2)


def test_constant_velocity_is_recovered_from_position_fixes():
    """Init v=0; position fixes on moving truth recover the velocity estimate."""
    p0 = torch.tensor([0.0, 0.0, 1.5])
    v_true = torch.tensor([1.0, -0.5, 0.0])  # m/s
    eskf = Eskf(position=p0, attitude=_IDENT, device="cpu")
    accel = torch.tensor([0.0, 0.0, _G])  # level, zero kinematic accel
    gyro = torch.zeros(3)
    dt = 0.01
    for k in range(600):
        eskf.predict(accel, gyro, dt)
        eskf.update_position(p0 + v_true * ((k + 1) * dt))
    s = eskf.state()
    assert torch.allclose(s.position[0], p0 + v_true * (600 * dt), atol=0.05)
    assert torch.allclose(s.linear_velocity[0], v_true, atol=0.1)


def test_state_shapes_and_bias_corrected_rate():
    """state() returns a batched (n=1) State; rate = last gyro - estimated gyro bias."""
    eskf = Eskf(
        position=torch.zeros(3),
        attitude=_IDENT,
        device="cpu",
        config=EskfConfig(),
    )
    gyro = torch.tensor([0.05, -0.02, 0.10])
    eskf.predict(torch.tensor([0.0, 0.0, _G]), gyro, dt=0.01)
    s = eskf.state()
    assert s.position.shape == (1, 3)
    assert s.attitude.shape == (1, 4)
    assert s.angular_velocity.shape == (1, 3)
    # no correction yet -> gyro bias is ~0, so the reported rate is the raw gyro
    assert torch.allclose(s.angular_velocity[0], gyro - eskf.gyro_bias, atol=1e-6)


def test_batched_envs_are_independent():
    """Env i in an N-batch matches running that env alone -- no cross-talk."""
    p0 = torch.tensor([0.0, 0.0, 1.5])
    vels = torch.tensor([[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.3, 0.3, 0.2]])  # (3, 3)
    accel = torch.tensor([0.0, 0.0, _G])
    dt = 0.01

    batched = Eskf(position=p0.expand(3, 3), attitude=_IDENT.expand(3, 4), device="cpu")
    for k in range(200):
        batched.predict(accel.expand(3, 3), torch.zeros(3, 3), dt)
        batched.update_position(p0 + vels * ((k + 1) * dt))  # (3, 3) per-env fix

    single = Eskf(position=p0, attitude=_IDENT, device="cpu")  # just env 1, on its own
    for k in range(200):
        single.predict(accel, torch.zeros(3), dt)
        single.update_position(p0 + vels[1] * ((k + 1) * dt))

    b, s = batched.state(), single.state()
    assert torch.allclose(b.position[1], s.position[0], atol=1e-5)
    assert torch.allclose(b.linear_velocity[1], s.linear_velocity[0], atol=1e-5)


def test_reset_env_ids_reinits_only_those_rows():
    """reset(env_ids) row == a freshly constructed filter; other rows fly untouched."""
    p0 = torch.tensor([0.0, 0.0, 1.5])
    accel = torch.tensor([0.0, 0.0, _G])
    dt = 0.01
    vels = torch.tensor([[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.3, 0.3, 0.2]])  # (3, 3)

    def drive(f, steps, offset):
        for k in range(steps):
            f.predict(accel.expand(3, 3), torch.zeros(3, 3), dt)
            f.update_position(p0 + vels * ((k + offset + 1) * dt))

    ref = Eskf(position=p0.expand(3, 3), attitude=_IDENT.expand(3, 4), device="cpu")
    drive(ref, 150, 0)

    poked = Eskf(position=p0.expand(3, 3), attitude=_IDENT.expand(3, 4), device="cpu")
    drive(poked, 75, 0)
    p_new = torch.tensor([[5.0, 5.0, 2.0]])  # respawn env 1 far from its track
    poked.reset(torch.tensor([1]), position=p_new, attitude=_IDENT.expand(1, 4))
    drive(poked, 75, 75)

    # rows 0/2 never saw the reset -> identical to the untouched batch
    r, q = ref.state(), poked.state()
    for i in (0, 2):
        assert torch.allclose(q.position[i], r.position[i], atol=1e-6)
        assert torch.allclose(q.linear_velocity[i], r.linear_velocity[i], atol=1e-6)

    # row 1 == a fresh filter constructed at the new pose, fed the same 75 fixes
    fresh = Eskf(position=p_new, attitude=_IDENT.expand(1, 4), device="cpu")
    for k in range(75):
        fresh.predict(accel.expand(1, 3), torch.zeros(1, 3), dt)
        fresh.update_position((p0 + vels[1] * ((k + 75 + 1) * dt)).expand(1, 3))
    f = fresh.state()
    assert torch.allclose(q.position[1], f.position[0], atol=1e-5)
    assert torch.allclose(q.linear_velocity[1], f.linear_velocity[0], atol=1e-5)
    assert torch.allclose(q.attitude[1], f.attitude[0], atol=1e-5)


def test_per_env_config_sweeps_gains():
    """A tensor-valued EskfConfig gives each env its own filter -- the sweep knob."""
    p0 = torch.tensor([0.0, 0.0, 1.5])
    cfg = EskfConfig(position_meas_std=torch.tensor([0.02, 2.0]))  # tight vs loose R
    eskf = Eskf(
        position=p0.expand(2, 3), attitude=_IDENT.expand(2, 4), device="cpu", config=cfg
    )
    accel = torch.tensor([0.0, 0.0, _G]).expand(2, 3)
    fix = (p0 + torch.tensor([1.0, 0.0, 0.0])).expand(2, 3)  # 1 m offset fix, both envs
    for _ in range(50):
        eskf.predict(accel, torch.zeros(2, 3), 0.01)
        eskf.update_position(fix)

    est = eskf.state().position  # (2, 3)
    err_tight = float(torch.linalg.norm(est[0] - fix[0]))
    err_loose = float(torch.linalg.norm(est[1] - fix[1]))
    assert err_tight < err_loose  # tight R trusts the fix -> corrects harder


def test_mag_update_corrects_heading_only():
    """A compass residual pulls yaw to the truth and leaves roll/pitch alone."""
    import math

    from isaacsitl import so3
    from isaacsitl.sensors import Magnetometer

    psi = 0.3  # true heading; the filter starts at yaw 0 (a 0.3 rad heading error)
    q_true = torch.tensor([[0.0, 0.0, math.sin(psi / 2), math.cos(psi / 2)]])
    mag = Magnetometer(1, "cpu")
    # what a compass on the TRUE (yawed) body reads, noiseless
    field_body = so3.quat_rotate(so3.quat_conjugate(q_true), mag.field_enu.expand(1, 3))

    eskf = Eskf(position=torch.zeros(1, 3), attitude=_IDENT.expand(1, 4), device="cpu")
    for _ in range(50):
        eskf.update_mag(field_body, mag.field_enu)
    rpy = so3.quat_to_euler(eskf.q)[0]
    assert abs(float(rpy[2]) - psi) < 0.02  # yaw converged to the truth
    assert float(rpy[:2].abs().max()) < 1e-3  # roll/pitch untouched (yaw-only fusion)
