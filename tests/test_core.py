"""Core unit tests -- torch-only, no sim boot.

Covers the import boundary (the kernel must NOT pull isaaclab/isaacsim), the frame
conversions, the control allocation, the State NED accessors, and the airframe.
Run: env -u PYTHONPATH uv run --directory . pytest -q tests/test_kernel.py
"""

import sys

import torch

import isaacsitl
from isaacsitl import (
    Quadrotor,
    conversions,
)
from isaacsitl.motor import InstantMotor, LaggedMotor
from isaacsitl.state import State

# A simple 4-rotor test airframe, decoupled from the real conf/airframe/*.yaml specs.
_TEST_GEOM = {
    "rotor_x": (-0.1, 0.1, -0.1, 0.1),
    "rotor_y": (-0.08, -0.08, 0.08, 0.08),
    "yaw_signs": (-1.0, 1.0, 1.0, -1.0),
    "kappa": 0.0157,
}


def test_kernel_imports_without_isaac():
    """The package is torch-native: importing it must not boot Isaac Lab / Isaac Sim."""
    assert "isaaclab" not in sys.modules
    assert "isaacsim" not in sys.modules
    # the public surface is present
    for name in ("State", "Actuation", "WindField", "Quadrotor", "Eskf"):
        assert hasattr(isaacsitl, name)


def test_frame_conversions_are_self_inverse():
    """ENU<->NED, FLU<->FRD, and the aero<->isaac quat swap each undo themselves."""
    v = torch.randn(5, 3)
    assert torch.allclose(conversions.vec_enu_ned(conversions.vec_enu_ned(v)), v)
    assert torch.allclose(conversions.vec_flu_frd(conversions.vec_flu_frd(v)), v)
    q = torch.randn(5, 4)
    q /= torch.linalg.norm(q, dim=-1, keepdim=True)
    back = conversions.quat_aero_isaac(conversions.quat_aero_isaac(q))
    assert torch.allclose(back, q, atol=1e-6)


def test_euler_quat_round_trip():
    """euler_xyz -> quat -> euler_xyz recovers the angles within the principal range."""
    e = (torch.rand(8, 3) - 0.5) * torch.tensor([2.0, 1.0, 2.0])  # safe pitch range
    q = conversions.euler_xyz_to_quat_xyzw(e)
    e2 = conversions.quat_xyzw_to_euler_xyz(q)
    assert torch.allclose(e, e2, atol=1e-5)


def test_state_ned_accessors():
    """ENU (1,2,3) -> NED (2,1,-3); ENU-identity body -> NED yaw +pi/2 (East)."""
    state = State(
        position=torch.tensor([1.0, 2.0, 3.0]).expand(2, 3),
        attitude=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 4),
        linear_velocity=torch.zeros(2, 3),
        angular_velocity=torch.zeros(2, 3),
        linear_acceleration=torch.zeros(2, 3),
    )
    assert torch.allclose(state.position_ned()[0], torch.tensor([2.0, 1.0, -3.0]))
    # Identity FLU-in-ENU points East; NED yaw = +pi/2, roll/pitch 0.
    euler = state.euler_ned_frd()
    assert torch.allclose(euler[:, :2], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(euler[:, 2], torch.full((2,), torch.pi / 2), atol=1e-6)


def test_quadrotor_linear_motor_and_twr():
    """Quadrotor + InstantMotor: linear curve, no reaction, ~4:1 TWR; force-at-links."""
    quad = Quadrotor(**_TEST_GEOM, motor=InstantMotor(4.6))
    a = torch.tensor([[-1.0, -1, -1, -1], [1.0, 1, 1, 1], [0.0, 0, 0, 0]])
    thrust, reaction = quad._motor.step(a, 0.01)
    assert reaction is None  # InstantMotor models no dW reaction
    assert torch.allclose(thrust[0], torch.zeros(4), atol=1e-6)  # -1 -> 0
    assert torch.allclose(thrust[1], torch.full((4,), 4.6), atol=1e-6)  # +1 -> max
    assert torch.allclose(thrust[2], torch.full((4,), 2.3), atol=1e-6)  # 0 -> half
    assert 3.5 < quad.thrust_to_weight(0.4724) < 4.5  # ~4:1 TWR at 0.472 kg
    # force-at-links primitive: per-rotor thrust + yaw couple (no manual r x F mixing)
    th, yaw = quad.thrusts(torch.zeros(2, 4), 0.01)  # action 0 -> 2.3 N/rotor
    assert torch.allclose(th, torch.full((2, 4), 2.3), atol=1e-4)
    assert torch.allclose(yaw, torch.zeros(2), atol=1e-6)  # symmetric -> 0 yaw couple


def test_quadrotor_classical_motor_lag_and_dw_reaction():
    """Quadrotor + LaggedMotor: motor lag + the OQCRL dW yaw reaction (one airframe)."""
    q = Quadrotor(**_TEST_GEOM, motor=LaggedMotor(4.6, 2, 4, "cpu"))
    dt = 0.01
    # hover (action 0, w=0.5): per-rotor thrust 4.6*0.25, symmetric -> 0 yaw couple
    th, yaw = q.thrusts(torch.zeros(2, 4), dt)
    assert torch.allclose(th, torch.full((2, 4), 4.6 * 0.25), atol=1e-4)
    assert torch.allclose(yaw, torch.zeros(2), atol=1e-5)
    # full-throttle: the motor LAGS (w climbs 0.5 -> ~0.67, not instantly to 1.0)
    q.reset()
    q.thrusts(torch.ones(2, 4), dt)
    assert (q._motor._w > 0.5).all() and (q._motor._w < 0.9).all()
    # differential throttle -> a nonzero dW yaw couple (PhysX adds r x F)
    q.reset()
    _, yaw2 = q.thrusts(torch.tensor([[1.0, -1.0, -1.0, 1.0]]).expand(2, 4), dt)
    assert yaw2.abs().min() > 1e-3


def test_airframe_config_loads_and_builds():
    """conf/airframe/*.yaml -> AirframeCfg -> Quadrotor (data-driven airframe)."""
    from isaacsitl.airframe import build_quadrotor, load_airframe

    af = load_airframe("sourceone_racer")
    assert af.usd.endswith("racer.usd")
    assert af.motor.kind == "instant" and abs(af.motor.max_thrust - 4.6) < 1e-9
    quad = build_quadrotor(af, num_envs=1, device="cpu")
    rx, _ = quad.rotor_offsets
    want = torch.tensor([-0.1008, 0.1008, -0.1008, 0.1008])
    assert torch.allclose(rx, want, atol=1e-6)
    # a twr-only motor spec derives max_thrust from the (USD) mass
    iris = load_airframe("iris")
    assert abs(iris.motor.twr - 2.0) < 1e-9 and iris.motor.max_thrust is None
    qi = build_quadrotor(iris, 1, "cpu", mass=1.5)
    assert abs(qi._motor.max_thrust - 2.0 * 1.5 * 9.81 / 4) < 1e-4


def test_airframe_usd_resolves_to_vendored_asset():
    """Specs + assets ship INSIDE the package: no repo checkout or sibling needed."""
    from pathlib import Path

    import isaacsitl
    from isaacsitl.airframe import load_airframe

    pkg = Path(isaacsitl.__file__).resolve().parent
    for name in ("sourceone_racer", "iris"):
        usd = Path(load_airframe(name).usd)
        assert usd.is_absolute() and usd.exists(), f"{name}: missing USD {usd}"
        assert pkg in usd.parents, f"{name}: USD not packaged: {usd}"


def test_airframe_and_asset_errors_name_the_alternatives():
    """A typo'd airframe/asset fails fast, listing what IS available."""
    import pytest

    from isaacsitl.airframe import load_airframe
    from isaacsitl.assets import asset_path

    with pytest.raises(FileNotFoundError, match="sourceone_racer"):
        load_airframe("sourceone_race")  # typo'd name -> the known list names it
    with pytest.raises(FileNotFoundError, match=r"racer\.usd"):
        asset_path("racerr.usd")


def test_body_drag_opposes_velocity_and_wires_through_the_spec():
    """Drag is zero at rest, opposes v per-axis, and builds from the airframe spec."""
    from isaacsitl.airframe import AirframeCfg, DragCfg, MotorCfg, build_quadrotor
    from isaacsitl.quadrotor import BodyDrag

    drag = BodyDrag(linear=(0.3, 0.3, 0.4), quadratic=(0.02, 0.02, 0.0))
    assert torch.allclose(drag.force(torch.zeros(2, 3)), torch.zeros(2, 3))
    v = torch.tensor([[2.0, -1.0, 0.5]])
    f = drag.force(v)
    assert (f * v <= 0).all()  # every component opposes the motion
    want_x = -(0.3 + 0.02 * 2.0) * 2.0  # -(c1 + c2*|v|) * v, x-axis
    assert abs(float(f[0, 0]) - want_x) < 1e-6

    cfg = AirframeCfg(
        usd="racer.usd",
        rotor_x=[-0.1, 0.1, -0.1, 0.1],
        rotor_y=[-0.08, -0.08, 0.08, 0.08],
        yaw_signs=[-1.0, 1.0, 1.0, -1.0],
        motor=MotorCfg(max_thrust=4.6),
        drag=DragCfg(linear=[0.3, 0.3, 0.4]),
    )
    quad = build_quadrotor(cfg, num_envs=1, device="cpu")
    assert quad.has_drag
    assert (quad.drag_force(v) * v <= 0).all()
    # the all-zeros default spec builds WITHOUT a drag model (no-op preserved)
    cfg.drag = DragCfg()
    assert not build_quadrotor(cfg, 1, "cpu").has_drag


def test_motor_dr_randomizes_per_env_within_bands():
    """randomize() resamples ONLY the given envs, inside the band; obs sees it."""
    from isaacsitl.motor import InstantMotor, LaggedMotor, MotorDRConfig

    gen = torch.Generator().manual_seed(0)
    m = LaggedMotor(4.6, num_envs=8, n_rotors=4, device="cpu")
    nominal = m.step(torch.zeros(8, 4), dt=10.0)[0].clone()  # settled: w -> 0.5
    m.reset()

    cfg = MotorDRConfig(max_thrust=0.2, tau=0.3)
    m.randomize(torch.arange(4), cfg, gen)  # envs 0-3 only
    intr = m.intrinsics
    assert intr.shape == (8, 2)
    assert (intr[4:] - 1.0).abs().max() < 1e-12  # untouched envs stay nominal
    assert (intr[:4, 0] - 1.0).abs().max() <= 0.2 + 1e-6
    assert (intr[:4, 1] - 1.0).abs().max() <= 0.3 + 1e-6
    assert (intr[:4] - 1.0).abs().max() > 1e-6  # actually resampled
    scaled = m.step(torch.zeros(8, 4), dt=10.0)[0]
    assert torch.allclose(scaled[:4], nominal[:4] * intr[:4, :1], atol=1e-5)
    assert torch.allclose(scaled[4:], nominal[4:], atol=1e-5)

    inst = InstantMotor(4.6, num_envs=2, device="cpu")
    inst.randomize(None, MotorDRConfig(max_thrust=0.1), gen)
    full = inst.step(torch.ones(2, 4), dt=0.01)[0]
    assert torch.allclose(full, 4.6 * inst.intrinsics, atol=1e-5)

    bare = InstantMotor(4.6)  # built without a batch: DR must refuse loudly
    import pytest

    with pytest.raises(ValueError, match="num_envs"):
        bare.randomize(None, cfg)


def test_lagged_motor_matches_the_closed_form_lag():
    """Exact ZOH: after k held steps, w == u + (w0 - u) * exp(-k*dt/tau), any dt/tau.

    The dt/tau = 5.0 case is the discriminator: explicit Euler overshoots to the
    clamp there (w -> 1.0 exactly), while the exact update lands on the closed form
    -- the property that lets a micro motor's tau ~ dt integrate at any physics dt.
    """
    import math

    for dt, tau, steps in ((0.01, 0.03, 7), (0.002, 0.012, 7), (0.01, 0.002, 1)):
        m = LaggedMotor(
            4.6, num_envs=1, n_rotors=1, device="cpu", tau_inc=tau, tau_dec=tau
        )
        thrust = None
        for _ in range(steps):  # throttle 1.0, held (action +1)
            thrust, _ = m.step(torch.ones(1, 1), dt)
        w_want = 1.0 + (0.5 - 1.0) * math.exp(-steps * dt / tau)
        assert torch.allclose(m._w, torch.full((1, 1), w_want), atol=1e-6)
        assert torch.allclose(thrust, torch.full((1, 1), 4.6 * w_want**2), atol=1e-5)


def test_motor_command_for_inverts_the_curve():
    """command_for is the steady-state inverse: step(command_for(T)) settles at T."""
    inst = InstantMotor(4.6)
    want = torch.tensor([[0.0, 1.15, 2.3, 4.6]])
    assert torch.allclose(inst.step(inst.command_for(want), 0.01)[0], want, atol=1e-6)
    lag = LaggedMotor(4.6, num_envs=1, n_rotors=4, device="cpu")
    cmds = lag.command_for(want)
    thrust = None
    for _ in range(600):  # hold the command until the lag settles (~20 tau)
        thrust, _ = lag.step(cmds, 0.01)
    assert torch.allclose(thrust, want, atol=1e-3)
    # over-asking saturates at the command ceiling, not beyond it
    over = torch.tensor([[9.9]])
    assert float(inst.command_for(over)) <= 1.0 + 1e-9
    assert float(lag.command_for(over)) <= 1.0 + 1e-9
    assert float(inst.command_for(over)) > 0.99
    assert float(lag.command_for(over)) > 0.99


def test_motor_dr_tau_band_changes_lag_rate():
    """A larger tau scale lags harder: less spin gained in one step."""
    from isaacsitl.motor import LaggedMotor, MotorDRConfig

    m = LaggedMotor(4.6, num_envs=2, n_rotors=1, device="cpu")
    m._tau_scale = torch.tensor([[1.0], [2.0]])  # env1: twice the time constant
    _ = MotorDRConfig  # (scales set directly for a deterministic check)
    thrust = m.step(torch.ones(2, 1), dt=0.01)[0]
    assert float(thrust[1]) < float(thrust[0])  # slower env gained less thrust


def test_resolve_usd_bare_vendored_vs_explicit_path(tmp_path):
    """Bare filename -> vendored asset; an explicit relative/absolute path -> as-is."""
    import pytest

    from isaacsitl.airframe import resolve_usd
    from isaacsitl.assets import asset_path

    # bare filename routes to the packaged assets
    assert resolve_usd("racer.usd") == str(asset_path("racer.usd"))
    # an explicit path (here absolute) that exists is returned resolved, NOT forced
    # into the package -- a user's own craft outside isaacsitl/assets still works
    custom = tmp_path / "mycraft.usd"
    custom.write_bytes(b"x")
    assert resolve_usd(str(custom)) == str(custom.resolve())
    # a relative path with a directory part is a real path, not a vendored lookup
    rel = tmp_path / "robots" / "c.usd"
    rel.parent.mkdir()
    rel.write_bytes(b"x")
    assert resolve_usd(str(rel)) == str(rel.resolve())
    # missing meshes fail fast either way
    with pytest.raises(FileNotFoundError, match=r"racerr\.usd"):
        resolve_usd("racerr.usd")  # bare, not vendored
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_usd(str(tmp_path / "sub" / "missing.usd"))  # explicit, absent
