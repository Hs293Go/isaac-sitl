"""classical controller suite: pins the inertia-aware design CLAUDE.md must not regress.

Everything here is closed-form: hover equilibrium, torque linear in inertia (gains are
frequencies, not raw torques), integrator band/clamp/reset, tilt-cap saturation,
per-env batched gains, and the yaw-couple sign. No sim; a hand-built `State` drives
the controller exactly as the vehicle bridge would. The Wrench layer (PinvAllocator +
the CTBR RateController) is pinned at the end.
"""

import math

import pytest
import torch

from isaacsitl import (
    ClassicalControllerBackend,
    InertialProperties,
    PinvAllocator,
    RateController,
    State,
    Wrench,
)
from isaacsitl.gnc.pid_controller import PIDController, PIDControllerCfg

_RX = (-0.1008, 0.1008, -0.1008, 0.1008)
_RY = (-0.0766, -0.0766, 0.0766, 0.0766)
_MASS = 0.472
_INERTIA = (0.0025, 0.0025, 0.004)
_DT = 0.01
_G = 9.81
_HOVER = (0.0, 0.0, 1.5)


# The PID gains / loop constants that route into the controller cfg (vs the airframe
# facts: mass/inertia -> InertialProperties, rotor geometry/position -> the backend).
_CFG_KEYS = frozenset(PIDControllerCfg.__dataclass_fields__)


def _controller(**kw) -> ClassicalControllerBackend:
    """Assemble the DI backend from loose kwargs (test ergonomics): split into the
    `PIDController` cfg, the `InertialProperties`, and the backend geometry/setpoint."""
    rotor_x = kw.pop("rotor_x", _RX)
    rotor_y = kw.pop("rotor_y", _RY)
    position = kw.pop("position", _HOVER)
    mass = kw.pop("mass", _MASS)
    inertia = kw.pop("inertia", _INERTIA)
    cfg_kw = {"dt": _DT}
    cfg_kw.update({k: kw.pop(k) for k in list(kw) if k in _CFG_KEYS})
    assert not kw, f"unexpected _controller kwargs: {sorted(kw)}"
    ctrl = PIDController(PinvAllocator(rotor_x, rotor_y), PIDControllerCfg(**cfg_kw))
    inertial = InertialProperties(
        torch.as_tensor(mass, dtype=torch.float32),
        torch.as_tensor(inertia, dtype=torch.float32),
    )
    return ClassicalControllerBackend(ctrl=ctrl, inertial=inertial, position=position)


def _state(
    n=2, pos=_HOVER, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), omega=(0.0,) * 3
) -> State:
    return State(
        position=torch.tensor(pos).expand(n, 3).clone(),
        attitude=torch.tensor(quat).expand(n, 4).clone(),
        linear_velocity=torch.tensor(vel).expand(n, 3).clone(),
        angular_velocity=torch.tensor(omega).expand(n, 3).clone(),
        linear_acceleration=torch.zeros(n, 3),
    )


def _act(ctl: ClassicalControllerBackend, state: State):
    ctl.update_state(state)
    return ctl.actuation()


def test_actuation_before_update_state_raises():
    with pytest.raises(RuntimeError, match="update_state"):
        _controller().actuation()


def test_hover_equilibrium_is_gravity_over_four_rotors():
    """At the setpoint, at rest: each rotor lifts mg/4 and no torque is commanded."""
    act = _act(_controller(), _state())
    want = _MASS * _G / 4
    assert torch.allclose(act.rotor_thrust, torch.full((2, 4), want), atol=1e-5)
    assert torch.allclose(act.body_torque, torch.zeros(2, 3), atol=1e-6)


def test_torque_is_linear_in_inertia_so_gains_are_frequencies():
    """The don't-regress property: same gains command the same ANGULAR ACCELERATION.

    Doubling the inertia must exactly double the corrective torque (at zero body
    rate), i.e. the attitude gains are frequencies, portable across airframes -- the
    light-racer-NaN fix.
    """
    # a small roll error keeps every rotor thrust positive (no clamp in the way)
    tilt = (math.sin(0.025), 0.0, 0.0, math.cos(0.025))  # 0.05 rad about x, xyzw
    s = _state(quat=tilt)
    acts = [
        _act(_controller(inertia=tuple(k * i for i in _INERTIA)), s)
        for k in (1.0, 2.0, 3.0)
    ]
    d21 = acts[1].rotor_thrust - acts[0].rotor_thrust
    d31 = acts[2].rotor_thrust - acts[0].rotor_thrust
    assert d21.abs().max() > 1e-4  # the correction is actually present
    assert torch.allclose(d31, 2.0 * d21, atol=1e-6)
    assert torch.allclose(
        acts[2].body_torque - acts[0].body_torque,
        2.0 * (acts[1].body_torque - acts[0].body_torque),
        atol=1e-6,
    )


def test_thrust_scales_with_mass():
    """Gravity feed-forward comes from the mass, not from tuned gains."""
    a1 = _act(_controller(mass=_MASS), _state())
    a2 = _act(_controller(mass=2 * _MASS), _state())
    assert torch.allclose(a2.rotor_thrust, 2.0 * a1.rotor_thrust, atol=1e-5)


def test_integrator_clamps_and_zeroes_outside_band():
    """The error integral saturates at i_limit and resets outside i_band."""
    ctl = _controller(i_limit=0.05, i_band=0.3)
    near = _state(pos=(0.0, 0.0, 1.4))  # 0.1 m low: inside the band
    for _ in range(100):  # 100 steps * 0.1 m * 0.01 s = 0.1 m.s >> the 0.05 clamp
        _act(ctl, near)
    assert torch.allclose(ctl._ctrl._i[:, 2], torch.full((2,), 0.05), atol=1e-6)
    _act(ctl, _state(pos=(5.0, 0.0, 1.5)))  # far from the setpoint: band zeroes it
    assert torch.allclose(ctl._ctrl._i, torch.zeros(2, 3), atol=1e-9)


def test_reset_clears_only_the_given_envs():
    ctl = _controller()
    for _ in range(10):
        _act(ctl, _state(pos=(0.0, 0.0, 1.4)))
    assert ctl._ctrl._i[:, 2].min() > 0.0
    ctl.reset(torch.tensor([0]))
    assert ctl._ctrl._i[0].abs().max() < 1e-12 and ctl._ctrl._i[1, 2] > 0.0


def test_tilt_cap_saturates_large_position_errors():
    """Once capped, a 10 m and a 100 m error command the SAME (bounded) attitude."""
    near_sat = _act(_controller(position=(10.0, 0.0, 1.5)), _state())
    far_sat = _act(_controller(position=(100.0, 0.0, 1.5)), _state())
    assert torch.allclose(near_sat.rotor_thrust, far_sat.rotor_thrust, atol=1e-5)
    uncapped = [
        _act(_controller(position=(d, 0.0, 1.5), tilt_limit=False), _state())
        for d in (10.0, 100.0)
    ]
    assert not torch.allclose(
        uncapped[0].rotor_thrust, uncapped[1].rotor_thrust, atol=1e-3
    )


def test_yaw_setpoint_commands_signed_yaw_couple():
    """A positive (CCW, ENU) yaw reference yields a positive body-z torque."""
    ctl = _controller()
    ctl.set_reference(_HOVER, yaw=0.3)
    assert _act(ctl, _state()).body_torque[:, 2].min() > 0.0
    ctl.set_reference(_HOVER, yaw=-0.3)
    assert _act(ctl, _state()).body_torque[:, 2].max() < 0.0


def test_per_env_gain_batch_broadcasts():
    """(num_envs, 3) gains sweep per-env: doubling kp_z doubles the correction."""
    ctl = _controller(ki=(0.0, 0.0, 0.0))  # pure P so the correction is exactly kp*e
    ctl._ctrl.kp = torch.tensor([[6.0, 6.0, 8.0], [6.0, 6.0, 16.0]])
    act = _act(ctl, _state(pos=(0.0, 0.0, 1.4)))  # pure vertical error: no tilt
    hover = _MASS * _G / 4
    dev = act.rotor_thrust.sum(dim=-1) - 4 * hover  # collective above hover
    assert dev.min() > 0.0
    assert torch.allclose(dev[1], 2.0 * dev[0], atol=1e-5)


def test_set_reference_feed_forward_and_hold_semantics():
    """Velocity feed-forward shows up in the collective; omitting it resets to hold."""
    ctl = _controller()
    ctl.set_reference(_HOVER, velocity=(0.0, 0.0, 0.5))
    with_ff = _act(ctl, _state()).rotor_thrust.sum(dim=-1)
    ctl.set_reference(_HOVER)  # no velocity: feed-forward must NOT linger
    held = _act(ctl, _state()).rotor_thrust.sum(dim=-1)
    assert (with_ff - held).min() > 1e-3
    assert torch.allclose(held, torch.full((2,), 4 * _MASS * _G / 4), atol=1e-5)


def test_to_moves_the_whole_device_chain():
    """`.to()` must reach the injected allocator -- the mixer left on CPU under CUDA
    states was a live-sim crash the all-CPU suite never sees; `meta` stands in."""
    ctl = _controller().to("meta")
    assert ctl._ctrl.kp.device.type == "meta"
    assert ctl._ctrl._alloc._inv_mix3.device.type == "meta"
    assert ctl._inertial.mass.device.type == "meta"
    rc = RateController(PinvAllocator(_RX, _RY)).to("meta")
    assert rc.kw.device.type == "meta"
    assert rc._alloc._inv_mix3.device.type == "meta"


# --- the Wrench layer: PinvAllocator + RateController (the CTBR inner loop) ---


def _inertial(inertia=_INERTIA) -> InertialProperties:
    return InertialProperties(
        torch.as_tensor(_MASS, dtype=torch.float32),
        torch.as_tensor(inertia, dtype=torch.float32),
    )


def test_pinv_allocator_splits_collective_and_clamps_at_bounds():
    """Zero torque splits the collective evenly; the per-rotor bounds clamp."""
    alloc = PinvAllocator(_RX, _RY, thrust_max=1.5)
    even = alloc.allocate(Wrench(torch.full((2,), 4.0), torch.zeros(2, 3)))
    assert torch.allclose(even.rotor_thrust, torch.full((2, 4), 1.0), atol=1e-6)
    # over-asking hits the ceiling; a negative-demand rotor hits the floor (no pull)
    hard = alloc.allocate(
        Wrench(torch.full((2,), 8.0), torch.tensor([[0.0, 0.0, 0.0]]).expand(2, 3))
    )
    assert float(hard.rotor_thrust.max()) <= 1.5 + 1e-6
    assert (
        float(
            alloc.allocate(
                Wrench(torch.zeros(2), torch.tensor([[1.0, 0.0, 0.0]]).expand(2, 3))
            ).rotor_thrust.min()
        )
        >= 0.0
    )


def test_rate_controller_hover_allocation_and_axis_response():
    """Zero rate error splits the collective evenly; rate errors load the right axes."""
    rc = RateController(PinvAllocator(_RX, _RY))
    omega = torch.zeros(2, 3)
    inertial = _inertial()
    hold = rc.actuation(omega, torch.full((2,), 4.0), torch.zeros(2, 3), inertial)
    assert torch.allclose(hold.rotor_thrust, torch.full((2, 4), 1.0), atol=1e-6)
    assert torch.allclose(hold.body_torque, torch.zeros(2, 3), atol=1e-8)
    # +roll-rate setpoint -> +x torque -> the +y (left, _RY > 0) rotors load up
    roll = rc.actuation(
        omega,
        torch.full((2,), 4.0),
        torch.tensor([[1.0, 0.0, 0.0]]).expand(2, 3),
        inertial,
    )
    assert (roll.rotor_thrust[:, 2:] > roll.rotor_thrust[:, :2]).all()
    # +yaw-rate setpoint -> a pure body-z couple, NO rotor differential
    yaw = rc.actuation(
        omega,
        torch.full((2,), 4.0),
        torch.tensor([[0.0, 0.0, 1.0]]).expand(2, 3),
        inertial,
    )
    assert float(yaw.body_torque[0, 2]) > 0.0
    assert torch.allclose(yaw.rotor_thrust, torch.full((2, 4), 1.0), atol=1e-6)


def test_rate_controller_torque_is_inertia_scaled_plus_gyroscopic():
    """Tracked spin (sp == omega) leaves exactly the omega x (I omega) feed-forward."""
    inertia = torch.tensor([0.0025, 0.005, 0.004])  # deliberately non-isotropic
    rc = RateController(PinvAllocator(_RX, _RY))
    omega = torch.tensor([[0.5, -0.3, 0.8]])
    act = rc.actuation(omega, torch.full((1,), 4.0), omega.clone(), _inertial(inertia))
    want = torch.cross(omega, inertia * omega, dim=-1)[0]
    rx, ry = torch.tensor(_RX), torch.tensor(_RY)
    tau_x = (ry * act.rotor_thrust).sum(dim=-1)  # reconstruct from the allocation
    tau_y = (-rx * act.rotor_thrust).sum(dim=-1)
    assert torch.allclose(tau_x[0], want[0], atol=1e-6)
    assert torch.allclose(tau_y[0], want[1], atol=1e-6)
    assert torch.allclose(act.body_torque[0, 2], want[2], atol=1e-6)
    # the allocation preserves the commanded collective
    assert torch.allclose(act.rotor_thrust.sum(dim=-1), torch.tensor([4.0]), atol=1e-5)


def test_rate_controller_torque_is_linear_in_live_inertia():
    """Per-call InertialProperties: doubling inertia doubles the corrective torque."""
    rc = RateController(PinvAllocator(_RX, _RY))
    omega = torch.zeros(1, 3)
    sp = torch.tensor([[0.0, 0.0, 2.0]])  # yaw-rate step: torque shows up on body-z
    tau1 = rc.actuation(omega, torch.ones(1), sp, _inertial()).body_torque[0, 2]
    doubled = tuple(2.0 * i for i in _INERTIA)
    tau2 = rc.actuation(omega, torch.ones(1), sp, _inertial(doubled)).body_torque[0, 2]
    assert torch.allclose(tau2, 2.0 * tau1, atol=1e-7)
