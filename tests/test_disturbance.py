"""Disturbance unit tests -- torch-only, no sim boot.

Pins the L1-experiment contract: onset envelope semantics, seed determinism
(the paired-A/B requirement), the exact-zero clean path, bias gating, the
frame discipline of drag (opposes AIR-relative motion in the world frame,
attitude-independent for symmetric coefficients), and JSON-able provenance.
"""

import json
import math

import pytest
import torch

from isaacsitl import Disturbance, DisturbanceCfg, State
from isaacsitl.wind import WindCfg


def _state(vel=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), n=2) -> State:
    return State(
        position=torch.zeros(n, 3),
        attitude=torch.tensor(quat).expand(n, 4),
        linear_velocity=torch.tensor(vel).expand(n, 3),
        angular_velocity=torch.zeros(n, 3),
        linear_acceleration=torch.zeros(n, 3),
    )


def test_envelope_semantics():
    d = Disturbance(
        DisturbanceCfg(force_enu=(1.0, 0.0, 0.0), t_on=2.0, ramp=1.0), 1, "cpu", dt=0.01
    )
    assert d.gain(0.0) == pytest.approx(0.0)
    assert d.gain(1.999) == pytest.approx(0.0)
    assert d.gain(2.5) == pytest.approx(0.5)
    assert d.gain(3.0) == pytest.approx(1.0)
    assert d.gain(100.0) == pytest.approx(1.0)
    step = Disturbance(DisturbanceCfg(force_enu=(1.0, 0.0, 0.0)), 1, "cpu", dt=0.01)
    assert step.gain(0.0) == pytest.approx(1.0)  # t_on=0, ramp=0 -> on immediately


def test_clean_cfg_is_exactly_zero():
    cfg = DisturbanceCfg()
    assert not cfg.active()
    d = Disturbance(cfg, 2, "cpu", dt=0.01)
    f, tau = d.wrench(0.0, _state(vel=(3.0, -1.0, 0.5)))
    assert torch.equal(f, torch.zeros(2, 3))
    assert torch.equal(tau, torch.zeros(2, 3))


def test_bias_gated_by_onset():
    cfg = DisturbanceCfg(
        force_enu=(0.0, 0.0, -0.981), torque_flu=(0.0, 0.0, 0.02), t_on=1.0
    )
    d = Disturbance(cfg, 2, "cpu", dt=0.01)
    f0, t0 = d.wrench(0.5, _state())
    assert torch.equal(f0, torch.zeros(2, 3)) and torch.equal(t0, torch.zeros(2, 3))
    f1, t1 = d.wrench(1.5, _state())
    assert torch.allclose(f1, torch.tensor([0.0, 0.0, -0.981]).expand(2, 3))
    assert torch.allclose(t1, torch.tensor([0.0, 0.0, 0.02]).expand(2, 3))


def test_same_seed_same_wrench_sequence():
    cfg = {
        "wind": WindCfg(steady=(4.0, 0.0, 0.0), gust_std=1.5),
        "drag_linear": (0.2, 0.2, 0.0),
        "seed": 7,
    }
    a = Disturbance(DisturbanceCfg(**cfg), 2, "cpu", dt=0.01)
    b = Disturbance(DisturbanceCfg(**cfg), 2, "cpu", dt=0.01)
    s = _state(vel=(1.0, 0.0, 0.0))
    for k in range(50):
        fa, ta = a.wrench(k * 0.01, s)
        fb, tb = b.wrench(k * 0.01, s)
        assert torch.equal(fa, fb) and torch.equal(ta, tb)


def test_different_seed_differs():
    mk = lambda seed: Disturbance(  # noqa: E731
        DisturbanceCfg(
            wind=WindCfg(gust_std=1.5), drag_linear=(0.2, 0.2, 0.2), seed=seed
        ),
        2,
        "cpu",
        dt=0.01,
    )
    a, b = mk(0), mk(1)
    s = _state(vel=(1.0, 0.0, 0.0))
    diff = any(
        not torch.equal(a.wrench(k * 0.01, s)[0], b.wrench(k * 0.01, s)[0])
        for k in range(10)
    )
    assert diff


def test_onset_does_not_perturb_gust_sequence():
    """Gating scales the OUTPUT; the seeded gust state advances identically."""
    mk = lambda t_on: Disturbance(  # noqa: E731
        DisturbanceCfg(
            wind=WindCfg(gust_std=1.0), drag_linear=(0.3, 0.3, 0.3), t_on=t_on, seed=3
        ),
        2,  # must match the _state batch: wrench enforces shape agreement
        "cpu",
        dt=0.01,
    )
    a, b = mk(0.0), mk(0.2)  # same seed, different onset
    s = _state(vel=(2.0, 0.0, 0.0))
    for k in range(60):
        fa, _ = a.wrench(k * 0.01, s)
        fb, _ = b.wrench(k * 0.01, s)
        if k * 0.01 >= 0.2:  # past both onsets the sequences must coincide
            assert torch.equal(fa, fb)


def test_drag_opposes_air_relative_motion():
    c = 0.4
    cfg = DisturbanceCfg(drag_linear=(c, c, c))
    d = Disturbance(cfg, 2, "cpu", dt=0.01)
    # still air, moving +x at 3 m/s -> world force -c*3 along x
    f, _ = d.wrench(0.0, _state(vel=(3.0, 0.0, 0.0)))
    assert torch.allclose(f, torch.tensor([-c * 3.0, 0.0, 0.0]).expand(2, 3), atol=1e-6)
    # steady wind faster than the (still) craft -> pushed DOWNWIND (+x)
    windy = DisturbanceCfg(wind=WindCfg(steady=(5.0, 0.0, 0.0)), drag_linear=(c, c, c))
    dw = Disturbance(windy, 2, "cpu", dt=0.01)
    fw, _ = dw.wrench(0.0, _state())
    assert torch.allclose(fw, torch.tensor([c * 5.0, 0.0, 0.0]).expand(2, 3), atol=1e-6)


def test_drag_world_force_is_attitude_independent_for_symmetric_coeffs():
    c = 0.25
    d = Disturbance(DisturbanceCfg(drag_linear=(c, c, c)), 2, "cpu", dt=0.01)
    vel = (2.0, -1.0, 0.5)
    f_id, _ = d.wrench(0.0, _state(vel=vel))
    yaw90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))  # xyzw
    f_rot, _ = d.wrench(0.0, _state(vel=vel, quat=yaw90))
    assert torch.allclose(f_id, f_rot, atol=1e-5)


def test_cfg_validation():
    with pytest.raises(ValueError, match="t_on"):
        DisturbanceCfg(t_on=-1.0)
    with pytest.raises(ValueError, match="ramp"):
        DisturbanceCfg(ramp=-0.1)
    with pytest.raises(ValueError, match="drag_linear"):
        DisturbanceCfg(drag_linear=(-0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="wind needs drag_linear"):
        DisturbanceCfg(wind=WindCfg(steady=(5.0, 0.0, 0.0)))


def test_provenance_is_jsonable_and_complete():
    cfg = DisturbanceCfg(
        wind=WindCfg(steady=(6.0, 0.0, 0.0), gust_std=1.5),
        drag_linear=(0.1, 0.2, 0.0),
        force_enu=(0.0, 0.0, -0.981),
        t_on=8.0,
        ramp=0.5,
        seed=42,
    )
    p = cfg.provenance()
    s = json.dumps(p)  # must not raise
    back = json.loads(s)
    assert back["seed"] == 42
    assert back["t_on"] == pytest.approx(8.0)
    assert back["wind"]["gust_std"] == pytest.approx(1.5)
    assert back["force_enu"] == pytest.approx([0.0, 0.0, -0.981])
