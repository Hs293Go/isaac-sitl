"""Wind-field unit tests -- torch-only, no sim boot.

Pins the OU gust's exact-discretization statistics (stationary mean/std for any dt,
exponential autocorrelation), the steady-only degeneration, per-env reset isolation,
the DR steady resampling, and the quadratic drag law.
"""

import math

import pytest
import torch

from isaacsitl import State
from isaacsitl.wind import WindCfg, WindField, quadratic_drag


def test_ou_gust_stationary_stats():
    """Stationary mean -> steady, std -> gust_std, autocorr(tau) -> 1/e."""
    n, dt, tau, std = 8192, 0.01, 0.5, 1.5
    steady = torch.tensor([3.0, -1.0, 0.5])
    gen = torch.Generator().manual_seed(0)
    cfg = WindCfg(steady=(3.0, -1.0, 0.5), gust_std=std, gust_tau=tau)
    wind = WindField(cfg, n, "cpu", generator=gen)
    v0 = wind.velocity.clone()
    lag = 50  # lag * dt = tau -> autocorrelation exp(-1)
    for _ in range(lag):
        wind.step(dt)
    v1 = wind.velocity
    assert torch.allclose(v1.mean(dim=0), steady, atol=0.1)
    assert abs(float(v1.std(dim=0).mean()) - std) < 0.05 * std
    g0, g1 = v0 - steady, v1 - steady
    corr = float((g0 * g1).mean() / (g0.std() * g1.std()))
    assert abs(corr - math.exp(-1.0)) < 0.05


def test_ou_std_holds_for_any_dt():
    """The exact discretization keeps the stationary std at gust_std for ANY dt."""
    for dt in (0.001, 0.01, 0.2):
        gen = torch.Generator().manual_seed(1)
        wind = WindField(WindCfg(gust_std=2.0, gust_tau=0.3), 8192, "cpu", gen)
        for _ in range(100):
            wind.step(dt)
        assert abs(float(wind.velocity.std()) - 2.0) < 0.1, f"dt={dt}"


def test_steady_only_is_constant():
    """gust_std=0 degenerates to a constant steady wind (and never draws RNG)."""
    wind = WindField(WindCfg(steady=(5.0, 0.0, 0.0)), 4, "cpu")
    v = wind.step(0.01).clone()
    assert torch.equal(v, wind.step(0.01))
    assert torch.allclose(v, torch.tensor([5.0, 0.0, 0.0]).expand(4, 3))


def test_reset_touches_only_given_envs():
    """reset(env_ids) redraws those envs' gust + DR steady; others keep theirs."""
    gen = torch.Generator().manual_seed(2)
    cfg = WindCfg(gust_std=1.0, steady_speed_range=(1.0, 3.0))
    wind = WindField(cfg, 4, "cpu", generator=gen)
    before = wind.velocity.clone()
    wind.reset(torch.tensor([1, 3]))
    after = wind.velocity
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    assert not torch.equal(after[[1, 3]], before[[1, 3]])


def test_steady_speed_range_resamples_horizontal_headings():
    """DR steady wind: per-env speed inside the range, level, headings spread."""
    gen = torch.Generator().manual_seed(3)
    wind = WindField(WindCfg(steady_speed_range=(2.0, 5.0)), 512, "cpu", gen)
    steady = wind.velocity  # gust_std=0 -> velocity IS the steady field
    speed = steady[:, :2].norm(dim=-1)
    assert float(speed.min()) >= 2.0 - 1e-5
    assert float(speed.max()) <= 5.0 + 1e-5
    assert torch.allclose(steady[:, 2], torch.zeros(512))
    heading = torch.atan2(steady[:, 1], steady[:, 0])
    assert float(heading.std()) > 1.0  # uniform-on-circle std ~ pi/sqrt(3) ~ 1.8


def test_per_env_gust_std_sweeps_conditions():
    """EskfConfig-style knob: an (N,) gust_std runs N gust climates in one batch."""
    gen = torch.Generator().manual_seed(4)
    sigma = torch.tensor([0.0, 0.5, 2.0])
    wind = WindField(WindCfg(gust_std=sigma, gust_tau=0.3), 3, "cpu", generator=gen)
    samples = torch.stack([wind.step(0.01).clone() for _ in range(10_000)])
    stds = samples.std(dim=0).mean(dim=-1)  # (3,) per-env time-std
    assert float(stds[0]) < 1e-9  # a zero-sigma env stays gust-free
    assert abs(float(stds[1]) - 0.5) < 0.15 * 0.5
    assert abs(float(stds[2]) - 2.0) < 0.15 * 2.0


def test_gust_std_range_resamples_per_episode():
    """gust_std_range: reset redraws each env's gust LEVEL inside the band."""
    gen = torch.Generator().manual_seed(5)
    cfg = WindCfg(gust_std_range=(0.0, 1.5))
    wind = WindField(cfg, 64, "cpu", generator=gen)
    first = wind._sigma.clone()
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.5
    assert float(first.std()) > 0.1  # levels actually vary across envs
    wind.reset(torch.arange(0, 8))
    assert not torch.equal(wind._sigma[:8], first[:8])  # redrawn
    assert torch.equal(wind._sigma[8:], first[8:])  # others keep their level
    # calm-air episodes exist: the running field stays finite and bounded
    for _ in range(50):
        wind.step(0.01)
    assert bool(torch.isfinite(wind.velocity).all())


def test_quadratic_drag_direction_and_magnitude():
    """F = 0.5*rho*CdA*|v|*v: zero at rest, downwind, quadratic in speed."""
    v = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    f = quadratic_drag(v, cda=0.1)
    want = 0.5 * 1.225 * 0.1 * 9.0
    assert torch.allclose(f[0], torch.tensor([want, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(f[1], torch.zeros(3))
    assert torch.allclose(quadratic_drag(2.0 * v, cda=0.1)[0], 4.0 * f[0], atol=1e-5)


def test_wind_cfg_validation():
    """A bad wind spec fails at construction, naming the field."""
    with pytest.raises(ValueError, match="gust_std"):
        WindCfg(gust_std=-0.1)
    with pytest.raises(ValueError, match="gust_tau"):
        WindCfg(gust_tau=0.0)
    with pytest.raises(ValueError, match="steady_speed_range"):
        WindCfg(steady_speed_range=(3.0, 1.0))


def test_state_payload_fields_are_optional():
    """State stays backward compatible: payload fields default to None."""
    base = {
        "position": torch.zeros(2, 3),
        "attitude": torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 4),
        "linear_velocity": torch.zeros(2, 3),
        "angular_velocity": torch.zeros(2, 3),
        "linear_acceleration": torch.zeros(2, 3),
    }
    assert State(**base).payload_position is None
    rigged = State(**base, payload_position=torch.zeros(2, 3))
    assert rigged.payload_position is not None and rigged.payload_velocity is None
