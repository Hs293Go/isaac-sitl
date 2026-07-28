"""Wind field: per-env steady wind + Ornstein-Uhlenbeck gusts, and quadratic drag.

An INPUT model, not a plant (CLAUDE.md "never ship an analytic simulation plant"): the
OU process integrates its own exogenous gust state -- never any vehicle ground truth --
and its output only shapes drag forces COMMANDED onto the articulation
(`Actuation.body_force` / `payload_force`); PhysX still integrates every body.

The gust is the exact discretization of the per-axis OU SDE
`dv = -v/tau dt + sigma*sqrt(2/tau) dW`:

    v[k+1] = a * v[k] + sigma * sqrt(1 - a^2) * randn,   a = exp(-dt / tau)

so the stationary std is `gust_std` for ANY dt (a first-order Dryden surrogate; full
MIL-HDBK-1797 Dryden shaping is deliberately out of scope). The steady vector carries
the mean; `reset(env_ids)` draws the stationary gust (no spin-up transient) and, when
`steady_speed_range` is set, resamples each env's steady speed + horizontal heading --
the wind domain-randomization knob for training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

_RHO_ISA = 1.225  # ISA sea-level air density [kg/m^3]


@dataclass
class WindCfg:
    """A per-env wind field spec (ENU, SI units).

    Every numeric field is a scalar shared by the batch OR a per-env tensor (the
    `EskfConfig`-style sweep knob) -- a condition sweep runs N wind climates in one
    batch. The env-cfg path (`LabVectorizedEnvCfg.wind`) uses plain scalars; tensors
    are for code constructing a `WindField` directly.

    Attributes:
        steady: mean ENU wind velocity [m/s] -- (3,) shared, or (N, 3) per-env
            (overridden per env when `steady_speed_range` is set).
        gust_std: OU gust stationary std per axis [m/s], scalar or (N,)
            (0 = steady-only).
        gust_tau: OU gust correlation time [s], scalar or (N,).
        steady_speed_range: optional (lo, hi) [m/s]; when set, `reset` resamples each
            env's steady wind as a random horizontal heading at a speed uniform in
            the range (the DR knob) instead of the `steady` vector.
        gust_std_range: optional (lo, hi) [m/s]; when set, `reset` resamples each
            env's gust std uniform in the range (per-episode gust DR -- so a policy
            also flies calm air), overriding `gust_std`.
    """

    steady: tuple[float, float, float] | torch.Tensor = (0.0, 0.0, 0.0)
    gust_std: float | torch.Tensor = 0.0
    gust_tau: float | torch.Tensor = 1.0
    steady_speed_range: tuple[float, float] | None = None
    gust_std_range: tuple[float, float] | None = None

    def __post_init__(self):
        """Validate the spec (fail at construction, not mid-rollout)."""
        if bool((torch.as_tensor(self.gust_std) < 0.0).any()):
            raise ValueError(f"WindCfg.gust_std must be >= 0, got {self.gust_std}")
        if bool((torch.as_tensor(self.gust_tau) <= 0.0).any()):
            raise ValueError(f"WindCfg.gust_tau must be > 0, got {self.gust_tau}")
        for name in ("steady_speed_range", "gust_std_range"):
            band = getattr(self, name)
            if band is not None:
                lo, hi = band
                if lo < 0.0 or hi < lo:
                    raise ValueError(
                        f"WindCfg.{name} needs 0 <= lo <= hi, got {lo, hi}"
                    )


class WindField:
    """Batched ENU wind velocity: per-env steady vector + OU gust state."""

    def __init__(
        self,
        cfg: WindCfg,
        num_envs: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ):
        """Allocate the (num_envs, 3) steady/gust buffers and draw the initial state.

        Args:
            cfg: the wind spec.
            num_envs: leading batch dim.
            device: torch device for the buffers.
            generator: optional RNG (must live on `device`) for reproducible wind.
        """
        self._cfg = cfg
        self._gen = generator

        def as_t(v) -> torch.Tensor:
            return torch.as_tensor(v, dtype=torch.float32, device=device)

        self._steady = as_t(cfg.steady).broadcast_to(num_envs, 3).clone()
        # sigma/tau kept (N, 1) so scalar and per-env specs share one code path.
        self._sigma = as_t(cfg.gust_std).broadcast_to(num_envs).clone()[:, None]
        self._tau = as_t(cfg.gust_tau).broadcast_to(num_envs).clone()[:, None]
        self._has_gust = bool((self._sigma > 0.0).any()) or (
            cfg.gust_std_range is not None and cfg.gust_std_range[1] > 0.0
        )
        self._gust = torch.zeros(num_envs, 3, device=device)
        self.reset()

    @property
    def velocity(self) -> torch.Tensor:
        """The current (num_envs, 3) ENU wind velocity [m/s] (no time advance)."""
        return self._steady + self._gust

    def step(self, dt: float) -> torch.Tensor:
        """Advance the gust by `dt` [s] (exact OU update); returns `velocity`."""
        if self._has_gust:
            a = torch.exp(-dt / self._tau)  # (N, 1) decay per env
            noise = torch.randn(
                self._gust.shape, generator=self._gen, device=self._gust.device
            )
            self._gust = a * self._gust + self._sigma * (1.0 - a * a).sqrt() * noise
        return self.velocity

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Redraw the given envs' stationary gust (and steady wind under DR)."""
        ids = slice(None) if env_ids is None else env_ids
        n = self._gust[ids].shape[0]
        dev = self._gust.device
        cfg = self._cfg
        if cfg.gust_std_range is not None:  # per-episode gust-level DR
            lo, hi = cfg.gust_std_range
            self._sigma[ids] = lo + (hi - lo) * torch.rand(
                (n, 1), generator=self._gen, device=dev
            )
        if self._has_gust:  # stationary draw: the gust starts converged
            self._gust[ids] = self._sigma[ids] * torch.randn(
                (n, 3), generator=self._gen, device=dev
            )
        else:
            self._gust[ids] = 0.0
        if cfg.steady_speed_range is not None:  # DR: random speed + heading, level
            lo, hi = cfg.steady_speed_range
            speed = lo + (hi - lo) * torch.rand(n, generator=self._gen, device=dev)
            heading = 2.0 * math.pi * torch.rand(n, generator=self._gen, device=dev)
            self._steady[ids, 0] = speed * torch.cos(heading)
            self._steady[ids, 1] = speed * torch.sin(heading)
            self._steady[ids, 2] = 0.0


def quadratic_drag(
    v_rel: torch.Tensor, cda: float, rho: float = _RHO_ISA
) -> torch.Tensor:
    """Bluff-body drag force [N] along the (..., 3) relative wind.

    `F = 0.5 * rho * Cd*A * |v_rel| * v_rel` with `v_rel = v_wind - v_body`, so the
    force pushes the body downwind (and, in still air, opposes its motion). `rho`
    defaults to the ISA sea-level air density 1.225 kg/m^3.
    """
    return 0.5 * rho * cda * v_rel.norm(dim=-1, keepdim=True) * v_rel
