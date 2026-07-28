"""Deterministic plant disturbances: wind/drag + wrench bias, with onset control.

The L1-validation experiment shape: a MATCHED (force/torque channel), PERSISTENT,
DETERMINISTIC disturbance the plant feels independently of the commanded
actuation. `DisturbanceCfg` is the recordable spec — `provenance()` goes
verbatim into a run report or fixture header, so a disturbed run is
reconstructable from its header alone. `Disturbance` is the runtime that turns
it into a per-step external wrench for
`ArticulationVehicle.set_external_wrench` (applied automatically by `DroneSim`
when a disturbance is set).

Determinism contract: same cfg (incl. `seed`) + same step sequence ⇒ identical
wrench sequence, run to run — the paired-A/B requirement. The wind field
advances every step regardless of the onset gate (the gate scales the OUTPUT),
so onset timing never perturbs the underlying gust sequence.

Frames: `force_enu` is ENU WORLD [N] (rotated into body FLU at the plant seam,
with the attitude current at apply time); `torque_flu` is body FLU [N·m]; drag
coefficients are body-FLU, applied against the AIR-relative velocity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from isaacsitl.quadrotor import BodyDrag
from isaacsitl.so3 import quat_conjugate, quat_rotate
from isaacsitl.wind import WindCfg, WindField

if TYPE_CHECKING:
    from isaacsitl.state import State


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


@dataclass
class DisturbanceCfg:
    """A recordable disturbance spec (all components off by default).

    Attributes:
        wind: optional ENU wind field (steady + seeded OU gusts). Wind acts on
            the body only through `drag_linear`, so setting it without drag is a
            spec error (a silent no-op would fake a disturbed run).
        drag_linear: body-FLU linear drag coefficients [N/(m/s)] applied against
            the AIR-relative velocity (still-air drag when `wind` is None).
        force_enu: constant ENU WORLD force bias [N] — e.g. ``(0, 0, -dm * 9.81)``
            is the payload-pickup surrogate (exact in the gravity channel; the
            inertial dm·a term is the documented second-order approximation).
        torque_flu: constant body-FLU torque bias [N·m] (matched to the
            attitude/rate loop).
        t_on: onset time [s]; the whole disturbance is gated 0 before it.
        ramp: linear 0→1 ramp duration [s] after `t_on` (0 = step onset).
        seed: RNG seed for the gust sequence (steady-only wind draws no RNG).
    """

    wind: WindCfg | None = None
    drag_linear: tuple[float, float, float] | None = None
    force_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_flu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    t_on: float = 0.0
    ramp: float = 0.0
    seed: int = 0

    def __post_init__(self):
        """Validate the spec (fail at construction, not mid-rollout)."""
        if self.t_on < 0.0:
            raise ValueError(f"DisturbanceCfg.t_on must be >= 0, got {self.t_on}")
        if self.ramp < 0.0:
            raise ValueError(f"DisturbanceCfg.ramp must be >= 0, got {self.ramp}")
        if self.drag_linear is not None and any(c < 0.0 for c in self.drag_linear):
            raise ValueError(
                f"DisturbanceCfg.drag_linear must be >= 0, got {self.drag_linear}"
            )
        if self.wind is not None and self.drag_linear is None:
            raise ValueError(
                "DisturbanceCfg.wind needs drag_linear — wind acts on the body "
                "only through drag, and a silent no-op would fake a disturbed run"
            )

    def active(self) -> bool:
        """Whether any component is set (False ⇒ the clean path, exactly)."""
        return (
            self.wind is not None
            or self.drag_linear is not None
            or any(self.force_enu)
            or any(self.torque_flu)
        )

    def provenance(self) -> dict:
        """The exact parameters, JSON-able — write this into the run header."""
        return _jsonable(asdict(self))


class Disturbance:
    """Runtime: turns a `DisturbanceCfg` into a per-step external wrench."""

    def __init__(
        self,
        cfg: DisturbanceCfg,
        num_envs: int,
        device: torch.device | str,
        dt: float,
    ):
        """Allocate the wind field (seeded on `device`) and the bias tensors."""
        self.cfg = cfg
        self._dt = dt
        self._wind = None
        if cfg.wind is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.seed)
            self._wind = WindField(cfg.wind, num_envs, device, generator=gen)
        self._drag = (
            BodyDrag(linear=cfg.drag_linear).to(device)
            if cfg.drag_linear is not None
            else None
        )
        as_row = lambda v: torch.as_tensor(  # noqa: E731
            v, dtype=torch.float32, device=device
        ).broadcast_to(num_envs, 3)
        self._force = as_row(cfg.force_enu)
        self._torque = as_row(cfg.torque_flu)

    def gain(self, t: float) -> float:
        """The onset envelope: 0 before `t_on`, linear ramp to 1, then 1."""
        if t < self.cfg.t_on:
            return 0.0
        if self.cfg.ramp <= 0.0:
            return 1.0
        return min(1.0, (t - self.cfg.t_on) / self.cfg.ramp)

    def wrench(self, t: float, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """One step's external wrench: ((N, 3) force ENU, (N, 3) torque FLU).

        Advances the wind field by one `dt` — call exactly once per physics step.
        The onset gate scales the output only, so the gust sequence under a given
        seed is identical whatever the onset.
        """
        wind_vel = self._wind.step(self._dt) if self._wind is not None else None
        g = self.gain(t)
        force = g * self._force
        torque = g * self._torque
        if self._drag is not None and g > 0.0:
            # A fresh tensor either way -- never mutate the caller's State.
            if wind_vel is None:
                v_air = state.linear_velocity.clone()
            else:
                v_air = state.linear_velocity - wind_vel
            q = state.attitude
            f_body = self._drag.force(quat_rotate(quat_conjugate(q), v_air))
            force += g * quat_rotate(q, f_body)  # force is fresh (g * bias) above
        return force, torque
