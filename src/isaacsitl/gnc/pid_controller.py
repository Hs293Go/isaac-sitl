"""Pure math geometric PID controller for quadrotors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from isaacsitl import so3
from isaacsitl.actuation import Actuation
from isaacsitl.gnc.control_utils import (
    limit_thrust_by_tilt,
    thrust_and_yaw_to_desired_attitude,
)
from isaacsitl.gnc.controller import (
    Allocator,
    InertialProperties,
    QuadrotorController,
    QuadrotorControlSetpoint,
    Wrench,
)
from isaacsitl.state import State


@dataclass
class PIDControllerCfg:
    """Control POLICY for the geometric PID controller (gains + loop constants).

    The tunables and the world/actuation constants the loops need -- NOT airframe
    facts. Mass, inertia, and the rotor allocation are the craft's (live from the
    articulation / its geometry) and arrive at `PIDController.compute` as
    `InertialProperties` + `allocation`, never baked in here.
    """

    # Gains: (3,) shared across the batch, or an (N, 3) tensor per-env (a gain sweep).
    kp: tuple[float, float, float] | torch.Tensor = (6.0, 6.0, 8.0)  # position, ENU
    kv: tuple[float, float, float] | torch.Tensor = (4.0, 4.0, 8.0)  # velocity, ENU
    ki: tuple[float, float, float] | torch.Tensor = (0.8, 0.8, 1.0)  # integral, ENU
    kr: tuple[float, float, float] | torch.Tensor = (120.0, 120.0, 40.0)  # att [1/s^2]
    kw: tuple[float, float, float] | torch.Tensor = (16.0, 16.0, 8.0)  # rate [1/s]
    i_limit: float = 1.0  # symmetric per-axis integrator clamp [m.s] (anti-windup)
    i_band: float = 0.3  # integrate the position error only within this radius [m]
    dt: float = 0.01  # control period [s] (physics step), integrates the position error
    gravity: float = 9.81  # gravitational acceleration [m/s^2]
    tilt_max: float = 0.52  # max commanded tilt [rad] (~30 deg) when the cap is on
    tilt_limit: bool = True  # cap the horizontal force so tilt stays <= tilt_max


class PIDController(QuadrotorController):
    """Pure two-loop geometric position/attitude controller; no Backend, no I/O."""

    def __init__(self, alloc: Allocator, cfg: PIDControllerCfg | None = None):
        """Build from a `PIDControllerCfg`; the integrator is sized lazily."""
        cfg = PIDControllerCfg() if cfg is None else cfg
        # Gains: (3,) shared across the batch, or (N, 3) per-env (a gain sweep). The
        # loop math (`self.kp * e_pos` etc.) broadcasts against the (N, 3) errors.
        self.kp = torch.as_tensor(cfg.kp, dtype=torch.float32)
        self.kv = torch.as_tensor(cfg.kv, dtype=torch.float32)
        self.ki = torch.as_tensor(cfg.ki, dtype=torch.float32)
        self.kr = torch.as_tensor(cfg.kr, dtype=torch.float32)
        self.kw = torch.as_tensor(cfg.kw, dtype=torch.float32)
        self.i_limit = float(cfg.i_limit)
        self.i_band = float(cfg.i_band)
        self.dt = float(cfg.dt)
        self.gravity = float(cfg.gravity)
        self._tilt_max = float(cfg.tilt_max)
        self._tilt_limit = bool(cfg.tilt_limit)
        self._i: torch.Tensor | None = None  # position-error integral, lazily sized
        self._alloc = alloc  # wrench -> per-rotor thrust allocation (geometry mixer)

    def to(self, device: torch.device | str) -> PIDController:
        """Move the gains, the allocator, and any integrator to `device`."""
        for name in ("kp", "kv", "ki", "kr", "kw"):
            setattr(self, name, getattr(self, name).to(device))
        self._alloc.to(device)
        if self._i is not None:
            self._i = self._i.to(device)
        return self

    def reset(self, env_ids: Any = None) -> None:
        """Clear the position-error integrator for the given envs (all if None)."""
        if self._i is None:
            return
        if env_ids is None:
            self._i.zero_()
        else:
            self._i[env_ids] = 0.0

    def compute(
        self,
        state: State,
        setpoint: QuadrotorControlSetpoint,
        inertial: InertialProperties,
    ) -> Actuation:
        """Run one geometric control step; return the per-rotor-thrust `Actuation`.

        Args:
            state: current batched `State` (ENU world pos/vel, FLU attitude/body-rate).
            setpoint: the tracking reference + feed-forward terms.
            inertial: live mass / inertia (batch-aware); the gravity feed-forward and
                the angular-accel -> torque map.
        """
        s = state
        sp = setpoint
        # Lazily size the integrator to (num_envs, 3) on first use / a batch change.
        if self._i is None or self._i.shape[0] != s.position.shape[0]:
            self._i = torch.zeros_like(s.position)
        rot = so3.quat_to_matrix(s.attitude)  # body (FLU) -> world (ENU); cols = axes

        # ---- Loop 1: position/velocity (+ integral) -> thrust + desired attitude ----
        e_pos = sp.position - s.position
        e_vel = sp.velocity - s.linear_velocity  # velocity feed-forward (0 = hold)
        # Anti-windup: integrate the position error only WITHIN `i_band` of the setpoint
        # (zero farther out). Unconditional integration saturated each transit -> the
        # error grew leg-over-leg and the horizontal loop diverged.
        near = (e_pos.norm(dim=-1, keepdim=True) < self.i_band).to(e_pos.dtype)
        self._i = near * (self._i + e_pos * self.dt).clamp(-self.i_limit, self.i_limit)
        f_des = self.kp * e_pos + self.kv * e_vel + self.ki * self._i + sp.acceleration
        f_des[..., 2] += self.gravity  # gravity feed-forward (ENU up)
        force_des = inertial.mass[..., None] * f_des  # (n, 3)
        if self._tilt_limit:
            force_des = limit_thrust_by_tilt(force_des, self._tilt_max)

        b3 = rot[..., 2]  # body-up in world
        thrust = (force_des * b3).sum(dim=-1)  # project onto body-up

        rot_des = thrust_and_yaw_to_desired_attitude(force_des, sp.yaw)

        # ---- Loop 2: SO(3) attitude error -> angular accel -> torque via inertia ----
        err = rot_des.mT @ rot - rot.mT @ rot_des
        e_rot = 0.5 * so3.vee(err)
        e_omega = s.angular_velocity - sp.body_rate
        ang_acc = -self.kr * e_rot - self.kw * e_omega
        jw = inertial.inertia * s.angular_velocity
        torque = inertial.inertia * ang_acc + torch.cross(
            s.angular_velocity, jw, dim=-1
        )

        return self._alloc.allocate(Wrench(thrust=thrust, torque=torque))
