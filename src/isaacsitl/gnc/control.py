"""Geometric position/attitude control: a pure `PIDController` + a `Backend` adapter.

Two pieces, split along the framework boundary:

  - `PIDController` -- the MATH. A torch-native, batch-native two-loop geometric
    controller (Lee et al. SE(3)) built on `so3`. `compute(state, setpoint, inertial)`
    is a pure function of its inputs (plus the position-error integral it carries):
    the live airframe facts arrive as `InertialProperties` (mass/inertia, from the
    articulation) and the geometry lives in the `Allocator` injected at construction
    -- so one controller is airframe-agnostic.
  - `ClassicalControllerBackend` -- the `Backend` ADAPTER. Holds the tracking setpoint
    and the inertial facts, keeps the setpoint API (`set_reference`/`nudge_*`) and the
    Backend lifecycle (`update_state`/`actuation`/`reset`), and delegates the math to
    the injected `QuadrotorController`.

    Loop 1 (position/velocity + integral)  ->  desired thrust [N] + desired attitude
    Loop 2 (SO(3) attitude error)  ->  desired body angular accel -> torque via inertia
    Allocation ([Fz, tau_x, tau_y] -> per-rotor) + yaw couple -> Actuation

The attitude loop is INERTIA-AWARE: it commands an angular acceleration and converts
it to a torque through the body inertia (plus the gyroscopic term), so the gains are
frequencies (1/s^2, 1/s) and airframe-independent -- the same gains hold a 0.5 kg
racer and a heavier craft (raw torque gains do not).

Frames (see `state.py`): position/velocity ENU world; attitude FLU-body <- ENU-world;
angular velocity FLU body. Thrust acts along body +Z (FLU up); torque is body-frame.
The non-RL counterpart to an RL `Backend`: same data contract, no blocking round-trip,
batch-native so one controller can hold N vehicles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaacsitl.actuation import Actuation
from isaacsitl.gnc.controller import (
    InertialProperties,
    QuadrotorController,
    QuadrotorControlSetpoint,
)

if TYPE_CHECKING:
    from isaacsitl.state import State


class ClassicalControllerBackend:
    """`Backend` adapter over a pure `QuadrotorController`: setpoint + live facts."""

    def __init__(
        self,
        ctrl: QuadrotorController,
        inertial: InertialProperties,
        position: tuple[float, float, float] = (0.0, 0.0, 1.5),
        yaw: float = 0.0,
    ):
        """Construct the adapter.

        Args:
            ctrl: the `QuadrotorController` (e.g. `PIDController`); it owns its
                `Allocator`, so the geometry never enters here.
            inertial: live mass / inertia (batch-aware, from the articulation).
            position: initial ENU position setpoint [m].
            yaw: initial yaw setpoint [rad].
        """
        # Control POLICY -> the pure controller; airframe FACTS (live mass/inertia)
        # arrive per-compute so one controller is airframe-agnostic.
        self._ctrl = ctrl
        self._inertial = inertial
        self._sp_pos = torch.tensor(position, dtype=torch.float32)
        self._sp_vel = torch.zeros(3)  # velocity feed-forward (0 = hold; set to track)
        self._sp_acc = torch.zeros(3)  # acceleration feed-forward (0 = hold)
        self._sp_body_rate = torch.zeros(3)  # body-rate feed-forward (0 = hold)
        self._sp_yaw = float(yaw)
        self._state: State | None = None

    def to(self, device: torch.device | str) -> ClassicalControllerBackend:
        """Move the controller, airframe facts, and setpoint to `device`."""
        self._ctrl.to(device)
        self._inertial.to(device)
        self._sp_pos = self._sp_pos.to(device)
        self._sp_vel = self._sp_vel.to(device)
        self._sp_acc = self._sp_acc.to(device)
        self._sp_body_rate = self._sp_body_rate.to(device)
        return self

    # ------------------------------------------------------------------ Backend hooks
    def update_state(self, state: State) -> None:
        """Latch the State for the next `actuation()`."""
        self._state = state

    def actuation(self) -> Actuation:
        """Assemble the setpoint and delegate one step to the controller."""
        s = self._state
        if s is None:
            raise RuntimeError("update_state() must be called before actuation()")
        sp = QuadrotorControlSetpoint(
            position=self._sp_pos,
            velocity=self._sp_vel,
            acceleration=self._sp_acc,
            body_rate=self._sp_body_rate,
            yaw=self._sp_yaw,
        )
        return self._ctrl.compute(s, sp, self._inertial)

    def reset(self, env_ids: Any = None) -> None:
        """Clear the position-error integrator for the given envs."""
        self._ctrl.reset(env_ids)

    # ------------------------------------------------------------------ setpoint API
    def set_reference(
        self,
        position: torch.Tensor | tuple[float, float, float],
        velocity: torch.Tensor | tuple[float, float, float] | None = None,
        acceleration: torch.Tensor | tuple[float, float, float] | None = None,
        body_rate: torch.Tensor | tuple[float, float, float] | None = None,
        yaw: torch.Tensor | float | None = None,
    ) -> None:
        """Set the full tracking reference: position + feed-forward terms + yaw.

        The per-step trajectory feed (what `classical_nav`/`classical_sweep` call each
        control step). Each vector is (3,) shared or (num_envs, 3) per-env. An omitted
        `velocity`/`acceleration` resets that feed-forward to zero (hold semantics --
        no stale feed-forward outlives its trajectory); an omitted `yaw` keeps the
        current yaw setpoint.
        """
        dev = self._sp_pos.device
        self._sp_pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
        self._sp_vel = (
            torch.zeros(3, device=dev)
            if velocity is None
            else torch.as_tensor(velocity, dtype=torch.float32, device=dev)
        )
        self._sp_acc = (
            torch.zeros(3, device=dev)
            if acceleration is None
            else torch.as_tensor(acceleration, dtype=torch.float32, device=dev)
        )
        self._sp_body_rate = (
            torch.zeros(3, device=dev)
            if body_rate is None
            else torch.as_tensor(body_rate, dtype=torch.float32, device=dev)
        )
        if yaw is not None:
            self._sp_yaw = float(yaw)

    def set_position(self, position: tuple[float, float, float]) -> None:
        """Set the ENU position setpoint [m]."""
        self._sp_pos = torch.tensor(
            position, dtype=torch.float32, device=self._sp_pos.device
        )

    def nudge_position(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Shift the ENU position setpoint by (dx, dy, dz) metres."""
        self._sp_pos += torch.tensor(
            [dx, dy, dz], dtype=torch.float32, device=self._sp_pos.device
        )

    def nudge_yaw(self, dyaw: float) -> None:
        """Shift the yaw setpoint by dyaw radians."""
        self._sp_yaw += float(dyaw)
