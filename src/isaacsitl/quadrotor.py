"""Quadrotor: the multirotor airframe -- rotor geometry + a pluggable `Motor`.

ONE airframe for both plants; the only difference is the `Motor` it owns
(`InstantMotor` -> the linear curve; `LaggedMotor` -> first-order motor lag + the
OQCRL `i_rotor*dW` yaw reaction). `thrusts` runs the motor and returns per-rotor
thrust + the yaw couple; the vehicle applies each thrust as a force at the rotor
link (PhysX forms r x F roll/pitch) and the couple on the body -- force-at-links.

The airframe owns its geometry directly and is NOT a mixer: the rotor positions
are realized by the USD rotor links, and `thrusts` needs only yaw-signs + kappa.
Control allocation (a desired wrench -> per-rotor commands) lives in the wrench-based
controller (`ClassicalControllerBackend`), built from this geometry. Mass / inertia are
the Articulation's; `hover_thrust` / `thrust_to_weight` derive hover + TWR from a mass.
"""

from __future__ import annotations

import torch

from isaacsitl.motor import Motor, MotorDRConfig

# Airframe geometry is DATA now (conf/airframe/*.yaml -> isaacsitl.airframe).
# This class stays a plain geometry + motor holder; specs build it via build_quadrotor.


class BodyDrag:
    """Diagonal body-frame drag: `F = -(c_lin + c_quad * |v|) * v` (FLU, per-axis).

    The standard rotor-drag / parasite-drag shape (the piece HANDOFF flagged as the
    residual vs IDR's effectiveness model). A FORCE model, not a plant: it only shapes
    the wrench COMMANDED onto the articulation (`Actuation.body_force`); PhysX still
    integrates the ground-truth state.
    """

    def __init__(
        self,
        linear: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        quadratic: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Per-axis body-FLU coefficients: linear [N/(m/s)], quadratic [N/(m/s)^2]."""
        self._c1 = torch.as_tensor(linear, dtype=torch.float32)
        self._c2 = torch.as_tensor(quadratic, dtype=torch.float32)

    def force(self, vel_body: torch.Tensor) -> torch.Tensor:
        """Body-FLU drag force [N] opposing the (..., 3) body-frame velocity."""
        return -(self._c1 + self._c2 * vel_body.abs()) * vel_body

    def to(self, device: torch.device | str) -> BodyDrag:
        """Move the coefficients to `device`; returns self."""
        self._c1 = self._c1.to(device)
        self._c2 = self._c2.to(device)
        return self


def mixer_allocation(
    rotor_x: torch.Tensor | tuple[float, ...],
    rotor_y: torch.Tensor | tuple[float, ...],
) -> torch.Tensor:
    """Rotor geometry -> control allocation (the pinv of the force-at-links mixer).

    Builds the mixer `[1, ry, -rx]` (each rotor's +z force gives collective + r x F
    roll/pitch) and returns its pseudo-inverse: the (n_rotors, 3) map
    `[Fz, tau_x, tau_y] -> per-rotor thrust`. The ONE builder -- `PinvAllocator`
    wraps it as the allocation object, so the geometry math lives in a single place.
    Yaw is a separate body couple, so no kappa enters here.
    """
    rx = torch.as_tensor(rotor_x, dtype=torch.float32)
    ry = torch.as_tensor(rotor_y, dtype=torch.float32)
    mix3 = torch.stack([torch.ones_like(rx), ry, -rx])  # (3, n_rotors)
    return torch.linalg.pinv(mix3)  # (n_rotors, 3)


class Quadrotor:
    """Multirotor airframe: rotor geometry + a pluggable `Motor` (force-at-links)."""

    def __init__(
        self,
        rotor_x: torch.Tensor | tuple[float, ...],
        rotor_y: torch.Tensor | tuple[float, ...],
        yaw_signs: torch.Tensor | tuple[float, ...],
        kappa: float,
        motor: Motor,
        drag: BodyDrag | None = None,
    ):
        """Define the rotor geometry, the motor model, and (optionally) body drag.

        Args:
            rotor_x: (n_rotors,) body-FLU x rotor offset [m]. The layout; the force is
                applied at the USD rotor links, not via this position.
            rotor_y: (n_rotors,) body-FLU y rotor offset [m].
            yaw_signs: (n_rotors,) per-rotor yaw-reaction sign in FRD-z (+1 / -1).
            kappa: drag-to-thrust ratio cq/ct (steady yaw reaction per unit thrust).
            motor: the command->thrust model (`InstantMotor` linear, `LaggedMotor`
                classical) -- the only thing that differs between the two plants.
            drag: optional `BodyDrag`; the runner applies its force at the base COM
                (`Actuation.body_force`) alongside the rotor thrusts. None = no drag.
        """
        self._rx = torch.as_tensor(rotor_x, dtype=torch.float32)
        self._ry = torch.as_tensor(rotor_y, dtype=torch.float32)
        self._yaw = torch.as_tensor(yaw_signs, dtype=torch.float32)
        self._kappa = float(kappa)
        self._motor = motor
        self._drag = drag
        self._n_rotors = self._rx.shape[0]

    def thrusts(
        self, action: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Command [-1,1] (..., n) -> (per-rotor thrust [N], yaw couple [N*m]).

        The force-at-links primitive: the vehicle applies each thrust as +z at its
        rotor link (PhysX forms r x F roll/pitch), and the yaw couple
        `-Sum yaw_i*(kappa*T_i + i_rotor*dW_i)` as a body torque (no +z makes yaw).
        No manual mixing -- replicates IDR's `_apply_classical`.
        """
        thrust, reaction = self._motor.step(action, dt)
        react = reaction if reaction is not None else 0.0
        yaw = -(self._yaw * (self._kappa * thrust + react)).sum(dim=-1)  # FLU body-z
        return thrust, yaw

    def command_for(self, thrust: torch.Tensor) -> torch.Tensor:
        """Per-rotor thrust [N] -> motor command [-1, 1] (the motor's inverse curve).

        The CTBR path: an inner rate loop allocates thrusts, this maps them onto the
        command space, and `thrusts` remains the single authority for what results.
        """
        return self._motor.command_for(thrust)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the motor state (a no-op for the stateless `InstantMotor`)."""
        self._motor.reset(env_ids)

    def randomize(
        self,
        env_ids: torch.Tensor | None,
        cfg: MotorDRConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        """Resample the motor's per-env intrinsics (thrust / tau bands) at reset."""
        self._motor.randomize(env_ids, cfg, generator)

    @property
    def intrinsics(self) -> torch.Tensor | None:
        """The motor's (num_envs, k) DR scales, for a privileged/intrinsics obs."""
        return self._motor.intrinsics

    @property
    def motor_state(self) -> torch.Tensor | None:
        """The motor's per-rotor state for the obs (None for a stateless motor)."""
        return self._motor.state

    @property
    def has_drag(self) -> bool:
        """True when the airframe carries a `BodyDrag` model."""
        return self._drag is not None

    def drag_force(self, vel_body: torch.Tensor) -> torch.Tensor | None:
        """Body-FLU drag force for the (..., 3) body velocity; None without drag."""
        return None if self._drag is None else self._drag.force(vel_body)

    @property
    def rotor_offsets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The (rotor_x, rotor_y) positions [m] -- the canonical rotor order."""
        return self._rx, self._ry

    def to(self, device: torch.device | str) -> Quadrotor:
        """Move the geometry and the motor state to `device`; returns self."""
        self._rx = self._rx.to(device)
        self._ry = self._ry.to(device)
        self._yaw = self._yaw.to(device)
        self._motor.to(device)
        if self._drag is not None:
            self._drag.to(device)
        return self

    def hover_thrust(self, mass: torch.Tensor | float, gravity: float = 9.81):
        """Per-rotor thrust to hover at `mass` [N] (derived from the live mass)."""
        return mass * gravity / self._n_rotors

    def thrust_to_weight(self, mass: torch.Tensor | float, gravity: float = 9.81):
        """Max thrust-to-weight ratio for `mass` (n_rotors * max_thrust / m*g)."""
        return self._n_rotors * self._motor.max_thrust / (mass * gravity)
