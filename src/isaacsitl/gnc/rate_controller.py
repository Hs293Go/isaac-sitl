"""RateController: the CTBR inner loop, a Wrench-layer law over an `Allocator`."""

from __future__ import annotations

import torch

from isaacsitl.actuation import Actuation
from isaacsitl.gnc.controller import (
    Allocator,
    InertialProperties,
    Wrench,
)


class RateController:
    """Body-rate P loop -> torque via inertia -> allocation (the CTBR inner loop).

    The inner two-thirds of the PID cascade, standalone: track a body-rate setpoint
    with an airframe-independent frequency gain (angular accel -> torque through the
    live body inertia + the gyroscopic term), then hand the `Wrench` to the injected
    `Allocator`. This is the sim twin of an autopilot's rate controller (PX4 acro /
    offboard body-rate), so a CTBR RL policy trains against the loop it will command
    on hardware.

    Deliberately NOT a `QuadrotorController`: its primary input (collective thrust)
    is not expressible in a tracking setpoint -- in CTBR mode the POLICY is the outer
    law, and this class is the `Wrench`-layer stage under it. Stateless (no
    integrator), so there is no `reset`.
    """

    def __init__(
        self,
        alloc: Allocator,
        kw: tuple[float, float, float] = (16.0, 16.0, 8.0),
    ):
        """Build the rate loop over an allocation.

        Args:
            alloc: wrench -> per-rotor allocation (the airframe geometry's).
            kw: body-rate gains (3,) [1/s] -- the closed-loop rate bandwidth,
                airframe-independent (the PID cascade's inner-loop gains).
        """
        self._alloc = alloc
        self.kw = torch.as_tensor(kw, dtype=torch.float32)

    def to(self, device: torch.device | str) -> RateController:
        """Move the gains and the allocation to `device`; returns self."""
        self.kw = self.kw.to(device)
        self._alloc.to(device)
        return self

    def actuation(
        self,
        angular_velocity: torch.Tensor,
        thrust: torch.Tensor,
        body_rate_sp: torch.Tensor,
        inertial: InertialProperties,
    ) -> Actuation:
        """One rate-loop step: rate error -> torque -> allocated `Actuation`.

        Args:
            angular_velocity: (N, 3) body rates [rad/s] (FLU).
            thrust: (N,) collective thrust command [N] (body +z).
            body_rate_sp: (N, 3) body-rate setpoint [rad/s].
            inertial: live inertia (batch-aware) for the accel -> torque map.

        Returns:
            Per-rotor-thrust `Actuation` (yaw torque as the body couple).
        """
        ang_acc = self.kw * (body_rate_sp - angular_velocity)
        jw = inertial.inertia * angular_velocity
        torque = inertial.inertia * ang_acc + torch.cross(angular_velocity, jw, dim=-1)
        return self._alloc.allocate(Wrench(thrust=thrust, torque=torque))
