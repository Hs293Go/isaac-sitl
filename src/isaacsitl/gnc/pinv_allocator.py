"""Pseudoinverse-based control allocation for quadrotors."""

from __future__ import annotations

import torch

from isaacsitl.actuation import Actuation
from isaacsitl.gnc.controller import Allocator, Wrench
from isaacsitl.quadrotor import mixer_allocation


class PinvAllocator(Allocator):
    """Wrench -> per-rotor thrust via the pseudoinverse of the geometry mixer.

    Allocates `[Fz, tau_x, tau_y]` to per-rotor +z thrusts (force-at-links: PhysX
    forms r x F, so the map is `mixer_allocation`'s pinv -- the ONE builder the
    airframe geometry defines); yaw stays a pure body couple (a +z rotor force makes
    no yaw). Per-rotor bounds clamp the allocation: pass the motor's ceiling as
    `thrust_max` where known -- saturation then happens HERE, visibly, instead of
    downstream in the motor's command clamp.
    """

    def __init__(
        self,
        rotor_x: torch.Tensor | tuple[float, ...],
        rotor_y: torch.Tensor | tuple[float, ...],
        thrust_min: float = 0.0,
        thrust_max: float = torch.inf,
    ):
        """Build the allocation from the rotor geometry (+ per-rotor thrust bounds).

        Args:
            rotor_x: (n_rotors,) body-FLU x rotor offset [m].
            rotor_y: (n_rotors,) body-FLU y rotor offset [m].
            thrust_min: per-rotor floor [N] (rotors cannot pull).
            thrust_max: per-rotor ceiling [N] (the motor's max_thrust where known).
        """
        self._inv_mix3 = mixer_allocation(rotor_x, rotor_y)  # (n_rotors, 3)
        self._thrust_min = float(thrust_min)
        self._thrust_max = float(thrust_max)

    def to(self, device: torch.device | str) -> PinvAllocator:
        """Move the allocation map to `device`; returns self."""
        self._inv_mix3 = self._inv_mix3.to(device)
        return self

    def allocate(self, wrench: Wrench) -> Actuation:
        """Allocate a wrench (collective + body torque) to per-rotor thrusts.

        Args:
            wrench: `Wrench` with (N,) collective thrust and (N, 3) body torque.

        Returns:
            Per-rotor-thrust `Actuation`; the yaw torque rides as the body couple.
        """
        thrust, torque = wrench.thrust, wrench.torque
        w = torch.stack([thrust, torque[..., 0], torque[..., 1]], dim=-1)
        rotor_thrust = (w @ self._inv_mix3.T).clamp(self._thrust_min, self._thrust_max)
        body_torque = torch.zeros_like(torque)
        body_torque[..., 2] = torque[..., 2]
        return Actuation(rotor_thrust=rotor_thrust, body_torque=body_torque)
