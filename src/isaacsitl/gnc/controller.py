"""Abstractions for classical control: setpoints, references, and controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import torch

if TYPE_CHECKING:
    from isaacsitl.actuation import Actuation
    from isaacsitl.state import State


@dataclass
class QuadrotorControlSetpoint:
    """Tracking reference: position + feed-forward (vel / acc / body-rate) + yaw.

    Each vector is (3,) shared or (num_envs, 3) per-env; an unset feed-forward defaults
    to zero (hold semantics). `yaw` is the ENU heading setpoint [rad].
    """

    position: torch.Tensor
    velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    acceleration: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    body_rate: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    yaw: float = 0.0


@dataclass
class InertialProperties:
    """Live, batch-aware rigid-body properties read from the articulation.

    Mass / inertia are the Articulation's (read live from the USD), never baked into
    the airframe spec -- so a slung-payload delta or a per-env inertia
    domain-randomization just changes these tensors. Batch shapes: `mass` () or (N,);
    `inertia` (3,) or (N, 3), principal [Ixx, Iyy, Izz] kg.m^2.
    """

    mass: torch.Tensor
    inertia: torch.Tensor

    def to(self, device: torch.device | str) -> InertialProperties:
        """Move mass / inertia to `device`; returns self."""
        self.mass = self.mass.to(device)
        self.inertia = self.inertia.to(device)
        return self


@dataclass
class Wrench:
    """Pre-allocation control wrench."""

    thrust: torch.Tensor  # (N,) collective, along body +z (FLU)
    torque: torch.Tensor  # (N, 3) body FLU [N.m]


class Allocator(ABC):
    """Abstract wrench -> per-rotor thrust allocation."""

    @abstractmethod
    def to(self, device: torch.device | str) -> Self:
        """Move the allocation to `device`; returns self.

        Part of the contract because the OWNING controller's `to` must move it --
        a CPU-resident mixer under CUDA states is the classic silent-until-live bug.
        """

    @abstractmethod
    def allocate(self, wrench: Wrench) -> Actuation:
        """Allocate a wrench to per-rotor thrusts.

        Args:
            wrench: Wrench with collective thrust and body torque.

        Returns:
            Per-rotor thrusts (N, n_rotors).
        """


class QuadrotorController(ABC):
    """Abstract classical quadrotor controller: setpoint -> action.

    The layer for SETPOINT-tracking laws (position/attitude cascades). A law whose
    primary input the setpoint cannot express -- a CTBR inner loop takes collective
    thrust directly -- lives one level down, at the `Wrench` layer, composing an
    `Allocator` itself.
    """

    @abstractmethod
    def to(self, device: torch.device | str) -> Self:
        """Move the controller's parameters to `device`; returns self."""

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the controller's internal state (e.g. PID integrators)."""

    @abstractmethod
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
