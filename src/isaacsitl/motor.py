"""Motor: the pluggable per-rotor command->thrust model the airframe owns.

One seam, two implementations, so the `Quadrotor` airframe is a single class:
  - `InstantMotor` -- ideal: thrust tracks the command, no state, no reaction.
  - `LaggedMotor`  -- a real motor: first-order spin-up lag (asymmetric rise/fall time
    constants, as in isaaclab_contrib's Thruster) + the rotor angular acceleration that
    drives the OQCRL reaction torque `i_rotor * dW`.

`step` returns `(per-rotor thrust [N], yaw reaction [N*m] or None)`. The airframe
sums the thrust into the wrench and folds the reaction into the yaw (with the geometry
yaw-signs); PhysX integrates the rotation. A `None` reaction means "no motor-accel term"
(the instant case), so the airframe's wrench is genuinely one code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class MotorDRConfig:
    """Per-env motor domain randomization: multiplicative uniform bands.

    Each field is the half-width of a uniform band around the nominal value
    (0.1 = +/-10%), resampled per env by `Motor.randomize` -- call it from the env
    reset so every episode flies a slightly different powertrain (the OmniDrones-style
    intrinsics DR). Zero (the default) leaves that parameter nominal.
    """

    max_thrust: float = 0.0  # thrust-coefficient band (KF)
    tau: float = 0.0  # motor time-constant band (lagged motor only)


def _uniform_band(
    rows: int, band: float, device, generator: torch.Generator | None
) -> torch.Tensor:
    """(rows, 1) multiplicative scales, uniform in [1 - band, 1 + band]."""
    u = torch.rand((rows, 1), device=device, generator=generator)
    return 1.0 + band * (2.0 * u - 1.0)


class Motor(ABC):
    """Base: per-rotor command [-1, 1] -> per-rotor thrust [N] (+ optional dW)."""

    def __init__(
        self,
        max_thrust: float,
        num_envs: int | None = None,
        device: torch.device | str = "cpu",
    ):
        """Store the thrust ceiling [N]; with `num_envs`, allocate the DR scales."""
        self.max_thrust = float(max_thrust)
        self._num_envs = num_envs
        self._thrust_scale: torch.Tensor | None = (
            None if num_envs is None else torch.ones(num_envs, 1, device=device)
        )

    @property
    def state(self) -> torch.Tensor | None:
        """Per-rotor motor state for the obs ([-1, 1]); None for a stateless motor."""
        return None

    @abstractmethod
    def step(
        self, action: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Command (..., n) -> (per-rotor thrust [N], yaw reaction [N*m] | None)."""

    @abstractmethod
    def command_for(self, thrust: torch.Tensor) -> torch.Tensor:
        """Invert the steady-state curve: desired thrust [N] -> command in [-1, 1].

        The CTBR inner loop allocates per-rotor thrusts; this maps them back onto the
        command space so `step` stays the one authority for the thrust that actually
        results (lag, DR, reaction all still apply). Uses the NOMINAL max_thrust: a
        DR'd powertrain deliberately tracks the request imperfectly -- that mismatch
        is the sim2real robustness the DR exists to train.
        """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset any per-rotor state (no-op for a stateless motor)."""
        _ = env_ids

    def randomize(
        self,
        env_ids: torch.Tensor | None,
        cfg: MotorDRConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        """Resample this motor's per-env intrinsics inside `cfg`'s bands.

        Call at episode reset (env_ids = the resetting envs; None = all). Needs the
        batch size: build the motor with `num_envs` (build_motor does).
        """
        if self._thrust_scale is None:
            raise ValueError("motor was built without num_envs; DR needs the batch")
        if cfg.max_thrust:
            idx = slice(None) if env_ids is None else env_ids
            rows = self._num_envs if env_ids is None else len(env_ids)
            dev = self._thrust_scale.device
            self._thrust_scale[idx] = _uniform_band(
                rows, cfg.max_thrust, dev, generator
            )

    @property
    def intrinsics(self) -> torch.Tensor | None:
        """(num_envs, k) current per-env DR scales (privileged obs); None without DR."""
        return self._thrust_scale

    def to(self, device: torch.device | str) -> Motor:
        """Move any state to `device` (no-op for a stateless motor); returns self."""
        if self._thrust_scale is not None:
            self._thrust_scale = self._thrust_scale.to(device)
        return self


class InstantMotor(Motor):
    """Idealized motor: thrust = 0.5*(a+1)*max_thrust, no lag, no reaction (dW None)."""

    def step(self, action: torch.Tensor, dt: float) -> tuple[torch.Tensor, None]:
        """Linear curve: -1 -> 0, +1 -> max_thrust; the hover command is emergent."""
        thrust = (0.5 * (action + 1.0) * self.max_thrust).clamp(min=0.0)
        if self._thrust_scale is not None:
            thrust *= self._thrust_scale  # per-env DR (ones until randomized)
        return thrust, None

    def command_for(self, thrust: torch.Tensor) -> torch.Tensor:
        """Exact inverse of the linear curve (clamped to the command range)."""
        return (2.0 * thrust / self.max_thrust - 1.0).clamp(-1.0, 1.0)


class LaggedMotor(Motor):
    """Real motor: first-order spin-up lag + the OQCRL `i_rotor * dW` reaction torque.

    A throttle command spins each rotor toward its target with asymmetric time constants
    (`tau_inc` accelerating, `tau_dec` coasting down -- a real motor drives up
    faster than it spins down; the idea is borrowed from isaaclab_contrib's Thruster).
    `thrust = max_thrust * w**2` (quadratic curve -> hover at w=0.5, i.e. action 0).
    The rotor angular acceleration `dW` gives the reaction torque `i_rotor * dW`.
    Defaults are symmetric (`tau_inc == tau_dec`) -- the validated classical plant.

    The lag is integrated with the exact ZOH discretization
    `w += (throttle - w) * (1 - exp(-dt/tau))`, so any tau is resolved at any physics
    dt (a micro motor's tau ~ dt breaks explicit Euler; this never can).
    """

    def __init__(
        self,
        max_thrust: float,
        num_envs: int,
        n_rotors: int,
        device: torch.device | str,
        *,
        tau_inc: float = 0.03,
        tau_dec: float = 0.03,
        w_max: float = 3000.0,
        i_rotor: float = 6.19e-6,
    ):
        """Build the motor state (w=0.5 ~ hover spin) and the lag/reaction params.

        Args:
            max_thrust: per-rotor thrust at full throttle [N].
            num_envs: batch size -- the motor state is (num_envs, n_rotors).
            n_rotors: rotors per vehicle.
            device: where the motor state lives.
            tau_inc: spin-up time constant [s] (a 5-inch racing motor ~0.02-0.04).
            tau_dec: spin-down time constant [s]; equal to tau_inc by default (= the
                validated symmetric plant).
            w_max: full-throttle rotor speed [rad/s] (~3000 for a 5-inch prop); scales
                the normalized dw into the physical reaction torque.
            i_rotor: rotor+prop polar inertia [kg*m^2] -- the dW reaction term.
        """
        super().__init__(max_thrust, num_envs, device)
        self._w = torch.full((num_envs, n_rotors), 0.5, device=device)
        self._tau_inc = float(tau_inc)
        self._tau_dec = float(tau_dec)
        self._tau_scale = torch.ones(num_envs, 1, device=device)
        self._w_max = float(w_max)
        self._i_rotor = float(i_rotor)

    def step(
        self, action: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lag toward throttle, thrust = max_thrust*w**2, reaction = i_rotor*dW."""
        throttle = 0.5 * (action + 1.0)  # [-1, 1] -> [0, 1]
        tau = torch.where(throttle >= self._w, self._tau_inc, self._tau_dec)
        tau *= self._tau_scale  # per-env DR (ones until randomized)
        # Exact ZOH step of dw = (throttle - w)/tau (see class docstring).
        w_new = self._w + (throttle - self._w) * (1.0 - torch.exp(-dt / tau))
        d_w = (w_new - self._w) / dt  # per-step mean rate: the reaction impulse / dt
        self._w = torch.clamp(w_new, 0.0, 1.0)
        thrust = self.max_thrust * self._w**2
        if self._thrust_scale is not None:
            thrust *= self._thrust_scale  # per-env DR
        reaction = self._i_rotor * d_w * self._w_max  # i_rotor * dW [N*m], per rotor
        return thrust, reaction

    def command_for(self, thrust: torch.Tensor) -> torch.Tensor:
        """Steady-state inverse of the quadratic curve: cmd = 2*sqrt(T/max) - 1."""
        w = (thrust / self.max_thrust).clamp(min=0.0).sqrt()
        return (2.0 * w - 1.0).clamp(-1.0, 1.0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the motor state to ~hover spin (w=0.5 at TWR 4) for `env_ids`."""
        idx = slice(None) if env_ids is None else env_ids
        self._w[idx] = 0.5

    def randomize(
        self,
        env_ids: torch.Tensor | None,
        cfg: MotorDRConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        """Resample thrust AND time-constant scales inside `cfg`'s bands."""
        super().randomize(env_ids, cfg, generator)
        if cfg.tau:
            idx = slice(None) if env_ids is None else env_ids
            rows = self._num_envs if env_ids is None else len(env_ids)
            self._tau_scale[idx] = _uniform_band(
                rows, cfg.tau, self._tau_scale.device, generator
            )

    @property
    def intrinsics(self) -> torch.Tensor:
        """(num_envs, 2): [thrust scale, tau scale] -- the privileged DR obs."""
        return torch.cat([self._thrust_scale, self._tau_scale], dim=-1)

    def to(self, device: torch.device | str) -> LaggedMotor:
        """Move the motor state to `device`; returns self."""
        super().to(device)
        self._w = self._w.to(device)
        self._tau_scale = self._tau_scale.to(device)
        return self

    @property
    def state(self) -> torch.Tensor:
        """The lagged motor state, normalized to [-1, 1] for the obs (real motor_w)."""
        return 2.0 * self._w - 1.0
