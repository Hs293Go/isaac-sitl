"""Sensor: turns the vehicle State into a measurement, gated by its own update rate.

Mirrors Pegasus's split between physics-rate sensors (IMU / GPS / barometer /
magnetometer) and render-rate graphical sensors (camera / lidar). The kernel lesson from
the stress test: the sensor compute must be BATCH-capable for the vectorized runner --
SITL needs one real IMU stream, batched RL needs N (or none, when the task uses
geometric observations). The update-at-rate pattern (each sensor decides whether enough
time has elapsed to emit) is preserved.

Concrete sensors (`Imu`, `LocalPosition`) are torch-native and batched, ported in spirit
from `~/src/autopilot` (`estimators/sensor_data.hpp`, `simulator/sensors.hpp`): the IMU
is a gravity-compensated strapdown (body specific force + angular rate), and local
position is a GPS-like ENU fix. Each has an OPTIONAL noise model (white noise +
first-order Gauss-Markov bias + turn-on bias) with defaults tracing the autopilot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import torch

from isaacsitl.so3 import quat_conjugate, quat_rotate

if TYPE_CHECKING:
    from isaacsitl.state import State

_G = 9.81  # gravity magnitude [m/s^2]; ENU gravity vector is (0, 0, -_G)


class Sensor(ABC):
    """A sensor that outputs measurements at `update_rate` Hz (0 = every step)."""

    sensor_type: str = "sensor"
    update_rate: float = 0.0

    def __init__(self) -> None:
        """Init the rate accumulator (subclasses must call this)."""
        self._elapsed = 0.0

    def _due(self, dt: float) -> bool:
        """True when >= one period has elapsed (update_rate <= 0 -> every step)."""
        if self.update_rate <= 0.0:
            return True
        self._elapsed += dt
        period = 1.0 / self.update_rate
        if self._elapsed + 1e-9 >= period:
            self._elapsed -= period
            return True
        return False

    @abstractmethod
    def update(self, state: State, dt: float) -> Any | None:
        """Return a new (batched) measurement, or None if the rate has not elapsed."""

    def reset(self, env_ids: Any = None) -> None:
        """Reset internal accumulators for the given envs.

        The rate clock is batch-shared by design (the vectorized runner steps every
        env together), so only a FULL reset (`env_ids=None`) rewinds it -- a per-env
        reset must not perturb the emission timing of the envs still running.
        """
        if env_ids is None:
            self._elapsed = 0.0


# --------------------------------------------------------------------- measurements
@dataclass
class ImuMeasurement:
    """Strapdown IMU reading (autopilot `ImuData`)."""

    accel: torch.Tensor  # (..., 3) body-frame specific force [m/s^2]
    gyro: torch.Tensor  # (..., 3) body-frame angular rate [rad/s]


@dataclass
class LocalPositionMeasurement:
    """ENU local position fix (autopilot `LocalPositionData`)."""

    position: torch.Tensor  # (..., 3) ENU position [m]


@dataclass
class MagnetometerMeasurement:
    """Body-frame magnetic field reading -- a heading reference for the ESKF."""

    field: torch.Tensor  # (..., 3) body-frame magnetic field [unit]


@dataclass
class BarometerMeasurement:
    """Static-pressure reading + the MSL altitude it encodes (a HEIGHT reference)."""

    pressure_hpa: torch.Tensor  # (...,) absolute pressure [hPa]
    altitude_m: torch.Tensor  # (...,) MSL altitude [m]
    temperature_c: float  # ambient temperature [degC] (ISA sea level)


# ------------------------------------------------------------------------ noise model
@dataclass
class ImuNoiseConfig:
    """IMU noise densities / biases (defaults trace to autopilot `ImuNoiseConfig`)."""

    gyro_noise_density: float = math.radians(2.0 * 35.0 / 3600.0)  # rad/s/sqrt(Hz)
    gyro_random_walk: float = math.radians(2.0 * 4.0 / 3600.0)
    gyro_bias_correlation_time: float = 1.0e3  # s
    gyro_turn_on_bias_sigma: float = math.radians(0.5)
    accel_noise_density: float = 2.0 * 2.0e-3  # m/s^2/sqrt(Hz)
    accel_random_walk: float = 2.0 * 3.0e-3
    accel_bias_correlation_time: float = 300.0
    accel_turn_on_bias_sigma: float = 20.0e-3 * _G


@dataclass
class LocalPositionNoiseConfig:
    """ENU position noise (default trace to autopilot `reported_pos_std_dev`)."""

    pos_std: tuple[float, float, float] = (0.5, 0.5, 0.5)  # per-axis std [m]


@dataclass
class MagnetometerNoiseConfig:
    """Magnetometer white-noise std (unit field); small -> a firm heading reference."""

    field_std: float = 0.01


@dataclass
class BarometerNoiseConfig:
    """Barometer white-noise std [hPa].

    Even a "clean" rig wants a small dither: a perfectly constant pressure reads as a
    STUCK sensor to PX4's data validator, baro fusion never starts, and it never arms.
    """

    pressure_std: float = 0.005


class _GaussMarkovNoise:
    """Batched white noise + Gauss-Markov bias + turn-on bias, shape (n, dim).

    A faithful port of the autopilot's `NoiseProcess::corrupt`: the bias propagates as a
    discrete Gauss-Markov process and a per-step white term is added, so the corrupted
    value is `true + bias + white`. Per-env/axis; `reset` resamples the turn-on bias.
    """

    def __init__(
        self,
        num_envs: int,
        dim: int,
        *,
        noise_density: float,
        random_walk: float,
        correlation_time: float,
        turn_on_sigma: float,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ):
        """Configure the densities/correlation and sample the initial turn-on bias."""
        self._nd = noise_density
        self._rw = random_walk
        self._tau = correlation_time
        self._turn_on = turn_on_sigma
        self._gen = generator
        self._shape = (num_envs, dim)
        self._device = torch.device(device)
        self.bias = torch.zeros(self._shape, device=self._device)
        self.reset()

    def _randn(self, rows: int | None = None) -> torch.Tensor:
        shape = self._shape if rows is None else (rows, self._shape[1])
        return torch.randn(shape, device=self._device, generator=self._gen)

    def reset(self, env_ids: Any = None) -> None:
        """Resample the turn-on bias (all envs, or just `env_ids`)."""
        if env_ids is None:
            self.bias = self._turn_on * self._randn()
        else:
            rows = env_ids.shape[0] if torch.is_tensor(env_ids) else len(env_ids)
            self.bias[env_ids] = self._turn_on * self._randn(rows)

    def corrupt(self, true_value: torch.Tensor, dt: float) -> torch.Tensor:
        """`true + bias + white`, propagating the Gauss-Markov bias one step."""
        decay = math.exp(-dt / self._tau)
        # discretized random-walk std: sqrt(-rw^2 * tau/2 * (e^(-2dt/tau) - 1))
        walk_var = (
            -(self._rw**2) * self._tau / 2.0 * (math.exp(-2.0 * dt / self._tau) - 1.0)
        )
        self.bias = decay * self.bias + math.sqrt(max(walk_var, 0.0)) * self._randn()
        white = (self._nd / math.sqrt(dt)) * self._randn()
        return true_value + self.bias + white


# ------------------------------------------------------------------ concrete sensors
class Imu(Sensor):
    """Strapdown IMU: gravity-compensated body specific force + body angular rate.

    `accel = R_world->body (a_world - g_world)` with `g_world = (0, 0, -G)` (ENU), so at
    rest and level it reads `(0, 0, +G)` (the reaction to gravity); `gyro` is the body
    angular rate. Pass an `ImuNoiseConfig` for autopilot-grade white + Gauss-Markov-bias
    noise; `noise=None` (default) gives the clean measurement.
    """

    sensor_type = "imu"

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        update_rate: float = 0.0,
        noise: ImuNoiseConfig | None = None,
        generator: torch.Generator | None = None,
    ):
        """Build the IMU; if `noise` is given, set up the accel/gyro noise processes."""
        super().__init__()
        self.update_rate = update_rate
        self._device = torch.device(device)
        self._gravity = torch.zeros(3, device=self._device)
        self._gravity[2] = _G  # add to a_world to cancel gravity -> specific force
        if noise is None:
            self._accel_noise = None
            self._gyro_noise = None
        else:
            self._accel_noise = _GaussMarkovNoise(
                num_envs,
                3,
                noise_density=noise.accel_noise_density,
                random_walk=noise.accel_random_walk,
                correlation_time=noise.accel_bias_correlation_time,
                turn_on_sigma=noise.accel_turn_on_bias_sigma,
                device=self._device,
                generator=generator,
            )
            self._gyro_noise = _GaussMarkovNoise(
                num_envs,
                3,
                noise_density=noise.gyro_noise_density,
                random_walk=noise.gyro_random_walk,
                correlation_time=noise.gyro_bias_correlation_time,
                turn_on_sigma=noise.gyro_turn_on_bias_sigma,
                device=self._device,
                generator=generator,
            )

    def update(self, state: State, dt: float) -> ImuMeasurement | None:
        """Body specific force + angular rate; None if the rate has not elapsed."""
        if not self._due(dt):
            return None
        spec_world = state.linear_acceleration + self._gravity  # a_world - (0,0,-G)
        accel = quat_rotate(quat_conjugate(state.attitude), spec_world)  # world -> body
        gyro = state.angular_velocity.clone()
        if self._accel_noise is not None and self._gyro_noise is not None:
            accel = self._accel_noise.corrupt(accel, dt)
            gyro = self._gyro_noise.corrupt(gyro, dt)
        return ImuMeasurement(accel=accel, gyro=gyro)

    def reset(self, env_ids: Any = None) -> None:
        """Reset the rate clock and (if noisy) the bias processes."""
        super().reset(env_ids)
        if self._accel_noise is not None and self._gyro_noise is not None:
            self._accel_noise.reset(env_ids)
            self._gyro_noise.reset(env_ids)


class LocalPosition(Sensor):
    """GPS-like ENU local position fix; optional Gaussian noise (autopilot std 0.5 m).

    Pass a `LocalPositionNoiseConfig` for per-axis Gaussian noise; the default (None)
    returns the true ENU position.
    """

    sensor_type = "local_position"

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        update_rate: float = 0.0,
        noise: LocalPositionNoiseConfig | None = None,
        generator: torch.Generator | None = None,
    ):
        """Build the local-position sensor; cache the per-axis noise std if given."""
        super().__init__()
        self._n = num_envs
        self.update_rate = update_rate
        self._device = torch.device(device)
        self._gen = generator
        self._std = (
            None if noise is None else torch.tensor(noise.pos_std, device=self._device)
        )

    def update(self, state: State, dt: float) -> LocalPositionMeasurement | None:
        """ENU position (+ Gaussian noise if configured); None if not yet due."""
        if not self._due(dt):
            return None
        # Must clone in all branches to avoid in-place modification of the state tensor
        pos = state.position.clone()
        if self._std is not None:
            noise = torch.randn(pos.shape, device=pos.device, generator=self._gen)
            pos += self._std * noise
        return LocalPositionMeasurement(position=pos)


class Barometer(Sensor):
    """ISA static pressure at MSL altitude = `alt_ref_m` (home) + local ENU up.

    Reporting MSL rather than bare local `up` keeps the baro CONSISTENT with a
    geodetic GPS -- a local-up baro sits ~home-altitude below the GPS fix and the
    autopilot EKF cannot reconcile the offset ("height estimate not stable" -> arm
    denied). The pressure model is the ISA troposphere (valid below 11 km).
    """

    sensor_type = "barometer"

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        alt_ref_m: float = 0.0,
        update_rate: float = 0.0,
        noise: BarometerNoiseConfig | None = None,
        generator: torch.Generator | None = None,
    ):
        """Build the barometer; `alt_ref_m` is the MSL altitude of the ENU origin."""
        super().__init__()
        self._n = num_envs
        self.update_rate = update_rate
        self._device = torch.device(device)
        self._alt_ref = float(alt_ref_m)
        self._gen = generator
        self._std = None if noise is None else float(noise.pressure_std)

    def update(self, state: State, dt: float) -> BarometerMeasurement | None:
        """ISA pressure (+ dither if configured); None if the rate has not elapsed."""
        if not self._due(dt):
            return None
        alt = self._alt_ref + state.position[..., 2]
        pressure = 1013.25 * (1.0 - 2.25577e-5 * alt) ** 5.25588  # ISA below 11 km
        if self._std is not None:
            pressure += self._std * torch.randn(
                pressure.shape, device=pressure.device, generator=self._gen
            )
        return BarometerMeasurement(
            pressure_hpa=pressure, altitude_m=alt, temperature_c=15.0
        )


class Magnetometer(Sensor):
    """Body-frame magnetic field `R_world->body @ field_enu` -- a HEADING reference.

    The Earth field's horizontal component rotates with yaw in the body frame, so the
    ESKF can OBSERVE heading -- unobservable from IMU + GPS alone (yaw otherwise
    dead-reckons + drifts). `field_enu` is the reference field in world ENU (default
    North + 45 deg down, unit); pass a `MagnetometerNoiseConfig` for white noise, None
    for the clean reading.
    """

    sensor_type = "magnetometer"

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        field_enu: tuple[float, float, float] = (0.0, 0.707, -0.707),
        update_rate: float = 0.0,
        noise: MagnetometerNoiseConfig | None = None,
        generator: torch.Generator | None = None,
    ):
        """Build the magnetometer; normalize + cache the world-ENU reference field."""
        super().__init__()
        self._n = num_envs
        self.update_rate = update_rate
        self._device = torch.device(device)
        self._gen = generator
        f = torch.tensor(field_enu, dtype=torch.float32, device=self._device)
        self._field = f / f.norm().clamp_min(1e-9)  # unit world-ENU reference
        self._std = None if noise is None else float(noise.field_std)

    def update(self, state: State, dt: float) -> MagnetometerMeasurement | None:
        """Body-frame field (+ white noise if configured); None if not yet due."""
        if not self._due(dt):
            return None
        world = self._field.expand(state.attitude.shape[0], 3)
        field = quat_rotate(quat_conjugate(state.attitude), world)  # world -> body
        if self._std is not None:
            field += self._std * torch.randn(
                field.shape, device=field.device, generator=self._gen
            )
        return MagnetometerMeasurement(field=field)

    @property
    def field_enu(self) -> torch.Tensor:
        """The unit world-ENU reference field (the ESKF mag update needs it)."""
        return self._field
