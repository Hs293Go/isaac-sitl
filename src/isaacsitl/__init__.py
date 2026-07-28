"""isaacsitl — a single-vehicle physics SITL simulator on Isaac Lab.

The plant (PhysX articulation), autopilot-grade synthetic sensors, and the
verified classical GNC reference stack (ESKF + SE(3) cascade + allocator +
min-snap trajectories), flattened out of the isaacflight program (v0.1.0).

Everything importable from this package root is torch-only — no sim boot
required (enforced by test). The Isaac-side plant lives in `isaacsitl.sim`,
importable only AFTER `isaacsitl.app.launch()` (or your own AppLauncher).
"""

from isaacsitl import conversions, so3
from isaacsitl.actuation import Actuation
from isaacsitl.airframe import AirframeCfg, build_quadrotor, load_airframe
from isaacsitl.gnc.control import ClassicalControllerBackend
from isaacsitl.gnc.controller import (
    InertialProperties,
    QuadrotorControlSetpoint,
    Wrench,
)
from isaacsitl.gnc.eskf import Eskf, EskfConfig
from isaacsitl.gnc.pid_controller import PIDController, PIDControllerCfg
from isaacsitl.gnc.pinv_allocator import PinvAllocator
from isaacsitl.gnc.rate_controller import RateController
from isaacsitl.motor import InstantMotor, LaggedMotor, Motor, MotorDRConfig
from isaacsitl.quadrotor import BodyDrag, Quadrotor
from isaacsitl.sensors import (
    Barometer,
    BarometerNoiseConfig,
    Imu,
    ImuMeasurement,
    ImuNoiseConfig,
    LocalPosition,
    LocalPositionNoiseConfig,
    Magnetometer,
    MagnetometerNoiseConfig,
)
from isaacsitl.state import State
from isaacsitl.wind import WindCfg, WindField, quadratic_drag

__all__ = [
    "Actuation",
    "AirframeCfg",
    "Barometer",
    "BarometerNoiseConfig",
    "BodyDrag",
    "ClassicalControllerBackend",
    "Eskf",
    "EskfConfig",
    "Imu",
    "ImuMeasurement",
    "ImuNoiseConfig",
    "InertialProperties",
    "InstantMotor",
    "LaggedMotor",
    "LocalPosition",
    "LocalPositionNoiseConfig",
    "Magnetometer",
    "MagnetometerNoiseConfig",
    "Motor",
    "MotorDRConfig",
    "PIDController",
    "PIDControllerCfg",
    "PinvAllocator",
    "Quadrotor",
    "QuadrotorControlSetpoint",
    "RateController",
    "State",
    "WindCfg",
    "WindField",
    "Wrench",
    "build_quadrotor",
    "conversions",
    "load_airframe",
    "quadratic_drag",
    "so3",
]
