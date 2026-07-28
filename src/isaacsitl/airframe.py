"""Airframe specs: data-driven drone configs (conf/airframe/*.yaml) -> a `Quadrotor`.

A drone is DATA, not code: `conf/airframe/<name>.yaml` (USD + rotor geometry + motor)
merged over the `AirframeCfg` schema in struct mode -- a typo'd key RAISES.
`load_airframe` reads one; `build_quadrotor` turns it into the kernel `Quadrotor`. The
USD lives in the spec too, so selecting an airframe swaps the WHOLE craft (geometry +
mesh + inertia). A bare `usd:` filename resolves to the vendored `isaacsitl.assets`;
an absolute path names a user-supplied craft. The firmware distinction (betaflight vs
PX4) is just the ordered fields here -- never a class.

Isaac-free (omegaconf + torch), so the kernel imports it without a sim boot; the USD is
a path the vehicle/scene consumes, not loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf

from isaacsitl.assets import asset_path
from isaacsitl.motor import InstantMotor, LaggedMotor, Motor
from isaacsitl.quadrotor import BodyDrag, Quadrotor

# Packaged next to this module so installed wheels and the repo checkout both work.
_CONF_DIR = Path(__file__).resolve().parent / "conf" / "airframe"
_GRAVITY = 9.81


@dataclass
class MotorCfg:
    """The command->thrust model: `instant` (linear) or `lagged` (lag + dW)."""

    kind: str = "instant"  # "instant" | "lagged"
    max_thrust: float | None = None  # fixed per-rotor thrust [N] ...
    twr: float | None = None  # ... OR derive max_thrust = twr * mass * g / n_rotors
    tau_inc: float = 0.03  # lagged: spin-up time constant [s]
    tau_dec: float = 0.03  # lagged: spin-down time constant [s]


@dataclass
class DragCfg:
    """Diagonal body-FLU drag coefficients (zeros = no drag force applied)."""

    linear: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # N/(m/s)
    quadratic: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # N/(m/s)^2


@dataclass
class AirframeCfg:
    """A drone spec: USD + rotor geometry (the canonical order) + motor + drag."""

    usd: str = MISSING  # multirotor USD; its rotor links are resolved by name (rotor.*)
    rotor_x: list[float] = MISSING  # (n,) body-FLU forward rotor offsets [m]
    rotor_y: list[float] = MISSING  # (n,) body-FLU left rotor offsets [m]
    yaw_signs: list[float] = MISSING  # (n,) per-rotor spin sign in FRD-z (+1/-1)
    kappa: float = 0.0157  # drag-to-thrust ratio cq/ct (steady yaw per unit thrust)
    motor: MotorCfg = field(default_factory=MotorCfg)
    drag: DragCfg = field(default_factory=DragCfg)


def resolve_usd(usd: str) -> str:
    """Resolve an airframe `usd:` field to an absolute on-disk path.

    A BARE filename (no directory part) maps to the vendored `isaacsitl.assets`; an
    explicit relative or absolute path is used as given (relative resolves against the
    CWD). Either way a missing mesh fails HERE -- with the path named -- not as a
    cryptic USD error at sim boot.
    """
    p = Path(usd)
    if p.parent == Path():  # bare filename -> vendored asset
        return str(asset_path(usd))
    if not p.exists():
        raise FileNotFoundError(f"airframe USD not found: {usd}")
    return str(p.resolve())


def load_airframe(name: str) -> AirframeCfg:
    """Load `conf/airframe/<name>.yaml` over the `AirframeCfg` schema (struct).

    The spec's `usd` is resolved on load via `resolve_usd` (bare filename -> vendored
    asset; an explicit path used as-is), so a bad mesh reference fails here with the
    airframe named, not as a cryptic USD error at sim boot.
    """
    path = _CONF_DIR / f"{name}.yaml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in _CONF_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"no airframe config '{name}' (known: {known})")
    merged = OmegaConf.merge(OmegaConf.structured(AirframeCfg), OmegaConf.load(path))
    cfg: AirframeCfg = OmegaConf.to_object(merged)  # resolves + validates MISSING
    cfg.usd = resolve_usd(cfg.usd)
    return cfg


def build_motor(
    cfg: MotorCfg, num_envs: int, device, mass: float | None = None, n_rotors: int = 4
) -> Motor:
    """Build the `Motor` from the spec (derives max_thrust from `twr * mass` if set)."""
    max_thrust = cfg.max_thrust
    if cfg.twr is not None:
        if mass is None:
            raise ValueError("motor.twr needs the airframe mass (read from the USD)")
        max_thrust = cfg.twr * float(mass) * _GRAVITY / n_rotors
    if max_thrust is None:
        raise ValueError("motor spec needs `max_thrust` or `twr`")
    if cfg.kind == "lagged":
        return LaggedMotor(
            max_thrust,
            num_envs,
            n_rotors,
            device,
            tau_inc=cfg.tau_inc,
            tau_dec=cfg.tau_dec,
        )
    if cfg.kind != "instant":
        raise ValueError(f"unknown motor kind '{cfg.kind}' (use 'instant' | 'lagged')")
    return InstantMotor(max_thrust, num_envs, device)


def build_quadrotor(
    cfg: AirframeCfg, num_envs: int, device, mass: float | None = None
) -> Quadrotor:
    """Data-driven `Quadrotor`: rotor geometry + `Motor` (+ drag), all from the spec."""
    motor = build_motor(
        cfg.motor, num_envs, device, mass=mass, n_rotors=len(cfg.rotor_x)
    )
    drag = None
    if any(cfg.drag.linear) or any(cfg.drag.quadratic):
        drag = BodyDrag(tuple(cfg.drag.linear), tuple(cfg.drag.quadratic))
    quad = Quadrotor(cfg.rotor_x, cfg.rotor_y, cfg.yaw_signs, cfg.kappa, motor, drag)
    return quad.to(device)
