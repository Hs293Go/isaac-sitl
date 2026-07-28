"""PD trajectory tracking on ESTIMATED state (an ESKF) -- or ground truth.

A `ClassicalControllerBackend` flies one free-flyer along a MINIMUM-SNAP trajectory,
closing the loop over ESTIMATED state: each step synthesizes an `Imu`, a noisy
`LocalPosition`, and a `Magnetometer` from the true `State`, feeds a basic `Eskf` (IMU
predict + GPS + compass correct), and the controller flies on `eskf.state()`. The
reference `r(t)` -- position AND velocity/acceleration/body-rate feed-forward -- is a
smooth min-snap spline through the SADDLE circuit (two Split-S loops on a figure-8),
built inline by `minsnap` at startup. The score is the standard min-snap TRACKING
error -- how tightly the flown path hugs `r(t)`, as position-error RMS and max over the
WHOLE lap (Mellinger & Kumar 2011; Lee et al. 2010). The knot is tied (closed loop), so
the drone spawns MOVING at `r'(0)`; the circuit peaks ~28 deg, flown with the PD tilt
limit OFF. The compass is always fused -- mag-less, the ESKF yaw drifts in near-hover
and this velocity-aligned path diverges. `--no-estimator` bypasses the ESKF to close
the loop on the TRUE state -- the control ceiling (best achievable tracking with
perfect feedback).

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
        python examples/nav.py --headless
"""

import argparse
import json
import math
from pathlib import Path
from typing import cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD trajectory tracking on ESKF state")
parser.add_argument("--usd", default=None, help="free-flyer USD (default: vendored)")
parser.add_argument(
    "--estimator",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="fly on the ESKF estimate (--no-estimator flies on the true state)",
)
parser.add_argument(
    "--trajectory",
    choices=["saddle", "circle", "figure8"],
    default="saddle",
    help="reference trajectory (default: saddle)",
)
parser.add_argument(
    "--wind", type=float, default=0.0, help="steady wind speed [m/s], blowing +x (ENU)"
)
parser.add_argument(
    "--gust", type=float, default=0.0, help="OU gust std [m/s] (tau 1 s) atop --wind"
)
parser.add_argument(
    "--force",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("FX", "FY", "FZ"),
    help="constant ENU force bias [N] (e.g. 0 0 -0.981 = a 100 g payload pickup)",
)
parser.add_argument(
    "--torque",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("TX", "TY", "TZ"),
    help="constant body-FLU torque bias [N.m]",
)
parser.add_argument(
    "--onset", type=float, default=0.0, help="disturbance onset [s] (0 = from spawn)"
)
parser.add_argument("--ramp", type=float, default=0.0, help="onset ramp duration [s]")
parser.add_argument("--seed", type=int, default=0, help="gust RNG seed (recorded)")
parser.add_argument(
    "--mass-scale",
    type=float,
    default=1.0,
    help="PLANT mass scale; the controller keeps the nominal (parametric mismatch)",
)
parser.add_argument("--out", default=None, help="verdict mirror (console-only default)")
parser.add_argument("--dump", default=None, help="save ts/ref/true/err to .npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacsitl import (  # noqa: E402
    ClassicalControllerBackend,
    DisturbanceCfg,
    Eskf,
    EskfConfig,
    Imu,
    ImuMeasurement,
    ImuNoiseConfig,
    LocalPosition,
    LocalPositionNoiseConfig,
    Magnetometer,
    MagnetometerNoiseConfig,
    PinvAllocator,
    State,
    WindCfg,
)
from isaacsitl.geomag import field_ned_gauss  # noqa: E402
from isaacsitl.gnc.pid_controller import PIDController, PIDControllerCfg  # noqa: E402
from isaacsitl.gnc.trajectory import circle, figure_8, saddle  # noqa: E402
from isaacsitl.sim import DroneSim  # noqa: E402

_DT = 0.01
_MAG_HZ = 50.0  # compass rate [Hz] -- realistic (not every 100 Hz step)
_GPS_HZ = 100.0  # GPS/correct rate [Hz]. Correcting every step is fine WITH R inflated
# (position_meas_std >> true GPS noise): gentle frequent corrections keep the estimate
# responsive (stable) AND smooth. Sparse GPS dead-reckons between fixes so the velocity
# lags and the closed loop wanders. So inflate R, don't slow the GPS.
_HOME_LAT, _HOME_LON = 47.4, 8.5  # deg (default home) for the WMM field
_G = 9.81
# Racer aero for the wind option: OQCRL's 5-inch rotor drag (isaac-drone-racing
# `dynamics.PARAMS_5INCH`: drag accel = -k_i * v_body_i * sum(W)), evaluated at the
# hover rotor speed -- k_w * sum(W^2) = g at hover, so sum(W)_hover = 2*sqrt(g/k_w) --
# and scaled by the craft mass into a linear body-FLU drag [N/(m/s)]. The rotor-drag
# model has no z term. Applied against the AIR-relative velocity when wind is on.
_KX, _KY, _KW = 4.85e-5, 7.28e-5, 2.49e-6  # OQCRL params_5inch


# With the estimator optional, its sensor suite (IMU + GPS + compass) and predict/
# update cycle live in one wrapper, built only when the ESKF is in the loop.
class Estimator:
    """The ESKF plus its sensor suite (IMU + GPS + compass); truth in, estimate out."""

    def __init__(self, init: State, num_envs: int, device: str):
        """Build the estimator's sensor suite for `num_envs` and `device`."""
        # Estimation: a noisy IMU + GPS feeding a basic ESKF, init'd from the spawn
        # pose. Calibrated IMU (no turn-on bias), autopilot-style: an R-inflated filter
        # can't observe a constant accel bias, so a turn-on bias would drift velocity.
        imu_noise = ImuNoiseConfig(
            accel_turn_on_bias_sigma=0.0, gyro_turn_on_bias_sigma=0.0
        )
        gps_noise = LocalPositionNoiseConfig(pos_std=(0.02, 0.02, 0.02))
        self._imu = Imu(num_envs=num_envs, device=device, noise=imu_noise)
        self._gps = LocalPosition(
            num_envs=num_envs, device=device, update_rate=_GPS_HZ, noise=gps_noise
        )
        # Compass (always fused): the WMM (geomag) world-ENU field -> heading (yaw)
        # fusion. Without it the ESKF's yaw is unobservable in near-hover and this
        # velocity-aligned path diverges once yaw runs away.
        bn, be, bd = field_ned_gauss(_HOME_LAT, _HOME_LON)
        mag_noise = MagnetometerNoiseConfig()
        self._mag = Magnetometer(
            num_envs=num_envs,
            device=device,
            update_rate=_MAG_HZ,
            field_enu=(be, bn, -bd),
            noise=mag_noise,
        )
        self._eskf = Eskf(
            position=init.position[0],
            attitude=init.attitude[0],
            velocity=init.linear_velocity[0],  # spawn MOVING -> no warmup transient
            device=device,
            # Sweep-paired estimator for the saddle: a moderately responsive R (0.054)
            # with a looser accel process noise -- the pairing the sweep found for this
            # circuit.
            config=EskfConfig(position_meas_std=0.054, accel_noise_density=1.768),
        )

    def _predict(self, truth: State):
        """Feed the ESKF with a new IMU measurement (from the true state)."""
        # rate 0 -> emits every step (never None)
        imu_m = cast(ImuMeasurement, self._imu.update(truth, _DT))
        self._eskf.predict(imu_m.accel[0], imu_m.gyro[0], _DT)

    def _update(self, truth: State):
        """Feed the ESKF with new GPS and compass measurements (from the true state)."""
        gps_m = self._gps.update(truth, _DT)
        mag_m = self._mag.update(truth, _DT)
        if gps_m is not None:
            self._eskf.update_position(gps_m.position[0])
        if mag_m is not None:  # compass heading -> yaw observable (else drift)
            self._eskf.update_mag(mag_m.field[0], self._mag.field_enu)

    def step(self, truth: State) -> State:
        """Return the ESKF's current state estimate."""
        self._predict(truth)
        self._update(truth)
        return self._eskf.state()


lines = []
try:
    # generate the closed circuit inline (pos/vel/accel + flatness body-rate ff)
    match args.trajectory:
        case "saddle":
            ts_np, pos_np, vel_np, acc_np, omega_np, yaw_np = saddle(_DT)
        case "circle":
            ts_np, pos_np, vel_np, acc_np, omega_np, yaw_np = circle(_DT)
        case "figure8":
            ts_np, pos_np, vel_np, acc_np, omega_np, yaw_np = figure_8(_DT)
    ref_pos = torch.tensor(pos_np, dtype=torch.float32, device=args.device)  # (T, 3)
    lines.append(
        f"circuit: {pos_np.shape[0]} steps ({pos_np.shape[0] * _DT:.1f}s), "
        f"length: {np.trapz(np.linalg.norm(np.diff(pos_np, axis=0), axis=1)):.2f} m, "
        f"velocity magnitude max={np.linalg.norm(vel_np, axis=1).max():.2f} m/s, "
        f"mean={np.linalg.norm(vel_np, axis=1).mean():.2f} m/s, "
    )
    ref_vel = torch.tensor(vel_np, dtype=torch.float32, device=args.device)
    ref_acc = torch.tensor(acc_np, dtype=torch.float32, device=args.device)
    ref_omega = torch.tensor(omega_np, dtype=torch.float32, device=args.device)
    ref_yaw = torch.tensor(yaw_np, dtype=torch.float32, device=args.device)
    steps = ref_pos.shape[0]
    r0, v0 = pos_np[0], vel_np[0]  # spawn ON the circuit, MOVING at r'(0) (knot tied)
    spawn = (float(r0[0]), float(r0[1]), float(r0[2]))
    spawn_vel = (float(v0[0]), float(v0[1]), float(v0[2]))

    # The plant: DroneSim owns the sim context, scene, USD spawn, and the live
    # mass/inertia readback (the USD is the authority for the physical numbers).
    dsim = DroneSim(
        "sourceone_racer",
        dt=_DT,
        device=args.device,
        usd=args.usd,
        spawn=spawn,
        spawn_vel=spawn_vel,
        mass_scale=args.mass_scale,
    )
    af = dsim.airframe
    max_thrust = dsim.max_thrust  # per-rotor cap [N]; motors saturate
    # classical_sweep optimum on the SADDLE circuit's tracking RMS (the min-snap hug):
    # the velocity/accel/body-rate feed-forward (set per step) does the bulk on the
    # aggressive 3D circuit -- the flatness body-rate term (ff on the attitude loop) is
    # what drops it to ~2.4 cm RMS on ground truth (~3 cm on the ESKF estimate, where
    # the filter's noise floor then dominates); kp/kv tighten the residual.
    pid_cfg = PIDControllerCfg(
        kp=(3.15, 3.15, 7.89),
        kv=(5.90, 5.90, 11.80),
        ki=(0.1, 0.1, 0.1),
        kr=(40.0, 40.0, 20.0),
        kw=(8.0, 8.0, 4.0),
        tilt_limit=False,  # the saddle's feed-forward needs up to ~28 deg tilt
        dt=_DT,
    )
    ctrl = PIDController(PinvAllocator(tuple(af.rotor_x), tuple(af.rotor_y)), pid_cfg)
    ctl = ClassicalControllerBackend(
        inertial=dsim.inertial,
        ctrl=ctrl,
        position=spawn,
    ).to(args.device)

    # Optional disturbance (off by default -- the still-air baseline stays
    # byte-identical to the documented numbers): wind felt through the OQCRL rotor
    # drag on the AIR-relative velocity, plus constant force/torque biases and the
    # plant-side mass mismatch, all onset-gated and seed-deterministic
    # (isaacsitl.disturbance -- the L1-validation experiment shape).
    dist_cfg = None
    if args.wind > 0.0 or args.gust > 0.0 or any(args.force) or any(args.torque):
        drag_linear = wind_cfg = None
        if args.wind > 0.0 or args.gust > 0.0:
            w_sum_hover = 2.0 * math.sqrt(_G / _KW)  # sum of the 4 hover rotor speeds
            drag_linear = (  # OQCRL rotor drag; aero scales with the DRONE, so nominal
                dsim.mass_nominal * _KX * w_sum_hover,
                dsim.mass_nominal * _KY * w_sum_hover,
                0.0,
            )
            wind_cfg = WindCfg(steady=(args.wind, 0.0, 0.0), gust_std=args.gust)
        dist_cfg = DisturbanceCfg(
            wind=wind_cfg,
            drag_linear=drag_linear,
            force_enu=tuple(args.force),
            torque_flu=tuple(args.torque),
            t_on=args.onset,
            ramp=args.ramp,
            seed=args.seed,
        )
        dsim.set_disturbance(dist_cfg)

    truth = dsim.state()
    est = (
        Estimator(truth, num_envs=dsim.num_envs, device=args.device)
        if args.estimator
        else None
    )
    inertia_str = [round(x, 6) for x in dsim.inertia.tolist()]
    fb = "ESKF estimate (mag=on)" if est is not None else "ground truth"
    mass_str = f"mass={dsim.mass:.4f} kg"
    if args.mass_scale != 1.0:  # noqa: RUF069 -- exact sentinel default
        mass_str += f" (controller told nominal {dsim.mass_nominal:.4f} kg)"
    lines.append(f"booted: {mass_str}, inertia_diag={inertia_str} feedback={fb}")
    if dist_cfg is not None:
        lines.append("disturbance: " + json.dumps(dist_cfg.provenance()))

    # Track the trajectory: set the per-step reference (pos + vel/accel feed-forward),
    # estimate, control, and log the TRUE tracking error ||p - r(t)|| (on truth).
    err = torch.zeros(steps, device=args.device)
    true_pos = torch.zeros(steps, 3, device=args.device)
    sat_steps = 0  # steps with any rotor pinned at max_thrust (the headroom gauge)
    broke: tuple[int, str] | None = None
    flown = steps
    for k in range(steps):
        state_feedback = est.step(truth) if est is not None else truth

        # per-step reference + feed-forward (trajectory track)
        ctl.set_reference(ref_pos[k], ref_vel[k], ref_acc[k], ref_omega[k], ref_yaw[k])
        ctl.update_state(state_feedback)
        act = ctl.actuation()
        act.rotor_thrust = act.rotor_thrust.clamp(max=max_thrust)  # motors saturate
        sat_steps += int(bool((act.rotor_thrust >= max_thrust - 1e-6).any()))
        # (any configured disturbance is applied inside dsim.step)

        err[k] = torch.linalg.norm(truth.position[0] - ref_pos[k])
        true_pos[k] = truth.position[0]
        if float(truth.position[0, 2]) < 0.15:
            broke = (k, "drone ground strike")
        if float(err[k]) > 2.0:
            broke = broke or (k, f"divergence (tracking error {float(err[k]):.2f} m)")
        if broke is not None:
            flown = k + 1
            break
        truth = dsim.step(act)  # advance one physics step; next iteration's truth

    e = err[:flown].cpu().numpy()  # ||p - r(t)|| over the flight (the min-snap hug)
    delta = true_pos[:flown].cpu().numpy() - pos_np[:flown]  # (T, 3) true - reference
    rms = float(np.sqrt((e**2).mean()))
    exy = float(np.sqrt((np.linalg.norm(delta[:, :2], axis=1) ** 2).mean()))
    ez = float(np.sqrt((delta[:, 2] ** 2).mean()))
    src = "the ESKF estimate" if est is not None else "ground-truth state"
    lines.append(f"trajectory: {flown} steps ({flown * _DT:.1f}s), tracked on {src}")
    # Standard min-snap tracking metric (Mellinger & Kumar 2011; Lee et al. 2010): how
    # tightly the flown path HUGS the reference over the whole flight (no steady state).
    lines.append(
        f"tracking error: RMS {rms * 100:.1f} cm, max {e.max() * 100:.1f} cm "
        f"(horizontal {exy * 100:.1f} cm, vertical {ez * 100:.1f} cm); "
        f"thrust saturation {100.0 * sat_steps / flown:.1f}% of steps"
    )
    k_on = int(args.onset / _DT)
    if 0 < k_on < flown:  # transient visibility: split the metric at the onset
        pre = float(np.sqrt((e[:k_on] ** 2).mean()))
        post = float(np.sqrt((e[k_on:] ** 2).mean()))
        lines.append(
            f"onset split @ {args.onset:.1f}s: pre {pre * 100:.1f} cm RMS | "
            f"post {post * 100:.1f} cm RMS, post peak {e[k_on:].max() * 100:.1f} cm"
        )
    if broke is not None:
        lines.append(f"BROKE at t={broke[0] * _DT:.2f} s: {broke[1]}")
    print("\n".join(f"[nav] {ln}" for ln in lines[-4:]), flush=True)
    if args.dump:
        np.savez(
            args.dump,
            ts=ts_np,
            ref=pos_np,
            true=true_pos.cpu().numpy(),
            err=e,
            rms=rms,
        )
        lines.append(f"dumped tracking geometry -> {args.dump}")
except Exception as ex:
    import traceback

    lines += ["[FAIL] " + repr(ex), traceback.format_exc()]

print("\n".join(lines), flush=True)  # console (Isaac shows flushed stdout)
if args.out:  # opt-in file mirror (agents/CI); default run is console-only
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
sim_app.close()
