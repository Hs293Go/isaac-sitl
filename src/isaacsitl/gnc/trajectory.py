"""Shared reference trajectories for the classical examples (Isaac-free: numpy/minsnap).

`classical_nav` tracks one; `classical_sweep` optimizes gains on the SAME one -- both
import from here so the sweep can never drift from the mission it is tuning for.
"""

from __future__ import annotations

from typing import NamedTuple

import minsnap_trajectories as ms
import numpy as np


class References(NamedTuple):
    """Reference trajectory for a classical mission.

    Attributes:
        ts: (T,) time samples [s]
        pos: (T, 3) position samples [m] in ENU
        vel: (T, 3) velocity samples [m/s] in ENU
        acc: (T, 3) acceleration samples [m/s^2] in ENU
        body_rates: (T, 3) body rates [rad/s] in XYZ (roll, pitch, yaw)
    """

    ts: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    body_rates: np.ndarray
    yaw: np.ndarray


def _condition_velocity_yaw(
    vel: np.ndarray, dt: float, min_speed: float = 0.5, max_rate: float = 4.0
) -> np.ndarray:
    """Velocity-heading yaw made safe for trajectories with near-vertical segments.

    `atan2(vy, vx)` is singular where horizontal speed vanishes (a Split-S apex flown
    near-vertical): the heading flips ~180 deg in one step and kicks the attitude loop.
    This holds the last valid heading through low-horizontal-speed segments and then
    slew-rate-limits the result (wrap-aware), so neither the setpoint nor its rate can
    spike. Planar trajectories (|v_xy| always > min_speed, gentle yaw) pass unchanged.

    Args:
        vel: (T, 3) ENU velocity samples.
        dt: sample period [s].
        min_speed: horizontal speed [m/s] below which the heading is ill-defined and
            held (nose direction is undefined flying straight down).
        max_rate: max yaw slew rate [rad/s] -- keep <= the attitude-loop bandwidth.

    Returns:
        (T,) continuous (unwrapped) yaw [rad].
    """
    speed = np.hypot(vel[:, 0], vel[:, 1])
    held = np.arctan2(vel[:, 1], vel[:, 0])
    for k in range(1, held.shape[0]):  # hold through near-vertical (ill-defined) spans
        if speed[k] < min_speed:
            held[k] = held[k - 1]
    out = np.empty_like(held)  # wrap-aware slew limiter: bounded setpoint AND rate
    out[0] = held[0]
    cap = max_rate * dt
    for k in range(1, held.shape[0]):
        step = (held[k] - out[k - 1] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
        out[k] = out[k - 1] + np.clip(step, -cap, cap)
    return out


def circle(
    dt: float,
    radius: float = 1.5,
    lap_s: float = 8.0,
    alt: float = 1.0,
    laps: int = 2,
    center: tuple[float, float] = (0.0, 0.0),
) -> References:
    """Analytic x-y circle at `alt`: the slung-payload reference (the PAYLOAD's ring).

    Starts at angle 0 -- (cx + r, cy, alt) moving +y -- and runs `laps` laps of
    `lap_s` seconds. Returns (ts, pos, vel, acc), each (T, 3) numpy in ENU sampled at
    `dt`; the derivatives are closed-form (a = -w^2 (p - c)), so the feed-forward
    chain is exact rather than spline-fit.
    """
    omega = 2.0 * np.pi / lap_s
    ts = np.arange(0.0, laps * lap_s + dt / 2, dt)
    th = omega * ts
    cx, cy = center
    zeros = np.zeros_like(th)
    pos = np.stack(
        [cx + radius * np.cos(th), cy + radius * np.sin(th), np.full_like(th, alt)],
        axis=-1,
    )
    vel = radius * omega * np.stack([-np.sin(th), np.cos(th), zeros], axis=-1)
    acc = -radius * omega**2 * np.stack([np.cos(th), np.sin(th), zeros], axis=-1)
    body_rates = np.stack([zeros, zeros, np.full_like(th, omega)], axis=-1)
    yaw = np.arctan2(vel[..., 1], vel[..., 0])
    return References(ts, pos, vel, acc, body_rates, yaw)


def figure_8(
    dt: float,
    length: float = 3.0,  # Total end-to-end length of the 8
    lap_s: float = 12.0,  # Time to complete one full figure 8
    alt: float = 1.0,
    laps: int = 2,
    center: tuple[float, float] = (0.0, 0.0),
) -> References:
    """Analytic x-y Figure-8 (Lemniscate of Bernoulli) at `alt`.

    The trajectory is parameterized such that it completes one full '8'
    every `lap_s` seconds. Body rates assume a nose-forward (tangential) heading.
    """
    # Omega for the full lap duration
    omega = 2.0 * np.pi / lap_s
    ts = np.arange(0.0, laps * lap_s + dt / 2, dt)

    # Scale parameter 'a' for Lemniscate: x^2 + y^2 = 2a^2*x*y
    # Distance from center to outer apex is length / 2
    # a = (length / 2) / np.sqrt(2)

    # We use a modified parameterization to make it a clean Lissajous-like 8
    # x(t) = a * sqrt(2) * cos(omega * t) / (1 + sin^2(omega * t))
    # Simplifying for a standard smooth 8-curve:
    cx, cy = center
    th = omega * ts

    # Analytical Positions
    # Using a standard, robust Lissajous figure-8 formulation for clean derivatives:
    # x scales with length/2, y scales with width (typically half the length scale)
    x_scale = length / 2
    y_scale = length / 4

    pos_x = cx + x_scale * np.sin(th)
    pos_y = cy + y_scale * np.sin(2 * th)
    pos_z = np.full_like(th, alt)
    pos = np.stack([pos_x, pos_y, pos_z], axis=-1)

    # Analytical Velocities (First derivatives)
    vel_x = x_scale * omega * np.cos(th)
    vel_y = y_scale * 2 * omega * np.cos(2 * th)
    vel_z = np.zeros_like(th)
    vel = np.stack([vel_x, vel_y, vel_z], axis=-1)

    # Analytical Accelerations (Second derivatives)
    acc_x = -x_scale * (omega**2) * np.sin(th)
    acc_y = -y_scale * (2 * omega) ** 2 * np.sin(2 * th)
    acc_z = np.zeros_like(th)
    acc = np.stack([acc_x, acc_y, acc_z], axis=-1)

    # Calculate Tangential Body Rates (Yaw Rate)
    # Target yaw: psi = atan2(vel_y, vel_x); d(psi)/dt = (vx*ay - vy*ax) / (vx^2 + vy^2)
    # -- the derivative of velocity is ACCELERATION, not jerk.
    numerator = vel_x * acc_y - vel_y * acc_x
    denominator = (vel_x**2) + (vel_y**2)

    # Handle the rare case of zero velocity to avoid divide-by-zero
    # (Though for this smooth curve parameterization, denominator > 0 everywhere)
    yaw_rate = np.zeros_like(th)
    idx = denominator > 1e-6
    yaw_rate[idx] = numerator[idx] / denominator[idx]

    # Flat-body assumption: Roll and pitch rates are 0, yaw rate drives the turning
    zeros = np.zeros_like(th)
    body_rates = np.stack([zeros, zeros, yaw_rate], axis=-1)
    yaw = np.arctan2(vel_y, vel_x)
    return References(ts, pos, vel, acc, body_rates, yaw)


def saddle(
    dt: float,
) -> References:
    """Min-snap CLOSED 3D "saddle" circuit -- two Split-S loops on a figure-8 footprint.

    A racing lap: two Split-S descending half-loops (x-z) at the ends, joined by two
    climbing, y-crossing diagonals (a figure-8 footprint). Tied into a CLOSED loop --
    the first and last waypoint coincide with a matched tangent velocity, so the seam is
    C-continuous (0 gap). Returns (ts, pos, vel, acc) -- each (T, 3) numpy in ENU; the
    drone spawns MOVING at vel[0]. Peaks ~28 deg tilt, so fly it with the
    ClassicalControllerBackend's tilt limit OFF (`tilt_limit=False`).
    """
    r, h, w, ll = 1.5, 4.0, 1.2, 3.0
    hb = h - 2 * r  # bottom altitude
    wps = [
        (-ll, -w, h),  # 0 start (+x, upper straight, y=-w) -- knot tied here
        (ll, -w, h),  # 1 upper straight end -> Split-S #1
        (ll + r, -w, h - r),  # 2 Split-S #1 apex (+x bulge)
        (ll, -w, hb),  # 3 Split-S #1 bottom (now -x, low)
        (0.0, -0.3, (h + hb) / 2),  # 4 climbing diagonal, crossing (y: -w -> +w)
        (-ll, w, h),  # 5 arrive upper (y=+w) -> Split-S #2
        (-ll - r, w, h - r),  # 6 Split-S #2 apex (-x bulge)
        (-ll, w, hb),  # 7 Split-S #2 bottom (now +x, low)
        (0.0, 0.3, (h + hb) / 2),  # 8 climbing diagonal back (y: +w -> -w)
        (-ll, -w, h),  # 9 close the knot (= wp0)
    ]
    times = [0.0, 2.5, 3.75, 5.0, 7.5, 10.0, 11.25, 12.5, 15.0, 17.5]
    v0 = np.array([2.4, 0.0, 0.0])  # tie the knot: matched tangent velocity at the ends
    refs = [
        ms.Waypoint(
            time=t,
            position=np.array(p),
            **({"velocity": v0} if i in {0, len(wps) - 1} else {}),
        )
        for i, (t, p) in enumerate(zip(times, wps, strict=True))
    ]
    polys = ms.generate_trajectory(
        refs,
        degree=8,
        idx_minimized_orders=(3, 4),
        num_continuous_orders=4,
        algorithm="closed-form",
    )
    ts = np.arange(0.0, times[-1] + dt / 2, dt)
    pva = ms.compute_trajectory_derivatives(polys, ts, 3)  # (3, T, 3): pos, vel, acc
    # Velocity-aligned yaw is SINGULAR where horizontal speed vanishes -- the saddle
    # flies near-vertical at Split-S apexes (|v_xy| -> 0.01 m/s), where atan2(vy, vx)
    # flips ~134 deg in a step and kicks the attitude loop (horizontal RMS 2 -> 14 cm).
    # Condition the heading (hold through the singular spans + slew-limit), then let
    # minsnap build attitude AND body rates consistent with it -- passing yaw_rate too,
    # which yaw="velocity" leaves at 0 (else the yaw-axis feed-forward is blind).
    yaw = _condition_velocity_yaw(pva[1], dt)
    yaw_rate = np.gradient(yaw, dt)
    # vehicle_mass = 1.0 kg -> the (discarded) thrust reference is per-unit-mass; keep
    # body rates (rad/s), now consistent with the conditioned yaw.
    quad_refs = ms.compute_quadrotor_trajectory(
        polys, ts, 1.0, yaw=yaw, yaw_rate=yaw_rate
    )
    return References(ts, pva[0], pva[1], pva[2], np.array(quad_refs.body_rates), yaw)
