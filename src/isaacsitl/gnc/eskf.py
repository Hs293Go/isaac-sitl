"""Error-state Kalman filter (ESKF): IMU prediction + position/heading correction.

A torch-native, BATCH-AWARE port of the eskf-baseline (`~/lidar_ws/src/eskf-baseline`):
a 15-D error state `[dp, dtheta, dv, dba, dbg]` over a nominal
`(p, q, v, accel_bias, gyro_bias, gravity)`, carried over a leading batch dim `N` so one
call steps N independent filters on the GPU. `predict` propagates from the IMU (body
specific force + body rate -- exactly the `Imu` sensor's output); `update_position`
corrects from a `LocalPosition` ENU fix; `update_mag` fuses a scalar compass heading.
The rotation primitives reuse `so3` (xyzw quaternions, active body->world, itself
batch-aware), matching the kernel convention; the baseline's hand-rolled quat helpers
are NOT duplicated.

`classical_nav` flies on `N=1` (the autopilot estimates `State` from noisy sensors
instead of reading ground truth); the vectorized runner / gain sweep flies on `N` in
the thousands, each env carrying its own `EskfConfig` -- pass tensor-valued fields to
sweep the noise densities per env. With IMU + position only, yaw is unobservable (no
heading reference) -- `update_mag` supplies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaacsitl import so3
from isaacsitl.state import State

_TANGENT = 15  # error state: [dp(3), dtheta(3), dv(3), dba(3), dbg(3)]


@dataclass
class EskfConfig:
    """IMU process-noise densities, position/heading meas noise, and initial covariance.

    Each field is a scalar (shared across the batch) OR an (N,) tensor (per-env, for a
    gain sweep). Process-noise defaults trace to the eskf-baseline `Config`;
    `position_meas_std` matches the `LocalPosition` sensor (autopilot
    `reported_pos_std_dev`).
    """

    # Filter tuning (NOT the true sensor noise): accel/gyro densities are high so the
    # filter trusts the frequent GPS and stays adaptive; the bias walks stay small so
    # a maneuver's acceleration is explained as velocity, not as a (mis-)bias.
    accel_noise_density: float | torch.Tensor = 0.5
    gyro_noise_density: float | torch.Tensor = 0.05
    accel_bias_random_walk: float | torch.Tensor = 0.001
    gyro_bias_random_walk: float | torch.Tensor = 0.0001
    position_meas_std: float | torch.Tensor = 0.1  # R inflated ~5x over injected GPS
    mag_meas_std: float | torch.Tensor = 0.05  # compass heading meas std [rad]
    # SMALL init covariance: classical_nav inits from the (known) spawn pose at rest, so
    # be confident. A large init-P + R-inflation makes the first corrections big and
    # poorly-constrained -> a spurious velocity is injected before P settles, and the
    # controller chases it into a boot-shoot.
    init_pos_std: float | torch.Tensor = 0.1
    init_att_std: float | torch.Tensor = 0.1  # rad
    init_vel_std: float | torch.Tensor = 0.1
    init_accel_bias_std: float | torch.Tensor = 0.1
    init_gyro_bias_std: float | torch.Tensor = 0.01


class Eskf:
    """Batch-aware ESKF over IMU prediction + ENU position/heading correction.

    Nominal state (all leading dim N): position `p` (ENU), orientation `q` (xyzw,
    FLU->ENU), velocity `v` (ENU), `accel_bias`/`gyro_bias` (body). The (N,15,15) error
    covariance `P` is propagated with the baseline's analytic Jacobians. `N` is inferred
    from `position`; a `(3,)` pose is treated as `N=1` (what `classical_nav` flies on).
    """

    def __init__(
        self,
        *,
        position: torch.Tensor,
        attitude: torch.Tensor,
        device: torch.device | str,
        velocity: torch.Tensor | None = None,
        config: EskfConfig | None = None,
        gravity: float = 9.81,
    ):
        """Init the nominal state from a reference pose (+ optional velocity)."""
        self.cfg = config or EskfConfig()
        dev = torch.device(device)
        self._dev = dev
        self.p = position.detach().to(dev).reshape(-1, 3).clone()  # (N, 3)
        n = self.p.shape[0]
        self._n = n
        self.q = attitude.detach().to(dev).reshape(-1, 4).clone()  # (N, 4)
        self.v = (
            torch.zeros(n, 3, device=dev)
            if velocity is None
            else velocity.detach().to(dev).reshape(-1, 3).clone()
        )
        self.accel_bias = torch.zeros(n, 3, device=dev)
        self.gyro_bias = torch.zeros(n, 3, device=dev)
        self.gravity = torch.tensor([0.0, 0.0, -gravity], device=dev)  # ENU points down
        self._last_gyro = torch.zeros(n, 3, device=dev)
        # Noise params as (N,) tensors: a shared scalar broadcasts, an (N,) sweeps per
        # env. Cached so predict/update don't re-broadcast every step.
        c = self.cfg
        self._accel_nd = self._bc(c.accel_noise_density)
        self._gyro_nd = self._bc(c.gyro_noise_density)
        self._accel_rw = self._bc(c.accel_bias_random_walk)
        self._gyro_rw = self._bc(c.gyro_bias_random_walk)
        self._pos_std = self._bc(c.position_meas_std)
        self._mag_std = self._bc(c.mag_meas_std)
        stds = (
            [c.init_pos_std] * 3
            + [c.init_att_std] * 3
            + [c.init_vel_std] * 3
            + [c.init_accel_bias_std] * 3
            + [c.init_gyro_bias_std] * 3
        )
        self._init_var = torch.stack(
            [self._bc(s) for s in stds], dim=-1
        ).square()  # (N, 15), kept so reset() can restore rows to the init covariance
        self.P = torch.diag_embed(self._init_var)  # (N, 15, 15)

    def _bc(self, x: float | torch.Tensor) -> torch.Tensor:
        """A config scalar or (N,) tensor -> (N,) on device (shared or per-env)."""
        t = torch.as_tensor(x, dtype=torch.float32, device=self._dev)
        return t.broadcast_to((self._n,))

    # ---------------------------------------------------------------------- lifecycle
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        position: torch.Tensor,
        attitude: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Re-init the filters for `env_ids` from a fresh reference pose, row-scoped.

        The episodic-RL seam: a vectorized env respawns a subset of envs each step,
        and those rows must restart exactly as a freshly-constructed filter would --
        new pose in, biases zeroed, covariance back at the init values -- while the
        surviving rows keep flying their own estimates untouched.

        Args:
            env_ids: (k,) rows to reset; None resets all N.
            position: (k, 3) spawn positions (ENU).
            attitude: (k, 4) spawn attitudes (xyzw).
            velocity: (k, 3) spawn velocities; zeros when None.
        """
        idx = slice(None) if env_ids is None else env_ids
        rows = self._n if env_ids is None else len(env_ids)
        dev = self._dev
        self.p[idx] = position.detach().to(dev).reshape(rows, 3)
        self.q[idx] = attitude.detach().to(dev).reshape(rows, 4)
        self.v[idx] = (
            torch.zeros(rows, 3, device=dev)
            if velocity is None
            else velocity.detach().to(dev).reshape(rows, 3)
        )
        self.accel_bias[idx] = 0.0
        self.gyro_bias[idx] = 0.0
        self._last_gyro[idx] = 0.0
        self.P[idx] = torch.diag_embed(self._init_var[idx])

    # --------------------------------------------------------------------- prediction
    def predict(self, accel: torch.Tensor, gyro: torch.Tensor, dt: float) -> None:
        """Strapdown propagate the nominal state and the error covariance by `dt`.

        Args:
            accel: (N, 3) body-frame specific force [m/s^2] (the `Imu` accel).
            gyro: (N, 3) body-frame angular rate [rad/s] (the `Imu` gyro).
            dt: timestep [s].
        """
        accel = accel.reshape(-1, 3)
        gyro = gyro.reshape(-1, 3)
        accel_ub = accel - self.accel_bias
        gyro_ub = gyro - self.gyro_bias
        d_theta = gyro_ub * dt  # (N, 3)
        r = so3.quat_to_matrix(self.q)  # (N, 3, 3) body -> world
        accel_world = torch.einsum("nij,nj->ni", r, accel_ub) + self.gravity

        # nominal motion (Euler; first-order quaternion update)
        self.p += self.v * dt
        self.q = so3.quat_multiply(self.q, so3.angle_axis_to_quat(d_theta))
        self.q /= self.q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.v += accel_world * dt
        self._last_gyro = gyro

        # error-state transition F and process noise Q (baseline motion_jacobians)
        eye3 = torch.eye(3, device=self._dev)
        f = torch.zeros(self._n, _TANGENT, _TANGENT, device=self._dev)
        f[:, 0:3, 0:3] = eye3
        f[:, 0:3, 6:9] = dt * eye3
        f[:, 3:6, 3:6] = so3.angle_axis_to_matrix(-d_theta)
        f[:, 3:6, 12:15] = -dt * eye3
        f[:, 6:9, 3:6] = -dt * (r @ so3.hat(accel_ub))  # d(vel)/d(theta)
        f[:, 6:9, 6:9] = eye3
        f[:, 6:9, 9:12] = -dt * r
        f[:, 9:12, 9:12] = eye3
        f[:, 12:15, 12:15] = eye3

        # Discrete process noise from continuous densities: variance = density^2 * dt
        # (NOT (density*dt)^2 -- extra dt makes Q ~100x too small at dt=1e-2, so the
        # filter goes overconfident, ignores the GPS, and rides its own drift).
        q = torch.zeros(self._n, _TANGENT, _TANGENT, device=self._dev)
        q[:, 3:6, 3:6] = (self._gyro_nd**2 * dt)[:, None, None] * eye3
        q[:, 6:9, 6:9] = (self._accel_nd**2 * dt)[:, None, None] * eye3
        q[:, 9:12, 9:12] = (self._accel_rw**2 * dt)[:, None, None] * eye3
        q[:, 12:15, 12:15] = (self._gyro_rw**2 * dt)[:, None, None] * eye3

        self.P = f @ self.P @ f.mT + q

    # --------------------------------------------------------------------- correction
    def update_position(self, position: torch.Tensor) -> None:
        """Correct with an ENU position fix (the `LocalPosition` sensor). (N, 3)."""
        position = position.reshape(-1, 3)
        h = torch.zeros(self._n, 3, _TANGENT, device=self._dev)
        h[:, 0:3, 0:3] = torch.eye(3, device=self._dev)
        r_meas = (self._pos_std**2)[:, None, None] * torch.eye(3, device=self._dev)

        innovation = position - self.p  # (N, 3)
        self._correct(h, r_meas, innovation)

    def update_mag(self, field_body: torch.Tensor, field_world: torch.Tensor) -> None:
        """Correct HEADING (yaw only) from a body field vs its world-ENU reference.

        Fuses a SCALAR heading, not the raw 3-vector. A 3-axis field update lets the
        filter explain the residual with spurious roll/pitch -- the (large) vertical
        field has more leverage on the horizontal components than yaw does -- corrupting
        the attitude the accel + GPS already pin down. Rotating the measured field to
        world with the current estimate, the yaw error is the heading gap between it and
        the reference; updating only the world-vertical (yaw) dof keeps the mag to its
        one observable direction (EKF mag-heading fusion, cf. PX4 ECL). Makes yaw
        observable (IMU + GPS leave it dead-reckoning + drifting). `field_world` is the
        reference Earth field, e.g. from `geo_mag` (shared (3,) or per-env (N, 3)).
        """
        field_body = field_body.reshape(-1, 3)
        field_world = field_world.reshape(-1, 3)
        r = so3.quat_to_matrix(self.q)  # (N, 3, 3) body -> world
        meas_world = torch.einsum("nij,nj->ni", r, field_body)  # field back to world
        a_meas = torch.atan2(meas_world[:, 1], meas_world[:, 0])  # ENU heading
        a_ref = torch.atan2(field_world[:, 1], field_world[:, 0])
        d = a_ref - a_meas
        innovation = torch.atan2(torch.sin(d), torch.cos(d))  # wrapped heading residual
        h = torch.zeros(self._n, 1, _TANGENT, device=self._dev)
        h[:, 0, 3:6] = r[:, 2, :]  # d(heading)/d(att err): world-up row of R
        r_meas = (self._mag_std**2).reshape(self._n, 1, 1)
        self._correct(h, r_meas, innovation[:, None])

    def _correct(
        self, h: torch.Tensor, r_meas: torch.Tensor, innovation: torch.Tensor
    ) -> None:
        """One Kalman correction: gain, inject, Joseph-form covariance update.

        `h` is the (N, m, 15) measurement Jacobian, `r_meas` the (N, m, m) measurement
        noise, `innovation` the (N, m) residual. Joseph form keeps P symmetric
        positive-definite in float32 (the plain (I-KH)P loses PD, then the next
        innovation-cov inverse goes singular); the jitter keeps the innovation
        covariance invertible across a wide gain sweep.
        """
        m = h.shape[1]
        s = h @ self.P @ h.mT + r_meas
        s += 1e-6 * torch.eye(m, device=self._dev)  # jitter -> stays invertible
        gain = self.P @ h.mT @ torch.linalg.inv(s)  # (N, 15, m)
        dx = torch.einsum("nij,nj->ni", gain, innovation)  # (N, 15)
        self._inject(dx)
        ikh = torch.eye(_TANGENT, device=self._dev) - gain @ h
        self.P = ikh @ self.P @ ikh.mT + gain @ r_meas @ gain.mT
        self.P = 0.5 * (self.P + self.P.mT)  # keep symmetric

    def _inject(self, dx: torch.Tensor) -> None:
        """Fold the (N, 15) error-state correction into the nominal state (boxplus)."""
        dp, d_theta, dv, dba, dbg = torch.split(dx, 3, dim=-1)
        self.p += dp
        self.q = so3.quat_multiply(self.q, so3.angle_axis_to_quat(d_theta))
        self.q /= self.q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.v += dv
        self.accel_bias += dba
        self.gyro_bias += dbg

    # ------------------------------------------------------------------------- output
    def state(self) -> State:
        """The estimated `State` (batched, leading dim N); the truth stand-in."""
        rate = self._last_gyro - self.gyro_bias
        return State(
            position=self.p.clone(),
            attitude=self.q.clone(),
            linear_velocity=self.v.clone(),
            angular_velocity=rate.clone(),
            linear_acceleration=torch.zeros(self._n, 3, device=self._dev),
        )
