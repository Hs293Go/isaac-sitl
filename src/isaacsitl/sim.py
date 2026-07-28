"""The Isaac-side plant: scene, articulation vehicle, and the `DroneSim` step API.

The ONLY module in isaacsitl that imports `isaaclab` at module load — import it
AFTER the Kit app boots (`isaacsitl.app.launch()`, or your own `AppLauncher`).
Everything else in the package is torch-only and imports without a sim.

`DroneSim` distills the boot/spawn/step boilerplate every consumer script used
to repeat: scene dressing, USD spawn from the airframe spec, live mass/inertia
readback (the USD is the authority, never the YAML), the write-after-reset
velocity fix, and the read-before-commit acceleration ordering. The plant is
the PhysX articulation — dynamics is never hand-rolled here.
"""

from __future__ import annotations

from isaaclab.assets import Articulation, ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz
import torch

from isaacsitl import conversions, so3
from isaacsitl.actuation import Actuation
from isaacsitl.airframe import AirframeCfg, load_airframe, resolve_usd
from isaacsitl.gnc.controller import InertialProperties
from isaacsitl.state import State

_GRAVITY = 9.81


def add_ground_and_lighting(prefix: str = "/World") -> None:
    """Ground plane + cool ambient DomeLight + a tilted DistantLight key."""
    # color=None keeps the default_environment.usd native blue grid (IDR look);
    # the GroundPlaneCfg default color (0,0,0) would override it to gray.
    ground = sim_utils.GroundPlaneCfg(color=None)
    ground.func(f"{prefix}/ground", ground)
    dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.80, 0.85, 0.92))
    dome.func(f"{prefix}/DomeLight", dome)
    # Directional key for 3D shading (IDR tilts its key by RotateXYZ(-55,0,35) deg).
    q = quat_from_euler_xyz(
        torch.tensor([-0.96]), torch.tensor([0.0]), torch.tensor([0.61])
    )
    key = sim_utils.DistantLightCfg(intensity=3000.0)
    key.func(f"{prefix}/KeyLight", key, orientation=tuple(q[0].tolist()))


class ArticulationVehicle:
    """Batched vehicle over an Isaac Lab `Articulation` (rotor force-at-links)."""

    def __init__(
        self,
        articulation: Articulation,
        env_origins: torch.Tensor,
        dt: float,
        body_name: str = "body",
        rotor_xy: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Wrap an `Articulation`.

        Args:
            articulation: the floating-base vehicle articulation.
            env_origins: (N, 3) per-env world origins (scene.env_origins); subtracted to
                give env-local ENU position.
            dt: physics dt, used to finite-difference linear acceleration.
            body_name: the base rigid body that rotor wrenches are applied to.
            rotor_xy: optional airframe (rotor_x, rotor_y) [m]; if given, assert the USD
                rotor-link order matches it on init (catches a silent link reorder that
                would scramble force-at-links roll/pitch/yaw).
        """
        self._art = articulation
        self._env_origins = env_origins
        self._dt = dt
        self._body_id = articulation.find_bodies(body_name)[0]
        # Force-at-links: apply each rotor's thrust at its rotor link, so PhysX forms
        # the r x F roll/pitch -- no manual moment-arm mixing. The combined
        # [rotor links..., base body] ids let one batch call carry the per-rotor
        # forces + the base-body yaw couple. Empty -> single-body USD, wrench-only.
        rotor_ids = articulation.find_bodies("rotor.*")[0]
        self._link_ids = (
            list(rotor_ids) + list(self._body_id) if len(rotor_ids) else None
        )
        if rotor_xy is not None and self._link_ids is not None:
            self._assert_rotor_order(rotor_xy)
        self._prev_vel = torch.zeros_like(articulation.data.root_lin_vel_w)

    def read_state(self) -> State:
        """Read the batched ENU-FLU `State` (PURE; call `commit_step` 1x/step).

        A consumer reads state several times per step (metrics / obs / control), so
        this is side-effect-free. Acceleration is finite-differenced against the
        velocity latched by `commit_step`, a clean one-step accel that is identical
        across repeated post-physics reads.
        """
        d = self._art.data
        acc = (d.root_lin_vel_w - self._prev_vel) / self._dt
        # position/attitude/accel are fresh tensors (subtraction / fancy-index / arith);
        # the two velocities are raw views into the data buffer, so clone them -- a
        # snapshotted State must not mutate when later physics updates the buffer.
        return State(
            position=d.root_pos_w - self._env_origins,
            attitude=conversions.quat_wxyz_to_xyzw(d.root_quat_w),
            linear_velocity=d.root_lin_vel_w.clone(),
            angular_velocity=d.root_ang_vel_b.clone(),
            linear_acceleration=acc,
        )

    def commit_step(self) -> None:
        """Latch current velocity as the next accel baseline (pre-physics, 1x/step)."""
        self._prev_vel = self._art.data.root_lin_vel_w.clone()

    def apply_actuation(self, actuation: Actuation) -> None:
        """Apply the rotor forces/wrench and any joint efforts/positions."""
        if actuation.rotor_thrust is not None and self._link_ids is not None:
            # +z thrust at each rotor link (PhysX forms the r x F roll/pitch +
            # collective) and the yaw couple on the base body, in one batch
            # set_forces_and_torques over [rotor links..., base body].
            t = actuation.rotor_thrust
            n = t.shape[-1]
            forces = t.new_zeros((*t.shape[:-1], len(self._link_ids), 3))
            forces[..., :n, 2] = t  # +z thrust at each rotor link (link-local frame)
            torques = forces.new_zeros(forces.shape)
            if actuation.body_force is not None:
                forces[..., n, :] = actuation.body_force  # COM body force (drag etc.)
            if actuation.body_torque is not None:
                torques[..., n, :] = actuation.body_torque  # yaw couple on the body
            self._art.permanent_wrench_composer.set_forces_and_torques(
                forces, torques, body_ids=self._link_ids
            )
        if actuation.wrench is not None:  # direct body-FLU wrench (effectiveness model)
            force, torque = actuation.wrench
            self._art.permanent_wrench_composer.set_forces_and_torques(
                force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._body_id
            )
        if actuation.joint_effort is not None:
            self._art.set_joint_effort_target(actuation.joint_effort)
        if actuation.joint_position is not None:
            self._art.set_joint_position_target(actuation.joint_position)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the finite-difference accel history for the given envs.

        Pose/velocity writes (the spawn distribution) are the caller's job via the
        tensor write API; this just keeps the first post-reset acceleration clean.
        """
        if env_ids is None:
            self._prev_vel.zero_()
        else:
            self._prev_vel[env_ids] = 0.0

    def _assert_rotor_order(self, rotor_xy: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Verify the USD rotor-link order matches the airframe's (rotor_x, rotor_y).

        Force-at-links applies thrust[i] at rotor link i, and yaw_signs + the action
        are indexed the same way; a USD link reorder would SILENTLY scramble
        roll/pitch/yaw. Compare the resolved links' body-frame (x, y) to the expected
        offsets (env 0; identical across envs). Best-effort -- skipped if the body
        pose isn't ready.
        """
        if self._link_ids is None:
            return
        try:
            d = self._art.data
            rotor_ids = self._link_ids[:-1]  # link_ids = [rotor links..., base body]
            body_id = self._body_id[0]
            off_w = d.body_pos_w[0, rotor_ids] - d.body_pos_w[0, body_id]  # world
            if off_w.norm() < 1e-4:
                return  # body pose not populated yet -- skip
            q = conversions.quat_wxyz_to_xyzw(d.body_quat_w[0, body_id])
            q = q.unsqueeze(0).expand(off_w.shape[0], 4)  # broadcast the base-body quat
            off_b = so3.quat_rotate(so3.quat_conjugate(q), off_w)  # into body frame
        except (AttributeError, IndexError, RuntimeError):
            return  # can't read the structure here -- skip the check
        want = torch.stack([rotor_xy[0].to(off_b), rotor_xy[1].to(off_b)], dim=-1)
        if not torch.allclose(off_b[:, :2], want, atol=0.02):
            got = [[round(v, 4) for v in p] for p in off_b[:, :2].tolist()]
            exp = [[round(v, 4) for v in p] for p in want.tolist()]
            raise ValueError(
                f"rotor-link order mismatch: USD links' body-frame (x, y) {got} do not "
                f"match the airframe rotor offsets {exp} (within 2 cm). The link order "
                "must match the airframe geometry / action order."
            )


class DroneSim:
    """Single-vehicle physics SITL: plant + scene + airframe facts, one step API.

    Boots nothing itself — call `isaacsitl.app.launch()` (or run your own
    `AppLauncher`) first. Construction builds the SimulationContext, dresses the
    scene, spawns the airframe's USD, reads mass/inertia live from PhysX, and
    wraps the articulation. `step(action)` advances one physics step and returns
    the next `State`; batch dims are preserved (leading env axis, N=1 today).
    """

    def __init__(
        self,
        airframe: str | AirframeCfg = "sourceone_racer",
        *,
        dt: float = 0.01,
        device: str = "cpu",
        usd: str | None = None,
        spawn: tuple[float, float, float] = (0.0, 0.0, 1.0),
        spawn_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        physx: PhysxCfg | None = None,
        prim_path: str = "/World/Robot",
    ):
        """Build sim context + scene + vehicle from an airframe spec.

        Args:
            airframe: `conf/airframe/<name>.yaml` name, or a built `AirframeCfg`.
            dt: physics step [s].
            device: torch/PhysX device ("cpu" is fastest single-env; see benchmarks).
            usd: optional USD path override (default: the airframe spec's).
            spawn: initial ENU position [m].
            spawn_vel: initial ENU velocity [m/s] (written post-reset — Isaac's
                `init_state.lin_vel` is NOT applied by `sim.reset()`).
            physx: optional PhysX overrides; the default shrinks the GPU contact
                buffers for a single vehicle (the defaults reserve ~GBs).
            prim_path: stage path for the spawned robot.
        """
        self.airframe = (
            load_airframe(airframe) if isinstance(airframe, str) else airframe
        )
        self.dt = dt
        self.device = device
        self.sim = SimulationContext(
            SimulationCfg(
                dt=dt,
                device=device,
                physx=physx
                or PhysxCfg(
                    gpu_found_lost_pairs_capacity=2**16,
                    gpu_max_rigid_contact_count=2**18,
                    gpu_max_rigid_patch_count=2**14,
                ),
            )
        )
        add_ground_and_lighting()
        usd_path = resolve_usd(usd) if usd is not None else self.airframe.usd
        cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=spawn, lin_vel=spawn_vel, joint_pos={}, joint_vel={}
            ),
            actuators={},
        )
        self.robot = Articulation(cfg)
        self.sim.reset()
        n = self.robot.num_instances
        self.num_envs = n
        # init_state.lin_vel isn't applied by reset; write it so the craft STARTS
        # moving (else it lags a moving reference and spikes at t=0).
        root_vel = torch.zeros(n, 6, device=device)
        root_vel[:, :3] = torch.as_tensor(spawn_vel, dtype=torch.float32, device=device)
        self.robot.write_root_velocity_to_sim(root_vel)
        # Airframe FACTS read live from PhysX (the USD is the authority): total mass
        # and the base body's principal inertia feed the inertia-aware controllers.
        masses = self.robot.root_physx_view.get_masses()[0]
        self.mass = float(masses.sum())
        bid = int(self.robot.find_bodies("body")[0][0])
        inertia_full = self.robot.root_physx_view.get_inertias()[0, bid].reshape(3, 3)
        self.inertia = torch.diagonal(inertia_full).to(device).clamp_min(1e-6)
        self.inertial = InertialProperties(
            mass=torch.as_tensor(self.mass, device=device), inertia=self.inertia
        )
        m = self.airframe.motor
        if m.max_thrust is not None:
            self.max_thrust = float(m.max_thrust)
        elif m.twr is not None:
            self.max_thrust = m.twr * self.mass * _GRAVITY / len(self.airframe.rotor_x)
        else:
            raise ValueError("airframe motor spec needs `max_thrust` or `twr`")
        rotor_xy = (
            torch.tensor(self.airframe.rotor_x),
            torch.tensor(self.airframe.rotor_y),
        )
        self.vehicle = ArticulationVehicle(
            self.robot,
            env_origins=torch.zeros(n, 3, device=device),
            dt=dt,
            body_name="body",
            rotor_xy=rotor_xy,
        )
        self.vehicle.commit_step()

    def state(self) -> State:
        """The current batched `State` (pure read)."""
        return self.vehicle.read_state()

    def reset(
        self,
        position: tuple[float, float, float] | None = None,
        velocity: tuple[float, float, float] | None = None,
    ) -> State:
        """Re-pose the vehicle (identity attitude) and return the fresh `State`.

        With no arguments, only the finite-difference accel history is cleared.
        """
        n = self.num_envs
        if position is not None or velocity is not None:
            pose = torch.zeros(n, 7, device=self.device)
            pose[:, :3] = torch.as_tensor(
                position if position is not None else (0.0, 0.0, 1.0),
                dtype=torch.float32,
                device=self.device,
            )
            pose[:, 3] = 1.0  # identity attitude, Isaac wxyz
            self.robot.write_root_pose_to_sim(pose)
            vel = torch.zeros(n, 6, device=self.device)
            if velocity is not None:
                vel[:, :3] = torch.as_tensor(
                    velocity, dtype=torch.float32, device=self.device
                )
            self.robot.write_root_velocity_to_sim(vel)
        self.vehicle.reset()
        self.vehicle.commit_step()
        return self.vehicle.read_state()

    def step(self, action: Actuation | torch.Tensor) -> State:
        """Advance one physics step; return the next `State`.

        Args:
            action: per-rotor thrusts (N, n_rotors) [N] — clamped to the airframe's
                motor ceiling (motors saturate) — or a full `Actuation` (a
                controller's output, applied as given: rotor thrusts + yaw couple +
                body forces).
        """
        if not isinstance(action, Actuation):
            thrust = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            ).clamp(min=0.0, max=self.max_thrust)
            action = Actuation(rotor_thrust=thrust)
        self.vehicle.apply_actuation(action)
        self.vehicle.commit_step()  # latch v_k for the NEXT accel finite-diff
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
        return self.vehicle.read_state()
