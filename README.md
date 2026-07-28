# isaac-sitl

**A single-vehicle physics SITL simulator on Isaac Lab — the articulation
plant, autopilot-grade synthetic sensors, and a verified classical GNC
reference stack, for validating flight estimation and control.**

Flattened out of the [isaacflight](../isaacflight) program (`v0.1.0`): the
classical stack, minus the framework. What it deliberately is **not**: an RL
environment (no tasks, rewards, or gym API), an autopilot bridge (no MAVLink),
or a framework (no ABCs, no plugin registry). The API is what its consumers
call, nothing speculative.

## Quick start

```bash
uv sync              # torch-only surface: sensors, ESKF, cascade, airframes — no sim
uv run pytest -q     # unit suite, seconds, CPU-only

uv sync --group isaaclab   # the Isaac-side plant (Isaac Sim 5.1 / Isaac Lab 2.3; RTX GPU)
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
    python examples/nav.py --headless
```

> [!TIP]
> A sourced ROS distro leaks its `PYTHONPATH` into the venv and breaks
> imports — prefix commands with `env -u PYTHONPATH`.

## The API

```python
from isaacsitl.app import launch

sim_app = launch(headless=True)  # boot Kit BEFORE importing isaacsitl.sim

import torch
from isaacsitl import Imu, ImuNoiseConfig
from isaacsitl.sim import DroneSim

sim = DroneSim("sourceone_racer", dt=0.01, device="cpu")
state = sim.state()
imu = Imu(num_envs=sim.num_envs, device=sim.device, noise=ImuNoiseConfig())

for _ in range(2000):
    meas = imu.update(state, sim.dt)
    thrust = my_controller(my_estimator(meas))  # your stack under test
    state = sim.step(thrust)  # (N, 4) rotor thrusts -> next State
```

`DroneSim` owns the scene, the USD spawn from a data-driven airframe spec
(`conf/airframe/*.yaml`), and the live mass/inertia readback — the USD is the
authority for physical numbers. The bundled reference stack (`isaacsitl.gnc`:
ESKF, SE(3) cascade, CTBR rate loop, pinv allocation, min-snap trajectories)
is the baseline your estimator/controller is scored against; `examples/nav.py`
flies it to ~2–3 cm RMS on the saddle circuit.

Frames: ENU world / FLU body, quaternions `(x, y, z, w)`; `State` exposes
NED-FRD accessors at the autopilot boundary. Everything is batch-native
(leading env dim, N=1 today); rotation math (`so3`) is held to the
[hs293math](https://github.com/Hs293Go/hs293math) golden record at a pinned
commit.

## License

MIT — see [LICENSE](LICENSE). Vendored USD provenance: `src/isaacsitl/assets/NOTICE.md`.
