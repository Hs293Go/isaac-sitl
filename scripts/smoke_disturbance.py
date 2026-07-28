"""Disturbance-physics smoke: F/m, tau/I, and the clean-path restore.

Exit-code-honest (0 = OK, 1 = FAIL) for the integration tier, which spawns this
as a subprocess — Isaac can't re-init in one interpreter and hangs on close, so
one boot per check with a hard exit is the only robust shape.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
        python scripts/smoke_disturbance.py --headless
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="external-wrench physics smoke")
parser.add_argument(
    "--out",
    default="/tmp/isaacsitl_smoke_disturbance.txt",  # noqa: S108 -- verdict mirror
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402

from isaacsitl.sim import DroneSim  # noqa: E402

ok = True
lines = []


def check(name: str, got: float, want: float, rel_tol: float) -> None:
    """Record a relative-tolerance check into the verdict."""
    global ok  # noqa: PLW0603 -- script-level verdict accumulator
    good = abs(got - want) <= rel_tol * abs(want)
    ok = ok and good
    lines.append(
        f"{name}: got {got:.6g} want {want:.6g} "
        f"(tol {rel_tol:.0%}) [{'OK' if good else 'FAIL'}]"
    )


try:
    dsim = DroneSim("sourceone_racer", dt=0.01, device=args.device)
    hover = torch.full((1, 4), dsim.mass * 9.81 / 4.0, device=args.device)
    state = dsim.step(hover)  # clean step: hover thrust cancels gravity

    # A known ENU force for one step: dv ~= F/m * dt in the world frame.
    fx = 0.5  # N
    dsim.vehicle.set_external_wrench(
        force_enu=torch.tensor([[fx, 0.0, 0.0]], device=args.device)
    )
    v0 = state.linear_velocity[0].clone()
    s1 = dsim.step(hover)
    dv = s1.linear_velocity[0] - v0
    want_dv = fx / dsim.mass * dsim.dt
    check("dvx = F/m*dt", float(dv[0]), want_dv, 0.2)
    lat_ok = abs(float(dv[1])) < 0.2 * want_dv
    ok = ok and lat_ok
    lines.append(
        f"lateral cross-talk dvy={float(dv[1]):.3e} [{'OK' if lat_ok else 'FAIL'}]"
    )
    dsim.vehicle.set_external_wrench(None, None)

    # A known body-FLU yaw torque for one step: dwz ~= tau/Izz * dt.
    tau_z = 0.02  # N.m
    dsim.vehicle.set_external_wrench(
        torque_flu=torch.tensor([[0.0, 0.0, tau_z]], device=args.device)
    )
    w0 = s1.angular_velocity[0].clone()
    s2 = dsim.step(hover)
    dw = s2.angular_velocity[0] - w0
    check(
        "dwz = tau/Izz*dt", float(dw[2]), tau_z / float(dsim.inertia[2]) * dsim.dt, 0.25
    )
    dsim.vehicle.set_external_wrench(None, None)

    # Clearing the latch restores the clean path: hover stays put.
    s3 = dsim.step(hover)
    dv3 = float((s3.linear_velocity[0] - s2.linear_velocity[0]).abs().max())
    clean_ok = dv3 < 0.5 * want_dv
    ok = ok and clean_ok
    lines.append(f"clean restore |dv|={dv3:.3e} [{'OK' if clean_ok else 'FAIL'}]")
    lines.append(f"[{'OK' if ok else 'FAIL'}] external-wrench physics")
except Exception as e:
    import traceback

    ok = False
    lines += ["[FAIL] " + repr(e), traceback.format_exc()]

print("\n".join(f"[smoke_disturbance] {ln}" for ln in lines), flush=True)
Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
os._exit(0 if ok else 1)  # exit code = verdict; skip Isaac's hanging close()
