"""External-wrench physics through a live Isaac Sim (the F/m check).

Run explicitly (GPU + the isaaclab group; excluded from the default suite):

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        uv run --group isaaclab pytest -m integration -q
"""

import pytest

pytestmark = pytest.mark.integration


def test_external_wrench_accelerates_f_over_m_and_tau_over_i():
    from isaaclab.app import AppLauncher

    _app = AppLauncher({"headless": True}).app

    import torch

    from isaacsitl.sim import DroneSim

    dsim = DroneSim("sourceone_racer", dt=0.01, device="cpu")
    hover = torch.full((1, 4), dsim.mass * 9.81 / 4.0)
    state = dsim.step(hover)  # clean step: hover thrust cancels gravity

    # A known ENU force for one step: dv ≈ F/m * dt in the world frame.
    fx = 0.5  # N
    dsim.vehicle.set_external_wrench(
        force_enu=torch.tensor([[fx, 0.0, 0.0]]), torque_flu=None
    )
    v0 = state.linear_velocity[0].clone()
    s1 = dsim.step(hover)
    dv = s1.linear_velocity[0] - v0
    want = fx / dsim.mass * dsim.dt
    assert abs(float(dv[0]) - want) < 0.2 * want, (float(dv[0]), want)
    assert abs(float(dv[1])) < 0.2 * want  # no lateral cross-talk
    dsim.vehicle.set_external_wrench(None, None)

    # A known body-FLU yaw torque for one step: dw_z ≈ tau/Izz * dt.
    tau_z = 0.02  # N.m
    dsim.vehicle.set_external_wrench(
        force_enu=None, torque_flu=torch.tensor([[0.0, 0.0, tau_z]])
    )
    w0 = s1.angular_velocity[0].clone()
    s2 = dsim.step(hover)
    dw = s2.angular_velocity[0] - w0
    want_w = tau_z / float(dsim.inertia[2]) * dsim.dt
    assert abs(float(dw[2]) - want_w) < 0.25 * want_w, (float(dw[2]), want_w)
    dsim.vehicle.set_external_wrench(None, None)

    # Clearing the latch restores the clean path: velocity change ≈ 0 at hover.
    s3 = dsim.step(hover)
    dv3 = s3.linear_velocity[0] - s2.linear_velocity[0]
    assert float(dv3.abs().max()) < 0.5 * want
