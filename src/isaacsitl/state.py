"""State: the vehicle state — Isaac ENU-FLU, with NED-FRD accessors. torch, batched.

Mirrors Pegasus's `State` (position / attitude / velocities / acceleration) but every
field is a torch tensor with a leading batch dim, and the frame accessors use the
kernel's torch `conversions` rather than per-vehicle scipy. ENU-FLU is the Isaac-native
convention the RL runner reads; the NED-FRD accessors feed autopilot/SITL backends.
"""

from dataclasses import dataclass

import torch

from isaacsitl import conversions


@dataclass
class State:
    """Vehicle state in ENU-FLU; fields are (..., 3)/(..., 4) torch tensors.

    Attributes:
        position: (..., 3) ENU world position.
        attitude: (..., 4) quaternion [x, y, z, w] — FLU body relative to ENU world.
        linear_velocity: (..., 3) world velocity (ENU).
        angular_velocity: (..., 3) body angular velocity (FLU).
        linear_acceleration: (..., 3) world acceleration (ENU).
        payload_position: (..., 3) ENU position of the slung payload's center, or
            None for a vehicle without a payload rig (`PayloadCfg`).
        payload_velocity: (..., 3) ENU world velocity of the slung payload, or None.
    """

    position: torch.Tensor
    attitude: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    linear_acceleration: torch.Tensor
    payload_position: torch.Tensor | None = None
    payload_velocity: torch.Tensor | None = None

    def position_ned(self) -> torch.Tensor:
        """Position in NED (for autopilot/SITL backends)."""
        return conversions.vec_enu_ned(self.position)

    def attitude_ned_frd(self) -> torch.Tensor:
        """Attitude quaternion [x, y, z, w] — FRD body relative to NED world."""
        return conversions.quat_aero_isaac(self.attitude)

    def linear_velocity_ned(self) -> torch.Tensor:
        """World velocity in NED."""
        return conversions.vec_enu_ned(self.linear_velocity)

    def angular_velocity_frd(self) -> torch.Tensor:
        """Body angular velocity in FRD."""
        return conversions.vec_flu_frd(self.angular_velocity)

    def euler_ned_frd(self) -> torch.Tensor:
        """Roll/pitch/yaw (aerospace RPY) of the FRD body in the NED world."""
        return conversions.quat_xyzw_to_euler_xyz(self.attitude_ned_frd())
