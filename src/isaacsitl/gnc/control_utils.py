"""Utility functions for classical control."""

import math

import torch


def limit_thrust_by_tilt(force_des: torch.Tensor, tilt_max: float) -> torch.Tensor:
    """Cap the horizontal force so tilt stays <= tilt_max.

    A large position error otherwise demands an extreme tilt (1.4 m -> ~41 deg) that
    overshoots + clamps the allocator, diverging the horizontal loop. OFF for aggressive
    tracking whose feed-forward already needs a big tilt.
    """
    f_xy = force_des[..., :2]
    f_xy_max = force_des[..., 2:3].clamp(min=1e-3) * math.tan(tilt_max)
    fn = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (f_xy_max / fn).clamp(max=1.0)
    return torch.cat([f_xy * scale, force_des[..., 2:3]], dim=-1)


def thrust_and_yaw_to_desired_attitude(
    force_des: torch.Tensor, yaw: float | torch.Tensor
) -> torch.Tensor:
    """Generates a desired attitude from a desired thrust vector and yaw."""
    dev = force_des.device
    norm = force_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b3_des = force_des / norm
    yaw = torch.tensor(yaw, device=dev)
    c1 = torch.stack([yaw.cos(), yaw.sin(), torch.zeros((), device=dev)])
    b2_des = torch.cross(b3_des, c1.expand_as(b3_des), dim=-1)
    b2_des /= b2_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b1_des = torch.cross(b2_des, b3_des, dim=-1)
    return torch.stack([b1_des, b2_des, b3_des], dim=-1)  # (n, 3, 3)
