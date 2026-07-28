"""Actuation: torch, batched, command over a vehicle's actuators — rotors AND joints.

The backend stress test (PX4 SITL · racing RL · aerial manipulator) showed that
Pegasus's rotor-only `input_reference()` cannot express an aerial manipulator.
`Actuation` generalizes it: per-rotor commands (mapped to a body wrench by the plant)
plus optional joint efforts/positions for an arm (driven onto the `Articulation`
joints). A free-flyer racer is the no-joints case; a manipulator is the full case.
Batch-native: every field carries a leading env dim.
"""

from dataclasses import dataclass

import torch


@dataclass
class Actuation:
    """What a `Backend` emits each step (the generalization of `input_reference()`).

    Attributes:
        rotor: (..., n_rotors) per-rotor command (normalized motor / thrust); the plant
          maps it to a body wrench (``set_forces_and_torques``). DEPRECATED path
          (manual mixing) -- prefer ``rotor_thrust``. None for joint-only vehicles.
        rotor_thrust: (..., n_rotors) per-rotor thrust [N] applied as a +z FORCE AT EACH
          ROTOR LINK (force-at-links); PhysX forms the r x F roll/pitch from the link
          positions, so no manual moment-arm math. The favored, _apply_classical-style
          path. Pairs with ``body_torque`` for the yaw couple a +z force can't make.
        body_force: (..., 3) body-FLU force at the base COM (e.g. body drag / ground
          contact), applied alongside the per-rotor ``rotor_thrust`` -- the part of a
          force-at-links wrench that isn't a rotor thrust.
        body_torque: (..., 3) body-FLU couple applied at the base body (the yaw reaction
          -Sum yaw_i*(kappa*T_i + I_rotor*dW_i) -- no +z force makes yaw). Pairs
          with ``rotor_thrust``.
        wrench: optional ``(force, torque)`` body-FLU wrench applied DIRECTLY (each
          (..., 3)), bypassing the rotor->plant map -- for a controller / effectiveness
          model that already produces a body wrench. Mutually exclusive with ``rotor``.
        joint_effort: (..., n_joints) joint torque/effort targets for the arm; None for
          free-flyers.
        joint_position: (..., n_joints) optional joint position targets
          (position-controlled arms); mutually exclusive with ``joint_effort`` per
          joint.
        payload_force: (..., 3) force at the slung payload's COM -- ENU WORLD frame,
          unlike the body-FLU fields above (wind drag is naturally a world-frame
          force; the vehicle applies it ``is_global``). None without a payload rig.
    """

    rotor: torch.Tensor | None = None
    rotor_thrust: torch.Tensor | None = None
    body_force: torch.Tensor | None = None
    body_torque: torch.Tensor | None = None
    wrench: tuple[torch.Tensor, torch.Tensor] | None = None
    joint_effort: torch.Tensor | None = None
    joint_position: torch.Tensor | None = None
    payload_force: torch.Tensor | None = None
