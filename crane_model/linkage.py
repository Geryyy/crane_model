"""
Where the cylinder chains sit -- for the picture, never for the physics.

URDF is a tree, so the boom four-bar and the two arm cylinders hang off it as
dangling chains. `mujoco_plant.to_mujoco_xml` freezes them and pinocchio leaves
them at the same neutral, which is the configuration the parity tests check --
but frozen they render detached from the machine they drive, while Gazebo, which
closes the loops, shows them mounted. This closes them in closed form instead,
for a display model to pose. Nothing here reaches a force.

The four-bar is `symbolic.boom_ratio`'s, in numpy: a viewer should not import
casadi and pinocchio to place a mesh. Both read `config/hydraulics.yaml`.

Each chain angle is counted from a frame the description does not state, so
`_ZERO` carries the constant that turns a geometric quantity into the joint
value the URDF counts. Fitted once against
`epsilon_crane_description/config/initialization_*.yaml` -- the description's own
answer for the closed chain -- and it reproduces every shipped pose to 2.4 mrad
and 0.3 mm except `abovelog`, which disagrees with the other six by up to 0.09
rad and 46 mm and is taken as stale. Four of the six constants land within
1.3 mrad of a multiple of pi/2, which is the convention they encode.
"""

from __future__ import annotations

import numpy as np
import yaml

from .conventions import default_hydraulics_path

#: The chain joints, in the order `cylinder_joints` returns them. The left arm
#: cylinder mirrors the right one: same plane, same stroke, mirrored mount.
BOOM_BARREL = "boom_cylinder_mounting_on_slewing_column"
BOOM_STROKE = "boom_cylinder_piston_in_barrel_linear_joint"
BOOM_BIG = "boom_cylinder_linkage_big_mounting_on_slewing_column"
BOOM_SMALL = "boom_cylinder_linkage_small_mounting_on_boom"
ARM_BARREL = (
    "arm_cylinder_mounting_on_boom_right",
    "arm_cylinder_mounting_on_boom_left",
)
ARM_STROKE = (
    "arm_cylinder_piston_in_barrel_linear_joint_right",
    "arm_cylinder_piston_in_barrel_linear_joint_left",
)

#: Every joint this module places. `to_mujoco_xml` leaves exactly these movable
#: on the display model and freezes them everywhere else.
DISPLAY_JOINTS = frozenset(
    (BOOM_BARREL, BOOM_STROKE, BOOM_BIG, BOOM_SMALL) + ARM_BARREL + ARM_STROKE
)

#: Joint value minus geometric quantity, rad or m. See the module docstring.
_ZERO = {
    BOOM_BARREL: -1.570842,
    BOOM_STROKE: 0.005490,
    BOOM_BIG: -0.000563,
    BOOM_SMALL: 1.571952,
    ARM_BARREL[0]: -1.570793,
    ARM_STROKE[0]: 0.000039,
}


def _rotate(angle: float, value):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1]]
    )


def _direction(value) -> float:
    return float(np.arctan2(value[1], value[0]))


def load_geometry(path=None) -> dict:
    """Read the `boom` and `arm` sections of `config/hydraulics.yaml`, once."""
    with open(path or default_hydraulics_path(), encoding="utf-8") as handle:
        constants = yaml.safe_load(handle)
    return {key: constants[key] for key in ("boom", "arm")}


def cylinder_joints(geometry: dict, q2: float, q3: float) -> dict:
    """
    Place the boom four-bar and the arm cylinders at boom angle `q2`, arm `q3`.

    Planar throughout: the arm cylinder's lateral offset enters its length and
    nothing else. Outside the four-bar's reach the coupler point does not exist
    and `delta` goes negative -- same domain as `symbolic.boom_ratio`, and the
    NaN it returns there is the honest answer, not a pose.
    """
    boom, arm = geometry["boom"], geometry["arm"]

    foot = np.array([boom["pS0"]["x"], boom["pS0"]["y"]])
    pivot = np.array([boom["pS1"]["x"], boom["pS1"]["y"]])
    r13, r23 = float(boom["r13"]), float(boom["r23"])
    # The four-bar's attachment on the boom, in the column's plane.
    attachment = _rotate(q2, (boom["a2"] + boom["pS2"]["x"], boom["pS2"]["y"]))

    span = attachment - pivot
    span_squared = float(span @ span)
    # Coupler point: the two circles r13 about the pivot and r23 about the
    # attachment, on the branch `boom_ratio` fixes -- the negative root.
    along = (r13 * r13 - r23 * r23 + span_squared) / (2.0 * span_squared)
    delta = ((r13 + r23) ** 2 - span_squared) * (span_squared - (r13 - r23) ** 2)
    across = -np.sqrt(delta) / (2.0 * span_squared)
    coupler = pivot + along * span + across * np.array([-span[1], span[0]])

    cylinder = coupler - foot
    # The arm cylinder is direct: foot on the boom, attachment rotating with q3.
    reach = _rotate(q3, (arm["a3"] + arm["pS4"]["x"], -arm["pS4"]["z"])) - np.array(
        [arm["pS3"]["x"], arm["pS3"]["y"]]
    )
    lateral = arm["pS4"]["y"] - arm["pS3"]["z"]

    barrel = _direction(reach) + _ZERO[ARM_BARREL[0]]
    stroke = float(np.sqrt(reach @ reach + lateral * lateral)) + _ZERO[ARM_STROKE[0]]
    return {
        BOOM_BARREL: _direction(cylinder) + _ZERO[BOOM_BARREL],
        BOOM_STROKE: float(np.linalg.norm(cylinder)) + _ZERO[BOOM_STROKE],
        BOOM_BIG: _direction(coupler - pivot) + _ZERO[BOOM_BIG],
        # Counted on the boom, so q2 comes back off it.
        BOOM_SMALL: _direction(coupler - attachment) - q2 + _ZERO[BOOM_SMALL],
        ARM_BARREL[0]: barrel,
        ARM_BARREL[1]: barrel,
        ARM_STROKE[0]: stroke,
        ARM_STROKE[1]: stroke,
    }
