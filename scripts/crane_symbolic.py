"""
The crane's symbolic model, stated once in Python.

This is the module `docs/features/cbs-ocp-python/grill.md` D8 asks for: the cpin
model build, the mimic projection, the transmission and the output map, written
once and *imported* by every export rather than restated in each of them.
`crane_mpc/scripts/export_ocp.py` and `crane_planning/scripts/export_timing_ocp.py`
are its consumers; `scripts/export_model_fixture.py` in this package is the third,
and its output is what `test/test_symbolic_parity.cpp` compares against the
numeric `crane_model`.

It is a **library**. It has no `main`, it writes nothing, and it imports neither
`acados_template` nor anything from ROS -- so the parity test needs a solver
neither to build nor to run, and a disagreement with the numeric model is about
the dynamics rather than about a toolchain.

What it builds is what `src/symbolic_graph.cpp` builds today, at the dimensions
issue 068 left behind:

* Pinocchio's own `crba` and `nonLinearEffects`, over `casadi.SX`, on the parsed
  description -- not a restatement of the equations of motion;
* the mimic projection **P**, because Pinocchio drops `<mimic>` and
  `q5_small_telescope` and `q11_right_rail_joint` have to be reconstructed;
* the damping of `wiki/robot_model.md` §1, read out of the description and then
  overridden where `config/hydraulics.yaml` says to -- `q4_big_telescope`'s URDF
  entry is a simulation number (`wiki/implementation/parameters.md` §5) and a
  faithful URDF read is the *wrong* model here;
* the payload body at the mount frame, carried as a **symbol**: what it is bound
  to is the OCP's business (issue 072);
* the cylinder transmission of `wiki/hydraulics.md` §2--§4, from the same
  `config/hydraulics.yaml` the C++ `cylinder_geometry.hpp` is evaluated with;
* the Schur complement of `wiki/robot_model.md` §3.1 that separates the passive
  rows from the actuated ones;
* the output map **z**, with the `A±(v) sqrt(v² + eps²)` and `tanh(eps_v)`
  smoothing `wiki/mpc.md` §3.1 requires of constraint 7.

Nothing here is deleted from the C++ side and nothing here is an OCP: no cost, no
constraint, no horizon, no solver.

## The two coordinate sets, and why there are two

`crane_model`'s own contract still carries **eight** canonical coordinates and a
sixteen-state; issue 068's reduction is at the *OCP boundary*. So this module
carries both, and says which is which everywhere:

* `q`, `dq` -- the canonical eight of the model API contract §2, in the order
  `theta1_slewing`, `theta2_boom`, `theta3_arm`, `q4_big_telescope`,
  `theta6_tip`, `theta7_tilt`, `theta8_rotator`, tool. Indices `K_ACTUATED_ROWS`
  are actuated and `K_PASSIVE_ROWS` are the two passive sway coordinates.
* `x`, `u` -- what the OCPs plan: `NX = 14` and `NU = 5`. The tool coordinate is
  not a decision variable, because the gripper is opened and closed by the
  low-level controller; it arrives in the parameter vector `p` and its rate and
  acceleration are zero. **Its link keeps its mass**, which is the whole reason
  it is pinned rather than removed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import yaml

# --- the canonical eight of the model API contract §2 -------------------------

K_GENERALIZED_DOF = 8
K_ACTUATED_DOF = 6
K_PASSIVE_DOF = 2

# I_a and I_u of `wiki/robot_model.md` §0.1. Neither class is contiguous, so
# every block of M and every row of h is gathered through these rather than
# sliced -- exactly as `detail::kActuatedRows` and `detail::kPassiveRows` are.
K_ACTUATED_ROWS = (0, 1, 2, 3, 6, 7)
K_PASSIVE_ROWS = (4, 5)

# The tool's slot among the actuated six, i.e. `cylinder::kToolAxis`.
K_TOOL_AXIS = 5

# --- what the OCPs plan, after issue 068 --------------------------------------
#
# The actuated six minus the tool. `K_PLANNED_ROWS` is in canonical indices and
# `K_PLANNED_AXES` in actuated ones, because the transmission and the output map
# are written per *axis* and the mass matrix per *coordinate*.
K_PLANNED_DOF = K_ACTUATED_DOF - 1
K_PLANNED_ROWS = K_ACTUATED_ROWS[:K_PLANNED_DOF]
K_PLANNED_AXES = tuple(range(K_PLANNED_DOF))

NX = 2 * K_PLANNED_DOF + 2 * K_PASSIVE_DOF
NU = K_PLANNED_DOF

# The blocks of `x`, in the order `crane_mpc/src/ocp.cpp` reduces to and expands
# from: planned positions, passive positions, planned rates, passive rates.
X_PLANNED_POSITION = 0
X_PASSIVE_POSITION = K_PLANNED_DOF
X_PLANNED_VELOCITY = K_PLANNED_DOF + K_PASSIVE_DOF
X_PASSIVE_VELOCITY = 2 * K_PLANNED_DOF + K_PASSIVE_DOF

# --- the parameter vector -----------------------------------------------------
#
# One pinned tool coordinate and one payload body. The payload is ten numbers --
# mass, the centre of mass in K8, and the six independent entries of Theta_L --
# and it is a *symbol* here in both senses: this module never decides what it is,
# and an OCP that wants it as a runtime acados parameter (issue 072) binds it
# without rebuilding the model.
P_TOOL_POSITION = 0
P_PAYLOAD_MASS = 1
P_PAYLOAD_COM = 2
P_PAYLOAD_INERTIA = 5
NP = 11

# The order the six independent entries of a symmetric 3x3 are packed in.
INERTIA_ENTRIES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# --- the output map z of `wiki/nomenclature.md` §10 ---------------------------
#
# Four blocks of six, the offsets of `crane_model/symbolic/casadi_graph.hpp`. It
# stays six wide although the OCPs plan five: the tool cylinder is still in the
# model, and reporting what it carries at the pinned configuration is how "the
# transmission did not leave with the coordinate" stays checkable.
K_OUTPUT_DOF = 24
K_ACTUATED_FORCE_OFFSET = 0
K_CYLINDER_FORCE_OFFSET = 6
K_PISTON_VELOCITY_OFFSET = 12
K_AXIS_FLOW_OFFSET = 18

# The link a payload attaches to (`wiki/robot_model.md` §5).
PAYLOAD_MOUNT_LINK = "K8_rotator_lower_part"

# The two tools, spelled as `config/hydraulics.yaml`'s `joints.tool` spells them.
TOOLS = ("pzs100", "epsilon7040")


# ------------------------------------------------------------- hydraulics.yaml


@dataclass
class Constants:
    """
    `config/hydraulics.yaml`, as read.

    One field per entry of the C++ `crane_model::hydraulics::Constants`, with the
    same name, because the two are the same file read twice. A number that
    appears here and not in the file, or in the file and not here, is the drift
    that file exists to make impossible.
    """

    r_gear: float = 0.0
    v_m: float = 0.0

    slewing_a_a: float = 0.0
    boom_a_a: float = 0.0
    boom_a_b: float = 0.0
    arm_a_a: float = 0.0
    arm_a_b: float = 0.0
    telescope_a_a: float = 0.0
    telescope_a_b: float = 0.0
    tool_a_a: float = 0.0
    tool_a_b: float = 0.0

    boom_ps0_x: float = 0.0
    boom_ps0_y: float = 0.0
    boom_ps1_x: float = 0.0
    boom_ps1_y: float = 0.0
    boom_a2: float = 0.0
    boom_ps2_x: float = 0.0
    boom_ps2_y: float = 0.0
    r13: float = 0.0
    r23: float = 0.0

    arm_ps3_x: float = 0.0
    arm_ps3_y: float = 0.0
    arm_ps3_z: float = 0.0
    arm_a3: float = 0.0
    arm_ps4_x: float = 0.0
    arm_ps4_y: float = 0.0
    arm_ps4_z: float = 0.0

    jaw_p0: float = 0.0
    jaw_p1: float = 0.0
    jaw_p2: float = 0.0
    jaw_a9: float = 0.0
    jaw_a10: float = 0.0
    jaw_a11: float = 0.0
    jaw_a12: float = 0.0
    jaw_ps7_x: float = 0.0
    jaw_ps7_y: float = 0.0
    jaw_ps8_x: float = 0.0
    jaw_ps8_y: float = 0.0

    eps_abs: float = 0.0
    eps_v: float = 0.0

    pzs100_joints: tuple = ()
    epsilon7040_joints: tuple = ()
    damping_overrides: tuple = ()

    # The sums the linkages actually rotate. Not measurements: each is a
    # statement about which frame the other two numbers are expressed in, which
    # is why `cylinder_geometry.hpp` writes them out too rather than folding
    # them into the file.
    def boom_link_x(self) -> float:
        return self.boom_a2 + self.boom_ps2_x

    def arm_link_x(self) -> float:
        return self.arm_a3 + self.arm_ps4_x

    def arm_link_y(self) -> float:
        return -self.arm_ps4_z

    def arm_lateral_offset(self) -> float:
        return self.arm_ps4_y - self.arm_ps3_z

    def joints(self, tool: str) -> tuple:
        return self.pzs100_joints if tool == "pzs100" else self.epsilon7040_joints


def default_hydraulics_path() -> Path:
    """`config/hydraulics.yaml` beside this package's `scripts/`."""
    return Path(__file__).resolve().parent.parent / "config" / "hydraulics.yaml"


def _at(root, key: str, path: Path):
    node = root
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{path}: missing key {key}")
        node = node[part]
    return node


def _number(root, key: str, path: Path) -> float:
    value = _at(root, key, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}: key {key} is not a number")
    return float(value)


def load_constants(path: Path | None = None) -> Constants:
    """
    Read `config/hydraulics.yaml`.

    A missing key is an exception and never a default, for the reason the file's
    own header gives: a wrong hydraulic constant is a wrong force limit.
    """
    path = Path(path) if path is not None else default_hydraulics_path()
    with open(path) as stream:
        root = yaml.safe_load(stream)
    if not isinstance(root, dict):
        raise ValueError(f"{path}: not a mapping")

    out = Constants()
    out.r_gear = _number(root, "r_gear", path)
    out.v_m = _number(root, "V_m", path)

    out.slewing_a_a = _number(root, "areas.q1_slewing.A_A", path)
    out.boom_a_a = _number(root, "areas.q2_boom.A_A", path)
    out.boom_a_b = _number(root, "areas.q2_boom.A_B", path)
    out.arm_a_a = _number(root, "areas.q3_arm.A_A", path)
    out.arm_a_b = _number(root, "areas.q3_arm.A_B", path)
    out.telescope_a_a = _number(root, "areas.q4_telescope.A_A", path)
    out.telescope_a_b = _number(root, "areas.q4_telescope.A_B", path)
    out.tool_a_a = _number(root, "areas.q8_tool.A_A", path)
    out.tool_a_b = _number(root, "areas.q8_tool.A_B", path)

    out.boom_ps0_x = _number(root, "boom.pS0.x", path)
    out.boom_ps0_y = _number(root, "boom.pS0.y", path)
    out.boom_ps1_x = _number(root, "boom.pS1.x", path)
    out.boom_ps1_y = _number(root, "boom.pS1.y", path)
    out.boom_a2 = _number(root, "boom.a2", path)
    out.boom_ps2_x = _number(root, "boom.pS2.x", path)
    out.boom_ps2_y = _number(root, "boom.pS2.y", path)
    out.r13 = _number(root, "boom.r13", path)
    out.r23 = _number(root, "boom.r23", path)

    out.arm_ps3_x = _number(root, "arm.pS3.x", path)
    out.arm_ps3_y = _number(root, "arm.pS3.y", path)
    out.arm_ps3_z = _number(root, "arm.pS3.z", path)
    out.arm_a3 = _number(root, "arm.a3", path)
    out.arm_ps4_x = _number(root, "arm.pS4.x", path)
    out.arm_ps4_y = _number(root, "arm.pS4.y", path)
    out.arm_ps4_z = _number(root, "arm.pS4.z", path)

    out.jaw_p0 = _number(root, "jaw.mirror.p0", path)
    out.jaw_p1 = _number(root, "jaw.mirror.p1", path)
    out.jaw_p2 = _number(root, "jaw.mirror.p2", path)
    out.jaw_a9 = _number(root, "jaw.a9", path)
    out.jaw_a10 = _number(root, "jaw.a10", path)
    out.jaw_a11 = _number(root, "jaw.a11", path)
    out.jaw_a12 = _number(root, "jaw.a12", path)
    out.jaw_ps7_x = _number(root, "jaw.pS7.x", path)
    out.jaw_ps7_y = _number(root, "jaw.pS7.y", path)
    out.jaw_ps8_x = _number(root, "jaw.pS8.x", path)
    out.jaw_ps8_y = _number(root, "jaw.pS8.y", path)

    out.eps_abs = _number(root, "smoothing.eps_abs", path)
    out.eps_v = _number(root, "smoothing.eps_v", path)

    shared = _at(root, "joints.shared", path)
    if len(shared) != K_GENERALIZED_DOF - 1:
        raise ValueError(
            f"{path}: joints.shared is not {K_GENERALIZED_DOF - 1} entries"
        )
    out.pzs100_joints = (*shared, _at(root, "joints.tool.pzs100", path))
    out.epsilon7040_joints = (*shared, _at(root, "joints.tool.epsilon7040", path))

    overrides = root.get("damping_overrides") or []
    out.damping_overrides = tuple(
        (str(entry["joint"]), float(entry["d"])) for entry in overrides
    )
    return out


# ---------------------------------------------------- the cylinder transmission
#
# `src/cylinder_geometry.hpp`, term for term. Every linkage is planar, so a
# planar vector is a two-tuple of scalars and the arithmetic is written out --
# which is what makes the two readable side by side, and the reason the C++ is a
# template rather than three functions in `model.cpp` in the first place.
#
# Nothing branches on the value of a coordinate. A configuration at which a
# linkage does not close leaves a non-finite ratio rather than raising a flag,
# because the symbolic caller has no configuration to raise one at.


@dataclass
class AxisAreas:
    """
    Effective areas per axis.

    `a_a`/`a_b` are the force-producing areas of `wiki/hydraulics.md` §4;
    `a_eff_pos`/`a_eff_neg` the direction-dependent pump draw of §3. The rotator
    carries V_m in all four, in m³/rad.
    """

    a_a: float = 0.0
    a_b: float = 0.0
    a_eff_pos: float = 0.0
    a_eff_neg: float = 0.0


def axis_areas(constants: Constants) -> list:
    """Compose the areas of §6.1 per axis, as the circuit topology of §2 and §3 does."""
    areas = [AxisAreas() for _ in range(K_ACTUATED_DOF)]
    # q1 slewing: two cylinders, symmetric circuit, so both chambers see 2 A_A.
    areas[0] = AxisAreas(
        2.0 * constants.slewing_a_a,
        2.0 * constants.slewing_a_a,
        2.0 * constants.slewing_a_a,
        2.0 * constants.slewing_a_a,
    )
    # q2 boom: a single differential cylinder.
    areas[1] = AxisAreas(
        constants.boom_a_a, constants.boom_a_b, constants.boom_a_a, constants.boom_a_b
    )
    # q3 arm: two differential cylinders, one expression (§2.3).
    areas[2] = AxisAreas(
        2.0 * constants.arm_a_a,
        2.0 * constants.arm_a_b,
        2.0 * constants.arm_a_a,
        2.0 * constants.arm_a_b,
    )
    # q4 telescope: regenerative on extension. Rod-side oil is fed back to the
    # piston side, so only the annulus difference is drawn from the pump, while
    # the force still follows the physical areas -- which is why a_a and
    # a_eff_pos differ on this axis alone.
    areas[3] = AxisAreas(
        constants.telescope_a_a,
        constants.telescope_a_b,
        constants.telescope_a_a - constants.telescope_a_b,
        constants.telescope_a_b,
    )
    # q7 rotator: a motor, where V_m takes the role the piston area plays
    # elsewhere (§2.5). Symmetric in both directions.
    areas[4] = AxisAreas(constants.v_m, constants.v_m, constants.v_m, constants.v_m)
    # q8 tool: a cylinder axis on the same pump, areas per §6.1.
    areas[5] = AxisAreas(
        constants.tool_a_a, constants.tool_a_b, constants.tool_a_a, constants.tool_a_b
    )
    return areas


def _rotate(angle, value):
    cosine = ca.cos(angle)
    sine = ca.sin(angle)
    return (cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1])


def _perpendicular(value):
    """S_perp v, the planar 90 degree rotation of `wiki/nomenclature.md` §7."""
    return (-value[1], value[0])


def _mirrored(value):
    """diag(-1, 1) v, the reflection the 7040's outer jaw arm carries (§2.6)."""
    return (-value[0], value[1])


def _sub(left, right):
    return (left[0] - right[0], left[1] - right[1])


def _add(left, right):
    return (left[0] + right[0], left[1] + right[1])


def _scale(factor, value):
    return (factor * value[0], factor * value[1])


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


def _squared_norm(value):
    return _dot(value, value)


def _norm(value):
    return ca.sqrt(_squared_norm(value))


def boom_ratio(constants: Constants, q2):
    """
    ds_2/dq_2, `wiki/hydraulics.md` §2.2.

    The cylinder drives the coupler point p_J of a four-bar and not the boom
    directly, so s_2 is the distance from the cylinder foot p_S0 to that coupler
    point and the derivative is the analytic chain rule through p_J(d²(q2)) --
    exact, not a difference quotient. A q2 at which the triangle inequality on
    d, r_13 and r_23 fails leaves `delta` negative, and the ratio not finite.
    """
    p_s0 = (constants.boom_ps0_x, constants.boom_ps0_y)
    p_s1 = (constants.boom_ps1_x, constants.boom_ps1_y)

    p_s2 = _rotate(q2, (constants.boom_link_x(), constants.boom_ps2_y))
    d_pivot = _sub(p_s2, p_s1)
    d_pivot_rate = _perpendicular(p_s2)  # d(p_S2)/dq2

    d_squared = _squared_norm(d_pivot)
    d_squared_rate = 2.0 * _dot(d_pivot, d_pivot_rate)

    reach_sum = constants.r13 + constants.r23
    reach_difference = constants.r13 - constants.r23
    delta = (reach_sum * reach_sum - d_squared) * (
        d_squared - reach_difference * reach_difference
    )

    link_difference = constants.r13 * constants.r13 - constants.r23 * constants.r23
    alpha = (link_difference + d_squared) / (2.0 * d_squared)
    alpha_rate = -link_difference / (2.0 * d_squared * d_squared) * d_squared_rate

    root = ca.sqrt(delta)
    delta_gradient = (
        reach_sum * reach_sum + reach_difference * reach_difference - 2.0 * d_squared
    )
    # The branch is fixed by the derivation: the intersection whose local y is
    # negative in the frame with p_S1 at the origin and p_S2 on +x (§2.2). Since
    # S_perp d points along local +y, that is the negative root, and it is the
    # branch whose stroke spans the s_2 limits of §6.3.
    beta = -root / (2.0 * d_squared)
    beta_rate = (
        -(delta_gradient * d_squared / root - 2.0 * root)
        / (4.0 * d_squared * d_squared)
        * d_squared_rate
    )

    p_j = _add(
        _add(p_s1, _scale(alpha, d_pivot)), _scale(beta, _perpendicular(d_pivot))
    )
    p_j_rate = _add(
        _add(_scale(alpha_rate, d_pivot), _scale(alpha, d_pivot_rate)),
        _add(
            _scale(beta_rate, _perpendicular(d_pivot)),
            _scale(beta, _perpendicular(d_pivot_rate)),
        ),
    )

    c_cyl = _sub(p_j, p_s0)
    return _dot(c_cyl, p_j_rate) / _norm(c_cyl)


def arm_ratio(constants: Constants, q3):
    """
    ds_3/dq_3, `wiki/hydraulics.md` §2.3.

    A direct cylinder: the in-plane part rotates with q3 while the out-of-plane
    offset stays constant. That offset keeps the stroke away from zero, so this
    axis has no degenerate configuration.
    """
    moving = _rotate(q3, (constants.arm_link_x(), constants.arm_link_y()))
    in_plane = _sub(moving, (constants.arm_ps3_x, constants.arm_ps3_y))
    in_plane_rate = _perpendicular(moving)

    lateral = constants.arm_lateral_offset()
    stroke = ca.sqrt(_squared_norm(in_plane) + lateral * lateral)
    return _dot(in_plane, in_plane_rate) / stroke


def jaw_ratio(constants: Constants, q8):
    """
    ds_8/dq_8 for the 7040, `wiki/hydraulics.md` §2.6.

    The commanded outer-jaw angle q8 drives the inner jaw through the quadratic
    fit phi_sim,9(q8) and the cylinder spans the two jaws, so its length depends
    on both angles. Planar, exactly as the deployed model is. The reflection
    applies to the outer arm and to its rate alike, and *after* S_perp in both --
    diag(-1, 1) and S_perp do not commute.
    """
    mirror_angle = (constants.jaw_p0 * q8 + constants.jaw_p1) * q8 + constants.jaw_p2
    mirror_rate = 2.0 * constants.jaw_p0 * q8 + constants.jaw_p1

    outer_arm = _rotate(
        q8, (constants.jaw_a10 + constants.jaw_ps7_x, constants.jaw_ps7_y)
    )
    inner_arm = _rotate(
        mirror_angle, (constants.jaw_a12 + constants.jaw_ps8_x, constants.jaw_ps8_y)
    )

    c_cyl = _add(
        _sub(inner_arm, _mirrored(outer_arm)),
        (constants.jaw_a11 + constants.jaw_a9, 0.0),
    )
    c_cyl_rate = _sub(
        _scale(mirror_rate, _perpendicular(inner_arm)),
        _mirrored(_perpendicular(outer_arm)),
    )
    return _dot(c_cyl, c_cyl_rate) / _norm(c_cyl)


def jacobian_diagonal(constants: Constants, tool: str, q2, q3, q8) -> list:
    """
    Return the diagonal of J_cyl, which is the whole of it.

    The geometry does not couple the axes at all (`wiki/hydraulics.md` §5.7), so
    the off-diagonal entries are structurally zero and are never formed. Only q2,
    q3 and -- for the 7040 -- q8 enter; the other three axes are a constant
    radius or one-to-one.
    """
    diagonal = [None] * K_ACTUATED_DOF
    # q1 slewing: rack and pinion, the only constant transmission (§2.1).
    diagonal[0] = ca.SX(constants.r_gear)
    diagonal[1] = boom_ratio(constants, q2)
    diagonal[2] = arm_ratio(constants, q3)
    # q4 telescope: the cylinder is one-to-one with the joint coordinate; the
    # factor of two sits between q4 and the tip travel, not here (§2.4).
    diagonal[3] = ca.SX(1.0)
    # q7 rotator: a motor, so the motor angle is the joint coordinate (§2.5).
    diagonal[4] = ca.SX(1.0)
    # q8 tool (§2.6). The PZS100 rail cylinder is one-to-one with q8 and there is
    # no linkage to solve; the 7040 jaw is a four-bar whose cylinder spans both
    # jaws, so its ratio is configuration-dependent like the boom's.
    diagonal[5] = ca.SX(1.0) if tool == "pzs100" else jaw_ratio(constants, q8)
    return diagonal


# ------------------------------------------------------------ the parsed model


@dataclass
class JointSlot:
    """
    One canonical coordinate's place in the parsed model.

    `unbounded` is the `continuous` rotator: Pinocchio stores such a joint as
    (cos, sin), so it occupies two configuration entries and one velocity entry.
    """

    config_index: int
    velocity_index: int
    unbounded: bool


@dataclass
class CoupledJoint:
    """
    A joint the description declares as a `<mimic>` of a canonical one.

    Two of them matter: `q5_small_telescope` mimics `q4_big_telescope`, the second
    telescope stage of `wiki/robot_model.md` §0, and the PZS100's
    `q11_right_rail_joint` mimics `q9_left_rail_joint`. The multiplier and the
    offset are read out of the description and not assumed.
    """

    name: str
    slot: JointSlot
    source: int
    multiplier: float
    offset: float


@dataclass
class Description:
    """The description as parsed, together with the constants it does not carry."""

    tool: str
    constants: Constants
    model: pin.Model
    neutral: np.ndarray
    joints: list = field(default_factory=list)
    coupled: list = field(default_factory=list)
    # P of `src/symbolic_graph.hpp`, one entry per canonical coordinate: a list
    # of `(velocity_index, weight)`, the coordinate's own row first and then one
    # row per `<mimic>` that follows it. dq_full = P dq, so M = Pᵀ M_full P and
    # h = Pᵀ h_full -- which is how the second telescope stage and the mirrored
    # rail get their inertia counted on the coordinate that drives them.
    drives: list = field(default_factory=list)
    damping: list = field(default_factory=list)
    mount_joint: int = 0
    mount_placement: pin.SE3 = None


def _bind(model: pin.Model, name: str) -> JointSlot:
    if not model.existJointName(name):
        raise ValueError(f"robot description is missing joint {name}")
    joint = model.joints[model.getJointId(name)]
    if joint.nv != 1 or joint.nq not in (1, 2):
        raise ValueError(f"joint {name} is not a single degree of freedom")
    return JointSlot(int(joint.idx_q), int(joint.idx_v), joint.nq == 2)


def parse(
    description_xml: str, tool: str, constants: Constants, gravity
) -> Description:
    """
    Parse one description onto the canonical eight coordinates.

    The description is read twice on purpose, exactly as `detail::parse` reads it
    twice: Pinocchio builds the kinematic tree and **drops `<mimic>`** -- with
    mimic parsing on it refuses the PZS100 outright, which declares the right
    rail as a mimic of a joint that comes later in its own depth-first order --
    so the coupled joints come from the XML itself rather than being assumed.
    """
    if tool not in TOOLS:
        raise ValueError(f"tool {tool} is not one of {TOOLS}")

    model = pin.buildModelFromXML(description_xml)
    model.gravity.linear = np.asarray(gravity, dtype=float)
    tree = ET.fromstring(description_xml)

    out = Description(
        tool=tool,
        constants=constants,
        model=model,
        neutral=pin.neutral(model),
    )

    canonical = constants.joints(tool)
    out.joints = [_bind(model, name) for name in canonical]

    for joint in tree.iter("joint"):
        mimic = joint.find("mimic")
        name = joint.get("name")
        if mimic is None or name is None:
            continue
        driver = mimic.get("joint")
        if driver not in canonical or not model.existJointName(name):
            continue
        out.coupled.append(
            CoupledJoint(
                name=name,
                slot=_bind(model, name),
                source=canonical.index(driver),
                multiplier=float(mimic.get("multiplier", 1.0)),
                offset=float(mimic.get("offset", 0.0)),
            )
        )

    out.drives = [[(slot.velocity_index, 1.0)] for slot in out.joints]
    for coupled in out.coupled:
        out.drives[coupled.source].append(
            (coupled.slot.velocity_index, coupled.multiplier)
        )

    # D of `wiki/robot_model.md` §1. The description is its source
    # (`wiki/implementation/parameters.md` §1) and §5 says those entries are the
    # identified values on the actuated axes and the hand-tuned per-tool values
    # on the two passive ones -- which is why reading them off the selected
    # description gets the tool dependence right without a table here.
    damping = {}
    for joint in tree.iter("joint"):
        dynamics = joint.find("dynamics")
        if dynamics is not None and dynamics.get("damping") is not None:
            damping[joint.get("name")] = float(dynamics.get("damping"))
    out.damping = [damping.get(name, 0.0) for name in canonical]

    # Except where `config/hydraulics.yaml` says otherwise. The override table is
    # data carrying its own reason, so this loop has no idea that the joint on it
    # today is the telescope: it applies whatever the file lists, and a table
    # naming a joint that is not one of the canonical eight is a failure rather
    # than a line that quietly does nothing.
    for name, value in constants.damping_overrides:
        if name not in canonical:
            raise ValueError(
                f"the damping override for {name} names no canonical coordinate "
                f"of this description"
            )
        out.damping[canonical.index(name)] = value

    if not model.existFrame(PAYLOAD_MOUNT_LINK):
        raise ValueError(f"robot description carries no link {PAYLOAD_MOUNT_LINK}")
    frame = model.frames[model.getFrameId(PAYLOAD_MOUNT_LINK)]
    out.mount_joint = int(frame.parentJoint)
    out.mount_placement = frame.placement
    return out


# ---------------------------------------------------------------- the SX model


class CraneSymbolicModel:
    """
    The equations of motion of one description, as `casadi.SX`.

    Built once and read many times. Every expression below is an attribute, so a
    consumer that wants something this module does not wrap in a `Function` --
    an OCP writing its own residual, say -- reaches the graph directly instead of
    rebuilding it.
    """

    def __init__(
        self,
        description_xml: str,
        tool: str,
        constants: Constants | None = None,
        gravity=(0.0, 0.0, -9.81),
    ):
        self.constants = constants if constants is not None else load_constants()
        self.description = parse(description_xml, tool, self.constants, gravity)
        self.tool = tool

        # --- the symbols -----------------------------------------------------
        self.x = ca.SX.sym("x", NX)
        self.u = ca.SX.sym("u", NU)
        self.p = ca.SX.sym("p", NP)
        self.q_tool = self.p[P_TOOL_POSITION]
        self.payload = self.p[P_PAYLOAD_MASS:NP]

        # The canonical eight, put back together out of the fourteen this
        # problem plans and the pinned tool coordinate. The tool's rate and its
        # acceleration are zero -- the gripper is held by the low-level
        # controller, so no plan written against this model can move it.
        self.q = ca.SX.zeros(K_GENERALIZED_DOF)
        self.dq = ca.SX.zeros(K_GENERALIZED_DOF)
        self.ddq_a = ca.SX.zeros(K_ACTUATED_DOF)
        for axis, row in enumerate(K_PLANNED_ROWS):
            self.q[row] = self.x[X_PLANNED_POSITION + axis]
            self.dq[row] = self.x[X_PLANNED_VELOCITY + axis]
            self.ddq_a[axis] = self.u[axis]
        for index, row in enumerate(K_PASSIVE_ROWS):
            self.q[row] = self.x[X_PASSIVE_POSITION + index]
            self.dq[row] = self.x[X_PASSIVE_VELOCITY + index]
        self.q[K_ACTUATED_ROWS[K_TOOL_AXIS]] = self.q_tool

        # --- the equations of motion ------------------------------------------
        self.mass, self.bias = self.equations(self.q, self.dq, self.payload)

        m_aa, m_au, m_ua, m_uu, h_a, h_u = self._partition(self.mass, self.bias)
        self.mass_uu = m_uu
        self.mass_ua = m_ua[:, K_PLANNED_AXES]
        self.bias_u = h_u

        # `wiki/robot_model.md` §3.1: ddq_u = -M_uu⁻¹ (M_ua ddq_a + h_u), with
        # h_u already carrying D_uu dq_u. Two by two, so the inverse is the
        # adjugate and there is no factorisation to branch on -- a configuration
        # where M_uu is singular leaves a non-finite expression.
        right_hand = h_u + ca.mtimes(m_ua, self.ddq_a)
        determinant = m_uu[0, 0] * m_uu[1, 1] - m_uu[0, 1] * m_uu[1, 0]
        self.ddq_u = ca.vertcat(
            -(m_uu[1, 1] * right_hand[0] - m_uu[0, 1] * right_hand[1]) / determinant,
            -(m_uu[0, 0] * right_hand[1] - m_uu[1, 0] * right_hand[0]) / determinant,
        )

        # tau_a of §3.3, the actuated rows of M ddq + h at the consistent passive
        # acceleration. It equals M_eff u + h_eff of §3.4 and is the quantity
        # `wiki/mpc.md` §2's effort term and §3's constraint 6 are written in.
        self.tau_a = h_a + ca.mtimes(m_aa, self.ddq_a) + ca.mtimes(m_au, self.ddq_u)

        # The rows of xdot in the reduced order. The tool's two rows are
        # dq_tool = 0 and ddq_tool = 0 by construction and are simply not carried.
        self.xdot = ca.vertcat(
            self.x[X_PLANNED_VELOCITY : X_PLANNED_VELOCITY + K_PLANNED_DOF],
            self.x[X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + K_PASSIVE_DOF],
            self.u,
            self.ddq_u,
        )

        self.cylinder_jacobian = ca.vertcat(
            *jacobian_diagonal(
                self.constants, tool, self.q[1], self.q[2], self.q[K_ACTUATED_ROWS[-1]]
            )
        )
        self.z = self._output_map()

    # -- the two Pinocchio algorithms, over SX --------------------------------

    def equations(self, q, dq, payload):
        """
        M and h + D dq in the canonical eight coordinates.

        The same `crba` and `nonLinearEffects` the numeric path calls, on the
        same parsed description, through the same projection P, with the same
        damping and the same payload body. Nothing here restates a term of the
        equations of motion.
        """
        model = cpin.Model(self.description.model)
        mount = self.description.mount_joint
        inertia = ca.SX.zeros(3, 3)
        for entry, (row, column) in enumerate(INERTIA_ENTRIES):
            inertia[row, column] = payload[P_PAYLOAD_INERTIA - P_PAYLOAD_MASS + entry]
            inertia[column, row] = inertia[row, column]
        # Theta_L is about the payload's own centre of mass with the axes of K8,
        # which is the URDF `<inertial>` convention; `act` carries the whole body
        # from K8 into the frame of the joint that moves it.
        body = cpin.Inertia(
            payload[0],
            payload[
                P_PAYLOAD_COM - P_PAYLOAD_MASS : P_PAYLOAD_INERTIA - P_PAYLOAD_MASS
            ],
            inertia,
        )
        model.inertias[mount] = model.inertias[mount] + cpin.SE3(
            self.description.mount_placement
        ).act(body)

        configuration, velocity = self._expand(q, dq)
        data = model.createData()
        cpin.crba(model, data, configuration)
        cpin.nonLinearEffects(model, data, configuration, velocity)

        # `crba` fills the upper triangle of data.M only, so the lower half is
        # read back transposed rather than mirrored into place -- the same
        # reading `Model::Impl::mass_entry` does.
        def entry(row, column):
            return data.M[row, column] if row <= column else data.M[column, row]

        drives = self.description.drives
        mass = ca.SX.zeros(K_GENERALIZED_DOF, K_GENERALIZED_DOF)
        for row in range(K_GENERALIZED_DOF):
            for column in range(row, K_GENERALIZED_DOF):
                total = ca.SX.zeros()
                for left_index, left_weight in drives[row]:
                    for right_index, right_weight in drives[column]:
                        total += (
                            left_weight * right_weight * entry(left_index, right_index)
                        )
                mass[row, column] = total
                mass[column, row] = total

        bias = ca.SX.zeros(K_GENERALIZED_DOF)
        for row in range(K_GENERALIZED_DOF):
            total = ca.SX.zeros()
            for index, weight in drives[row]:
                total += weight * data.nle[index]
            bias[row] = total + self.description.damping[row] * dq[row]
        return mass, bias

    def _expand(self, q, dq):
        """
        Spread q and dq of the canonical eight over the parsed model's rows.

        The mimics follow their source and every joint outside both sets keeps its
        neutral value and zero velocity: the four cylinder sub-chains and the
        7040's driven inner jaw are loops the description closes only in Gazebo.
        """
        model = self.description.model
        configuration = ca.SX(self.description.neutral)
        velocity = ca.SX.zeros(model.nv)

        def write(slot, value):
            if slot.unbounded:
                configuration[slot.config_index] = ca.cos(value)
                configuration[slot.config_index + 1] = ca.sin(value)
            else:
                configuration[slot.config_index] = value

        for index, slot in enumerate(self.description.joints):
            write(slot, q[index])
        for coupled in self.description.coupled:
            write(coupled.slot, coupled.multiplier * q[coupled.source] + coupled.offset)
        for index, drives in enumerate(self.description.drives):
            for velocity_index, weight in drives:
                velocity[velocity_index] = weight * dq[index]
        return configuration, velocity

    @staticmethod
    def _partition(mass, bias):
        """M and h split by the actuated/passive partition, gathered by index."""
        actuated = list(K_ACTUATED_ROWS)
        passive = list(K_PASSIVE_ROWS)
        return (
            mass[actuated, actuated],
            mass[actuated, passive],
            mass[passive, actuated],
            mass[passive, passive],
            bias[actuated],
            bias[passive],
        )

    def _output_map(self):
        """
        z, the algebraic output of `wiki/nomenclature.md` §10, in four blocks of six.

        `wiki/mpc.md` §3.1 requires both non-smooth pieces of constraint 7 to be
        smoothed for a gradient-based solver: |v| by sqrt(v² + eps²), and the step
        in A^± at v = 0 by a tanh of width eps_v. `Model::transmission` keeps the
        physical step, because it is not being differentiated.
        """
        areas = axis_areas(self.constants)
        dq_a = ca.vertcat(*[self.dq[row] for row in K_ACTUATED_ROWS])

        cylinder_force = []
        piston_velocity = []
        axis_flow = []
        for axis in range(K_ACTUATED_DOF):
            ratio = self.cylinder_jacobian[axis]
            velocity = ratio * dq_a[axis]
            magnitude = ca.sqrt(
                velocity * velocity + self.constants.eps_abs * self.constants.eps_abs
            )
            mean_area = 0.5 * (areas[axis].a_eff_pos + areas[axis].a_eff_neg)
            half_step = 0.5 * (areas[axis].a_eff_pos - areas[axis].a_eff_neg)
            effective_area = mean_area + half_step * ca.tanh(
                velocity / self.constants.eps_v
            )

            cylinder_force.append(self.tau_a[axis] / ratio)
            piston_velocity.append(velocity)
            axis_flow.append(effective_area * magnitude)
        return ca.vertcat(
            self.tau_a,
            ca.vertcat(*cylinder_force),
            ca.vertcat(*piston_velocity),
            ca.vertcat(*axis_flow),
        )

    # -- what a consumer takes away -------------------------------------------

    def chamber_force(self, p_a, p_b):
        """
        F_i = A_A p_A - A_B p_B (`wiki/hydraulics.md` §4).

        For the rotator both areas are V_m, so that entry is a torque in N m
        rather than a force in N.
        """
        areas = axis_areas(self.constants)
        return ca.vertcat(
            *[
                areas[axis].a_a * p_a[axis] - areas[axis].a_b * p_b[axis]
                for axis in range(K_ACTUATED_DOF)
            ]
        )

    def functions(self, prefix: str) -> list:
        """
        Wrap the model as `casadi.Function`s, named `<prefix>_<what>`.

        One evaluation carries the dynamics, the reduction and the output map
        together, because they are one expression graph and splitting them would
        put three copies of `crba` in the generated C. The small maps beside it
        are separate because each is asserted on its own: a projection error can
        cancel in a mass matrix, and a damping override that never arrived is
        invisible in every quantity except the damping.
        """
        q = ca.SX.sym("q", K_GENERALIZED_DOF)
        dq = ca.SX.sym("dq", K_GENERALIZED_DOF)
        p_a = ca.SX.sym("p_a", K_ACTUATED_DOF)
        p_b = ca.SX.sym("p_b", K_ACTUATED_DOF)

        out = [
            ca.Function(
                f"{prefix}_evaluate",
                [self.x, self.u, self.p],
                [
                    ca.densify(self.xdot),
                    ca.densify(self.z),
                    ca.densify(self.mass),
                    ca.densify(self.bias),
                    ca.densify(self.mass_uu),
                    ca.densify(self.mass_ua),
                    ca.densify(self.bias_u),
                ],
                ["x", "u", "p"],
                ["xdot", "z", "mass", "bias", "mass_uu", "mass_ua", "bias_u"],
            ),
            ca.Function(
                f"{prefix}_cylinder_jacobian",
                [q],
                [
                    ca.densify(
                        ca.vertcat(
                            *jacobian_diagonal(
                                self.constants, self.tool, q[1], q[2], q[7]
                            )
                        )
                    )
                ],
                ["q"],
                ["j_cyl"],
            ),
            ca.Function(
                f"{prefix}_chamber_force",
                [p_a, p_b],
                [ca.densify(self.chamber_force(p_a, p_b))],
                ["p_a", "p_b"],
                ["f_cyl"],
            ),
            ca.Function(
                f"{prefix}_damping",
                [dq],
                [ca.densify(ca.vertcat(*self.description.damping) * dq)],
                ["dq"],
                ["d_dq"],
            ),
        ]

        # One function per `<mimic>` the description declares, named after the
        # joint it reconstructs. Pinocchio dropped these, so an export that
        # forgot to put them back builds a model whose telescope is half as long
        # -- and the failure would be a plausible mass matrix rather than a
        # missing symbol. Naming the joint in the symbol is what makes the
        # absence a link error instead.
        for coupled in self.description.coupled:
            out.append(
                ca.Function(
                    f"{prefix}_mimic_{coupled.name}",
                    [q, dq],
                    [
                        ca.densify(
                            ca.vertcat(
                                coupled.multiplier * q[coupled.source] + coupled.offset,
                                coupled.multiplier * dq[coupled.source],
                            )
                        )
                    ],
                    ["q", "dq"],
                    ["mimic"],
                )
            )
        return out
