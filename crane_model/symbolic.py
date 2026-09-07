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

What it builds is the symbolic model used for code generation:

* Pinocchio's own `crba` and `nonLinearEffects`, over `casadi.SX`, on the parsed
  description -- not a restatement of the equations of motion;
* the mimic projection **P**, because Pinocchio drops `<mimic>` and
  `q5_small_telescope` and `q11_right_rail_joint` have to be reconstructed;
* the damping of `wiki/robot_model.md` §1, read out of the description and then
  overridden where `config/hydraulics.yaml` says to -- the table is empty since
  2026-09-07, because the description now carries the same C3 fit `k` comes
  from and a faithful URDF read is the right model again;
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
* `x`, `u` -- what the OCPs plan. Without an actuator that is the rigid
  `NX_RIGID = 14` and `NU = 5`; with one it is C3's `NX = 25` -- the rigid
  fourteen, the command lag, the progress pair of
  `docs/features/mpc-full-authority/brief.md` 2.1 and the force state -- over
  `NU_PROGRESS = 6`. The tool coordinate is
  not a decision variable, because the gripper is opened and closed by the
  low-level controller; it arrives in the parameter vector `p` and its rate and
  acceleration are zero. **Its link keeps its mass**, which is the whole reason
  it is pinned rather than removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import yaml

from .conventions import Tool, default_actuator_path, default_hydraulics_path
from .description import parse as parse_description

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

# The rigid-body half of `x`: what the state was before the actuator arrived,
# and what a consumer built without one still gets.
NX_RIGID = 2 * K_PLANNED_DOF + 2 * K_PASSIVE_DOF

# The planned joint-velocity commands. This is what `u` is for a consumer built
# without an actuator -- `crane_planning`'s timing OCP -- and it stays the name
# of that block after the progress input arrives below, so nothing downstream
# that means "one command per planned axis" has to be re-read.
NU = K_PLANNED_DOF

# The blocks of `x`, in the order `crane_mpc/src/ocp.cpp` reduces to and expands
# from: planned positions, passive positions, planned rates, passive rates.
X_PLANNED_POSITION = 0
X_PASSIVE_POSITION = K_PLANNED_DOF
X_PLANNED_VELOCITY = K_PLANNED_DOF + K_PASSIVE_DOF
X_PASSIVE_VELOCITY = 2 * K_PLANNED_DOF + K_PASSIVE_DOF

# --- C3, `wiki/hydraulic_actuator_model.md` §1 -------------------------------
#
# The fit's own keys for the five planned axes, in `K_PLANNED_AXES` order. The
# gripper is in the file and is not planned here, so it is not in this tuple.
K_AXIS_KEYS = ("sw", "ha", "ka", "sa", "ro")


@dataclass(frozen=True)
class ActuatorFit:
    """
    C3's two fitted numbers per planned axis, in `K_PLANNED_AXES` order.

    `k` is block 3's force-state stiffness and `tau_v` block 2's command lag.
    Block 1, the 60 ms transport delay, is **not** here: it is common to every
    axis and it lives in the node's predictor, not on the shooting grid
    (`docs/features/mpc-full-authority/brief.md` §2.2). Damping is not here
    either -- it reaches the dynamics through pinocchio's `model.damping` off
    the description and stays there. It is the same C3 fit `k` below comes
    from: `d_i` and `k_i` are one identification and only pair with each
    other.
    """

    k: tuple[float, ...]
    tau_v: tuple[float, ...]
    dead_time_s: float

    @property
    def lag_axes(self) -> tuple[int, ...]:
        """The axes that get a `u_f` state. `tau_v = 0` is a pole at infinity."""
        return tuple(axis for axis in K_PLANNED_AXES if self.tau_v[axis] > 0.0)


def load_actuator_fit(path=None) -> ActuatorFit:
    """
    Read `config/c3_full_model.json`, the fit of `hydraulic_actuator_model.md` §2.

    A missing axis or a non-positive `k` is an exception and never a default, for
    the same reason `load_constants` refuses one: a wrong actuator constant is a
    wrong command, and it would be invisible in every quantity except the
    response.
    """
    path = Path(path) if path is not None else Path(default_actuator_path())
    with open(path) as stream:
        root = json.load(stream)
    axes = root.get("axes")
    if not isinstance(axes, dict):
        raise ValueError(f"{path}: carries no axes mapping")

    k, tau_v = [], []
    for key in K_AXIS_KEYS:
        if key not in axes:
            raise ValueError(f"{path}: no fit for axis {key!r}")
        entry = axes[key]
        stiffness = float(entry["k"])
        lag = float(entry["tau_v"])
        if not (stiffness > 0.0) or lag < 0.0:
            raise ValueError(
                f"{path}: axis {key!r} has k={stiffness} tau_v={lag}; k must be "
                f"positive and tau_v non-negative"
            )
        k.append(stiffness)
        tau_v.append(lag)
    return ActuatorFit(
        k=tuple(k),
        tau_v=tuple(tau_v),
        dead_time_s=float(root["dead_time_common_ms"]) * 1.0e-3,
    )


#: The fit as shipped. Read once, at import, because `NX` is derived from it.
K_ACTUATOR_FIT = load_actuator_fit()

#: The axes that carry a command-lag state, derived from the fit the way
#: `K_PLANNED_AXES` is derived from the partition -- never written out by hand.
#: `ka` (the arm) is fitted at `tau_v = 0`, so it has no `u_f` state and its
#: `u_f` *is* `u`.
K_LAG_AXES = K_ACTUATOR_FIT.lag_axes
K_COMMAND_LAG_DOF = len(K_LAG_AXES)

# The C3 state, extending the rigid-body offsets above rather than renumbering
# them: the lagged command on the axes that have one, then the force state on
# every planned axis.
X_COMMAND_LAG = NX_RIGID

# --- the progress pair, `docs/features/mpc-full-authority/brief.md` 2.1 --------
#
# `s` is **virtual time**, in seconds of nominal plan, and `v_s` is how fast the
# plan is being spent -- seconds of plan per second of wall clock, so `v_s = 1`
# is exactly time-indexed tracking. This is the *time-scaling* form and not the
# contouring one: the reference is a spline in `time_from_start`, so there is no
# arclength anywhere and no contour error to decompose. `docs/features/
# corridor-mpc/brief.md` 5.1 carries a progress variable that means spatial
# progress with a linear reward and no tracking term; the two are not composable
# and this is the other one.
#
# The rate is a state and its **acceleration** is the input, one order above
# Marc's `s_dot`-as-input, so the plan's speed cannot step between cycles.
#
# They sit *before* the force state and not at the end of `x` on purpose:
# `v_s >= 0` ("the machine does not run the plan backwards") is a box, the force
# states deliberately carry none, and acados' `idxbx` is only readable as "the
# state row" while the boxed rows are a contiguous prefix.
X_PROGRESS = NX_RIGID + K_COMMAND_LAG_DOF
X_PROGRESS_RATE = X_PROGRESS + 1
K_PROGRESS_DOF = 2

X_ACTUATED_FORCE = X_PROGRESS + K_PROGRESS_DOF
NX = NX_RIGID + K_COMMAND_LAG_DOF + K_PROGRESS_DOF + K_PLANNED_DOF

# The boxed rows of `x`: everything up to the force state, which is bounded by
# `wiki/mpc.md` 3's constraint 6 instead of by a box.
NBX = X_ACTUATED_FORCE

# `u` with the progress acceleration on the end, which is what a consumer with
# an actuator plans. `NU` above stays the joint-command block.
U_PROGRESS_ACCEL = NU
NU_PROGRESS = NU + 1

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
# Four blocks of six, the output-map offsets used by the generated model. It
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

    eps_abs: float = 0.0
    eps_v: float = 0.0

    joints: tuple = ()
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
    path = Path(path) if path is not None else Path(default_hydraulics_path())
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

    out.eps_abs = _number(root, "smoothing.eps_abs", path)
    out.eps_v = _number(root, "smoothing.eps_v", path)

    joints = _at(root, "joints", path)
    if len(joints) != K_GENERALIZED_DOF:
        raise ValueError(f"{path}: joints is not {K_GENERALIZED_DOF} entries")
    out.joints = tuple(joints)

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


def jacobian_diagonal(constants: Constants, q2, q3) -> list:
    """
    Return the diagonal of J_cyl, which is the whole of it.

    The geometry does not couple the axes at all (`wiki/hydraulics.md` §5.7), so
    the off-diagonal entries are structurally zero and are never formed. Only q2
    and q3 enter; the other four axes are a constant radius or one-to-one.
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
    # q8 tool (§2.6): the PZS100 rail cylinder is one-to-one with q8 and there
    # is no linkage to solve.
    diagonal[5] = ca.SX(1.0)
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
    # P of the numeric model, one entry per canonical coordinate: a list
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
    # The canonical mapping -- joint binding, the `<mimic>` scan, P and the
    # per-joint damping -- belongs to `crane_model.description`, which the C++
    # library's Python counterpart uses too. Read there, not restated here.
    shared = parse_description(
        description_xml, Tool(tool), gravity, names=tuple(constants.joints)
    )
    model = shared.model
    canonical = list(shared.names)

    out = Description(
        tool=tool,
        constants=constants,
        model=model,
        neutral=shared.neutral,
    )
    out.joints = [
        JointSlot(drives[0].slot.idx_q, drives[0].slot.idx_v, drives[0].slot.nq == 2)
        for drives in shared.drives
    ]
    for source, drives in enumerate(shared.drives):
        for drive in drives[1:]:
            out.coupled.append(
                CoupledJoint(
                    name="",
                    slot=JointSlot(
                        drive.slot.idx_q, drive.slot.idx_v, drive.slot.nq == 2
                    ),
                    source=source,
                    multiplier=drive.multiplier,
                    offset=drive.offset,
                )
            )
    out.drives = [
        [(drive.slot.idx_v, drive.multiplier) for drive in drives]
        for drives in shared.drives
    ]
    out.damping = list(shared.damping)

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
        actuator: ActuatorFit | None = None,
    ):
        self.constants = constants if constants is not None else load_constants()
        self.description = parse(description_xml, tool, self.constants, gravity)
        self.tool = tool
        self.actuator = actuator

        # --- the symbols -----------------------------------------------------
        # The progress pair rides with the actuator, because the C3 branch is the
        # only consumer that has a `u` to spend time with; `actuator=None` is
        # `crane_planning`'s timing OCP and its `x` and `u` are unchanged.
        self.x = ca.SX.sym("x", NX if actuator is not None else NX_RIGID)
        self.u = ca.SX.sym("u", NU_PROGRESS if actuator is not None else NU)
        self.p = ca.SX.sym("p", NP)
        self.q_tool = self.p[P_TOOL_POSITION]
        self.payload = self.p[P_PAYLOAD_MASS:NP]

        # The canonical eight, put back together out of what this problem plans
        # and the pinned tool coordinate. The tool's rate and its acceleration
        # are zero -- the gripper is held by the low-level controller, so no plan
        # written against this model can move it.
        self.q = ca.SX.zeros(K_GENERALIZED_DOF)
        self.dq = ca.SX.zeros(K_GENERALIZED_DOF)
        for axis, row in enumerate(K_PLANNED_ROWS):
            self.q[row] = self.x[X_PLANNED_POSITION + axis]
            self.dq[row] = self.x[X_PLANNED_VELOCITY + axis]
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

        planned = list(K_PLANNED_AXES)
        self.ddq_a = ca.SX.zeros(K_ACTUATED_DOF)
        if actuator is None:
            # No actuator: the input *is* the actuated acceleration, the passive
            # rows follow it, and `tau_a` is the inverse dynamics that produced
            # it. This is the map `crane_planning`'s timing OCP inverts.
            for axis in K_PLANNED_AXES:
                self.ddq_a[axis] = self.u[axis]

            # `wiki/robot_model.md` §3.1: ddq_u = -M_uu⁻¹ (M_ua ddq_a + h_u),
            # with h_u already carrying D_uu dq_u. Two by two, so the inverse is
            # the adjugate and there is no factorisation to branch on -- a
            # configuration where M_uu is singular leaves a non-finite expression.
            right_hand = h_u + ca.mtimes(m_ua, self.ddq_a)
            determinant = m_uu[0, 0] * m_uu[1, 1] - m_uu[0, 1] * m_uu[1, 0]
            self.ddq_u = ca.vertcat(
                -(m_uu[1, 1] * right_hand[0] - m_uu[0, 1] * right_hand[1])
                / determinant,
                -(m_uu[0, 0] * right_hand[1] - m_uu[1, 0] * right_hand[0])
                / determinant,
            )
            # tau_a of §3.3, the actuated rows of M ddq + h at the consistent
            # passive acceleration. It equals M_eff u + h_eff of §3.4 and is the
            # quantity `wiki/mpc.md` §2's effort term and §3's constraint 6 are
            # written in.
            self.tau_a = h_a + ca.mtimes(m_aa, self.ddq_a) + ca.mtimes(m_au, self.ddq_u)
            self.u_f = self.u
            self.progress = None
            self.progress_rate = None
            self.progress_accel = None
            self.command_lag_rate = ca.SX.zeros(0)
            self.actuated_force_rate = ca.SX.zeros(0)
            self.actuated_force_static = ca.substitute(
                self.tau_a[:K_PLANNED_DOF], self.u, ca.SX.zeros(NU)
            )
            # The rows of xdot in the reduced order. The tool's two rows are
            # dq_tool = 0 and ddq_tool = 0 by construction and are not carried.
            self.xdot = ca.vertcat(
                self.x[X_PLANNED_VELOCITY : X_PLANNED_VELOCITY + K_PLANNED_DOF],
                self.x[X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + K_PASSIVE_DOF],
                self.u,
                self.ddq_u,
            )
        else:
            # C3, `wiki/hydraulic_actuator_model.md` §1, blocks 2, 3 and 5. Block
            # 1, the 60 ms transport delay, is deliberately absent: it is the
            # node's predictor and putting a second copy here double-counts it.
            #
            # Block 2, the PT1 command lag. Only the axes the fit gives a
            # positive tau_v get a state; on the arm tau_v is zero, which is a
            # pole at infinity, so its lagged command is the command itself.
            dq_a = self.x[X_PLANNED_VELOCITY : X_PLANNED_VELOCITY + K_PLANNED_DOF]
            lag_slot = {axis: slot for slot, axis in enumerate(K_LAG_AXES)}
            self.u_f = ca.vertcat(
                *[
                    self.x[X_COMMAND_LAG + lag_slot[axis]]
                    if axis in lag_slot
                    else self.u[axis]
                    for axis in K_PLANNED_AXES
                ]
            )
            self.command_lag_rate = ca.vertcat(
                *[
                    (self.u[axis] - self.u_f[axis]) / actuator.tau_v[axis]
                    for axis in K_LAG_AXES
                ]
            )

            # Block 3, the force state. It is a *state* now, so `tau_a` is read
            # off `x` rather than assembled -- which is what turns `wiki/mpc.md`
            # §3's constraint 6 into `x_j / J_c,ii(q)` and drops crba out of it.
            tau_p = self.x[X_ACTUATED_FORCE : X_ACTUATED_FORCE + K_PLANNED_DOF]
            self.actuated_force_rate = ca.vertcat(
                *[
                    actuator.k[axis] * (self.u_f[axis] - dq_a[axis])
                    for axis in K_PLANNED_AXES
                ]
            )

            # Block 5, the load: forward dynamics, torque to acceleration. This
            # is ABA's arrow, but **not** `cpin.aba`: the parsed model has 18
            # velocity rows and a `<mimic>` projection P over them, so ABA on it
            # would let the four cylinder sub-chains and the mimicked telescope
            # half move freely. The projected system P'MP ddq + P'h = tau is what
            # `self.mass`/`self.bias` already are, so forward dynamics here is
            # one dense solve of it -- seven rows, because the tool coordinate is
            # pinned at ddq = 0 and its row is a constraint torque nobody reads.
            m_pp = m_aa[planned, planned]
            m_pu = m_au[planned, :]
            system = ca.blockcat([[m_pp, m_pu], [self.mass_ua, m_uu]])
            solution = ca.solve(system, ca.vertcat(tau_p - h_a[planned], -h_u))
            ddq_p = solution[:K_PLANNED_DOF]
            self.ddq_u = solution[K_PLANNED_DOF:]
            for axis in K_PLANNED_AXES:
                self.ddq_a[axis] = ddq_p[axis]

            # The tool row of tau_a is still the constraint torque that holds the
            # gripper still, and the output map reports all six. Only the planned
            # five come out of the state.
            tau_tool = (
                h_a[K_TOOL_AXIS]
                + ca.mtimes(m_aa[K_TOOL_AXIS, :], self.ddq_a)
                + ca.mtimes(m_au[K_TOOL_AXIS, :], self.ddq_u)
            )
            self.tau_a = ca.vertcat(tau_p, tau_tool)

            # The force state that holds the machine still at this configuration,
            # i.e. h_eff of `wiki/robot_model.md` §3.4. There is no force
            # measurement anywhere in the stack, so this is what a consumer seeds
            # the force state with; seeding it at zero would start every horizon
            # with the hydraulics switched off and the boom in free fall.
            determinant = m_uu[0, 0] * m_uu[1, 1] - m_uu[0, 1] * m_uu[1, 0]
            ddq_u_static = ca.vertcat(
                -(m_uu[1, 1] * h_u[0] - m_uu[0, 1] * h_u[1]) / determinant,
                -(m_uu[0, 0] * h_u[1] - m_uu[1, 0] * h_u[0]) / determinant,
            )
            self.actuated_force_static = h_a[planned] + ca.mtimes(m_pu, ddq_u_static)

            # The progress pair: `s` advances at `v_s` and `v_s` is driven by the
            # sixth input. Nothing else in the plant reads either row -- the
            # coupling to the machine is entirely through the cost, which is
            # where the reference is evaluated at `s`.
            self.progress = self.x[X_PROGRESS]
            self.progress_rate = self.x[X_PROGRESS_RATE]
            self.progress_accel = self.u[U_PROGRESS_ACCEL]

            self.xdot = ca.vertcat(
                self.x[X_PLANNED_VELOCITY : X_PLANNED_VELOCITY + K_PLANNED_DOF],
                self.x[X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + K_PASSIVE_DOF],
                ddq_p,
                self.ddq_u,
                self.command_lag_rate,
                self.progress_rate,
                self.progress_accel,
                self.actuated_force_rate,
            )

        self.cylinder_jacobian = ca.vertcat(
            *jacobian_diagonal(self.constants, self.q[1], self.q[2])
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
        neutral value and zero velocity: the four cylinder sub-chains are loops
        the description closes only in Gazebo.
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
                        ca.vertcat(*jacobian_diagonal(self.constants, q[1], q[2]))
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
