"""A MuJoCo plant off the same URDF, for tuning the planner and the MPC offline."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from .conventions import (
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    canonical_joints,
)
from .errors import CraneModelError, ErrorCode
from .linkage import DISPLAY_JOINTS, cylinder_joints, load_geometry

PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]

#: `symbolic.py`'s rigid state: q_a(5), q_u(2), dq_a(5), dq_u(2).
NX_RIGID = 14

#: MuJoCo's viewer segfaults on interpreter teardown; `leave` skips it (exit 139).
_VIEWER_OPENED = False

#: Frames per second fed to the viewer. `_show` costs ~4 ms -- run at the 100 Hz
#: control tick that is 40% of the wall clock, spent on frames no screen shows.
_FRAME_RATE = 60.0

_CAMERA = {
    "fovy": 60.0,
    "lookat": (1.5, 0.5, 1.0),
    "azimuth": 100.0,
    "elevation": -20.0,
    "distance": 16.7,
}

_MESH_URI = re.compile(r'filename="package://([^/"]+)/([^"]+)"')
_ROBOT_TAG = re.compile(r"(<robot\b[^>]*>)")

#: MuJoCo rejects these on a `fixed` joint; URDF ignores them.
_MOVABLE_ONLY = ("axis", "limit", "dynamics", "mimic", "safety_controller")


def leave(status: int) -> int:
    """Exit `status`, skipping teardown if a viewer ran. See `_VIEWER_OPENED`."""
    sys.stdout.flush()
    sys.stderr.flush()
    if _VIEWER_OPENED:
        os._exit(status)
    return status


def to_mujoco_xml(description_xml: str, hydraulics_config: str | None = None) -> str:
    """Give the URDF MuJoCo will accept: meshes resolved, linkages neutralised."""
    tree = ET.fromstring(description_xml)
    names = tuple(canonical_joints(hydraulics_config))
    driven = set(names) | {
        joint.get("name")
        for joint in tree.iter("joint")
        if (joint.find("mimic") is not None)
        and joint.find("mimic").get("joint") in names
    }
    # URDF is a tree: boom/arm cylinders are dangling chains, not closed loops --
    # left movable they swing free. Pinocchio also leaves them at this neutral.
    # Frozen they weld into their parent: right inertia, and they render detached.
    keep = driven
    for joint in tree.iter("joint"):
        if joint.get("type") == "fixed" or joint.get("name") in keep:
            continue
        joint.set("type", "fixed")
        for tag in _MOVABLE_ONLY:
            for child in joint.findall(tag):
                joint.remove(child)

    compiler = (
        "<mujoco><compiler inertiafromgeom='false' balanceinertia='true' "
        "strippath='false' discardvisual='false'/></mujoco>"
    )
    xml, count = _ROBOT_TAG.subn(
        r"\1\n" + compiler, ET.tostring(tree, encoding="unicode"), count=1
    )
    if count != 1:
        raise CraneModelError(ErrorCode.INVALID_ROBOT_DESCRIPTION, "no <robot> element")

    # MuJoCo has no `package://` resolver, and the description names two.
    shares: dict[str, str] = {}

    def absolute(match: re.Match) -> str:
        package = match.group(1)
        if package not in shares:
            from ament_index_python.packages import get_package_share_directory

            shares[package] = get_package_share_directory(package)
        return f'filename="{shares[package]}/{match.group(2)}"'

    return _MESH_URI.sub(absolute, xml)


_SCENE_XML = """
<mujoco model='scene'>
  <asset>
    <texture name='sky' type='skybox' builtin='gradient' rgb1='0.5 0.6 0.8'
             rgb2='0 0 0' width='256' height='256'/>
    <texture name='ground_tex' type='2d' builtin='checker' rgb1='0.6 0.6 0.6'
             rgb2='0.4 0.4 0.4' width='512' height='512'/>
    <material name='ground_mat' texture='ground_tex' texrepeat='10 10'/>
  </asset>
  <worldbody>
    <light name='sun' pos='0 0 20' dir='0 0.5 -1' directional='true'
           diffuse='1 1 1' ambient='0.5 0.5 0.5'/>
    <geom name='ground' type='plane' pos='0 0 -1.1407' size='10 10 0.1'
          material='ground_mat' contype='0' conaffinity='0'/>
  </worldbody>
</mujoco>
"""


def _cylinder_chain(mujoco, model, description_xml: str):
    """Body, joint name, kind and axis for each cylinder joint `to_mujoco_xml` froze."""
    chain = []
    for joint in ET.fromstring(description_xml).iter("joint"):
        if joint.get("name") not in DISPLAY_JOINTS:
            continue
        body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, joint.find("child").get("link")
        )
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        chain.append((body, joint.get("name"), joint.get("type") == "prismatic", axis))
    return chain


def with_scenery(mujoco, xml: str):
    """Attach `xml` into `_SCENE_XML` and compile: floor, sky and light."""
    scene = mujoco.MjSpec.from_string(_SCENE_XML)
    crane = mujoco.MjSpec.from_string(xml)
    scene.attach(crane, frame=scene.worldbody.add_frame(), prefix="")
    return scene.compile()


class MujocoPlant:
    """One crane, integrated by MuJoCo, addressed in the canonical eight."""

    def __init__(
        self,
        description_xml: str,
        timestep: float = 5.0e-4,
        hydraulics_config: str | None = None,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = with_scenery(
            mujoco, to_mujoco_xml(description_xml, hydraulics_config)
        )
        self.model.opt.timestep = float(timestep)
        # Boom damping is 1.66e5; Euler takes that explicitly and caps the step.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # A `<mimic>` is rigid; MuJoCo's default equality is a 20 ms spring-damper.
        # Left soft, q5 holds its own share: telescope at -3.3 m/s^2 standing still.
        self.model.eq_solref[:] = [2.0 * self.model.opt.timestep, 1.0]
        self.model.eq_solimp[:, :2] = 0.9999
        # MuJoCo files URDF visuals in group 1 and collision geoms in group 0,
        # and renders both: park collision in 3, which the viewer hides. Before
        # the line below, which is what still tells the two apart.
        self.model.geom_group[self.model.geom_contype != 0] = 3
        # Clearance is proven against Coal; MuJoCo must not re-decide it.
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.data = mujoco.MjData(self.model)

        # By name: the two stacks publish different joint sets in different orders.
        slots = [self._slot(name) for name in canonical_joints(hydraulics_config)]
        self._qpos = np.array([slot[0] for slot in slots])
        self._dof = np.array([slot[1] for slot in slots])
        self._joint = np.array([slot[2] for slot in slots])
        self._planned_dof = self._dof[list(PLANNED_INDICES)]
        self._planned_qpos = self._qpos[list(PLANNED_INDICES)]
        self._diverged = 0
        self._residual = 0.0
        self._viewer = None
        self._scratch = None
        self._realtime = 1.0
        self._deadline = 0.0
        self._frame = 0.0
        # Frozen, MuJoCo welds each cylinder link in at its neutral offset, which
        # is the URDF origin with an identity rotation. `_pose_cylinders` writes
        # the closed linkage's offset over it; these are what it writes back.
        self._geometry = load_geometry(hydraulics_config)
        self._chain = _cylinder_chain(mujoco, self.model, description_xml)
        self._welded = np.array([body for body, _, _, _ in self._chain])
        self._neutral = (
            self.model.body_pos[self._welded].copy(),
            self.model.body_quat[self._welded].copy(),
        )
        self.forward()

    def _slot(self, name: str) -> tuple[int, int, int]:
        joint = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise CraneModelError(
                ErrorCode.MISSING_JOINT,
                f"the description does not carry the canonical joint '{name}'",
            )
        return (
            int(self.model.jnt_qposadr[joint]),
            int(self.model.jnt_dofadr[joint]),
            int(joint),
        )

    # --- state ---------------------------------------------------------------

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self._qpos].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self._dof].copy()

    @property
    def projection(self) -> np.ndarray:
        """`symbolic.py`'s P: MuJoCo's dofs onto the canonical eight."""
        matrix = np.zeros((self.model.nv, GENERALIZED_DOF))
        matrix[self._dof, range(GENERALIZED_DOF)] = 1.0
        for index in range(self.model.neq):
            follower, driver = self.model.eq_obj1id[index], self.model.eq_obj2id[index]
            if self.model.eq_type[index] != self._mj.mjtEq.mjEQ_JOINT or driver < 0:
                continue
            column = int(np.flatnonzero(self._dof == self.model.jnt_dofadr[driver])[0])
            matrix[self.model.jnt_dofadr[follower], column] = self.model.eq_data[index][
                1
            ]
        return matrix

    @property
    def state(self) -> np.ndarray:
        """The rigid fourteen, in `crane_model.symbolic`'s order."""
        q, dq = self.q, self.dq
        return np.concatenate(
            [
                q[list(PLANNED_INDICES)],
                q[list(PASSIVE_INDICES)],
                dq[list(PLANNED_INDICES)],
                dq[list(PASSIVE_INDICES)],
            ]
        )

    def _write(self, data, q, dq=None) -> None:
        """Place the canonical eight in `data`, mimic followers included."""
        q = np.asarray(q, dtype=float).reshape(GENERALIZED_DOF)
        data.qpos[self._qpos] = q
        data.qvel[self._dof] = (
            0.0 if dq is None else np.asarray(dq, dtype=float).reshape(GENERALIZED_DOF)
        )
        # A mimic is an equality, not an assignment: a stale follower fights it.
        for index in range(self.model.neq):
            follower, driver = self.model.eq_obj1id[index], self.model.eq_obj2id[index]
            if self.model.eq_type[index] != self._mj.mjtEq.mjEQ_JOINT or driver < 0:
                continue
            offset, multiplier = self.model.eq_data[index][:2]
            position, rate = self.model.jnt_qposadr, self.model.jnt_dofadr
            data.qpos[position[follower]] = (
                offset + multiplier * data.qpos[position[driver]]
            )
            data.qvel[rate[follower]] = multiplier * data.qvel[rate[driver]]

    def set_state(self, q, dq=None) -> None:
        self._write(self.data, q, dq)
        self.forward()

    def positions_of(self, body: str, q_rows) -> np.ndarray:
        """
        World position of `body` at each canonical-eight row. FK, nothing steps.

        On a scratch MjData: the plant is mid-run and the viewer renders its own.
        """
        rows = np.atleast_2d(np.asarray(q_rows, dtype=float))
        index = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, body)
        if index < 0:
            raise CraneModelError(
                ErrorCode.INVALID_ROBOT_DESCRIPTION,
                f"the description carries no body '{body}'",
            )
        if self._scratch is None:
            self._scratch = self._mj.MjData(self.model)
        out = np.empty((len(rows), 3))
        for slot, row in enumerate(rows):
            self._write(self._scratch, row)
            self._mj.mj_kinematics(self.model, self._scratch)
            out[slot] = self._scratch.xpos[index]
        return out

    def set_rigid_state(self, x, q_tool: float) -> None:
        x = np.asarray(x, dtype=float).reshape(NX_RIGID)
        q, dq = np.zeros(GENERALIZED_DOF), np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)], q[list(PASSIVE_INDICES)] = x[0:5], x[5:7]
        q[TOOL_INDEX] = q_tool
        dq[list(PLANNED_INDICES)], dq[list(PASSIVE_INDICES)] = x[7:12], x[12:14]
        self.set_state(q, dq)

    def forward(self) -> None:
        self._mj.mj_forward(self.model, self.data)

    # --- viewer --------------------------------------------------------------

    @property
    def viewer(self):
        """The passive viewer while one is open, else None."""
        return self._viewer

    def open_viewer(self, realtime: float = 1.0, key_callback=None):
        """
        Open the passive viewer. It renders; it never steps. See `leave`.

        `key_callback` is handed each key press on the viewer's own thread --
        it runs there, so it sets a flag and returns, it does not drive anything.
        """
        import mujoco.viewer

        global _VIEWER_OPENED
        _VIEWER_OPENED = True
        # fovy is a model property, so it has to be set before the scene is built.
        self.model.vis.global_.fovy = _CAMERA["fovy"]
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_callback
        )
        with self._viewer.lock():
            self._viewer.cam.lookat[:] = _CAMERA["lookat"]
            self._viewer.cam.azimuth = _CAMERA["azimuth"]
            self._viewer.cam.elevation = _CAMERA["elevation"]
            self._viewer.cam.distance = _CAMERA["distance"]
        self._realtime = float(realtime)
        self._deadline = self._frame = time.perf_counter()
        return self._viewer

    def _pose_cylinders(self) -> bool:
        """
        Swing the frozen cylinder links out to where the closed linkage puts them.

        Reproduces the joints MuJoCo no longer has, to float precision: each
        offset is a rotation about the joint axis or a slide along it.
        """
        placed = cylinder_joints(self._geometry, self.q[1], self.q[2])
        # Outside the four-bar's reach the coupler point does not exist; a NaN
        # offset would take the whole model with it. Leave it welded.
        if not np.isfinite(list(placed.values())).all():
            return False
        for row, (body, name, prismatic, axis) in enumerate(self._chain):
            if prismatic:
                self.model.body_pos[body] = self._neutral[0][row] + placed[name] * axis
            else:
                self._mj.mju_axisAngle2Quat(
                    self.model.body_quat[body], axis, placed[name]
                )
        self._mj.mj_kinematics(self.model, self.data)
        return True

    def _show(self) -> None:
        # `sync` writes the ctrl-drag wrench into `xfrc_applied` and leaves the
        # last one there on release: clear it first, so a pull lasts exactly as
        # long as the mouse is down.
        self.data.xfrc_applied[:] = 0.0
        # Picture only: a link's placement feeds the composite inertia, so the
        # weld goes back before anything steps again.
        posed = self._pose_cylinders()
        self._viewer.sync()
        if posed:
            self.model.body_pos[self._welded] = self._neutral[0]
            self.model.body_quat[self._welded] = self._neutral[1]

    def hold_viewer(self, until=None) -> bool:
        """
        Keep rendering until `until()` fires or the window goes away.

        True means `until` fired and the window is still there; False that it
        closed, which is the caller's signal to stop feeding it.
        """
        try:
            while self._viewer is not None and self._viewer.is_running():
                if until is not None and until():
                    return True
                self._show()
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return False

    def _render(self, duration: float) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        # Capped, not once per call: the caller ticks at the control rate, which
        # is faster than anything can be looked at and `_show` is not cheap.
        now = time.perf_counter()
        if now - self._frame >= 1.0 / _FRAME_RATE:
            self._frame = now
            self._show()
        if self._realtime <= 0.0:
            return
        self._deadline += duration / self._realtime
        remaining = self._deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            # Behind: drop the lost time rather than sprint, or screen != simulated.
            self._deadline = time.perf_counter()

    # --- driving it ----------------------------------------------------------

    def _advance(self, duration: float, force) -> None:
        # MuJoCo steps only by `opt.timestep`; the remainder is carried, not
        # dropped -- a dropped 0.25 ms on a 0.1 ms step loses a fifth per interval.
        self._residual += duration
        steps = int(self._residual / self.model.opt.timestep + 1.0e-9)
        self._residual -= steps * self.model.opt.timestep
        for _ in range(steps):
            self.data.qfrc_applied[:] = 0.0
            force()
            self._mj.mj_step(self.model, self.data)
        # On divergence MuJoCo resets the body and carries on: only the counter tells.
        bad = int(self.data.warning[self._mj.mjtWarning.mjWARN_BADQACC].number)
        if bad != self._diverged:
            self._diverged = bad
            raise CraneModelError(
                ErrorCode.SINGULAR_CONFIGURATION,
                "the plant diverged -- lower the drive bandwidth or the timestep",
            )
        self._render(duration)

    def _resistance(self, dof):
        """Gravity and Coriolis, own damping, and the mimic's pull, at `dof`."""
        return (
            self.data.qfrc_bias[dof]
            - self.data.qfrc_passive[dof]
            - self.data.qfrc_constraint[dof]
        )

    @property
    def holding_force(self) -> np.ndarray:
        """
        What the five planned axes need to stand still in this pose.

        Seeds C3's force state: the cylinders hold the crane up before anyone
        commands anything, so a run started at tau = 0 opens by dropping the
        boom. Same role as `effort_0` in the Gazebo description.

        Not `qfrc_bias` at the planned rows. Two things break that: the mimicked
        telescope half carries its own share, which only the projection picks up,
        and the passive pair is *not* held, so the seed is the one that leaves
        ddq_a = 0 while the sway rows accelerate freely. Same solve as
        `symbolic.py`'s static seed -- `h_a + M_au ddq_u`, `ddq_u = -M_uu^-1 h_u`
        -- and it has to be, or the two disagree at t = 0.
        """
        inertia = np.zeros((self.model.nv, self.model.nv))
        self._mj.mj_fullM(self.model, self.data, inertia)
        projection = self.projection
        mass = projection.T @ inertia @ projection
        bias = projection.T @ (self.data.qfrc_bias - self.data.qfrc_passive)
        planned, passive = list(PLANNED_INDICES), list(PASSIVE_INDICES)
        sway = np.linalg.solve(mass[np.ix_(passive, passive)], -bias[passive])
        return bias[planned] + mass[np.ix_(planned, passive)] @ sway

    @property
    def effort_limits(self) -> np.ndarray:
        """
        The description's `effort` on the five planned axes -- C3's block 4.

        MuJoCo parses URDF `effort` into `jnt_actfrcrange`, but enforces it only
        on real `<actuator>` elements and this plant has none: it drives
        `qfrc_applied`, which is unbounded. So this binds nothing until someone
        hands it to `C3Actuator(tau_max=...)`, and an unclamped force state is
        not C3 -- it is C3 with block 4 deleted.
        """
        rows = self.model.jnt_actfrcrange[self._joint[list(PLANNED_INDICES)]]
        return np.abs(np.asarray(rows)).max(axis=1)

    @property
    def continuous_axes(self) -> np.ndarray:
        """
        Which planned axes wrap -- an unlimited hinge, `type="continuous"`.

        The rotator is one, and it is the reason this exists: the JTC takes its
        position error through `angles::shortest_angular_distance`, so a loop
        that subtracts plainly commands the opposite direction once the axis has
        been spun past pi. `VelocityLoop` needs this to stay a transcription.
        """
        joints = self._joint[list(PLANNED_INDICES)]
        hinge = self.model.jnt_type[joints] == self._mj.mjtJoint.mjJNT_HINGE
        return np.asarray(hinge & (self.model.jnt_limited[joints] == 0))

    def step(self, tau_planned, duration: float) -> np.ndarray:
        """
        Hold a generalized force on the five planned axes for `duration`.

        N on prismatic, N*m on rotary -- C3's `X_ACTUATED_FORCE` rows as they
        are. Via `qfrc_applied`, not an actuator, which would add a transmission.
        """
        tau = np.asarray(tau_planned, dtype=float).reshape(len(PLANNED_INDICES))

        def apply() -> None:
            self.data.qfrc_applied[self._planned_dof] = tau

        self._advance(duration, apply)
        return self.state

    def drive(self, actuator, u, duration: float) -> np.ndarray:
        """
        Hold the velocity command `u` for `duration`, through C3.

        Zero-order hold, as ros2_control gives it: `u` is constant over the
        controller tick while `actuator` integrates at the plant step. The
        actuator's step must be this plant's, or the dead time is not what it
        says -- pass `MujocoPlant.model.opt.timestep` when constructing it.
        """
        u = np.asarray(u, dtype=float).reshape(len(PLANNED_INDICES))

        def apply() -> None:
            tau = actuator.step(u, self.data.qvel[self._planned_dof])
            self.data.qfrc_applied[self._planned_dof] = tau

        self._advance(duration, apply)
        return self.state

    def follow(
        self,
        q_ref,
        dq_ref,
        ddq_ref,
        duration: float,
        bandwidth_rad_s: float = 100.0,
    ) -> np.ndarray:
        """
        Drive the five planned axes along a reference and let the load swing.

        `M_ii (ddq_ref + w^2 e + 2 w edot)` over cancelled resistance, one
        bandwidth, no axis tuned by hand; a pure PD understates the sway. Tool
        left unheld: its gravity bias is zero only because `dh_trans9` and
        `dh_trans11` mirror each other (roll +-pi/2, y -+0.08) at mimic
        multiplier 1, so the two 100 kg rails cancel at every pose -- change a
        mass or an rpy and that goes silently.
        """
        q_ref = np.asarray(q_ref, dtype=float).reshape(len(PLANNED_INDICES))
        dq_ref = np.asarray(dq_ref, dtype=float).reshape(len(PLANNED_INDICES))
        ddq_ref = np.asarray(ddq_ref, dtype=float).reshape(len(PLANNED_INDICES))
        inertia = np.zeros((self.model.nv, self.model.nv))
        omega = float(bandwidth_rad_s)

        def apply() -> None:
            self._mj.mj_fullM(self.model, self.data, inertia)
            mass = np.abs(np.diag(inertia)[self._planned_dof])
            error = q_ref - self.data.qpos[self._planned_qpos]
            rate = dq_ref - self.data.qvel[self._planned_dof]
            self.data.qfrc_applied[self._planned_dof] = mass * (
                ddq_ref + omega * omega * error + 2.0 * omega * rate
            ) + self._resistance(self._planned_dof)

        self._advance(duration, apply)
        return self.state


# --- main -------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DESCRIPTION = os.path.join(_HERE, "../test/description/pzs100.urdf")


def main(argv: list[str] | None = None) -> int:
    """
    Open the viewer on `presets.OUTSIDE` and hold it there.

    Held by the shipped velocity loop through C3, not pinned: the pose is a
    setpoint the controller has to keep, so what the viewer shows is the loop
    standing still under gravity, droop and all.
    """
    from .actuator import C3Actuator
    from .presets import OUTSIDE
    from .velocity_loop import VelocityLoop, load_velocity_loop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        "-d",
        default=_DEFAULT_DESCRIPTION,
        help="URDF file to load (default: %(default)s)",
    )
    parser.add_argument(
        "--hydraulics",
        "-H",
        default=None,
        help="Hydraulics config YAML file (default: none)",
    )
    parser.add_argument("--realtime", type=float, default=1.0)
    arguments = parser.parse_args(argv)

    with open(arguments.description, encoding="utf-8") as handle:
        description_xml = handle.read()

    plant = MujocoPlant(description_xml, hydraulics_config=arguments.hydraulics)
    plant.set_state(OUTSIDE)

    gains_by_joint, rate_hz = load_velocity_loop()
    names = canonical_joints(arguments.hydraulics)
    control_step = 1.0 / rate_hz
    # Seeded with the pose's holding force: from tau = 0 the boom drops before
    # the loop has an error to answer.
    actuator = C3Actuator(
        timestep=plant.model.opt.timestep, tau_max=plant.effort_limits
    )
    actuator.reset(tau=plant.holding_force)
    loop = VelocityLoop(
        [gains_by_joint[names[index]] for index in PLANNED_INDICES],
        control_step,
        continuous=plant.continuous_axes,
    )

    planned = list(PLANNED_INDICES)
    q_ref = plant.q[planned].copy()
    dq_ref = np.zeros_like(q_ref)

    viewer = plant.open_viewer(arguments.realtime)
    try:
        while viewer.is_running():
            command = loop.step(q_ref, dq_ref, plant.q[planned], plant.dq[planned])
            plant.drive(actuator, command, control_step)
    except KeyboardInterrupt:
        pass
    viewer.close()
    return leave(0)


if __name__ == "__main__":
    raise SystemExit(main())
