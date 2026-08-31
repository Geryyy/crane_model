"""
Collision geometry and distance queries, on Coal.

The fitted link primitives and the self pairs not worth checking are read from
config/collision_model.yaml. The C++ library compiles the same table in because
it is ROS-free and cannot resolve a `package://` mesh URI; Python has no such
constraint and reads the file.

Every *placement* still comes from the description at runtime: each primitive is
attached to the link frame Pinocchio parsed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import coal
import numpy as np
import pinocchio as pin
import yaml

from .conventions import Frame, Tool
from .errors import CraneModelError, ErrorCode


@dataclass
class CollisionPrimitive:
    """
    One convex body in K0_mounting_base.

    `dimensions_m` is the extent along each axis of the primitive's own frame --
    not a half extent and not a radius. A box carries its three side lengths, a
    cylinder (2r, 2r, length), and a sphere its diameter three times.
    """

    id: str
    shape: str  # box | cylinder | sphere
    pose_in_mounting_base: pin.SE3 = field(default_factory=pin.SE3.Identity)
    dimensions_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    structural: bool = False
    #: This body is held by the tool. It is then **not** checked against the
    #: links that hold it -- they are gripping it, and reporting that as a
    #: collision refuses every pose -- and it **is** checked against the rest of
    #: the scene, which a scene body otherwise never is. A caller that carries
    #: something has to place it: the pose is read at the configuration being
    #: checked, so it travels with the tool rather than standing where the
    #: machine set off from.
    attached_to_tool: bool = False


@dataclass(frozen=True)
class CollisionResult:
    collision: bool
    minimum_distance_m: float
    other_id: str
    witness_on_robot_m: np.ndarray
    witness_on_other_m: np.ndarray


def default_collision_model_path() -> str:
    """Locate config/collision_model.yaml, installed first, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("crane_model"), "config", "collision_model.yaml"
        )
    except Exception:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, "config", "collision_model.yaml")


def _scene_geometry(primitive: CollisionPrimitive):
    extents = np.asarray(primitive.dimensions_m, dtype=float).reshape(-1)
    if extents.size != 3 or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        raise CraneModelError(
            ErrorCode.INVALID_SCENE, f"'{primitive.id}' has no finite positive extents"
        )
    if primitive.shape == "box":
        return coal.Box(*extents)
    # A cylinder or sphere whose extents disagree is refused rather than
    # silently reinterpreted, so a caller who passed a radius is told.
    if primitive.shape == "cylinder":
        if not np.isclose(extents[0], extents[1]):
            raise CraneModelError(
                ErrorCode.INVALID_SCENE,
                f"cylinder '{primitive.id}' has disagreeing cross-section extents",
            )
        return coal.Cylinder(0.5 * extents[0], extents[2])
    if primitive.shape == "sphere":
        if not (
            np.isclose(extents[0], extents[1]) and np.isclose(extents[1], extents[2])
        ):
            raise CraneModelError(
                ErrorCode.INVALID_SCENE,
                f"sphere '{primitive.id}' has disagreeing extents",
            )
        return coal.Sphere(0.5 * extents[0])
    raise CraneModelError(ErrorCode.INVALID_SCENE, f"unknown shape '{primitive.shape}'")


def _transform(pose: pin.SE3) -> coal.Transform3s:
    return coal.Transform3s(pose.rotation, pose.translation)


def _distance(robot, robot_pose, other, other_pose, other_id: str) -> CollisionResult:
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    result = coal.DistanceResult()
    try:
        coal.distance(
            robot,
            _transform(robot_pose),
            other,
            _transform(other_pose),
            request,
            result,
        )
    except Exception as exc:
        raise CraneModelError(
            ErrorCode.COLLISION_BACKEND_FAILURE,
            f"Coal failed the query against {other_id}: {exc}",
        ) from exc
    near = np.array(result.getNearestPoint1()), np.array(result.getNearestPoint2())
    if not (
        np.isfinite(result.min_distance) and np.all(np.isfinite(np.concatenate(near)))
    ):
        raise CraneModelError(
            ErrorCode.COLLISION_BACKEND_FAILURE,
            f"Coal returned no usable distance for {other_id}",
        )
    return CollisionResult(
        collision=bool(result.min_distance < 0.0),
        minimum_distance_m=float(result.min_distance),
        other_id=other_id,
        witness_on_robot_m=near[0],
        witness_on_other_m=near[1],
    )


class LinkGeometry:
    """The machine's own fitted primitives, bound to the parsed description."""

    def __init__(self, model, tool: Tool, path: str | None = None) -> None:
        with open(path or default_collision_model_path(), encoding="utf-8") as handle:
            table = yaml.safe_load(handle)

        self.bodies = []
        for entry in table["primitives"]:
            if entry["tool"] != tool.value or not model.existFrame(entry["link"]):
                continue
            extents = entry["extents_m"]
            if entry["shape"] == "capsule":
                geometry = coal.Capsule(extents[0], 2.0 * extents[1])
            else:
                geometry = coal.Box(*(2.0 * np.asarray(extents, dtype=float)))
            quaternion = pin.Quaternion(*np.roll(entry["orientation_xyzw"], 1))
            placement = pin.SE3(
                quaternion.matrix(), np.asarray(entry["origin_m"], dtype=float)
            )
            self.bodies.append(
                (entry["link"], model.getFrameId(entry["link"]), geometry, placement)
            )

        # Which of those bodies are the hand rather than the arm. A payload is
        # excluded from exactly these, and the set is derived from the
        # description rather than named here: a body is carrying if the joint
        # that places it is at or below the joint a payload mounts on.
        mount = None
        if model.existFrame(Frame.ROTATOR_LOWER_PART.value):
            mount = model.frames[
                model.getFrameId(Frame.ROTATOR_LOWER_PART.value)
            ].parentJoint
        self.detached = [
            index
            for index, (_, frame, _, _) in enumerate(self.bodies)
            if mount is None
            or mount not in list(model.supports[model.frames[frame].parentJoint])
        ]

        allowed = {tuple(sorted(pair)) for pair in table["allowed_self_pairs"]}
        self.self_pairs = [
            (i, j)
            for i in range(len(self.bodies))
            for j in range(i + 1, len(self.bodies))
            if tuple(sorted((self.bodies[i][0], self.bodies[j][0]))) not in allowed
            and self.bodies[i][0] != self.bodies[j][0]
        ]


def validate_scene(primitives) -> None:
    """Ids must be non-empty and unique; the model never repairs a scene."""
    seen = set()
    for primitive in primitives:
        if not primitive.id:
            raise CraneModelError(
                ErrorCode.INVALID_SCENE, "a scene primitive has an empty id"
            )
        if primitive.id in seen:
            raise CraneModelError(
                ErrorCode.INVALID_SCENE,
                f"the scene carries '{primitive.id}' more than once",
            )
        seen.add(primitive.id)


def queries(model, data, geometry: LinkGeometry, configuration, scene) -> list:
    """
    Check the scene and the machine against itself.

    One result per scene primitive, in scene order, then one for the machine
    against itself. Each is the closest pair found against that object.

    A primitive marked `attached_to_tool` is answered differently, and the two
    halves go together. It is not checked against the links that hold it, which
    would report the grip itself as a collision and refuse every pose. And it
    *is* checked against the other scene primitives, which an ordinary scene
    body never is -- because what a carried body is for is hitting the world,
    and a payload checked only against the machine that carries it is a payload
    that has not been checked at all.
    """
    validate_scene(scene)
    pin.forwardKinematics(model, data, configuration)
    pin.updateFramePlacements(model, data)
    base = data.oMf[model.getFrameId(Frame.MOUNTING_BASE.value)]
    placed = [
        (link, shape, base.actInv(data.oMf[frame]) * offset)
        for link, frame, shape, offset in geometry.bodies
    ]
    detached = [placed[index] for index in geometry.detached]
    shapes = {other.id: _scene_geometry(other) for other in scene}

    def closest(candidates):
        return min(candidates, key=lambda result: result.minimum_distance_m)

    results = []
    for other in scene:
        pairs = [
            _distance(
                shape,
                pose,
                shapes[other.id],
                other.pose_in_mounting_base,
                other.id,
            )
            for _, shape, pose in (detached if other.attached_to_tool else placed)
        ]
        if other.attached_to_tool:
            pairs += [
                _distance(
                    shapes[other.id],
                    other.pose_in_mounting_base,
                    shapes[obstacle.id],
                    obstacle.pose_in_mounting_base,
                    obstacle.id,
                )
                for obstacle in scene
                if obstacle.id != other.id and not obstacle.attached_to_tool
            ]
        if pairs:
            results.append(closest(pairs))
    if geometry.self_pairs:
        results.append(
            closest(
                _distance(
                    placed[i][1],
                    placed[i][2],
                    placed[j][1],
                    placed[j][2],
                    f"{placed[i][0]}|{placed[j][0]}",
                )
                for i, j in geometry.self_pairs
            )
        )
    if not results:
        raise CraneModelError(ErrorCode.INVALID_SCENE, "there is nothing to check")
    return results


def query(model, data, geometry: LinkGeometry, configuration, scene) -> CollisionResult:
    """Reduce the whole scene and the machine itself to the one pair that matters."""
    return min(
        queries(model, data, geometry, configuration, scene),
        key=lambda result: result.minimum_distance_m,
    )
