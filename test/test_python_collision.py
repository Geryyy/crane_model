"""Collision geometry and distance queries of the Python model."""

import os

import numpy as np
import pinocchio as pin
import pytest
import yaml
from crane_model import (
    CollisionPrimitive,
    CraneModel,
    CraneModelError,
    ErrorCode,
    Frame,
    Tool,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HYDRAULICS = os.path.join(ROOT, "config", "hydraulics.yaml")
COLLISION = os.path.join(ROOT, "config", "collision_model.yaml")

DESCRIPTIONS = {
    Tool.PZS100: os.path.join(HERE, "description", "pzs100.urdf"),
    Tool.EPSILON_7040: os.path.join(HERE, "description", "epsilon_7040.urdf"),
}


def build(tool):
    with open(DESCRIPTIONS[tool], encoding="utf-8") as handle:
        return CraneModel(
            handle.read(),
            tool,
            hydraulics_config=HYDRAULICS,
            collision_config=COLLISION,
        )


def box(identifier, position, side=0.6):
    return CollisionPrimitive(
        identifier,
        "box",
        pin.SE3(np.eye(3), np.asarray(position, dtype=float)),
        np.full(3, side),
    )


@pytest.mark.parametrize("tool", list(Tool))
def test_every_fitted_primitive_names_a_link_the_description_carries(tool):
    """A description that moves a link would otherwise keep a stale fit silently."""
    with open(COLLISION, encoding="utf-8") as handle:
        table = yaml.safe_load(handle)
    model = build(tool)._model
    fitted = [e for e in table["primitives"] if e["tool"] == tool.value]
    assert fitted, f"the table carries no primitives for {tool.value}"
    for entry in fitted:
        assert model.existFrame(entry["link"]), entry["link"]


def test_a_block_at_the_tool_collides_and_a_distant_one_does_not():
    model = build(Tool.PZS100)
    q = np.zeros(8)
    tcp = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP).position_m
    assert model.collision_query(q, [box("block", tcp)]).collision
    far = model.collision_query(q, [box("far", [0.0, 0.0, 50.0])])
    assert not far.collision and far.minimum_distance_m > 0.0


def test_the_arm_folded_back_over_the_base_is_refused_against_an_empty_scene():
    """Self collision is checked even with nothing in the scene; the pairs the
    fit calls not worth checking are the only ones excluded."""
    model = build(Tool.PZS100)
    folded = np.zeros(8)
    folded[1], folded[2] = -1.2, -0.91
    result = model.collision_query(folded, [])
    assert result.collision
    assert "|" in result.other_id  # a self pair, not a scene id


def test_distance_tracks_the_gap_and_is_not_merely_a_flag():
    """The reported distance is a signed distance, so it has to track the
    geometry rather than only change sign at contact. Taken along +x, clear of
    the machine; the per-primitive form is used because the machine's own 25 mm
    self clearance is a separate result that would otherwise mask the scene."""
    model = build(Tool.PZS100)
    q = np.zeros(8)
    gaps = np.array(
        [
            model.collision_queries(q, [box("block", [x, 0.0, 0.0])])[
                0
            ].minimum_distance_m
            for x in (10.0, 11.0, 12.0)
        ]
    )
    assert np.all(gaps > 0.0)
    # One metre of travel is one metre of clearance, give or take the witness
    # point sliding along the nearest link.
    assert np.all(np.abs(np.diff(gaps) - 1.0) < 0.05)


def test_a_scene_that_repeats_an_id_or_leaves_one_empty_is_refused():
    model = build(Tool.PZS100)
    q = np.zeros(8)
    for scene in (
        [box("same", [5, 0, 5]), box("same", [6, 0, 5])],
        [box("", [5, 0, 5])],
    ):
        with pytest.raises(CraneModelError) as refusal:
            model.collision_query(q, scene)
        assert refusal.value.code is ErrorCode.INVALID_SCENE


def test_a_cylinder_given_a_radius_instead_of_an_extent_is_refused():
    """dimensions_m is the extent along each axis, so a cylinder is (2r, 2r, l);
    a caller who passed a radius is told rather than silently reinterpreted."""
    model = build(Tool.PZS100)
    wrong = CollisionPrimitive(
        "log",
        "cylinder",
        pin.SE3(np.eye(3), np.array([3.0, 0.0, 0.0])),
        np.array([0.2, 0.4, 1.0]),
    )
    with pytest.raises(CraneModelError) as refusal:
        model.collision_query(np.zeros(8), [wrong])
    assert refusal.value.code is ErrorCode.INVALID_SCENE


def test_a_carried_body_ignores_the_grip_and_still_sees_the_world():
    """
    `attached_to_tool` is two rules, and both of them have to hold.

    A payload sits *inside* the gripper -- that is what being gripped means --
    so a body checked against the links holding it reports a collision at every
    pose the machine can reach, and the planner refuses everything. Excluding
    those links is only half the answer, though: a scene body is otherwise never
    checked against another scene body, so the exclusion on its own would leave
    the payload checked against nothing that matters.
    """
    model = build(Tool.PZS100)
    q = np.zeros(8)
    q[[0, 1, 2, 3, 6]] = (0.0, -0.2, 0.4, 1.0, 0.0)
    q[7] = 0.30
    q[[4, 5]] = model.passive_equilibrium(q[[0, 1, 2, 3, 6, 7]])
    mount = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.ROTATOR_LOWER_PART)
    at_tool = pin.XYZQUATToSE3(
        np.concatenate([mount.position_m, mount.orientation_xyzw])
    )

    def payload(attached):
        return CollisionPrimitive(
            "payload",
            "box",
            at_tool,
            np.array([0.9, 0.6, 0.6]),
            attached_to_tool=attached,
        )

    # Unattached, it is buried in its own gripper.
    assert model.collision_query(q, [payload(False)]).minimum_distance_m < 0.0
    # Attached, the links holding it stop counting.
    assert model.collision_query(q, [payload(True)]).minimum_distance_m > 0.0

    # And it is still a body in the world: an obstacle where it is, is a hit
    # that names the obstacle, while one out of reach is not.
    inside = box("rock", at_tool.translation, side=0.4)
    result = model.collision_queries(q, [payload(True), inside])[0]
    assert result.minimum_distance_m < 0.0
    assert result.other_id == "rock"
    away = box("far_rock", at_tool.translation + np.array([6.0, 0.0, 0.0]), side=0.4)
    assert model.collision_queries(q, [payload(True), away])[0].minimum_distance_m > 0.0
