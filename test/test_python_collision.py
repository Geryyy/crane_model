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

DESCRIPTION = os.path.join(HERE, "description", "pzs100.urdf")


def build(tool):
    with open(DESCRIPTION, encoding="utf-8") as handle:
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


def test_every_fitted_primitive_names_a_link_the_description_carries():
    """A description that moves a link would otherwise keep a stale fit silently."""
    with open(COLLISION, encoding="utf-8") as handle:
        table = yaml.safe_load(handle)
    tool = Tool.PZS100
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
    """Self collision is checked even against an empty scene."""
    model = build(Tool.PZS100)
    folded = np.zeros(8)
    folded[1], folded[2] = -1.2, -0.91
    result = model.collision_query(folded, [])
    assert result.collision
    assert "|" in result.other_id  # a self pair, not a scene id


def test_distance_tracks_the_gap_and_is_not_merely_a_flag():
    """A signed distance must track geometry, not only flip sign at contact.  Per
    primitive: the machine's own 25 mm self clearance would otherwise mask it."""
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
    # 1 m of travel is 1 m of clearance, give or take witness-point slide.
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
    """dimensions_m is the extent per axis: a cylinder is (2r, 2r, l), not r."""
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
    """`attached_to_tool` is two rules and both have to hold: a payload sits
    *inside* the gripper, so unexcluded it collides at every reachable pose, but
    scene bodies are never checked against each other, so excluding alone would
    leave it checked against nothing."""
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

    inside = box("rock", at_tool.translation, side=0.4)
    result = model.collision_queries(q, [payload(True), inside])[0]
    assert result.minimum_distance_m < 0.0
    assert result.other_id == "rock"
    away = box("far_rock", at_tool.translation + np.array([6.0, 0.0, 0.0]), side=0.4)
    assert model.collision_queries(q, [payload(True), away])[0].minimum_distance_m > 0.0


def test_a_restricted_query_splits_the_machine_at_the_upper_hinge():
    """The sway envelope is owed by what hangs on the hinges, so the halves must
    be askable separately. A restricted answer carries no self row."""
    model = build(Tool.PZS100)
    q = np.zeros(8)
    q[7] = 0.35
    tcp = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP).position_m
    scene = [box("at_tool", tcp), box("at_base", [0.0, 0.0, -1.5])]
    whole = model.collision_queries(q, scene)
    swinging = model.collision_queries(q, scene, swinging=True)
    rigid = model.collision_queries(q, scene, swinging=False)
    assert len(whole) == 3 and len(swinging) == 2 and len(rigid) == 2
    assert swinging[0].minimum_distance_m == whole[0].minimum_distance_m
    assert rigid[0].minimum_distance_m > swinging[0].minimum_distance_m + 0.5
    assert rigid[1].minimum_distance_m == whole[1].minimum_distance_m
    assert swinging[1].minimum_distance_m > rigid[1].minimum_distance_m + 0.5


def test_the_broad_phase_answers_exactly_what_checking_every_pair_would():
    """Self pairs are culled by a bounding-sphere lower bound -- 49 exact Coal
    queries per configuration reduced to a handful. Inflating the radii makes every
    bound useless and forces the exhaustive path through the same code, so the two
    answers must agree on the distance *and* on which pair it came from."""
    model = build(Tool.PZS100)
    geometry = model._link_geometry()
    exact = geometry.bound_radius.copy()

    rng = np.random.default_rng(0)
    lower = np.array([model._model.lowerPositionLimit[j] for j in range(8)])
    upper = np.array([model._model.upperPositionLimit[j] for j in range(8)])
    lower[~np.isfinite(lower)], upper[~np.isfinite(upper)] = -np.pi, np.pi

    checked = 0
    for _ in range(60):
        q = rng.uniform(lower, upper)
        culled = model.collision_queries(q, [])[-1]
        geometry.bound_radius = np.full_like(exact, 1.0e6)  # nothing can cull
        try:
            exhaustive = model.collision_queries(q, [])[-1]
        finally:
            geometry.bound_radius = exact
        assert culled.minimum_distance_m == exhaustive.minimum_distance_m
        assert culled.other_id == exhaustive.other_id
        checked += 1
    assert checked == 60


def test_the_broad_phase_bound_never_exceeds_the_distance_it_bounds():
    """The cull is only sound while `|c_i - c_j| - (r_i + r_j)` is a *lower* bound;
    a radius measured about the wrong origin would silently break that."""
    model = build(Tool.PZS100)
    geometry = model._link_geometry()
    for (centre, radius), (_, _, shape, _) in zip(geometry.bounds, geometry.bodies):
        shape.computeLocalAABB()
        # Centre is the AABB's own centre, so both extreme corners sit at the
        # half-diagonal and bound every other point of the shape.
        for corner in (shape.aabb_local.min_, shape.aabb_local.max_):
            assert radius >= float(np.linalg.norm(np.asarray(corner) - centre)) - 1e-9
