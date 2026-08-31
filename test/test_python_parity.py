"""The Python model against the recorded-trajectory fixture.

Same oracle as test_recorded_parity.cpp: the generated Maple/MATLAB model
evaluated at 64 configurations taken from the machine recordings. An
independent implementation, so agreement is evidence and not a tautology.
"""

import os

import numpy as np
import pytest
from crane_model import CraneModel, CraneModelError, ErrorCode, Frame, Payload, Tool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(ROOT, "config", "hydraulics.yaml")
TOLERANCE_M = 1e-9

DESCRIPTIONS = {
    Tool.PZS100: os.path.join(HERE, "description", "pzs100.urdf"),
    Tool.EPSILON_7040: os.path.join(HERE, "description", "epsilon_7040.urdf"),
}


def build(tool):
    with open(DESCRIPTIONS[tool], encoding="utf-8") as handle:
        return CraneModel(handle.read(), tool, hydraulics_config=CONFIG)


@pytest.fixture(scope="module")
def fixture_rows():
    path = os.path.join(HERE, "recorded_parity_fixture.txt")
    with open(path, encoding="utf-8") as handle:
        rows = [
            line.split() for line in handle if line.strip() and not line.startswith("#")
        ]
    return np.array(rows, dtype=float)


@pytest.mark.parametrize(
    "frame,columns",
    [
        (Frame.ROTATOR_LOWER_PART, slice(18, 21)),
        (Frame.TIP, slice(30, 33)),
    ],
)
def test_forward_kinematics_matches_the_recorded_fixture(fixture_rows, frame, columns):
    model = build(Tool.PZS100)
    for row in fixture_rows:
        pose = model.forward_kinematics(row[2:10], Frame.MOUNTING_BASE, frame)
        assert np.allclose(pose.position_m, row[columns], atol=TOLERANCE_M)


def test_k8_orientation_matches_the_recorded_fixture(fixture_rows):
    model = build(Tool.PZS100)
    for row in fixture_rows:
        pose = model.forward_kinematics(
            row[2:10], Frame.MOUNTING_BASE, Frame.ROTATOR_LOWER_PART
        )
        assert np.allclose(
            pose.as_matrix()[:3, :3], row[21:30].reshape(3, 3), atol=1e-9
        )


@pytest.mark.parametrize("tool", list(Tool))
def test_the_telescope_moves_the_tool_twice(tool):
    """q5_small_telescope mimics q4, so the tool travels 2*q4 and Jacobian
    column 3 carries both stages. Contract section 2; issue 030."""
    model = build(tool)
    q = np.zeros(8)
    rest = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP).position_m
    q[3] = 0.4
    out = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP).position_m
    assert np.linalg.norm(out - rest) == pytest.approx(0.8, abs=1e-9)
    assert np.linalg.norm(model.jacobian(q, Frame.TCP).value[:, 3]) == pytest.approx(
        2.0, abs=1e-9
    )


def test_tool_contact_is_a_frame_on_the_7040_and_a_refusal_on_the_pzs100():
    """Same call, real pose on one tool and a refusal on the other; a planner
    that assumes it works refuses every PZS100 plan. Issues 030, 040."""
    q = np.zeros(8)
    pose = build(Tool.EPSILON_7040).forward_kinematics(
        q, Frame.MOUNTING_BASE, Frame.TOOL_CONTACT
    )
    assert np.all(np.isfinite(pose.position_m))
    with pytest.raises(CraneModelError) as refusal:
        build(Tool.PZS100).forward_kinematics(
            q, Frame.MOUNTING_BASE, Frame.TOOL_CONTACT
        )
    assert refusal.value.code is ErrorCode.FRAME_UNAVAILABLE


def test_the_rotator_wraps_rather_than_stopping(fixture_rows):
    """The recordings reach -7.8 rad on theta8, well outside one turn: the
    joint is `continuous` and the model must take it as given."""
    rotator = fixture_rows[:, 8]
    assert rotator.min() < -np.pi
    model = build(Tool.PZS100)
    q = np.zeros(8)
    q[6] = rotator.min()
    wrapped = q.copy()
    wrapped[6] = rotator.min() + 2.0 * np.pi
    assert np.allclose(
        model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP).position_m,
        model.forward_kinematics(wrapped, Frame.MOUNTING_BASE, Frame.TCP).position_m,
        atol=1e-12,
    )


@pytest.mark.parametrize("tool", list(Tool))
def test_a_description_without_the_canonical_joints_is_refused(tool):
    with pytest.raises(CraneModelError) as refusal:
        CraneModel(
            '<robot name="stub"><link name="only"/></robot>',
            tool,
            hydraulics_config=CONFIG,
        )
    assert refusal.value.code in (
        ErrorCode.MISSING_JOINT,
        ErrorCode.INVALID_ROBOT_DESCRIPTION,
    )


def test_a_non_finite_configuration_is_refused():
    model = build(Tool.PZS100)
    q = np.zeros(8)
    q[2] = np.nan
    with pytest.raises(CraneModelError) as refusal:
        model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    assert refusal.value.code is ErrorCode.NON_FINITE_INPUT


def test_passive_equilibrium_matches_the_recorded_fixture(fixture_rows):
    model = build(Tool.PZS100)
    for row in fixture_rows:
        settled = model.passive_equilibrium(CraneModel.actuated(row[2:10]))
        assert np.allclose(settled, row[47:49], atol=1e-9)


def test_the_equilibrium_is_the_hanging_well_and_not_the_standing_saddle(fixture_rows):
    """h_u = 0 has the tool standing up as a second root, and a solver's own
    residual cannot tell it from the hanging one. The stiffness sign is what
    separates them, so the answer must be a minimum of the potential."""
    model = build(Tool.PZS100)
    for row in fixture_rows[:8]:
        q_a = CraneModel.actuated(row[2:10])
        settled = model.passive_equilibrium(q_a)
        q = np.zeros(8)
        q[[0, 1, 2, 3, 6, 7]] = q_a
        _, stiffness, _ = model._passive_gravity(q, settled)
        np.linalg.cholesky(stiffness)  # raises if the answer is a saddle


def test_a_configuration_with_nowhere_to_hang_is_refused():
    """Where pi/2 - (q2 + q3) leaves the tip joint's +-pi/2 range the tool would
    rest against a stop rather than hang, and a plausible-looking q_u would be
    worse than none. Issue 032; q_a is a measured case off the swept ranges."""
    model = build(Tool.PZS100)
    q_a = np.array([0.0, -1.2, -0.91, 0.0, 0.0, 0.0])
    assert abs(np.pi / 2.0 - (q_a[1] + q_a[2])) > np.pi / 2.0
    with pytest.raises(CraneModelError) as refusal:
        model.passive_equilibrium(q_a)
    assert refusal.value.code is ErrorCode.SINGULAR_CONFIGURATION


def test_a_payload_changes_where_the_tool_hangs():
    model = build(Tool.PZS100)
    q_a = np.array([0.2, 0.7, 0.4, 1.0, -1.5, 0.3])
    bare = model.passive_equilibrium(q_a)
    loaded = model.passive_equilibrium(
        q_a,
        Payload(
            mass_kg=400.0, center_of_mass_k8_m=np.array([0.3, 0.0, -0.5]), valid=True
        ),
    )
    assert not np.allclose(bare, loaded, atol=1e-6)
    # The payload is restored after the call, so the next answer is the bare one.
    assert np.allclose(model.passive_equilibrium(q_a), bare, atol=1e-12)
