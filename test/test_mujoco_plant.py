"""The MuJoCo plant against the model it exists to cross-check.

`mujoco_plant` rewrites the description -- meshes, compiler flags, the cylinder
joints fixed -- any of which can silently change the machine. Oracles: gravity,
computed by both sides from the same URDF, and where the load hangs. Skipped
where MuJoCo is absent: a tuning dependency, nothing deployed imports it."""

import os

import numpy as np
import pinocchio as pin
import pytest
from crane_model import CraneModel, Tool
from crane_model.conventions import ACTUATED_INDICES, PASSIVE_INDICES, Frame
from crane_model.description import parse
from crane_model.presets import OUTSIDE

mujoco = pytest.importorskip("mujoco")

from crane_model.mujoco_plant import (  # noqa: E402
    PLANNED_INDICES,
    MujocoPlant,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "hydraulics.yaml")
DESCRIPTION = os.path.join(HERE, "description", "pzs100.urdf")

#: Radians. The two agree to ~1e-4 on tilt; a conversion mistake is far larger.
TOLERANCE_RAD = 1.0e-3

#: The sway period is order 2 s and the passive damping is light.
SETTLE_S = 20.0

POSES = (
    OUTSIDE,
    np.array([0.2, 0.9, 1.4, 0.05, 0.4, 1.4, 0.7, 0.1]),
    np.array([-1.0, 0.2, 0.3, 1.9, 0.6, 1.7, -0.5, 0.4]),
)


def description() -> str:
    with open(DESCRIPTION, encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def plant() -> MujocoPlant:
    return MujocoPlant(description(), hydraulics_config=CONFIG)


def test_only_the_canonical_and_coupled_joints_move(plant):
    """Ten degrees of freedom: the canonical eight plus the two mimics."""
    assert plant.model.nv == 10
    # Both mimics arrive as equalities from MuJoCo's own URDF parser.
    assert plant.model.neq == 2
    names = {
        mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(plant.model.njnt)
    }
    assert not [name for name in names if "cylinder" in name]


@pytest.mark.parametrize("q", POSES)
def test_gravity_matches_pinocchio(plant, q):
    """Same URDF, same gravity, to float precision. Catches an invented body: on
    MuJoCo's default `inertiafromgeom` the links with no `<inertial>` picked up
    8.2 kg of mesh and every row came out 1% high."""
    model = parse(description(), Tool.PZS100, hydraulics_config=CONFIG)
    expected = model.projection.T @ pin.rnea(
        model.model,
        model.model.createData(),
        model.configuration(q),
        np.zeros(model.model.nv),
        np.zeros(model.model.nv),
    )

    plant.set_state(q)
    assert np.allclose(plant.projection.T @ plant.data.qfrc_bias, expected, atol=1.0e-6)


def test_the_load_hangs_where_pinocchio_says_it_does(plant):
    model = CraneModel(description(), Tool.PZS100, hydraulics_config=CONFIG)
    expected = model.passive_equilibrium(OUTSIDE[list(ACTUATED_INDICES)])

    plant.set_state(OUTSIDE)
    still = np.zeros(len(PLANNED_INDICES))
    plant.follow(OUTSIDE[list(PLANNED_INDICES)], still, still, SETTLE_S)

    settled = plant.q[list(PASSIVE_INDICES)]
    assert np.allclose(settled, expected, atol=TOLERANCE_RAD), (
        f"MuJoCo settles at {settled}, pinocchio at {expected}"
    )
    assert np.allclose(plant.dq[list(PASSIVE_INDICES)], 0.0, atol=1.0e-3)


#: `axes.sa` of `config/c3_full_model.json`. The URDF spells the damping to four
#: significant figures, so the reflected total matches the fit to ~1e-4, not exactly.
TELESCOPE = 3
DAMPING_RTOL = 1.0e-3


def test_the_telescope_reflects_the_inertia_of_both_stages(plant):
    """`q5` mimics `q4` at multiplier 1, so the mass distal of `q5` moves at
    `2 qdot4` and reflects four times over. Reading the CRBA diagonal instead --
    pinocchio drops `<mimic>` -- understated the telescope 3.4x and is what the
    C3 telescope fit was built on before `c3/rbd.py` learned to project."""
    model = parse(description(), Tool.PZS100, hydraulics_config=CONFIG)
    mass = pin.crba(model.model, model.model.createData(), model.configuration(OUTSIDE))
    mass = np.triu(mass) + np.triu(mass, 1).T  # crba fills the upper triangle only
    expected = model.projection.T @ mass @ model.projection

    plant.set_state(OUTSIDE)
    plant.forward()
    full = np.zeros((plant.model.nv, plant.model.nv))
    mujoco.mj_fullM(plant.model, plant.data, full)
    actual = plant.projection.T @ full @ plant.projection

    # The coupled column is the point and it is exact. The rest of the matrix
    # goes through MuJoCo's generated XML, whose inertia literals cost ~1e-5 on
    # the small off-diagonals.
    assert actual[TELESCOPE, TELESCOPE] == expected[TELESCOPE, TELESCOPE]
    assert np.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-3)
    # The number itself, so a mass edit that quietly halves it fails here too.
    assert actual[TELESCOPE, TELESCOPE] == pytest.approx(2191.5, rel=1.0e-4)
    # ... and it is the projection that earns it, not the driver's own column.
    driver = model.model.joints[model.model.getJointId("q4_big_telescope")].idx_v
    assert actual[TELESCOPE, TELESCOPE] == pytest.approx(
        3.403 * mass[driver, driver], rel=1.0e-3
    )


def test_the_telescope_reflects_the_fitted_damping(plant):
    """The check that would have caught the 14 % over-damp: `q4` carried the
    whole fitted `d` while `q5` added 8000 on top of it through the mimic, so
    the plant damped the commanded coordinate harder than the fit said."""
    import json
    import os

    model = json.load(
        open(
            os.path.join(os.path.dirname(HERE), "config", "c3_full_model.json"),
            encoding="utf-8",
        )
    )
    fitted = model["axes"]["sa"]["d"]

    velocity = np.zeros(8)
    velocity[TELESCOPE] = 1.0
    plant.set_state(np.zeros(8), velocity)
    plant.forward()
    reflected = -(plant.projection.T @ plant.data.qfrc_passive)[TELESCOPE]

    assert reflected == pytest.approx(fitted, rel=DAMPING_RTOL)


def _jointed(q):
    """The same model with the cylinder joints left movable, posed from `linkage`.

    MuJoCo's own answer to what `_pose_cylinders` works out by hand. Passing them
    off as canonical is what keeps `to_mujoco_xml` from freezing them.
    """
    from crane_model import mujoco_plant as module
    from crane_model.linkage import DISPLAY_JOINTS, cylinder_joints, load_geometry

    canonical = module.canonical_joints
    kept = tuple(canonical(CONFIG)) + tuple(sorted(DISPLAY_JOINTS))
    module.canonical_joints = lambda config=None: kept
    try:
        model = mujoco.MjModel.from_xml_string(
            module.to_mujoco_xml(description(), CONFIG)
        )
    finally:
        module.canonical_joints = canonical

    data = mujoco.MjData(model)
    placed = dict(zip(canonical(CONFIG), q))
    placed.update(cylinder_joints(load_geometry(CONFIG), q[1], q[2]))
    for name, value in placed.items():
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[index]] = value
    mujoco.mj_kinematics(model, data)
    return model, data


@pytest.mark.parametrize("q", POSES)
def test_the_cylinders_are_drawn_where_their_joints_would_put_them(q):
    """The picture: welded links, placed by arithmetic instead of by a joint."""
    plant = MujocoPlant(description(), hydraulics_config=CONFIG)
    plant.set_state(q)
    assert plant._pose_cylinders()
    reference, posed = _jointed(q)
    for body, _, _, _ in plant._chain:
        name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_BODY, body)
        twin = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(plant.data.xpos[body], posed.xpos[twin], atol=1.0e-12)
        assert np.allclose(plant.data.xmat[body], posed.xmat[twin], atol=1.0e-12)


def test_drawing_the_cylinders_leaves_the_physics_alone():
    """`_show` welds them back, so gravity never sees the picture."""
    plant = MujocoPlant(description(), hydraulics_config=CONFIG)
    plant.set_state(OUTSIDE)
    plant.forward()
    before = plant.data.qfrc_bias.copy()
    assert plant._pose_cylinders()
    plant.model.body_pos[plant._welded] = plant._neutral[0]
    plant.model.body_quat[plant._welded] = plant._neutral[1]
    plant.forward()
    assert np.array_equal(plant.data.qfrc_bias, before)


def test_positions_of_is_the_plant_without_disturbing_it():
    """
    The overlays' kinematics are the plant's: same bodies, mimics applied.

    A scratch MjData written without the mimic leaves the inner telescope stage
    behind, which moves the tool by the whole stroke. And the live state has to
    come back untouched -- the viewer is rendering it while this is called.
    """
    plant = MujocoPlant(description(), hydraulics_config=CONFIG)
    body = Frame.TCP.value
    index = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, body)
    expected = []
    for q in POSES:
        plant.set_state(q)
        expected.append(plant.data.xpos[index].copy())

    plant.set_state(OUTSIDE)
    actual = plant.positions_of(body, np.vstack(POSES))
    assert np.allclose(actual, expected, atol=1.0e-12)
    assert np.array_equal(plant.q, OUTSIDE)
