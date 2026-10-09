"""The passive pendulum against free ring-downs on the real PZS100.

`~/Documents/2025-12-17_pzs`, five bags: manipulator static, the gripper nudged
by hand about the tip axis and left to decay for ~50 s. `/joint_states`
`theta6_tip_joint` carries the swing. The period gave issue 222 the gripper's
CoM height, the envelope its dissipation; the bags are outside the repo, so the
numbers they yielded are the fixture here.

Two things hold the fit together and nothing else asserts either: the inertials
in `pzs100_description` and the Coulomb term `symbolic.equations` adds beside
`damping`. Before 222 the model ran 4.4 % fast and, reading `damping` alone,
took 45 s to half amplitude against the bags' 33 s -- and never came to rest.

The MuJoCo plant is held to the same fixture, because the symbolic twin passing
says nothing about it and issue 244 read it as 2-8x over-damped off a probe that
was wrong twice. Both traps are in the Coulomb term: the envelope is *not* a
single exponential, so a fitted `tau` is a property of the amplitude and not of
the model (0.435 rad -> 36 s, 0.033 rad -> 3.4 s on one description), and C3 at
`u = 0` is not a free decay -- `tau_dot = k*(0 - dq)` makes the five planned
axes a velocity damper and costs the loaded tip mode 93 s -> 24 s. Measure a
decay at the bags' amplitude with the planned axes held (`follow`), compare
half-lives, and never an exponential `tau` across two amplitudes.
"""

import functools
import os

import casadi as ca
import numpy as np
import pytest
from crane_model.conventions import Tool
from crane_model.model import CraneModel
from crane_model.symbolic import (
    NP,
    NU,
    NX_RIGID,
    X_PASSIVE_POSITION,
    X_PASSIVE_VELOCITY,
    CraneSymbolicModel,
)

HERE = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=1)
def description():
    with open(os.path.join(HERE, "description", "pzs100.urdf"), encoding="utf-8") as f:
        return f.read()


# Medians of the actuated five over the bags, which agree to 1e-3 rad: slewing,
# boom, arm, telescope, rotator. All five bags share one manipulator pose.
BAG_POSE = (0.0279, 0.9143, -0.0753, 0.6878, 6.2452)
#: The hand excitation's own starting amplitude, rad.
BAG_AMPLITUDE_RAD = 0.435
#: Rail stroke -> (mean tip period, time to half amplitude) over the decay.
#: closed = theta6_2/theta7_2/theta7_3, open = theta6_3/theta7_1.
BAG = {0.0: (2.276, 32.6), 0.538: (2.466, 37.25)}

PERIOD_RTOL = 0.05
HALF_LIFE_RTOL = 0.30


def _ringdown(q9, duration=50.0, step=5.0e-3):
    """Free decay of the tip mode with the actuated axes held, (t, sway)."""
    model = CraneSymbolicModel(description(), Tool.PZS100.value)
    passive_acceleration = ca.Function(
        "ddq_u", [model.x, model.u, model.p], [model.ddq_u]
    )
    parameters, command = np.zeros(NP), np.zeros(NU)
    parameters[0] = q9  # the pinned tool coordinate is the rail stroke

    rest = CraneModel(description(), Tool.PZS100).passive_equilibrium(
        list(BAG_POSE) + [q9]
    )
    state = np.zeros(NX_RIGID)
    state[: len(BAG_POSE)] = BAG_POSE
    state[X_PASSIVE_POSITION : X_PASSIVE_POSITION + 2] = rest
    state[X_PASSIVE_POSITION] += BAG_AMPLITUDE_RAD

    def rhs(x):
        out = np.zeros(NX_RIGID)
        out[X_PASSIVE_POSITION : X_PASSIVE_POSITION + 2] = x[
            X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + 2
        ]
        out[X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + 2] = np.asarray(
            passive_acceleration(x, command, parameters)
        ).ravel()
        return out

    times, sway = [0.0], [BAG_AMPLITUDE_RAD]
    for index in range(int(duration / step)):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        state = state + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        times.append((index + 1) * step)
        sway.append(state[X_PASSIVE_POSITION] - rest[0])
    return np.array(times), np.array(sway)


def _period_and_half_life(times, sway):
    turning = [
        (times[k], sway[k])
        for k in range(1, len(sway) - 1)
        if (sway[k] - sway[k - 1]) * (sway[k + 1] - sway[k]) < 0
    ]
    period = np.mean(
        [turning[k + 2][0] - turning[k][0] for k in range(len(turning) - 2)]
    )
    amplitude = np.array(
        [0.5 * abs(turning[k + 1][1] - turning[k][1]) for k in range(len(turning) - 1)]
    )
    middle = np.array(
        [0.5 * (turning[k + 1][0] + turning[k][0]) for k in range(len(turning) - 1)]
    )
    # The envelope is monotone, so interpolate it reversed to get a rising x.
    return period, float(np.interp(-amplitude[0] / 2, -amplitude, middle))


@pytest.mark.parametrize("q9", sorted(BAG))
def test_the_free_tip_swing_matches_the_pzs100_bags(q9):
    measured_period, measured_half_life = BAG[q9]
    period, half_life = _period_and_half_life(*_ringdown(q9))
    assert period == pytest.approx(measured_period, rel=PERIOD_RTOL)
    assert half_life == pytest.approx(measured_half_life, rel=HALF_LIFE_RTOL)


def _mujoco_ringdown(q9, duration=50.0, step=1.0e-3):
    """`_ringdown` off the MuJoCo plant: same pose, same kick, (t, sway)."""
    from crane_model.conventions import PASSIVE_INDICES, PLANNED_INDICES
    from crane_model.mujoco_plant import TOOL_INDEX, MujocoPlant

    plant = MujocoPlant(description(), timestep=step)
    rest = CraneModel(description(), Tool.PZS100).passive_equilibrium(
        list(BAG_POSE) + [q9]
    )
    q = np.zeros(plant.q.size)
    q[list(PLANNED_INDICES)] = BAG_POSE
    q[list(PASSIVE_INDICES)] = rest
    q[TOOL_INDEX] = q9
    q[PASSIVE_INDICES[0]] += BAG_AMPLITUDE_RAD
    plant.set_state(q)
    # Held, not pinned: writing `qpos` back each step fights the mimic equality
    # and the model runs away inside 50 ms.
    held = plant.q[list(PLANNED_INDICES)].copy()
    still = np.zeros(len(PLANNED_INDICES))
    tick = 0.01
    times, sway = [0.0], [BAG_AMPLITUDE_RAD]
    for index in range(int(duration / tick)):
        plant.follow(held, still, still, tick)
        times.append((index + 1) * tick)
        sway.append(plant.q[PASSIVE_INDICES[0]] - rest[0])
    return np.array(times), np.array(sway)


@pytest.mark.parametrize("q9", sorted(BAG))
def test_the_mujoco_plants_free_tip_swing_matches_the_pzs100_bags(q9):
    """The offline plant every MPC number is tuned against, on the same oracle."""
    pytest.importorskip("mujoco")
    measured_period, measured_half_life = BAG[q9]
    period, half_life = _period_and_half_life(*_mujoco_ringdown(q9))
    assert period == pytest.approx(measured_period, rel=PERIOD_RTOL)
    assert half_life == pytest.approx(measured_half_life, rel=HALF_LIFE_RTOL)


def test_the_description_carries_the_coulomb_friction_the_decay_needs():
    """Dropping `friction` is what left the old model ringing for a minute."""
    model = CraneSymbolicModel(description(), Tool.PZS100.value)
    tip, tilt = (model.description.friction[row] for row in (4, 5))
    assert tip > 0.0 and tilt > 0.0
