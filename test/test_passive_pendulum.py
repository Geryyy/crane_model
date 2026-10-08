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


def test_the_description_carries_the_coulomb_friction_the_decay_needs():
    """Dropping `friction` is what left the old model ringing for a minute."""
    model = CraneSymbolicModel(description(), Tool.PZS100.value)
    tip, tilt = (model.description.friction[row] for row in (4, 5))
    assert tip > 0.0 and tilt > 0.0
