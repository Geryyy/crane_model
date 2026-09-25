"""The identified C3 fit against the machine it was identified on.

`test_actuator.py` holds `actuator.py` to `symbolic.py` and to the cascade the
fit was scored with -- three statements of one law, which agree to 1e-10 and
would go on agreeing if every constant in `c3_full_model.json` were wrong. This
is the other half: the constants against the crane, out of
`step_response_fixture.json`, which `scripts/derive_step_fixture.py` distils
from the HydraulicCalib staircases. Issue 161 lost two sessions to a simulated
plant that reached full velocity in 0.2 s where the identification says 0.85,
with 150 green tests throughout, because nothing anywhere compared a plant to
the machine.

The fixture is a *shape*: the measured velocity step normalised by the velocity
the axis settled at. `/setpoints_compensated` carries `u_norm` in [-1, 1], not
rad/s, so the amplitude is not recoverable here -- but C3 has unit DC gain and
is linear, so the shape is the whole of what it claims and the model can be
driven with a unit step. Psi drops out, which is correct: it is assumed exact
everywhere in this package.

`m_ii` comes out of the fixture rather than out of `pzs100.urdf`. It is the
inertia at the pose the bag was recorded at, on the description that bag was
recorded with -- a different machine end, 21-30 % heavier here on sw/ha/ka/sa
and 3x lighter on ro. Recomputing would test the tool swap.

The tolerances are **what the shipped fit does today**, not a target. Where
they are wide they say so, and the comment says which way the model is wrong;
widening one is a decision, not a fix. Re-fitting is issue 164's explicit
non-goal.
"""

import json
import os

import numpy as np
import pytest
from crane_model.actuator import C3Actuator
from crane_model.conventions import default_actuator_path
from crane_model.symbolic import K_AXIS_KEYS, K_PLANNED_DOF, load_actuator_fit

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "step_response_fixture.json")

#: The fit's own step, and the only rate the pinned 60 ms dead time divides.
DT = 0.01

#: Worst |model - machine| over an axis' instants, with a little headroom.
TOLERANCE = {
    # 0.23, all of it at 0.9 and 1.2 s: the machine's slew rings near 1.1 Hz
    # and the SISO model at omega_n/2pi = 1.38, so the two drift out of phase
    # after the first overshoot. Through 0.6 s they agree to 0.08. The extra
    # mode is the tool hanging off the passive pair, which C3 does not carry.
    "sw": 0.30,
    "ha": 0.30,  # 0.23, at 0.3 s: the model settles, the machine overshoots
    # 0.40, at 0.15 and 0.2 s: the machine is already at 0.7 where the model is
    # at 0.33. `ka` is fitted at tau_v = 0 on 5 bags, 2-3 of them training.
    "ka": 0.50,
    # 0.79. The telescope does not move for 250 ms -- four times the pinned
    # transport delay -- and then goes in one step. That is the load-holding
    # valve metering the c3 README names, not a lag, and no PT1 can be it. The
    # fit's own nrmse marks `sa` as its weakest axis; this says how weak.
    "sa": 0.90,
    "ro": 0.15,  # 0.09, the axis the fit reproduces
}

#: What "the dead time is real" means at the first instant, for every axis.
#: This is the assertion issue 161's plant would have failed on day one.
AT_DEAD_TIME = 0.10


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def damping() -> dict[str, float]:
    """`d` per axis out of the fit file; `ActuatorFit` deliberately omits it."""
    with open(default_actuator_path(), encoding="utf-8") as handle:
        return {
            key: float(entry["d"]) for key, entry in json.load(handle)["axes"].items()
        }


def step_response(axis: str, m_ii: float, times) -> np.ndarray:
    """C3 and the one-axis rigid body, unit step from rest, sampled at `times`.

    The same cascade `test_actuator.py` scores against `fit_full.simulate`:
    `M_ii ddq = tau - d dq` integrated symplectically against the force
    `C3Actuator.step` returns. Unit step and rest because the machine's edge
    left a settled level and the system is linear, so the increment obeys the
    same equations from zero -- which is also why the fixture's unknown
    amplitude does not matter.
    """
    row = K_AXIS_KEYS.index(axis)
    actuator = C3Actuator(DT, load_actuator_fit())
    command, rate = np.zeros(K_PLANNED_DOF), np.zeros(K_PLANNED_DOF)
    command[row] = 1.0
    d = damping()[axis]

    steps = int(round(max(times) / DT)) + 1
    walked = np.zeros(steps + 1)
    for sample in range(steps):
        tau = actuator.step(command, rate)
        rate[row] += DT / m_ii * (tau[row] - d * rate[row])
        walked[sample + 1] = rate[row]
    return np.array([walked[round(t / DT)] for t in times])


@pytest.mark.parametrize("axis", K_AXIS_KEYS)
def test_the_fit_still_reproduces_the_machines_step(fixture, axis):
    entry = fixture["axes"][axis]
    assert abs(entry["dt_s"] - DT) < 1e-4, "the bag did not run at the fit's rate"
    simulated = step_response(axis, entry["m_ii"], entry["times_s"])
    residual = simulated - np.asarray(entry["response"])
    worst = int(np.argmax(np.abs(residual)))
    assert np.max(np.abs(residual)) <= TOLERANCE[axis], (
        f"{axis}: {entry['bag']} sample {entry['samples']} at "
        f"{entry['times_s'][worst]} s -- machine {entry['response'][worst]:.3f} "
        f"+-{entry['spread'][worst]:.3f}, model {simulated[worst]:.3f}"
    )


@pytest.mark.parametrize("axis", K_AXIS_KEYS)
def test_no_axis_has_moved_when_the_transport_delay_expires(fixture, axis):
    """At 60 ms the command has not arrived yet, on the machine or in the model.

    The sharp half of this file. A plant that ignores `dead_time` or integrates
    on an accumulated period is already most of the way up here, which is
    exactly the shape of issue 161's three faults, and no tolerance on the
    settled value would have caught it.
    """
    entry = fixture["axes"][axis]
    assert entry["times_s"][0] == pytest.approx(0.06)
    assert abs(entry["response"][0]) <= AT_DEAD_TIME
    assert abs(step_response(axis, entry["m_ii"], [0.06])[0]) <= AT_DEAD_TIME
