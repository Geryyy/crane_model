"""C3 numerically against the two forms it has to agree with.

Oracle 1: `symbolic.py`'s CasADi `command_lag_rate`/`actuated_force_rate` -- the
same law in the other domain, block by block. Oracle 2: the whole cascade the
C3 fit was scored with, `timber_crane_mujoco_py/calibration/c3/fit_full.py`'s
`simulate`, transcribed here. A block-by-block check cannot see the order the
blocks run in, and that is what the second one catches."""

import json
import os

import casadi as ca
import numpy as np
import pytest
from crane_model import Tool
from crane_model.actuator import C3Actuator
from crane_model.conventions import default_actuator_path
from crane_model.symbolic import (
    K_LAG_AXES,
    K_PLANNED_DOF,
    NP,
    NU_PROGRESS,
    NX,
    X_ACTUATED_FORCE,
    X_COMMAND_LAG,
    X_PLANNED_VELOCITY,
    CraneSymbolicModel,
    load_actuator_fit,
)

# Not a declared dependency of the package: the ARX2 oracle is test-only.
signal = pytest.importorskip("scipy.signal")

HERE = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = os.path.join(HERE, "description", "pzs100.urdf")

DT = 0.01
#: The two forms are the same arithmetic, so only rounding separates them.
TOLERANCE = 1e-10

#: The telescope, `sa`: the stiffest axis, the one an ordering error shows on.
TELESCOPE = 3


@pytest.fixture(scope="module")
def force_rate():
    """`(x, u, p) -> actuated_force_rate` off the SX graph."""
    with open(DESCRIPTION, encoding="utf-8") as handle:
        model = CraneSymbolicModel(
            handle.read(), Tool.PZS100, actuator=load_actuator_fit()
        )
    return ca.Function(
        "force_rate", [model.x, model.u, model.p], [model.actuated_force_rate]
    )


def test_one_step_solves_the_symbolic_blocks(force_rate):
    """Blocks 2 and 3 of one `step` against the CasADi rates they discretize.

    Dead time off: `symbolic.py` deliberately omits block 1 (it lives in the
    node's predictor), so stepping with the delay on would compare two laws.
    Block 3 consumes the *post*-block-2 `u_f`, as the calibration cascade does,
    so its rate is read at the state block 2 leaves behind -- signal order
    within the sample, not a simultaneous sweep."""
    fit = load_actuator_fit()
    actuator = C3Actuator(DT, fit, dead_time_s=0.0)
    slot = {axis: index for index, axis in enumerate(K_LAG_AXES)}
    lag_axes = list(K_LAG_AXES)
    rng = np.random.default_rng(11)

    for _ in range(5):
        x = rng.normal(size=NX)
        u = rng.normal(size=NU_PROGRESS)
        # Only tau_v > 0 axes carry a u_f state; for the rest u_f *is* u.
        u_f = np.array(
            [
                x[X_COMMAND_LAG + slot[axis]] if axis in slot else u[axis]
                for axis in range(K_PLANNED_DOF)
            ]
        )
        dq = x[X_PLANNED_VELOCITY : X_PLANNED_VELOCITY + K_PLANNED_DOF]
        tau = x[X_ACTUATED_FORCE : X_ACTUATED_FORCE + K_PLANNED_DOF]

        actuator.reset(tau=tau, u=u_f)
        after = actuator.step(u[:K_PLANNED_DOF], dq)

        # Block 2 is the ZOH solution of the very ODE symbolic.py states as
        # `command_lag_rate = (u - u_f)/tau_v`: the same law, solved exactly
        # over the step rather than stepped, so the oracle is the analytic
        # answer and not a difference quotient.
        tau_v = np.array([fit.tau_v[axis] for axis in lag_axes])
        expected = u_f[lag_axes] + (1.0 - np.exp(-DT / tau_v)) * (
            u[lag_axes] - u_f[lag_axes]
        )
        assert np.allclose(actuator.u_f[lag_axes], expected, rtol=TOLERANCE, atol=0.0)

        advanced = x.copy()
        advanced[X_COMMAND_LAG : X_COMMAND_LAG + len(lag_axes)] = actuator.u_f[lag_axes]
        rate = np.asarray(force_rate(advanced, u, np.zeros(NP))).reshape(-1)
        assert np.allclose((after - tau) / DT, rate, rtol=TOLERANCE, atol=0.0)


def lagged(command, tau_v):
    """`fit_full.lagged`: exact-pole PT1, unit DC gain. tau_v == 0 passes through."""
    if tau_v <= 0.0:
        return command
    pole = np.exp(-DT / tau_v)
    return signal.lfilter([1.0 - pole], [1.0, -pole], command)


def delayed(command, n_d):
    """`fit_full.delayed`: pure dead time, n_d samples, zero before."""
    out = np.zeros_like(command)
    out[n_d:] = command[:-n_d]
    return out


def test_one_axis_reproduces_the_cascade_the_fit_was_scored_with():
    """The whole chain -- blocks 1, 2, 3 and the body -- against `fit_full`'s own
    `simulate`, transcribed above: `lfilter(arx, delayed(lagged(u)))`.

    Every constant here means what it means because it was identified against
    exactly this cascade, so this is the test that keeps the module worth the
    numbers. It is also the only one that sees the *order* the blocks run in: a
    sign flip in block 3, or a body reading tau_n instead of tau_{n+1}, leaves
    the per-block rates above intact and moves every pole."""
    with open(default_actuator_path()) as stream:
        axis = json.load(stream)["axes"]["sa"]
    k, damping, mass = float(axis["k"]), float(axis["d"]), float(axis["m_ii_median"])
    a = DT / mass

    fit = load_actuator_fit()
    actuator = C3Actuator(DT, fit)  # real 60 ms dead time, real tau_v
    n_d = round(fit.dead_time_s / DT)

    command = np.random.default_rng(3).normal(size=500)
    expected = signal.lfilter(
        [0.0, a * k * DT],
        [1.0, a * damping + a * k * DT - 2.0, 1.0 - a * damping],
        delayed(lagged(command, fit.tau_v[TELESCOPE]), n_d),
    )

    # One axis alone: the full five-vector, zero elsewhere, row 3 read back.
    drive, measured = np.zeros(K_PLANNED_DOF), np.zeros(K_PLANNED_DOF)
    achieved = np.zeros_like(command)
    for sample in range(len(command) - 1):
        drive[TELESCOPE] = command[sample]
        tau = actuator.step(drive, measured)
        measured[TELESCOPE] += a * (tau[TELESCOPE] - damping * measured[TELESCOPE])
        achieved[sample + 1] = measured[TELESCOPE]

    assert np.allclose(achieved, expected, rtol=1e-9, atol=1e-12)
