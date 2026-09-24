"""
The plant-side deviations the harness can be given, and the ones it refuses.

`sim_chain.py` drives MuJoCo through the same C3 fit the exported OCP carries,
so with nothing set here every number it produces is a nominal-plant number --
the MPC measured against its own model. This module is the other side: a Psi
gain error and a perturbed fit, applied to the *simulated machine* while the
solver keeps the shipped one. The asymmetry is the mismatch; perturbing both
sides measures nothing.

Nothing here writes `config/c3_full_model.json`. The export key digests that
file, so a perturbation on disk would move the solver with the plant, and the
failure would be silent.

Identity in every field by default: a run configuring nothing has to be the run
that was there before this module existed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .errors import CraneModelError, ErrorCode
from .symbolic import K_ACTUATOR_FIT, K_AXIS_KEYS, K_PLANNED_DOF, ActuatorFit

#: The fit keys of the five planned axes, which is what a refusal names. The
#: gripper is in `K_AXIS_KEYS` but not planned, so it never reaches C3.
PLANNED_KEYS = K_AXIS_KEYS[:K_PLANNED_DOF]


class PsiGain:
    """
    Psi's static gain error: `dq_a_set,i = g_i^+- dq_a_ff_fb,i`.

    Psi is assumed exact everywhere in this stack and on the boom it
    demonstrably is not -- the 2-D fit falls back to 1-D in simulation
    (wiki/control_architecture.md 2.1), so the boom there is a different plant
    than the hardware presents. The *shape* of that error is known -- static,
    per axis, per direction, since Psi is fitted per axis and mostly per
    direction -- and its size is not. Hence a multiplier and not noise.

    Sits on the sum, between the velocity loop's output and C3, which is where
    Psi itself sits. Identity by default, and exactly so: `1.0 * u == u`.
    """

    def __init__(self, positive=None, negative=None) -> None:
        self.positive = self._side(positive)
        self.negative = self._side(negative)

    @staticmethod
    def _side(value) -> np.ndarray:
        if value is None:
            return np.ones(K_PLANNED_DOF)
        return np.asarray(value, dtype=float).reshape(K_PLANNED_DOF).copy()

    def apply(self, u) -> np.ndarray:
        """`u` as the Paltronic receives it, once Psi got the gain wrong."""
        u = np.asarray(u, dtype=float)
        return np.where(u >= 0.0, self.positive, self.negative) * u


def perturb_fit(fit: ActuatorFit | None = None, *, k=None, d=None, lag_shift_s=0.0):
    """
    Build the fit the *plant* runs: the shipped one under per-axis multipliers.

    `k` and `d` are multipliers in `PLANNED_KEYS` order. `lag_shift_s` moves the
    lag/dead-time split at constant sum -- `tau_v,i + t_d` is what the staircase
    campaign identified, the 60 ms is pinned by convention, and the node's
    predictor carries only the dead-time half, so the split is a thing the model
    can be wrong about on its own.

    The dead time is *common* to all axes while `tau_v` is per axis, so a shift
    at constant sum has to move every axis at once -- and the arm is fitted at
    `tau_v = 0`, with nothing to give in either direction. On the shipped fit
    every non-zero shift is therefore refused, by that axis' name. The knob
    still means something on a refit that gives the arm a lag.

    Returns the perturbed `ActuatorFit` *and* `d`'s multipliers: damping is not
    an `ActuatorFit` field. It reaches the dynamics as MuJoCo's `dof_damping`
    (`actuator.py`), so scaling it is the caller's to do, on the plant.
    """
    fit = K_ACTUATOR_FIT if fit is None else fit
    shift = float(lag_shift_s)
    # `d_i` and `k_i` are one identification. Scaling `k` alone leaves zeta =
    # d/(2 sqrt(k M_ii)) somewhere no campaign measured, and nothing downstream
    # can see that it happened.
    if (k is None) != (d is None):
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT,
            "k and d are one identification and pair only with each other: "
            f"got k={k!r} with d={d!r}. Scaling k alone moves zeta silently -- "
            "pass both multipliers or neither",
        )
    scale_k = np.ones(K_PLANNED_DOF) if k is None else _multipliers(k, "k")
    scale_d = np.ones(K_PLANNED_DOF) if d is None else _multipliers(d, "d")
    stiffness = np.asarray(fit.k, dtype=float) * scale_k

    lagged = np.asarray(fit.tau_v, dtype=float) + shift
    # tau_v == 0 is a pole at infinity and gets no `u_f` state, so a sample that
    # gives an axis a lag it had not, or takes one away, changes K_LAG_AXES,
    # hence NX, hence the export key: the solver would then refuse the very fit
    # the plant is running. A *negative* tau_v keeps K_LAG_AXES and is worse for
    # it -- `C3Actuator` reads it as alpha = 1, so the lag silently clips to zero
    # while the dead time keeps the whole shift and the sum is no longer the
    # identified one.
    moved = [
        PLANNED_KEYS[axis]
        for axis in range(K_PLANNED_DOF)
        if (fit.tau_v[axis] > 0.0) != (lagged[axis] > 0.0) or lagged[axis] < 0.0
    ]
    if moved:
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"a {shift:+.4g} s lag shift takes tau_v across zero on axis "
            f"{moved}, which moves K_LAG_AXES, NX and the export key. The arm is "
            "fitted at tau_v = 0 and the dead time is common, so on the shipped "
            "fit the split cannot move in either direction",
        )

    perturbed = replace(
        fit,
        k=tuple(float(value) for value in stiffness),
        tau_v=tuple(float(value) for value in lagged),
        dead_time_s=float(fit.dead_time_s) - shift,
    )
    return perturbed, scale_d


def _multipliers(value, name: str) -> np.ndarray:
    """
    One positive, finite multiplier per planned axis, or a refusal naming the axis.

    Nothing downstream checks either: `C3Actuator` takes a non-positive `k` and
    builds force *against* the command, which reads as an instability rather than
    as a bad sample, and a non-finite `d` multiplier reaches MuJoCo's
    `dof_damping` and ends the run somewhere unrelated to the draw that caused it.
    """
    try:
        scale = np.asarray(value, dtype=float).reshape(K_PLANNED_DOF).copy()
    except ValueError as error:
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"{name} needs one multiplier per planned axis {PLANNED_KEYS}: {error}",
        ) from error
    bad = [
        f"{PLANNED_KEYS[axis]}={scale[axis]:.6g}"
        for axis in range(K_PLANNED_DOF)
        if not (np.isfinite(scale[axis]) and scale[axis] > 0.0)
    ]
    if bad:
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"the {name} multipliers leave axis {bad} outside the fit's own "
            "domain, which wants both positive and finite",
        )
    return scale


#: The three PI rungs the delay sweep runs. `p` is the integral action and `d`
#: the proportional one (`velocity_loop.py`), so dropping `p` alone leaves a
#: pure proportional loop and dropping both leaves feedforward. Two rungs would
#: conflate the two.
PI_RUNGS = ("full", "no-integral", "feedforward")


def pi_rung(gains, rung: str) -> list:
    """`AxisGains` per axis with the named rung's gains; `full` is as shipped."""
    if rung not in PI_RUNGS:
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT, f"unknown PI rung {rung!r}; one of {PI_RUNGS}"
        )
    if rung == "full":
        return list(gains)
    zeroed = {"p": 0.0} if rung == "no-integral" else {"p": 0.0, "d": 0.0}
    return [replace(axis, **zeroed) for axis in gains]
