"""
What the machine measures of C3's force state.

`X_ACTUATED_FORCE` is the one state nothing on the crane observes -- the MPC
carries it from the previous solution. But `/cranedata` reports both chamber
pressures per axis, and pressure times the force-producing area times the
cylinder transmission *is* that force. This is that conversion, numerically.

Deliberately **not** in `symbolic.py`: `solver.py`'s `export_key` digests that
whole file, so a line added there invalidates every cached acados export in the
workspace. Nothing here enters the OCP.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .errors import CraneModelError, ErrorCode
from .symbolic import (
    K_ACTUATED_DOF,
    K_PLANNED_DOF,
    Constants,
    axis_areas,
    jacobian_diagonal,
)

#: `/cranedata` reports chamber pressure in bar; the areas are m^2.
BAR_TO_PA = 1.0e5


def cylinder_ratios(constants: Constants, q2, q3) -> np.ndarray:
    """
    `jacobian_diagonal` as six floats -- its constant entries stay SX.

    Raises where the boom four-bar does not close: `jacobian_diagonal` takes
    `sqrt` of a negative there and CasADi returns NaN rather than raising, and
    a NaN force row propagates into the carried rows and then *disappears* from
    any score that masks on a magnitude. Loud beats silent.
    """
    ratio = np.array(
        [
            float(ca.evalf(ca.SX(entry)))
            for entry in jacobian_diagonal(constants, q2, q3)
        ]
    )
    if not np.all(np.isfinite(ratio)):
        raise CraneModelError(
            ErrorCode.SINGULAR_CONFIGURATION,
            f"no cylinder transmission at q2={q2}, q3={q3}: {ratio}",
        )
    return ratio


def actuated_force_from_pressure(constants: Constants, q2, q3, p_a, p_b) -> np.ndarray:
    """
    One sample of measured chamber pressure to `X_ACTUATED_FORCE`'s five rows.

    `tau = (A_A p_A - A_B p_B) J_cyl` -- `CraneSymbolicModel.chamber_force`
    times the transmission, which inverts the output map's
    `cylinder_force = tau_a / ratio`, so the result carries that state's own
    units and sign. Use `a_a`/`a_b`, not `a_eff_*`: those are direction-
    dependent pump draw and do not multiply pressure.

    **Pressures in Pa** -- scale `/cranedata` by `BAR_TO_PA`. `p_a`/`p_b` are
    six long in `K_ACTUATED_ROWS` order; the tool column is read and dropped.
    Pose enters only through the boom and arm angles, the two transmissions
    that are not constant.
    """
    p_a = np.asarray(p_a, dtype=float).reshape(K_ACTUATED_DOF)
    p_b = np.asarray(p_b, dtype=float).reshape(K_ACTUATED_DOF)
    areas = axis_areas(constants)
    ratio = cylinder_ratios(constants, q2, q3)
    return np.array(
        [
            (areas[axis].a_a * p_a[axis] - areas[axis].a_b * p_b[axis]) * ratio[axis]
            for axis in range(K_PLANNED_DOF)
        ]
    )
