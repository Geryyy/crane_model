"""
`actuated_force_from_pressure` against the closed forms the geometry already has.

Two axes need no linkage solved: slewing's transmission is the constant rack
radius `r_gear`, telescope's is one-to-one. Those pin the areas, the units and
the factor.

The boom and the arm have no such closed form, and **nothing else in this
package tests `jacobian_diagonal` numerically** -- the symbolic parity gtest
that once did went with 104's test-surface strip (`README.md`). So the oracle
for those two columns is the same graph evaluated the other way: built over
`ca.SX` symbols and called as a `ca.Function`, against `cylinder_ratios`'
build-over-floats-and-`evalf`. That is what `cylinder_ratios` adds and the only
thing it can get wrong; it does not re-derive the linkage.
"""

import casadi as ca
import numpy as np
import pytest
from crane_model import symbolic as cs
from crane_model.conventions import hydraulic_limits
from crane_model.errors import CraneModelError
from crane_model.pressure import (
    BAR_TO_PA,
    actuated_force_from_pressure,
    cylinder_ratios,
)

CONSTANTS = cs.load_constants()
AREAS = cs.axis_areas(CONSTANTS)
#: Any pose; only the boom and arm columns can notice it.
Q2, Q3 = 0.4, 0.7
RATIO = cylinder_ratios(CONSTANTS, Q2, Q3)[: cs.K_PLANNED_DOF]


def force(p_a, p_b):
    return actuated_force_from_pressure(CONSTANTS, Q2, Q3, p_a, p_b)


def loaded(axis, a_bar, b_bar):
    p_a, p_b = np.zeros(cs.K_ACTUATED_DOF), np.zeros(cs.K_ACTUATED_DOF)
    p_a[axis], p_b[axis] = a_bar * BAR_TO_PA, b_bar * BAR_TO_PA
    return p_a, p_b


def test_slewing_is_pressure_times_area_times_the_rack_radius():
    # Both chambers see 2 A_A, so only the pressure difference survives.
    tau = force(*loaded(0, 200.0, 50.0))
    expected = 2.0 * CONSTANTS.slewing_a_a * 150.0 * BAR_TO_PA * CONSTANTS.r_gear
    assert tau[0] == pytest.approx(expected)
    # At system pressure that is the axis' rated force, 31810 N m -- the number
    # the simulator's actuator clamps at, which fixes the unit scale as Pa.
    rated = force(*loaded(0, hydraulic_limits()["system_pressure_pa"] / BAR_TO_PA, 0.0))
    assert rated[0] == pytest.approx(31810.0, rel=1e-3)


def test_telescope_is_one_to_one_so_the_joint_force_is_the_cylinder_force():
    tau = force(*loaded(3, 120.0, 30.0))
    expected = (
        CONSTANTS.telescope_a_a * 120.0 * BAR_TO_PA
        - CONSTANTS.telescope_a_b * 30.0 * BAR_TO_PA
    )
    assert tau[3] == pytest.approx(expected)


def test_rotator_carries_the_motor_displacement_not_an_area():
    assert force(*loaded(4, 180.0, 20.0))[4] == pytest.approx(
        CONSTANTS.v_m * 160.0 * BAR_TO_PA
    )


def test_the_sign_follows_the_loaded_chamber_in_both_directions():
    # A high, B low is the extending force on all five; swapping the two
    # reverses it. Magnitudes differ -- the areas are not symmetric -- so only
    # the sign is asserted, and it carries the transmission's own sign.
    high = np.full(cs.K_ACTUATED_DOF, 200.0 * BAR_TO_PA)
    low = np.full(cs.K_ACTUATED_DOF, 5.0 * BAR_TO_PA)
    assert np.all(np.sign(force(high, low)) == np.sign(RATIO))
    assert np.all(np.sign(force(low, high)) == -np.sign(RATIO))


def symbolic_ratios(q2, q3):
    """The same transmission through CasADi's own evaluator, not `evalf`."""
    symbol = ca.SX.sym("q", 2)
    graph = ca.Function(
        "j",
        [symbol],
        [ca.vertcat(*cs.jacobian_diagonal(CONSTANTS, symbol[0], symbol[1]))],
    )
    return np.asarray(graph([q2, q3])).reshape(cs.K_ACTUATED_DOF)


@pytest.mark.parametrize("q2, q3", [(-0.4, 0.0), (0.0, 0.7), (0.9, 1.4), (1.5, 2.4)])
def test_the_boom_and_arm_columns_agree_with_the_symbolic_evaluator(q2, q3):
    assert cylinder_ratios(CONSTANTS, q2, q3) == pytest.approx(
        symbolic_ratios(q2, q3), rel=1e-12
    )


def test_a_boom_angle_the_four_bar_cannot_reach_raises_rather_than_returning_nan():
    # Below closure `boom_ratio` takes sqrt of a negative and CasADi hands back
    # NaN. Silently seeding that would poison the carried rows and then vanish
    # from any score that masks on a magnitude.
    assert np.isnan(symbolic_ratios(-1.3, 0.7)[1])
    zero = np.zeros(cs.K_ACTUATED_DOF)
    with pytest.raises(CraneModelError):
        actuated_force_from_pressure(CONSTANTS, -1.3, 0.7, zero, zero)


def test_the_transmission_multiplies_the_chamber_force_and_the_tool_is_dropped():
    # Direction, against a ratio this module did not compute: the boom column
    # must be chamber force *times* ds/dq, not divided by it -- the two differ
    # by a factor of 4.9 there and the output map fixes which.
    p_a = np.full(cs.K_ACTUATED_DOF, 175.0 * BAR_TO_PA)
    p_b = np.full(cs.K_ACTUATED_DOF, 60.0 * BAR_TO_PA)
    chamber = np.array(
        [
            AREAS[axis].a_a * p_a[axis] - AREAS[axis].a_b * p_b[axis]
            for axis in range(cs.K_PLANNED_DOF)
        ]
    )
    tau = force(p_a, p_b)
    assert tau.shape == (cs.K_PLANNED_DOF,)
    assert tau == pytest.approx(chamber * symbolic_ratios(Q2, Q3)[: cs.K_PLANNED_DOF])
