"""
The plant-mismatch knobs: what they refuse, and that nothing is on by default.

Each refusal here is silent if missed -- `C3Actuator` validates none of them and
a wrong actuator constant is invisible in every quantity except the response.
The default-path test is the one that keeps issue 156's first acceptance
criterion true: nothing configured has to be the run that was there before.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from crane_model.errors import CraneModelError
from crane_model.mismatch import PI_RUNGS, PsiGain, perturb_fit, pi_rung
from crane_model.symbolic import K_ACTUATOR_FIT, load_actuator_fit
from crane_model.velocity_loop import load_velocity_loop


def test_nothing_configured_is_the_shipped_plant():
    """Identity in every field, and exactly -- not merely close."""
    fit, damping = perturb_fit()
    assert fit == K_ACTUATOR_FIT
    assert np.array_equal(damping, np.ones(len(K_ACTUATOR_FIT.k)))
    command = np.array([0.4, -0.2, 0.0, 0.1, -0.7])
    assert np.array_equal(PsiGain().apply(command), command)
    # In memory, plant side only: the export key digests the file, so a
    # perturbation that reached it would move the solver with the plant.
    assert load_actuator_fit() == K_ACTUATOR_FIT


def test_a_positive_lag_shift_is_refused_by_axis_name():
    """
    The arm is fitted at `tau_v = 0`, so *any* positive shift gives it a state.

    That moves `K_LAG_AXES`, hence `NX`, hence the export key: the solver would
    then refuse the very fit the plant is running.
    """
    with pytest.raises(CraneModelError, match=r"across zero on axis \['ka'\]"):
        perturb_fit(lag_shift_s=0.01)


def test_a_negative_shift_is_refused_by_the_arm_too():
    """
    `tau_v = -0.01` keeps `K_LAG_AXES` and is worse for it.

    `C3Actuator` reads a non-positive `tau_v` as `alpha = 1` -- no lag at all --
    so the dead time keeps the whole shift, the sum stops being the identified
    one and the export key never notices. Measured: a run took
    `tau_v = [0.09, 0.015, -0.01, 0.04, 0.115]` over a 70 ms dead time.
    """
    with pytest.raises(CraneModelError, match=r"across zero on axis \['ka'\]"):
        perturb_fit(lag_shift_s=-0.01)


def test_the_split_is_settable_at_constant_sum():
    """
    The mechanism, on a fit that has a lag on every axis to move.

    The shipped fit is not one: the dead time is common and the arm sits at
    `tau_v = 0`, so a shift at constant sum has nowhere to come from there.
    """
    lagged = replace(K_ACTUATOR_FIT, tau_v=(0.1, 0.025, 0.02, 0.05, 0.125))
    shifted, _ = perturb_fit(lagged, lag_shift_s=-0.01)
    assert shifted.dead_time_s == pytest.approx(lagged.dead_time_s + 0.01)
    assert np.allclose(
        np.array(shifted.tau_v) + shifted.dead_time_s,
        np.array(lagged.tau_v) + lagged.dead_time_s,
    )
    assert shifted.lag_axes == lagged.lag_axes
    # Downwards the binding axis is then whichever has the least to give.
    with pytest.raises(CraneModelError, match=r"across zero on axis \['ka'\]"):
        perturb_fit(lagged, lag_shift_s=-0.02)


def test_k_without_d_is_refused():
    """`k_i` and `d_i` are one identification; alone, `k` moves zeta silently."""
    scale = [1.2, 1.0, 1.0, 1.0, 1.0]
    with pytest.raises(CraneModelError, match=r"pair only with each other"):
        perturb_fit(k=scale)
    with pytest.raises(CraneModelError, match=r"pair only with each other"):
        perturb_fit(d=scale)
    fit, damping = perturb_fit(k=scale, d=scale)
    assert fit.k[0] == pytest.approx(1.2 * K_ACTUATOR_FIT.k[0])
    assert damping[0] == pytest.approx(1.2)


@pytest.mark.parametrize("name", ["k", "d"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_a_multiplier_outside_the_fits_domain_is_refused_by_axis_name(name, bad):
    """
    Nothing downstream checks either of them.

    `C3Actuator` takes a non-positive `k` and builds force against the command;
    a non-finite `d` multiplier reaches MuJoCo's `dof_damping` and ends the run
    somewhere unrelated to the draw.
    """
    scale = {"k": [1.0] * 5, "d": [1.0] * 5}
    scale[name][3] = bad
    with pytest.raises(CraneModelError, match=rf"the {name} multipliers leave axis"):
        perturb_fit(k=scale["k"], d=scale["d"])


def test_the_rungs_separate_the_two_branches():
    """`p` is the integral action and `d` the proportional one, not the reverse."""
    gains, _ = load_velocity_loop()
    shipped = [gains["theta1_slewing_joint"], gains["theta2_boom_joint"]]
    assert pi_rung(shipped, "full") == shipped
    assert [axis.p for axis in pi_rung(shipped, "no-integral")] == [0.0, 0.0]
    assert [axis.d for axis in pi_rung(shipped, "no-integral")] == [
        axis.d for axis in shipped
    ]
    assert [axis.d for axis in pi_rung(shipped, "feedforward")] == [0.0, 0.0]
    with pytest.raises(CraneModelError, match=r"unknown PI rung"):
        pi_rung(shipped, "none")
    assert PI_RUNGS == ("full", "no-integral", "feedforward")


def test_the_psi_gain_is_per_axis_and_per_direction():
    """A non-unit gain has to change the command, and only on its own side."""
    psi = PsiGain(
        positive=[2.0, 1.0, 1.0, 1.0, 1.0], negative=[1.0, 0.5, 1.0, 1.0, 1.0]
    )
    driven = psi.apply([0.3, -0.4, 0.2, -0.1, 0.0])
    assert driven == pytest.approx([0.6, -0.2, 0.2, -0.1, 0.0])
