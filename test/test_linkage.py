"""The closed cylinder chains against the description's own answer for them.

Oracle: `epsilon_crane_description/config/initialization_*.yaml`, which spells
the four boom-chain joints and the two arm-cylinder ones at poses somebody
closed by hand. Six of the seven shipped files agree with this solve; the
seventh, `abovelog`, disagrees with the other six by up to 0.09 rad and 46 mm
and is left out as stale. `_ZERO` is fitted at `outside`, so that row is a
check of the fit and `horizontal`, `transport` and `truckbed` are checks of the
geometry.
"""

import numpy as np
import pytest
from crane_model.linkage import (
    ARM_BARREL,
    ARM_STROKE,
    BOOM_BARREL,
    BOOM_BIG,
    BOOM_SMALL,
    BOOM_STROKE,
    cylinder_joints,
    load_geometry,
)

#: q2, q3, then the yaml's barrel, stroke, big, small, arm angle, arm length.
POSES = (
    (
        "outside",
        0.523599,
        0.523602,
        0.000685,
        1.847141,
        0.253841,
        -0.355826,
        -1.796361,
        1.859136,
    ),
    (
        "horizontal",
        0.0,
        1.570800,
        -0.006120,
        1.614207,
        -0.158245,
        -0.087064,
        -1.603140,
        2.149353,
    ),
    (
        "transport",
        -1.0,
        4.601500,
        0.166374,
        1.266441,
        -0.924512,
        1.070169,
        -1.395477,
        1.247279,
    ),
    (
        "truckbed",
        1.174300,
        -0.612700,
        0.064649,
        2.096373,
        0.748704,
        -0.375024,
        -1.770831,
        1.341303,
    ),
)


@pytest.mark.parametrize("pose", POSES, ids=[row[0] for row in POSES])
def test_matches_the_shipped_initialisations(pose):
    """Within 2.4 mrad and 0.3 mm -- the yaml's own rounding is 1e-6."""
    _, q2, q3, barrel, stroke, big, small, arm_angle, arm_stroke = pose
    placed = cylinder_joints(load_geometry(), q2, q3)
    expected = {
        BOOM_BARREL: barrel,
        BOOM_STROKE: stroke,
        BOOM_BIG: big,
        BOOM_SMALL: small,
        ARM_BARREL[0]: arm_angle,
        ARM_BARREL[1]: arm_angle,
        ARM_STROKE[0]: arm_stroke,
        ARM_STROKE[1]: arm_stroke,
    }
    for joint, value in expected.items():
        assert placed[joint] == pytest.approx(value, abs=2.5e-3), joint


def test_out_of_reach_is_not_a_pose():
    """Past the four-bar's reach there is no coupler point, and it says so."""
    with np.errstate(invalid="ignore"):
        placed = cylinder_joints(load_geometry(), 3.0, 0.0)
    assert np.isnan(placed[BOOM_STROKE])
