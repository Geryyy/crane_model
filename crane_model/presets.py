"""
Where the standalone tuning runs start and what they aim at.

`OUTSIDE` is `epsilon_crane_description/config/initialization_outside.yaml` on
the canonical eight; its `q9_0` is the eighth coordinate, the tool axis. Goals
are TCP placements in `K0_mounting_base`, the frame `PlanMotion` carries.
"""

from __future__ import annotations

import numpy as np

#: theta1, theta2, theta3, q4, theta6 (tip), theta7 (tilt), theta8, q9 (tool).
#: The passive pair is taken as written and sits ~20 mrad off its own rest.
OUTSIDE = np.array(
    [0.785000, 0.523599, 0.523602, 0.250000, 0.546470, 1.570521, 0.000000, 0.210000]
)

#: Out and up: slew, boom and telescope at once.
GOAL_OUT = np.array([4.0, 0.0, 2.0])


def goal_here(model, q) -> np.ndarray:
    """Lift the start TCP to 2 m, leaving x and y alone."""
    from .conventions import Frame

    pose = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    return np.array([pose.position_m[0], pose.position_m[1], 2.0])
