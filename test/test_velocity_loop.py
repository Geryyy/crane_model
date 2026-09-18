"""The velocity loop against the two things it is a copy of.

Oracle 1: the deployed ros2_control yaml, field by field -- `velocity_loop.yaml`
is a second copy of gains the controller reads, and nothing but this test keeps
the two honest. Oracle 2: `joint_trajectory_controller_plugins`'
`pid_trajectory_plugin.cpp::compute_commands`, whose clamp bounds the PI branch
alone while both feedforward branches are added outside it."""

import os

import numpy as np
import pytest
import yaml
from crane_model.velocity_loop import AxisGains, VelocityLoop, load_velocity_loop

DEPLOYED = os.path.join(
    "concrete_block_behavior_tree",
    "config",
    "ros2_control",
    "crane_controller_hydraulic_a2b_jtc_pid_pzs100.ros2_control.yaml",
)

#: `pid_trajectory_plugin_parameters.yaml`. An absent field takes these.
DEFAULTS = {
    "ff_velocity_scale": 0.0,
    "p": 0.0,
    "i": 0.0,
    "d": 0.0,
    "u_clamp_min": -np.inf,
    "u_clamp_max": np.inf,
}


def deployed_gains():
    """The controller's own gains, or None -- a source checkout may lack the pkg."""
    here = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(here) != "concrete_block_stack":
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent
    path = os.path.join(here, DEPLOYED)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    table = root["trajectory_controller_a2b"]["ros__parameters"]["gains"]
    return {
        joint: AxisGains(
            **{
                key: float((entry or {}).get(key, value))
                for key, value in DEFAULTS.items()
            }
        )
        for joint, entry in table.items()
    }


def test_mirrors_ros2_control():
    """`crane_model` must not depend on the behaviour-tree package at runtime, so
    the gains are duplicated; this test is the only thing holding the copies
    together. Both directions, so an axis added to one side is not silently
    dropped by the other."""
    deployed = deployed_gains()
    if deployed is None:
        pytest.skip(f"{DEPLOYED} is not in this checkout")

    ours, _ = load_velocity_loop()
    assert set(ours) == set(deployed)
    for joint, gains in ours.items():
        assert gains == deployed[joint], (
            f"{joint}: {gains} vs deployed {deployed[joint]}"
        )


def test_the_clamp_bounds_the_pi_branch_alone():
    """The documented deviation from wiki/control_architecture.md 2.1, pinned: a
    saturated PI plus the feedforward overshoots `u_clamp_max` by exactly the
    feedforward, because the C++ adds both open-loop branches outside the clamp."""
    gains = AxisGains(
        ff_velocity_scale=0.5, p=2.0, i=0.25, d=0.125, u_clamp_min=-0.3, u_clamp_max=0.4
    )
    loop = VelocityLoop([gains], 0.01)

    # Small enough that the PI stays inside the clamp: the plain law, term by term.
    u = loop.step([0.05], [0.2], [0.0], [0.1], feedforward=[0.7])
    correction = gains.p * 0.05 + gains.i * (0.05 * 0.01) + gains.d * 0.1
    assert correction < gains.u_clamp_max
    assert u[0] == pytest.approx(0.5 * 0.2 + 0.7 + correction, abs=1e-12)

    # Large e_pos: the PI saturates and the feedforward rides on top of the clamp.
    loop.reset()
    u = loop.step([50.0], [0.2], [0.0], [0.0], feedforward=[0.7])
    assert u[0] == pytest.approx(gains.u_clamp_max + 0.5 * 0.2 + 0.7, abs=1e-12)
    assert u[0] > gains.u_clamp_max
