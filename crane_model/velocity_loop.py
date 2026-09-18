"""
The low-level velocity loop, as the machine runs it.

A transcription of `joint_trajectory_controller_plugins::PidTrajectoryPlugin`
(`compute_commands`) over `control_toolbox::Pid`, so a gain validated here is
the gain that ships. Gains and their provenance: `config/velocity_loop.yaml`.

    u = ff_velocity_scale * dq_ref + feedforward
        + clamp(p e_pos + i int(e_pos) + d e_vel, u_clamp_min, u_clamp_max)

`feedforward` is the trajectory's `effort` field, which under
`effort_field_is_feedforward` carries C3's inversion previewed by the dead time
(`crane_planning` writes it). Both open-loop branches are added *outside* the
clamp, exactly as the C++ does: the clamp bounds the PI alone, never the total.

Four deviations from wiki/control_architecture.md 2.1 are reproduced on purpose,
because the deployed loop has them. Do not "fix" them here -- that would leave
the sim validating a controller nobody runs:

1. `p` == K_I and `d` == K_P. On a velocity command interface int(e_qdot) dt is
   e_pos, so the position-error gain *is* the integral action. `i` stays 0.
2. the clamp bounds the PI output, not the command.
3. no anti-windup. At `i == 0` control_toolbox's integral term is identically
   zero, and the real integral action is `p e_pos` -- a plant state, which
   cannot be reset, clamped or back-calculated from inside the controller.
4. so an axis held against a stall accumulates e_pos at dq_ref and comes off the
   stall with P at its clamp. That is the failure mode, and it should show here.

Psi is not modelled: it is an input linearization that exists only on the
machine, assumed exact (wiki/control_architecture.md 119). `u_clamp_*` is still
Psi's identified domain, so it stays -- it is the only bound on the PI branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from .conventions import (
    ACTUATED_INDICES,
    canonical_joints,
    default_velocity_loop_path,
)
from .errors import CraneModelError, ErrorCode

#: Plugin schema defaults (`pid_trajectory_plugin_parameters.yaml`). An absent
#: field here means the deployed yaml leaves it absent too, so it takes these.
_DEFAULTS = {
    "ff_velocity_scale": 0.0,
    "p": 0.0,
    "i": 0.0,
    "d": 0.0,
    "u_clamp_min": -np.inf,
    "u_clamp_max": np.inf,
}


@dataclass(frozen=True)
class AxisGains:
    """One axis' gains, spelled as the plugin's parameter map spells them."""

    ff_velocity_scale: float
    p: float
    i: float
    d: float
    u_clamp_min: float
    u_clamp_max: float


def load_velocity_loop(path=None) -> tuple[dict[str, AxisGains], float]:
    """
    Read `config/velocity_loop.yaml`: gains by joint name, and the design rate.

    A missing axis is an exception, never a default: an untuned axis reaching
    the plant at gain 0 looks like a tracking result, not like a missing entry.
    """
    path = path if path is not None else default_velocity_loop_path()
    with open(path, encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    table = root.get("gains")
    if not isinstance(table, dict):
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT, f"{path}: carries no gains mapping"
        )
    names = canonical_joints()
    gains = {}
    for index in ACTUATED_INDICES:
        joint = names[index]
        if joint not in table:
            raise CraneModelError(
                ErrorCode.MISSING_JOINT,
                f"{path}: no gains for actuated joint {joint!r}",
            )
        entry = table[joint] or {}
        unknown = set(entry) - set(_DEFAULTS)
        if unknown:
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT,
                f"{path}: {joint} carries unknown gain(s) {sorted(unknown)}",
            )
        values = {
            key: float(entry.get(key, default)) for key, default in _DEFAULTS.items()
        }
        if values["u_clamp_min"] > values["u_clamp_max"]:
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT, f"{path}: {joint} has an inverted u_clamp"
            )
        gains[joint] = AxisGains(**values)
    return gains, float(root["rate_hz"])


class VelocityLoop:
    """The PidTrajectoryPlugin law over a fixed set of axes, fixed step."""

    def __init__(self, gains, timestep: float, continuous=None) -> None:
        self.gains = tuple(gains)
        self.timestep = float(timestep)
        if not (self.timestep > 0.0):
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT, f"timestep must be positive, got {timestep}"
            )
        self._ff = np.array([axis.ff_velocity_scale for axis in self.gains])
        self._p = np.array([axis.p for axis in self.gains])
        self._i = np.array([axis.i for axis in self.gains])
        self._d = np.array([axis.d for axis in self.gains])
        self._low = np.array([axis.u_clamp_min for axis in self.gains])
        self._high = np.array([axis.u_clamp_max for axis in self.gains])
        # The JTC takes a continuous joint's position error through
        # `angles::shortest_angular_distance` (joint_trajectory_controller.cpp,
        # `compute_error_for_joint`). The rotator is one, and `p` is its integral
        # action, so past pi a plain subtraction commands the other way round.
        self._wrap = (
            np.zeros(len(self.gains), dtype=bool)
            if continuous is None
            else np.asarray(continuous, dtype=bool).reshape(len(self.gains))
        )
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(len(self.gains))

    def step(self, q_ref, dq_ref, q, dq, feedforward=None) -> np.ndarray:
        """One controller tick. Returns `u`, the joint velocity at Psi's input."""
        width = len(self.gains)
        q_ref, dq_ref = np.asarray(q_ref, float), np.asarray(dq_ref, float)
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        error = q_ref.reshape(width) - q.reshape(width)
        error = np.where(self._wrap, (error + np.pi) % (2.0 * np.pi) - np.pi, error)
        rate = dq_ref.reshape(width) - dq.reshape(width)
        # Plain integration: with i == 0 it is inert, and control_toolbox's
        # anti-windup strategies only act on this term. Set i and they diverge.
        self.integral = self.integral + error * self.timestep
        correction = np.clip(
            self._p * error + self._i * self.integral + self._d * rate,
            self._low,
            self._high,
        )
        forward = self._ff * dq_ref.reshape(width)
        if feedforward is not None:
            forward = forward + np.asarray(feedforward, float).reshape(width)
        return forward + correction
