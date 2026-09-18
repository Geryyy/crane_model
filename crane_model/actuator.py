"""
C3 numerically: the actuation system the hydraulics present to the rigid body.

`symbolic.py` carries the same law as CasADi for the OCP; this is the integrated
form, for driving a plant. One law, two domains, and `test_actuator.py` holds
them together -- do not re-fit anything here.

Blocks 1-4 of wiki/hydraulics.md 8, in signal order:

    1  dead time    u_d[n] = u[n - n_d]                common 60 ms, transport
    2  PT1 lag      u_f   += (1 - e^-dt/tau_v)(u_d - u_f)    per axis, tau_v >= 0
    3  force state  tau   += k (u_f - dq) dt                 per axis
    4  clamp        |tau| <= tau_max                         per axis

The discretisation is the fit's, not a choice: `calibration/c3/fit_full.py`
runs block 2 as an exact-pole PT1 (`lagged`) and block 3 as explicit Euler
(`arx`), in that order within a sample, and `k` and `tau_v` mean what they mean
because they were identified against exactly that. Euler on block 2 would be a
different lag -- 21 % short on the boom at a 10 ms step -- and would make the
answer depend on the plant step. The exact pole is also the ZOH solution of
`symbolic.py`'s continuous `(u - u_f)/tau_v`, so the two agree as dt shrinks.

Block 1 is here and *not* in `symbolic.py`: there it would double-count the
node's predictor. A plant is the other side of that -- it has to answer late.

Block 4 is **off by default and you almost always want it on**. Nothing else
bounds this force: `MujocoPlant` drives `qfrc_applied`, which MuJoCo does not
limit, so an axis commanded into a joint limit winds `tau` without bound -- the
boom passes its rated 2e5 N m in 1.5 s at the harness's own default rate, and
the plant then reports motion the relief valve forbids. `MujocoPlant.effort_limits`
is the description's answer for `tau_max`. The OCP bounds the same state through
constraint 6, so leaving it off here makes the numeric domain the permissive one.

Blocks 5 (the load) and 6 (the integrator) belong to whoever integrates the body.
Damping `d` is **not here**: it is already in the description as joint damping,
byte-identical to the fit's `d` column, so a copy would double-count it.

`u` is a joint velocity at Psi's input, rad/s (m/s on the telescope). Psi itself
is not modelled -- it is an input linearization that only exists on the machine,
and every consumer here assumes it exact.
"""

from __future__ import annotations

import numpy as np

from .errors import CraneModelError, ErrorCode
from .symbolic import K_ACTUATOR_FIT, K_PLANNED_DOF, ActuatorFit


class C3Actuator:
    """C3 blocks 1-3 on the five planned axes, fixed step, explicit Euler."""

    def __init__(
        self,
        timestep: float,
        fit: ActuatorFit | None = None,
        dead_time_s: float | None = None,
        tau_max=None,
    ) -> None:
        self.fit = fit if fit is not None else K_ACTUATOR_FIT
        self.timestep = float(timestep)
        if not (self.timestep > 0.0):
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT, f"timestep must be positive, got {timestep}"
            )
        dead_time = (
            float(self.fit.dead_time_s) if dead_time_s is None else float(dead_time_s)
        )
        if dead_time < 0.0:
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT, f"dead_time_s is negative: {dead_time}"
            )
        # A fractional delay would be silently rounded and every phase margin
        # measured against it would be wrong. 60 ms divides both rates we run.
        samples = dead_time / self.timestep
        self._n_delay = int(round(samples))
        if abs(samples - self._n_delay) > 1.0e-9:
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT,
                f"dead time {dead_time} s is not a whole number of {self.timestep} s "
                f"steps ({samples}); pick a step that divides it",
            )
        self._k = np.asarray(self.fit.k, dtype=float)
        tau_v = np.asarray(self.fit.tau_v, dtype=float)
        # Exact pole, so block 2 is unconditionally stable and step-independent.
        # tau_v == 0 is a pole at infinity and falls out as alpha == 1: u_f is
        # then the command, not a state, with no branch needed.
        self._alpha = np.where(
            tau_v > 0.0,
            1.0 - np.exp(-self.timestep / np.where(tau_v > 0.0, tau_v, 1.0)),
            1.0,
        )
        self._tau_max = (
            None
            if tau_max is None
            else np.abs(np.asarray(tau_max, dtype=float)).reshape(K_PLANNED_DOF)
        )
        if self._tau_max is not None and not np.all(self._tau_max > 0.0):
            raise CraneModelError(
                ErrorCode.INVALID_ARGUMENT, "tau_max carries a non-positive entry"
            )
        self.reset()

    # --- state ---------------------------------------------------------------

    def reset(self, tau=None, u=None) -> None:
        """
        Clear the queue and both states. `tau` seeds a non-zero hold.

        The queue is filled with `u_f`, i.e. "the command has been there longer
        than the dead time". That is what lets `step` delay *before* the PT1
        while the fit (`fit_full.simulate`) delays after it: LTI blocks commute
        from a consistent state, and a consistent state is the only kind this
        can build. Reach past it into `_queue` and that stops being true.
        """
        self.tau = np.zeros(K_PLANNED_DOF) if tau is None else self._vector(tau, "tau")
        self.u_f = np.zeros(K_PLANNED_DOF) if u is None else self._vector(u, "u")
        self._queue = [self.u_f.copy() for _ in range(self._n_delay)]

    def _vector(self, value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(K_PLANNED_DOF)
        if not np.all(np.isfinite(vector)):
            raise CraneModelError(
                ErrorCode.NON_FINITE_INPUT, f"{name} carries a non-finite entry"
            )
        return vector.copy()

    # --- driving it ----------------------------------------------------------

    def step(self, u, dq) -> np.ndarray:
        """
        Advance one `timestep` under command `u` at measured rate `dq`.

        Returns the force *after* the step, so a body integrated symplectically
        against the return value reproduces the fit's ARX2 exactly (see
        `calibration/c3`'s `simulate`); `test_actuator.py` pins that.
        """
        command = self._vector(u, "u")
        rate = self._vector(dq, "dq")
        if self._n_delay:
            self._queue.append(command)
            command = self._queue.pop(0)
        self.u_f = self.u_f + self._alpha * (command - self.u_f)
        # Block 3 reads the u_f block 2 just produced, not the one it replaced:
        # within a sample the fit is a cascade, not a simultaneous sweep.
        self.tau = self.tau + self._k * (self.u_f - rate) * self.timestep
        # Block 4. Clamping the state, not the output: the force is what winds
        # up, so releasing an axis off a limit must not first unwind a decade.
        if self._tau_max is not None:
            self.tau = np.clip(self.tau, -self._tau_max, self._tau_max)
        return self.tau.copy()
