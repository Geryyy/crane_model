"""
Numeric crane model on Pinocchio, in canonical coordinates.

Planning-side counterpart of the C++ library: kinematics, passive equilibrium
and geometry, no realtime dynamics. Both read the same URDF and the same
config/hydraulics.yaml, so the two agree by construction rather than by
discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from .collision import LinkGeometry
from .collision import queries as _collision_queries
from .collision import query as _collision_query
from .conventions import (
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    Frame,
    Tool,
)
from .description import parse
from .errors import CraneModelError, ErrorCode

# Residual h_u is driven to, in N m. The passive rows carry of order 1e3 N m of
# gravity terms that cancel at the equilibrium, leaving a noise floor near
# 1e-13; against a restoring stiffness of 2e3 N m/rad this is under 1e-11 rad.
EQUILIBRIUM_RESIDUAL_NM = 1.0e-8
# A runaway guard, not a working range: Newton from inside the well reaches the
# residual in single digits of iterations.
EQUILIBRIUM_ITERATIONS = 32
# Samples per passive axis in the seed search, endpoints included. Five over the
# pi of tip range puts a sample well inside the quarter turn that separates the
# hanging well from the saddles either side of it.
EQUILIBRIUM_SAMPLES = 5


@dataclass(frozen=True)
class Pose:
    expressed_in: Frame
    position_m: np.ndarray
    orientation_xyzw: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return pin.XYZQUATToSE3(
            np.concatenate([self.position_m, self.orientation_xyzw])
        ).homogeneous


@dataclass(frozen=True)
class Jacobian:
    value: np.ndarray  # 6x8, LOCAL frame, contract section 7
    expressed_in: Frame


@dataclass
class Payload:
    """A rigid body carried at K8_rotator_lower_part; inertia about its own CoM."""

    mass_kg: float = 0.0
    center_of_mass_k8_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_k8_kg_m2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    valid: bool = False


def _finite(name: str, value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise CraneModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"{name} has {array.size} entries, expected {size}",
        )
    if not np.all(np.isfinite(array)):
        raise CraneModelError(ErrorCode.NON_FINITE_INPUT, f"{name} is not finite")
    return array


class CraneModel:
    """Model built once from URDF XML; every query is read-only."""

    def __init__(
        self,
        robot_description_xml: str,
        tool: Tool,
        gravity_m_s2=(0.0, 0.0, -9.81),
        hydraulics_config: str | None = None,
        collision_config: str | None = None,
    ) -> None:
        self._description = parse(
            robot_description_xml, tool, gravity_m_s2, hydraulics_config
        )
        self._model = self._description.model
        self._data = self._model.createData()
        self._passive_slots = [
            self._description.drives[i][0].slot for i in PASSIVE_INDICES
        ]
        self._passive_lower = np.array(
            [self._model.lowerPositionLimit[slot.idx_q] for slot in self._passive_slots]
        )
        self._passive_upper = np.array(
            [self._model.upperPositionLimit[slot.idx_q] for slot in self._passive_slots]
        )
        self._mount = self._payload_mount()
        # Loaded on the first collision query: kinematics users pay nothing for
        # the geometry and need no collision table on disk.
        self._collision_config = collision_config
        self._geometry = None

    @property
    def tool(self) -> Tool:
        return self._description.tool

    def _payload_mount(self):
        """Joint carrying K8_rotator_lower_part, its placement, and bare inertia."""
        name = Frame.ROTATOR_LOWER_PART.value
        if not self._model.existFrame(name):
            return None
        frame = self._model.frames[self._model.getFrameId(name)]
        return (
            frame.parentJoint,
            frame.placement,
            self._model.inertias[frame.parentJoint].copy(),
        )

    def _attach(self, payload: Payload | None) -> None:
        if payload is None or not payload.valid:
            return
        if not np.isfinite(payload.mass_kg) or payload.mass_kg < 0.0:
            raise CraneModelError(
                ErrorCode.INVALID_PAYLOAD,
                "a valid payload needs finite, non-negative mass",
            )
        if self._mount is None:
            raise CraneModelError(
                ErrorCode.FRAME_UNAVAILABLE,
                f"the description carries no {Frame.ROTATOR_LOWER_PART.value}, "
                "so a payload cannot be attached",
            )
        joint, placement, bare = self._mount
        body = pin.Inertia(
            float(payload.mass_kg),
            np.asarray(payload.center_of_mass_k8_m, dtype=float),
            np.asarray(payload.inertia_k8_kg_m2, dtype=float),
        )
        self._model.inertias[joint] = bare + placement.act(body)

    def _detach(self) -> None:
        if self._mount is not None:
            joint, _, bare = self._mount
            self._model.inertias[joint] = bare.copy()

    def _frame_id(self, frame: Frame) -> int:
        if not self._model.existFrame(frame.value):
            raise CraneModelError(
                ErrorCode.FRAME_UNAVAILABLE,
                f"the {self.tool.value} description carries no frame '{frame.value}'",
            )
        return self._model.getFrameId(frame.value)

    def forward_kinematics(self, q, frm: Frame, to: Frame) -> Pose:
        """Return the pose of `to`, expressed in `frm`."""
        q = _finite("q", q, GENERALIZED_DOF)
        source, target = self._frame_id(frm), self._frame_id(to)
        pin.forwardKinematics(
            self._model, self._data, self._description.configuration(q)
        )
        pin.updateFramePlacements(self._model, self._data)
        relative = self._data.oMf[source].actInv(self._data.oMf[target])
        return Pose(frm, np.array(relative.translation), pin.SE3ToXYZQUAT(relative)[3:])

    def jacobian(self, q, frame: Frame) -> Jacobian:
        """
        Return the 6x8 LOCAL Jacobian in canonical coordinates.

        Coupled joints are summed into their driving coordinate's column, so the
        telescope column carries both stages.
        """
        q = _finite("q", q, GENERALIZED_DOF)
        full = pin.computeFrameJacobian(
            self._model,
            self._data,
            self._description.configuration(q),
            self._frame_id(frame),
            pin.ReferenceFrame.LOCAL,
        )
        return Jacobian(full @ self._description.projection, frame)

    def _passive_gravity(self, q: np.ndarray, passive: np.ndarray):
        """Passive gravity torque and the stiffness (Hessian) block at `passive`."""
        q = q.copy()
        q[list(PASSIVE_INDICES)] = passive
        configuration = self._description.configuration(q)
        hessian = pin.computeGeneralizedGravityDerivatives(
            self._model, self._data, configuration
        )
        gravity = (self._description.projection.T @ self._data.g)[list(PASSIVE_INDICES)]
        rows = [slot.idx_v for slot in self._passive_slots]
        stiffness = np.array(hessian)[np.ix_(rows, rows)]
        if not (np.all(np.isfinite(gravity)) and np.all(np.isfinite(stiffness))):
            raise CraneModelError(
                ErrorCode.SINGULAR_CONFIGURATION,
                "the passive gravity torque is not finite at this configuration",
            )
        # The Hessian of a potential is symmetric; the off-diagonals come out of
        # different sweeps of the same algorithm, so they are averaged.
        coupling = 0.5 * (stiffness[0, 1] + stiffness[1, 0])
        stiffness[0, 1] = stiffness[1, 0] = coupling
        return gravity, stiffness, configuration

    def _passive_torque(self, configuration: np.ndarray) -> np.ndarray:
        """Return the passive rows of the inverse dynamics at rest: contract section 7."""
        zero = np.zeros(self._model.nv)
        pin.rnea(self._model, self._data, configuration, zero, zero)
        residual = (self._description.projection.T @ self._data.tau)[
            list(PASSIVE_INDICES)
        ]
        if not np.all(np.isfinite(residual)):
            raise CraneModelError(
                ErrorCode.SINGULAR_CONFIGURATION,
                "the passive rows of the inverse dynamics are not finite at this configuration",
            )
        return residual

    def passive_equilibrium(self, q_a, payload: Payload | None = None) -> np.ndarray:
        """
        Return the passive coordinates at which the tool hangs, for `q_a`.

        A two-hinge pendulum has four critical points on the torus and two are
        the tool standing *up*; they satisfy h_u = 0 exactly as well. The sign of
        the stiffness is what separates them, so the seed is chosen only among
        samples with a positive definite one, and the condition is rechecked at
        every iterate rather than at the end.
        """
        q_a = _finite("q_a", q_a, len(ACTUATED_INDICES))
        q = np.zeros(GENERALIZED_DOF)
        q[list(ACTUATED_INDICES)] = q_a

        self._attach(payload)
        try:
            seed, best = None, np.inf
            axes = [
                np.linspace(
                    self._passive_lower[i], self._passive_upper[i], EQUILIBRIUM_SAMPLES
                )
                for i in range(len(PASSIVE_INDICES))
            ]
            for tip in axes[0]:
                for tilt in axes[1]:
                    sample = np.array([tip, tilt])
                    try:
                        gravity, stiffness, _ = self._passive_gravity(q, sample)
                        np.linalg.cholesky(stiffness)
                    except (CraneModelError, np.linalg.LinAlgError):
                        continue
                    if np.linalg.norm(gravity) < best:
                        best, seed = np.linalg.norm(gravity), sample
            if seed is None:
                raise CraneModelError(
                    ErrorCode.SINGULAR_CONFIGURATION,
                    "no pose in the passive joint range has a restoring stiffness at this q_a, "
                    "so there is nowhere for the tool to hang",
                )

            passive = seed
            for _ in range(EQUILIBRIUM_ITERATIONS):
                _, stiffness, configuration = self._passive_gravity(q, passive)
                residual = self._passive_torque(configuration)
                try:
                    factor = np.linalg.cholesky(stiffness)
                except np.linalg.LinAlgError:
                    raise self._unreachable() from None
                if np.linalg.norm(residual) <= EQUILIBRIUM_RESIDUAL_NM:
                    return passive
                step = np.linalg.solve(factor.T, np.linalg.solve(factor, residual))
                passive = passive - step
                if not np.all(np.isfinite(passive)):
                    raise self._unreachable()
            raise self._unreachable()
        finally:
            self._detach()

    @staticmethod
    def _unreachable() -> CraneModelError:
        return CraneModelError(
            ErrorCode.SINGULAR_CONFIGURATION,
            "the passive joints reach no hanging pose from inside the range the description "
            "gives them at this q_a",
        )

    def _link_geometry(self) -> LinkGeometry:
        if self._geometry is None:
            self._geometry = LinkGeometry(
                self._model, self.tool, self._collision_config
            )
        return self._geometry

    def collision_query(self, q, scene):
        """
        Return the worst distance between the machine and `scene`.

        Scene poses are in K0_mounting_base. Self pairs the fit calls not worth
        checking are excluded; everything else is checked, so a configuration
        folded back over itself is refused against an empty scene too.
        """
        q = _finite("q", q, GENERALIZED_DOF)
        return _collision_query(
            self._model,
            self._data,
            self._link_geometry(),
            self._description.configuration(q),
            scene,
        )

    def collision_queries(self, q, scene) -> list:
        """One result per scene primitive, in scene order, then one for self."""
        q = _finite("q", q, GENERALIZED_DOF)
        return _collision_queries(
            self._model,
            self._data,
            self._link_geometry(),
            self._description.configuration(q),
            scene,
        )

    @staticmethod
    def actuated(q) -> np.ndarray:
        return np.asarray(q, dtype=float)[list(ACTUATED_INDICES)]

    @staticmethod
    def passive(q) -> np.ndarray:
        return np.asarray(q, dtype=float)[list(PASSIVE_INDICES)]
