"""
Parse one URDF onto the canonical eight coordinates.

The single owner of the canonical-to-description mapping. The description is
read twice on purpose: Pinocchio builds the kinematic tree and drops `<mimic>`
-- with mimic parsing on it refuses the PZS100 outright, which declares the
right rail as a mimic of a joint later in its own depth-first order -- so the
coupled joints come from the XML itself rather than from a table here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from .conventions import GENERALIZED_DOF, Tool, canonical_joints
from .errors import CraneModelError, ErrorCode


@dataclass(frozen=True)
class JointSlot:
    idx_q: int
    nq: int
    idx_v: int


@dataclass(frozen=True)
class Drive:
    """One description joint driven by a canonical coordinate."""

    slot: JointSlot
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class Description:
    tool: Tool
    model: object
    neutral: np.ndarray
    names: tuple[str, ...]
    drives: tuple[tuple[Drive, ...], ...]
    damping: tuple[float, ...]
    projection: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.projection = np.zeros((self.model.nv, GENERALIZED_DOF))
        for index, drives in enumerate(self.drives):
            for drive in drives:
                self.projection[drive.slot.idx_v, index] = drive.multiplier

    def configuration(self, q: np.ndarray) -> np.ndarray:
        """
        Map canonical q[8] onto the description's own configuration vector.

        Written off each joint's own nq, so a `continuous` rotator (nq == 2, a
        cos/sin pair) and a revolute one (nq == 1) both work unchanged.
        """
        q_pin = self.neutral.copy()
        for index, drives in enumerate(self.drives):
            for drive in drives:
                value = drive.multiplier * q[index] + drive.offset
                if drive.slot.nq == 1:
                    q_pin[drive.slot.idx_q] = value
                else:
                    q_pin[drive.slot.idx_q] = np.cos(value)
                    q_pin[drive.slot.idx_q + 1] = np.sin(value)
        return q_pin


def _bind(model, name: str) -> JointSlot:
    if not model.existJointName(name):
        raise CraneModelError(
            ErrorCode.MISSING_JOINT,
            f"the description does not carry the canonical joint '{name}'",
        )
    joint = model.joints[model.getJointId(name)]
    if joint.nv != 1 or joint.nq not in (1, 2):
        raise CraneModelError(
            ErrorCode.INVALID_ROBOT_DESCRIPTION,
            f"joint {name} is not a single degree of freedom",
        )
    return JointSlot(int(joint.idx_q), int(joint.nq), int(joint.idx_v))


def parse(
    description_xml: str,
    tool: Tool,
    gravity_m_s2=(0.0, 0.0, -9.81),
    hydraulics_config: str | None = None,
    names: tuple[str, ...] | None = None,
) -> Description:
    """Build the canonical description for `tool` out of URDF XML."""
    try:
        model = pin.buildModelFromXML(description_xml)
        tree = ET.fromstring(description_xml)
    except Exception as exc:
        raise CraneModelError(ErrorCode.INVALID_ROBOT_DESCRIPTION, str(exc)) from exc
    model.gravity.linear = np.asarray(gravity_m_s2, dtype=float)

    names = tuple(names) if names else canonical_joints(hydraulics_config)
    drives: list[list[Drive]] = [[Drive(_bind(model, name))] for name in names]

    # One physical degree of freedom, several description joints.
    for joint in tree.iter("joint"):
        mimic = joint.find("mimic")
        name = joint.get("name")
        if mimic is None or name is None or mimic.get("joint") not in names:
            continue
        if not model.existJointName(name):
            continue
        drives[names.index(mimic.get("joint"))].append(
            Drive(
                _bind(model, name),
                float(mimic.get("multiplier", 1.0)),
                float(mimic.get("offset", 0.0)),
            )
        )

    damping = {
        joint.get("name"): float(joint.find("dynamics").get("damping"))
        for joint in tree.iter("joint")
        if joint.find("dynamics") is not None
        and joint.find("dynamics").get("damping") is not None
    }

    return Description(
        tool=tool,
        model=model,
        neutral=pin.neutral(model),
        names=tuple(names),
        drives=tuple(tuple(entry) for entry in drives),
        damping=tuple(damping.get(name, 0.0) for name in names),
    )
