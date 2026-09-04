"""
Canonical coordinates, frames and tool maps.

Normative source: wiki/implementation/model_api_contract.md sections 2 and 3.

The canonical joint ordering is NOT restated here: it is read from
config/hydraulics.yaml, the same file src/model.cpp and scripts/crane_symbolic.py
read, so the three cannot drift.
"""

from __future__ import annotations

import enum
import os

import yaml

GENERALIZED_DOF = 8
ACTUATED_DOF = 6
PASSIVE_DOF = 2

#: Positions of the actuated and passive coordinates in the generalized vector.
ACTUATED_INDICES = (0, 1, 2, 3, 6, 7)
PASSIVE_INDICES = (4, 5)


class Tool(enum.Enum):
    """Tool identifier; the value is the tool's own spelling."""

    PZS100 = "pzs100"


class Frame(enum.Enum):
    """
    Model frame; values are the descriptions' own link names.

    These do not match the enum's historical spelling -- TIP is
    K5_inner_telescope, not K5_tip. Mirrors the frame table at
    src/model.cpp:337-348, which is the only other copy.
    """

    WORLD = "world"
    MOUNTING_BASE = "K0_mounting_base"
    SLEWING_COLUMN = "K1_slewing_column"
    BOOM = "K2_boom"
    ARM = "K3_arm"
    BIG_TELESCOPE = "K4_outer_telescope"
    TIP = "K5_inner_telescope"
    TILT = "K6_double_joint_link"
    ROTATOR = "K7_rotator_upper_part"
    ROTATOR_LOWER_PART = "K8_rotator_lower_part"
    TCP = "K8_tool_center_point"


def default_hydraulics_path() -> str:
    """Locate config/hydraulics.yaml, installed first, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("crane_model"), "config", "hydraulics.yaml"
        )
    except Exception:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, "config", "hydraulics.yaml")


def canonical_joints(config_path: str | None = None) -> tuple[str, ...]:
    """Read the eight canonical joint names in contract order."""
    with open(config_path or default_hydraulics_path(), encoding="utf-8") as handle:
        joints = tuple(yaml.safe_load(handle)["joints"])
    if len(joints) != GENERALIZED_DOF:
        raise ValueError(f"joints carries {len(joints)} names, expected 8")
    return joints
