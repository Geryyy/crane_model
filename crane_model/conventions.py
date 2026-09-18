"""
Canonical coordinates, frames and tool maps.

Joint ordering is NOT restated here: it is read from config/hydraulics.yaml,
the same file src/model.cpp and crane_model/symbolic.py read.
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

    They do not match the enum's spelling -- TIP is K5_inner_telescope, not
    K5_tip. Mirrors src/model.cpp:337-348, the only other copy.
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


def default_actuator_path() -> str:
    """
    Locate config/c3_full_model.json, installed first, else the source tree.

    Byte-for-byte copy of the C3 fit: an installed package cannot reach the
    source. Issue 118 replaces the copy with a single read.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = os.path.join(
            get_package_share_directory("crane_model"), "config", "c3_full_model.json"
        )
        if os.path.exists(installed):
            return installed
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config", "c3_full_model.json")


def canonical_joints(config_path: str | None = None) -> tuple[str, ...]:
    """Read the eight canonical joint names in contract order."""
    with open(config_path or default_hydraulics_path(), encoding="utf-8") as handle:
        joints = tuple(yaml.safe_load(handle)["joints"])
    if len(joints) != GENERALIZED_DOF:
        raise ValueError(f"joints carries {len(joints)} names, expected 8")
    return joints


# Control-safe box axis names against the canonical joints they bound.
# config/control_safe_limits.yaml keys rows by these, never by index.
CONTROL_SAFE_AXES = {
    "slewing": "theta1_slewing_joint",
    "boom": "theta2_boom_joint",
    "arm": "theta3_arm_joint",
    "telescope": "q4_big_telescope",
    "rotator": "theta8_rotator_joint",
    "tool": "q9_left_rail_joint",
}

CONTROL_SAFE_ROWS = ("q_a_lower", "q_a_upper", "dq_a_max", "q_a_margin")


def default_velocity_loop_path() -> str:
    """Locate config/velocity_loop.yaml, installed first, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = os.path.join(
            get_package_share_directory("crane_model"), "config", "velocity_loop.yaml"
        )
        if os.path.exists(installed):
            return installed
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config", "velocity_loop.yaml")


def default_control_safe_limits_path() -> str:
    """Locate config/control_safe_limits.yaml, installed first, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = os.path.join(
            get_package_share_directory("crane_model"),
            "config",
            "control_safe_limits.yaml",
        )
        if os.path.exists(installed):
            return installed
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config", "control_safe_limits.yaml")


def control_safe_limits(path: str | None = None) -> dict[str, dict[str, float]]:
    """
    Read the control-safe box: row name, then axis name, to the number.

    A missing row or axis raises, never defaults: the box is narrower than the
    description, which admits poses the machine cannot reach.
    """
    with open(path or default_control_safe_limits_path(), encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    out = {}
    for row in CONTROL_SAFE_ROWS:
        if row not in document:
            raise ValueError(f"control_safe_limits.yaml carries no {row}")
        out[row] = {}
        for axis in CONTROL_SAFE_AXES:
            if axis not in document[row]:
                raise ValueError(f"control_safe_limits.yaml: {row} carries no {axis}")
            out[row][axis] = float(document[row][axis]["value"])
    return out


def hydraulic_limits(hydraulics_path: str | None = None) -> dict[str, float]:
    """
    Read the hydraulic constants no description carries.

    Flat, named as `crane_mpc`/`crane_planning` declare them.
    A missing key is an exception, never a default: a wrong hydraulic constant
    is a wrong force limit.
    """
    path = hydraulics_path or default_hydraulics_path()
    with open(path, encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    if not isinstance(root, dict):
        raise ValueError(f"{path}: not a mapping")
    pump = root.get("pump")
    if not isinstance(pump, dict):
        raise ValueError(f"{path}: carries no pump block")
    limits = {
        "pump_flow_max": float(pump["flow_max"]),
        "pump_flow_planning_factor": float(pump["planning_factor"]),
        "system_pressure_pa": float(root["system_pressure_pa"]),
    }
    for name in ("pump_flow_max", "system_pressure_pa"):
        if not limits[name] > 0.0:
            raise ValueError(f"{path}: {name} must be positive, got {limits[name]}")
    factor = limits["pump_flow_planning_factor"]
    if not (0.0 < factor <= 1.0):
        raise ValueError(
            f"{path}: pump.planning_factor must be in (0, 1], got {factor}"
        )
    return limits
