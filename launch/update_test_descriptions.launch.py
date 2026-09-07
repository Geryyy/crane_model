"""Expand the current crane xacro into crane_model's test description."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import (
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare

DESCRIPTION_PACKAGE = "epsilon_crane_description"
DESCRIPTION_FILE = "crane_description.urdf.xacro"
TOOL = "pzs100_description"
# The defaults of urdf/inc/parameters.xacro, spelled out. `sim_hydraulics:=true`
# is what carries the cylinder attachment frames the linkage cross-check reads.
# test/test_description_snapshot.py re-expands with these, so they live here
# once rather than in both files.
XACRO_ARGUMENTS = (
    "gazebo:=false",
    "sim_hydraulics:=true",
    "rigid:=true",
    "minimal:=false",
    "rotator_joint_type:=continuous",
)


def _xacro_process(tool, output_name):
    """Create one expanded URDF for *tool*."""
    source = PathJoinSubstitution(
        [FindPackageShare(DESCRIPTION_PACKAGE), "urdf", DESCRIPTION_FILE]
    )
    output = PathJoinSubstitution([LaunchConfiguration("output_dir"), output_name])

    return ExecuteProcess(
        cmd=[
            FindExecutable(name="xacro"),
            source,
            *XACRO_ARGUMENTS,
            f"tool:={tool}",
            "-o",
            output,
        ],
        output="screen",
    )


def generate_launch_description():
    """Generate the URDF fixture from the current source xacro."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_dir",
                default_value="src/concrete_block_stack/crane_model/test/description",
                description=(
                    "Directory for the generated URDF files. Relative paths are "
                    "resolved from the directory where ros2 launch is run."
                ),
            ),
            LogInfo(
                msg=(
                    "Generating pzs100.urdf from "
                    "epsilon_crane_description/urdf/crane_description.urdf.xacro"
                )
            ),
            _xacro_process(TOOL, "pzs100.urdf"),
        ]
    )
