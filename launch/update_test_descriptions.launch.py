"""Expand the current crane xacros into crane_model's test descriptions."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


DESCRIPTION_PACKAGE = "epsilon_crane_description"
DESCRIPTION_FILE = "crane_description.urdf.xacro"


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
            "gazebo:=false",
            "sim_hydraulics:=true",
            "rigid:=true",
            "minimal:=false",
            "rotator_joint_type:=continuous",
            f"tool:={tool}",
            "-o",
            output,
        ],
        output="screen",
    )


def generate_launch_description():
    """Generate both URDF fixtures from the current source xacros."""
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
                    "Generating pzs100.urdf and epsilon_7040.urdf from "
                    "epsilon_crane_description/urdf/crane_description.urdf.xacro"
                )
            ),
            _xacro_process("pzs100_description", "pzs100.urdf"),
            _xacro_process("epsilon_7040_description", "epsilon_7040.urdf"),
        ]
    )
