#!/usr/bin/env python3
"""Derive the recorded-trajectory parity fixture from the machine recordings.

Offline, run once by hand, never part of the build or of CI.  It is the
counterpart of `scripts/derive_collision_model.py`: the expensive, unrepeatable
input is reduced to a checked-in table, and a test in the suite then compares
the new backend against that table without a bag, without ROS and without the
generated model on any dependency surface.

    source install/setup.bash
    python3 test/derive_recorded_parity.py \
        --description test/description/pzs100.urdf \
        --recordings /home/vscode/Documents/control_recordings \
        --output test/recorded_parity_fixture.txt

What it does:

1.  Reads `/joint_states` out of every mcap bag under `--recordings` and keeps
    the eight canonical coordinates of `wiki/implementation/model_api_contract.md`
    section 2, with their velocities.
2.  Selects `--samples` configurations by farthest-point sampling in the
    range-normalised sixteen-dimensional `(q, dq)` space, so the fixture spans
    the working range the recordings actually visit rather than a stretch of
    one manoeuvre.  The selection is deterministic: the seed is the sample
    closest to the mean, and ties go to the earliest.
3.  Compiles and runs `mp_crane_reference.cpp`, which evaluates the generated
    Maple/MATLAB model at those configurations.
4.  Writes the fixture, with the bag, the topic and the provenance of every
    column in its header.

The recordings are read-only and stay outside the repository.  Nothing here
copies, moves or commits one.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOPIC = "/joint_states"

# The canonical order of contract section 2, by the legacy URDF joint names the
# descriptions and the recordings both use.  `q9_left_rail_joint` is the
# PZS100's `q8`; the recordings carry the rail gripper, not the 7040 jaw.
JOINTS = (
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
)

COLUMNS = (
    ("bag", 1),
    ("t_s", 1),
    ("q", 8),
    ("dq", 8),
    ("mp_k8_position_m", 3),
    ("mp_k8_rotation", 9),
    ("mp_tip_position_m", 3),
    ("mp_ddq_u_gain", 10),
    ("mp_ddq_u_bias", 2),
    ("mp_ddq_u_bias_static", 2),
    ("mp_passive_equilibrium_rad", 2),
)


def read_recordings(root):
    """Return `(bags, stamps, q, dq)` read out of every bag under `root`."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState

    bags = sorted(path for path in root.iterdir() if path.is_dir())
    if not bags:
        raise SystemExit(f"no bag directories under {root}")

    bag_index = []
    stamps = []
    positions = []
    velocities = []
    for index, bag in enumerate(bags):
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        start = None
        kept = 0
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            if start is None:
                start = stamp
            if topic != TOPIC:
                continue
            message = deserialize_message(data, JointState)
            where = {name: at for at, name in enumerate(message.name)}
            if not all(name in where for name in JOINTS):
                continue
            if len(message.velocity) != len(message.name):
                continue
            sample = [message.position[where[name]] for name in JOINTS]
            rates = [message.velocity[where[name]] for name in JOINTS]
            if not np.isfinite(sample + rates).all():
                continue
            bag_index.append(index)
            stamps.append((stamp - start) * 1e-9)
            positions.append(sample)
            velocities.append(rates)
            kept += 1
        print(f"{bag.name}: {kept} usable {TOPIC} samples", file=sys.stderr)
    return (
        np.array(bag_index),
        np.array(stamps),
        np.array(positions),
        np.array(velocities),
    )


def farthest_point_selection(features, count):
    """Return `count` row indices spanning `features`, greedily and repeatably."""
    span = features.max(axis=0) - features.min(axis=0)
    span[span == 0.0] = 1.0
    scaled = features / span
    centre = scaled.mean(axis=0)
    chosen = [int(np.argmin(((scaled - centre) ** 2).sum(axis=1)))]
    distance = ((scaled - scaled[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < count:
        pick = int(np.argmax(distance))
        chosen.append(pick)
        distance = np.minimum(distance, ((scaled - scaled[pick]) ** 2).sum(axis=1))
    return sorted(chosen)


def prefix_of(package, workspace):
    """Return the install prefix that carries `package`.

    AMENT_PREFIX_PATH first, then the workspace's own install tree.  The
    fallback is not redundant: the retired packages are not in the CBS build
    set, so a workspace built for this overlay regenerates its setup files
    without them while their earlier install prefix is still on disk.
    """
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix and Path(prefix, "share", package).is_dir():
            return Path(prefix)
    fallback = workspace / "install" / package
    if (fallback / "share" / package).is_dir():
        return fallback
    raise SystemExit(f"{package} is neither on AMENT_PREFIX_PATH nor built under {fallback}")


def build_reference(source, into, workspace):
    """Compile the generated-model evaluator against the built workspace."""
    distro = Path("/opt/ros", os.environ.get("ROS_DISTRO", "humble"))
    mp_crane = prefix_of("mp_crane", workspace)
    parameter = prefix_of("epsilon_crane_parameter", workspace)
    binary = into / "mp_crane_reference"
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        str(source),
        "-o",
        str(binary),
        f"-I{mp_crane / 'include' / 'mp_crane'}",
        f"-I{parameter / 'include'}",
        f"-I{distro / 'include' / 'urdf'}",
        f"-I{distro / 'include' / 'urdfdom_headers'}",
        f"-I{distro / 'include' / 'urdfdom'}",
        f"-I{distro / 'include'}",
        f"-L{mp_crane / 'lib'}",
        f"-L{parameter / 'lib'}",
        f"-L{distro / 'lib'}",
        "-lmp_crane",
        "-lepsilon_crane_parameter",
        "-lurdf",
        "-lyaml-cpp",
        f"-Wl,-rpath,{mp_crane / 'lib'}",
        f"-Wl,-rpath,{parameter / 'lib'}",
        f"-Wl,-rpath,{distro / 'lib'}",
    ]
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)
    return binary


def evaluate(binary, description, answers, q, dq):
    """Return the generated model's answers, one row per sample."""
    request = "\n".join(
        " ".join(f"{value:.17g}" for value in np.concatenate((row, rate)))
        for row, rate in zip(q, dq)
    )
    finished = subprocess.run(
        [str(binary), str(description), str(answers)],
        input=request + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for note in (finished.stdout + finished.stderr).splitlines():
        print(f"mp_crane: {note}", file=sys.stderr)
    rows = [
        [float(field) for field in line.split()]
        for line in answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.array(rows)


def header(args, bags, q, dq, revision):
    """Return the fixture's provenance header, as a list of comment lines."""
    legend = []
    at = 0
    for name, count in COLUMNS:
        legend.append(f"#   {at:2d}..{at + count - 1:2d}  {name}")
        at += count
    ranges = [
        f"#   {name:24s} q [{q[:, index].min():+.4f}, {q[:, index].max():+.4f}]"
        f"   |dq| <= {np.abs(dq[:, index]).max():.4f}"
        for index, name in enumerate(JOINTS)
    ]
    return (
        [
            "# Recorded-trajectory parity fixture for `crane_model` (PRD user story 65).",
            "#",
            "# GENERATED, never hand-edited.  Regenerate with the workspace sourced and",
            "# the recordings mounted:",
            "#",
            "#     python3 test/derive_recorded_parity.py \\",
            f"#         --description {args.description} \\",
            f"#         --recordings {args.recordings} \\",
            f"#         --output {args.output}",
            "#",
            f"# Recordings: {args.recordings}",
            f"# Topic:      {TOPIC}  (sensor_msgs/msg/JointState, ~100 Hz)",
            "# Bags, by the index in column 0:",
        ]
        + [f"#   {index}  {bag.name}" for index, bag in enumerate(bags)]
        + [
            "#",
            "# `t_s` is seconds from the first message of that bag.  The eight `q` are",
            "# the canonical coordinates of model_api_contract.md section 2, read out of",
            f"# {TOPIC} by their legacy URDF joint names; the recorded machine carries the",
            "# PZS100 rail gripper, so `q8` is `q9_left_rail_joint`.  The six actuated",
            "# coordinates are encoder measurements; the two passive ones, `theta6_tip_joint`",
            "# and `theta7_tilt_joint`, are what the retained stack's own estimator",
            "# published on the same topic -- they are recorded, not measured.  Both models",
            "# are evaluated at the identical `q`, so that distinction changes which",
            "# configurations the comparison visits and not what it compares.",
            "#",
            f"# The `mp_*` columns are the generated Maple/MATLAB model at revision {revision},",
            f"# evaluated on {args.description} by test/mp_crane_reference.cpp with no",
            "# payload and gravity 9.81 m/s^2 -- read that file for which call produced",
            "# which column.  `mp_ddq_u_gain` is d(ddq_u)/d(ddq_a), 2 rows of 5 actuated",
            "# columns, row-major; `mp_ddq_u_bias` is ddq_u at ddq_a = 0 and the recorded",
            "# dq; `mp_ddq_u_bias_static` is the same at dq = 0, which leaves gravity",
            "# alone.  `mp_k8_rotation` is row-major.",
            "#",
            f"# {len(q)} samples, chosen by farthest-point selection in the range-normalised",
            "# (q, dq) space of every usable sample in every bag.  The range they span:",
        ]
        + ranges
        + [
            "#",
            "# Columns:",
        ]
        + legend
        + ["#"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    workspace = here.parents[2]
    bag_index, stamps, q, dq = read_recordings(args.recordings)
    print(f"{len(q)} usable samples in total", file=sys.stderr)

    chosen = farthest_point_selection(np.hstack((q, dq)), args.samples)
    bag_index, stamps, q, dq = (
        bag_index[chosen],
        stamps[chosen],
        q[chosen],
        dq[chosen],
    )

    revision = subprocess.run(
        [
            "git",
            "-C",
            str(workspace / "src" / "matlab_codegen" / "mp_crane"),
            "rev-parse",
            "--short",
            "HEAD",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    scratch = tempfile.mkdtemp(prefix="crane_model_parity_")
    try:
        binary = build_reference(here / "mp_crane_reference.cpp", Path(scratch), workspace)
        reference = evaluate(
            binary, args.description, Path(scratch, "answers.txt"), q, dq
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    expected = sum(count for name, count in COLUMNS if name.startswith("mp_"))
    if reference.shape != (len(q), expected):
        raise SystemExit(f"the evaluator returned {reference.shape}, expected {(len(q), expected)}")

    bags = sorted(path for path in args.recordings.iterdir() if path.is_dir())
    lines = header(args, bags, q, dq, revision or "unknown")
    for index in range(len(q)):
        row = np.concatenate(
            ([bag_index[index], stamps[index]], q[index], dq[index], reference[index])
        )
        lines.append(" ".join(f"{value:.17g}" for value in row))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(q)} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
