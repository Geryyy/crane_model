#!/usr/bin/env python3
"""
Derive the collision model of `crane_model` from the machine descriptions.

Two artefacts come out of this script and both are checked in:

* the per-link primitive table compiled into `src/model.cpp`, printed as C++ on
  stdout, and
* `config/allowed_collisions.srdf`, the allowed-collision list of
  `wiki/trajectory_planning.md` §4.2, written with `--write-srdf`.

It runs **once**, offline, and is not part of the build. The reason it has to is
that the descriptions carry their link collision geometry as `package://` STL
meshes. `crane_model` is ROS-free and `ModelConfig` carries the description XML
and nothing else, so the library cannot resolve a package URI or read a mesh
file at runtime. The primitives are therefore fitted here, where the meshes are
on disk, and the fit is checked in. `test_contract.cpp` ties each entry back to
the `<collision>` element it was fitted to, so a description that moves a
collision origin or swaps a mesh fails the build instead of silently keeping
this fit.

The queries below go through the same Coal that `src/model.cpp` links, so the
list is derived against the geometry it will filter and not against a
re-implementation of it.

Usage:

    ./scripts/derive_collision_model.py \
        --description pzs100=test/description/pzs100.urdf \
        --description 7040=test/description/epsilon_7040.urdf \
        --package epsilon_crane_description=../../src/epsilon_crane_description \
        --package pzs100_description=../../src/crane_tools_description/pzs100 \
        --package epsilon_7040_description=../../src/crane_tools_description/7040 \
        --write-srdf config/allowed_collisions.srdf
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coal
import numpy as np

# The eight canonical coordinates of the model API contract §2 per tool. Every
# other movable joint stays at its neutral value, exactly as `Model::Impl::
# write_configuration` leaves it: the cylinder sub-chains and the driven inner
# jaw are loops the description closes only in Gazebo and carry no collision
# geometry.
CANONICAL = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
]
TOOL_JOINT = {"pzs100": "q9_left_rail_joint", "7040": "theta10_outer_jaw_joint"}

# A `continuous` joint has no limit element; it turns all the way round.
CONTINUOUS_RANGE = (-math.pi, math.pi)


# --------------------------------------------------------------------- meshes


def read_stl(path: Path) -> np.ndarray:
    """Return the vertices of an STL file, binary or ASCII, as an (n, 3) array."""
    raw = path.read_bytes()
    if (
        not raw.lower().lstrip().startswith(b"solid")
        or b"facet normal" not in raw[:512]
    ):
        count = struct.unpack("<I", raw[80:84])[0]
        if len(raw) < 84 + 50 * count:
            raise ValueError(f"{path} is not a well formed binary STL")
        block = np.frombuffer(raw[84 : 84 + 50 * count], dtype=np.uint8).reshape(
            count, 50
        )
        floats = block[:, 12:48].copy().view("<f4").reshape(count * 3, 3)
        return floats.astype(float)
    vertices = [
        [float(value) for value in line.split()[1:4]]
        for line in raw.decode("utf-8", "replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    return np.asarray(vertices, dtype=float)


# ----------------------------------------------------------------------- URDF


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation, i.e. R_z R_y R_x."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def parse_origin(element) -> tuple[np.ndarray, np.ndarray]:
    origin = None if element is None else element.find("origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return np.asarray(xyz), np.asarray(rpy)


class Description:
    """The parts of a URDF this derivation needs: the tree, the limits, the shapes."""

    def __init__(self, path: Path, packages: dict[str, Path]):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.packages = packages
        self.joints = {joint.get("name"): joint for joint in self.root.findall("joint")}
        self.children: dict[str, list[str]] = {}
        self.parent_joint: dict[str, str] = {}
        for name, joint in self.joints.items():
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.children.setdefault(parent, []).append(child)
            self.parent_joint[child] = name
        links = {link.get("name") for link in self.root.findall("link")}
        roots = [link for link in links if link not in self.parent_joint]
        if len(roots) != 1:
            raise ValueError(f"{path} has {len(roots)} root links: {sorted(roots)}")
        self.root_link = roots[0]

    def limits(self, name: str) -> tuple[float, float]:
        joint = self.joints[name]
        if joint.get("type") == "continuous":
            return CONTINUOUS_RANGE
        limit = joint.find("limit")
        return float(limit.get("lower")), float(limit.get("upper"))

    def collisions(self) -> dict[str, list[dict]]:
        """Return the `<collision>` elements per link, with their geometry resolved."""
        found: dict[str, list[dict]] = {}
        for link in self.root.findall("link"):
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                xyz, rpy = parse_origin(collision)
                entry = {"origin_xyz": xyz, "origin_rpy": rpy}
                mesh = geometry.find("mesh")
                box = geometry.find("box")
                if mesh is not None:
                    scale = np.asarray(
                        [float(v) for v in mesh.get("scale", "1 1 1").split()]
                    )
                    uri = mesh.get("filename")
                    entry["signature"] = "mesh:{}@{}".format(
                        uri, ",".join(f"{s:g}" for s in scale)
                    )
                    entry["points"] = read_stl(self.resolve(uri)) * scale
                elif box is not None:
                    size = np.asarray([float(v) for v in box.get("size").split()])
                    entry["signature"] = "box:" + ",".join(f"{s:g}" for s in size)
                    entry["points"] = np.array(
                        [
                            [sx, sy, sz]
                            for sx in (-size[0] / 2, size[0] / 2)
                            for sy in (-size[1] / 2, size[1] / 2)
                            for sz in (-size[2] / 2, size[2] / 2)
                        ]
                    )
                else:
                    raise ValueError(
                        f"{link.get('name')} carries a collision shape this "
                        "derivation does not know"
                    )
                found.setdefault(link.get("name"), []).append(entry)
        return found

    def resolve(self, uri: str) -> Path:
        prefix = "package://"
        if not uri.startswith(prefix):
            return Path(uri)
        package, _, relative = uri[len(prefix) :].partition("/")
        if package not in self.packages:
            raise KeyError(f"no --package given for {package} (needed by {uri})")
        return self.packages[package] / relative

    def forward_kinematics(
        self, configuration: dict[str, float]
    ) -> dict[str, np.ndarray]:
        """Link placements as 4x4 matrices in the root link's frame."""
        placements = {self.root_link: np.eye(4)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for child in self.children.get(parent, []):
                joint = self.joints[self.parent_joint[child]]
                placements[child] = placements[parent] @ self._joint_transform(
                    joint, configuration
                )
                stack.append(child)
        return placements

    def _joint_transform(self, joint, configuration: dict[str, float]) -> np.ndarray:
        xyz, rpy = parse_origin(joint)
        transform = np.eye(4)
        transform[:3, :3] = rpy_matrix(*rpy)
        transform[:3, 3] = xyz
        kind = joint.get("type")
        if kind == "fixed":
            return transform
        value = configuration.get(joint.get("name"), 0.0)
        mimic = joint.find("mimic")
        if mimic is not None:
            value = float(mimic.get("multiplier", 1.0)) * configuration.get(
                mimic.get("joint"), 0.0
            ) + float(mimic.get("offset", 0.0))
        axis_element = joint.find("axis")
        axis = np.asarray(
            [
                float(v)
                for v in (
                    axis_element.get("xyz") if axis_element is not None else "1 0 0"
                ).split()
            ]
        )
        axis = axis / np.linalg.norm(axis)
        motion = np.eye(4)
        if kind == "prismatic":
            motion[:3, 3] = value * axis
        else:
            skew = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            motion[:3, :3] = (
                np.eye(3)
                + math.sin(value) * skew
                + (1.0 - math.cos(value)) * skew @ skew
            )
        return transform @ motion


# ------------------------------------------------------------------ primitives


class Primitive:
    """An enclosing convex primitive: its kind, its extents, its frame in the link."""

    def __init__(
        self, kind: str, extents: np.ndarray, rotation: np.ndarray, centre: np.ndarray
    ):
        self.kind = kind  # "capsule" or "box"
        self.extents = extents  # capsule: (radius, half length); box: half sides
        self.rotation = rotation
        self.centre = centre

    def volume(self) -> float:
        if self.kind == "capsule":
            radius, half_length = self.extents
            return math.pi * radius * radius * (2.0 * half_length + 4.0 * radius / 3.0)
        return 8.0 * float(np.prod(self.extents))

    def coal(self):
        if self.kind == "capsule":
            return coal.Capsule(self.extents[0], 2.0 * self.extents[1])
        return coal.Box(*(2.0 * self.extents))

    def transform(self, placement: np.ndarray) -> coal.Transform3s:
        pose = placement @ self.homogeneous()
        return coal.Transform3s(pose[:3, :3], pose[:3, 3])

    def homogeneous(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = self.centre
        return pose

    def quaternion(self) -> np.ndarray:
        """(x, y, z, w) of `rotation`, the spelling Eigen's Quaterniond reads."""
        trace = self.rotation.trace()
        if trace > 0.0:
            scale = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / scale
            x = (self.rotation[2, 1] - self.rotation[1, 2]) * scale
            y = (self.rotation[0, 2] - self.rotation[2, 0]) * scale
            z = (self.rotation[1, 0] - self.rotation[0, 1]) * scale
        else:
            index = int(np.argmax(np.diag(self.rotation)))
            other = [(index + 1) % 3, (index + 2) % 3]
            root = math.sqrt(
                1.0
                + self.rotation[index, index]
                - self.rotation[other[0], other[0]]
                - self.rotation[other[1], other[1]]
            )
            components = np.zeros(3)
            components[index] = 0.5 * root
            scale = 0.5 / root
            components[other[0]] = (
                self.rotation[other[0], index] + self.rotation[index, other[0]]
            ) * scale
            components[other[1]] = (
                self.rotation[other[1], index] + self.rotation[index, other[1]]
            ) * scale
            x, y, z = components
            w = (
                self.rotation[other[1], other[0]] - self.rotation[other[0], other[1]]
            ) * scale
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        return np.array([x, y, z, w]) / norm


def frame_from_axis(axis: np.ndarray) -> np.ndarray:
    """Return a right-handed rotation whose third column is `axis`, Coal's own."""
    third = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(third @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, third)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(third, first), third))


def fit_capsule(points: np.ndarray) -> Primitive:
    """
    Fit the tightest capsule about `points` on their principal direction.

    The radius is the largest distance from the axis. The spine is then the
    shortest segment that still encloses every point: a point at axial offset t
    and radial offset d is inside as long as the spine reaches t - sqrt(R^2 -
    d^2), so the two ends are the extreme values of that expression.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    radial = np.linalg.norm(centred - np.outer(along, axis), axis=1)
    radius = float(radial.max())
    reach = np.sqrt(np.maximum(0.0, radius * radius - radial * radial))
    upper = float((along - reach).max())
    lower = float((along + reach).min())
    if upper < lower:  # a blob, not a beam: the enclosing capsule is a sphere
        upper = lower = 0.5 * (upper + lower)
    return Primitive(
        kind="capsule",
        extents=np.array([radius, 0.5 * (upper - lower)]),
        rotation=frame_from_axis(axis),
        centre=centroid + 0.5 * (upper + lower) * axis,
    )


def fit_box(points: np.ndarray) -> Primitive:
    """
    Fit the bounding box on the principal axes of `points`.

    Exact for a link whose `<collision>` already is a `<box>`: the principal
    axes of the eight corners are the box's own axes.
    """
    centred = points - points.mean(axis=0)
    rotation = np.linalg.svd(centred, full_matrices=False)[2].T
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    local = centred @ rotation
    lower, upper = local.min(axis=0), local.max(axis=0)
    return Primitive(
        kind="box",
        extents=0.5 * (upper - lower),
        rotation=rotation,
        centre=points.mean(axis=0) + rotation @ (0.5 * (lower + upper)),
    )


def fit_primitive(points: np.ndarray) -> Primitive:
    """
    Return the smaller of the two enclosing primitives.

    trajectory_planning 4.2 asks for capsules, and for the beams -- boom, arm,
    both telescope stages -- that is what this returns. It is the wrong shape
    for the chunky links: the capsule around the mounting base has a 0.98 m
    radius, which would refuse motions that clear it by a metre. Both candidates
    enclose the geometry, so taking the smaller one is safe in the direction
    that matters and only ever tighter.
    """
    candidates = [fit_capsule(points), fit_box(points)]
    return min(candidates, key=lambda candidate: candidate.volume())


# ------------------------------------------------------------ collision matrix


def collision_links(shapes: dict[str, Primitive]) -> list[str]:
    return sorted(shapes)


def adjacent_pairs(description: Description, links: set[str]) -> set[tuple[str, str]]:
    """
    Return the pairs with no third collision link between them in the tree.

    Plain parent-and-child is not the right test here: the descriptions put
    unshaped helper links (`moved_with_q1`, the DH frames) between two links
    that really are neighbours, and a pair separated only by those cannot be
    pulled apart by any joint.
    """
    pairs: set[tuple[str, str]] = set()
    for link in links:
        walker = link
        while walker in description.parent_joint:
            walker = (
                description.joints[description.parent_joint[walker]]
                .find("parent")
                .get("link")
            )
            if walker in links:
                pairs.add(tuple(sorted((link, walker))))
                break
    return pairs


def derive_matrix(
    description: Description, shapes: dict[str, Primitive], samples: int, seed: int
) -> dict[tuple[str, str], str]:
    """Return the disabled pairs and why, in the categories the legacy SRDF uses."""
    links = collision_links(shapes)
    tool_joint = (
        TOOL_JOINT["pzs100"]
        if "q9_left_rail_joint" in description.joints
        else TOOL_JOINT["7040"]
    )
    movable = CANONICAL + [tool_joint]
    lower = np.array([description.limits(name)[0] for name in movable])
    upper = np.array([description.limits(name)[1] for name in movable])

    generator = np.random.default_rng(seed)
    draws = generator.uniform(lower, upper, size=(samples, len(movable)))
    # The neutral configuration the model itself starts from, clamped into the
    # limits so it is a configuration the machine can actually stand in.
    draws = np.vstack((np.clip(np.zeros(len(movable)), lower, upper), draws))

    geometries = {link: shapes[link].coal() for link in links}
    request = coal.CollisionRequest()
    hits = {
        tuple(sorted((a, b))): 0 for i, a in enumerate(links) for b in links[i + 1 :]
    }
    default_hit: set[tuple[str, str]] = set()
    for index, draw in enumerate(draws):
        placements = description.forward_kinematics(dict(zip(movable, draw)))
        poses = {link: shapes[link].transform(placements[link]) for link in links}
        for pair in hits:
            first, second = pair
            result = coal.CollisionResult()
            coal.collide(
                geometries[first],
                poses[first],
                geometries[second],
                poses[second],
                request,
                result,
            )
            if result.isCollision():
                hits[pair] += 1
                if index == 0:
                    default_hit.add(pair)

    adjacent = adjacent_pairs(description, set(links))
    disabled: dict[tuple[str, str], str] = {}
    for pair, count in sorted(hits.items()):
        if pair in adjacent:
            disabled[pair] = "Adjacent"
        elif count == 0:
            disabled[pair] = "Never"
        elif count == len(draws):
            disabled[pair] = "Always"
        elif pair in default_hit:
            disabled[pair] = "Default"
    return disabled


# ---------------------------------------------------------------------- output


def cxx_string(text: str, indent: str, width: int = 100) -> str:
    """
    Return `text` as adjacent string literals, none wider than `width`.

    One character of `width` is left for the comma the caller puts after the
    last literal, so the emitted C++ stays inside the 100-column style rule.
    """
    room = width - len(indent) - 3
    chunks, rest = [], text
    while len(rest) > room:
        cut = max((rest.rfind(mark, 1, room + 1) for mark in "/@"), default=-1)
        cut = cut + 1 if cut > room // 2 else room
        chunks.append(rest[:cut])
        rest = rest[cut:]
    chunks.append(rest)
    return "\n".join(f'{indent}"{chunk}"' for chunk in chunks)


def cxx_table(fits: dict[str, dict[str, tuple[Primitive, dict]]]) -> str:
    lines = []
    for tool in sorted(fits):
        for link in sorted(fits[tool]):
            shape, source = fits[tool][link]
            extents = np.zeros(3)
            extents[: len(shape.extents)] = shape.extents
            # + 0.0 so a rounded -0.0 does not reach the generated source.
            extents = extents + 0.0
            centre = shape.centre + 0.0
            quaternion = shape.quaternion() + 0.0
            lines.append(
                '  {{Tool::{}, "{}",\n'
                "    LinkShape::{},\n"
                "    {{{:.6f}, {:.6f}, {:.6f}}},\n"
                "    {{{:.6f}, {:.6f}, {:.6f}}},\n"
                "    {{{:.9f}, {:.9f}, {:.9f}, {:.9f}}},\n"
                "{},\n"
                "    {{{:.6f}, {:.6f}, {:.6f}, {:.9f}, {:.9f}, {:.9f}}}}},".format(
                    "Pzs100" if tool == "pzs100" else "Epsilon7040",
                    link,
                    "Capsule" if shape.kind == "capsule" else "Box",
                    *extents,
                    *centre,
                    *quaternion,
                    cxx_string(source["signature"], "    "),
                    *(source["origin_xyz"] + 0.0),
                    *(source["origin_rpy"] + 0.0),
                )
            )
    return "\n".join(lines)


def cxx_pairs(disabled: dict[tuple[str, str], str]) -> str:
    return "\n".join(
        f'  {{"{a}", "{b}"}},  // {reason}'
        for (a, b), reason in sorted(disabled.items())
    )


def srdf(disabled: dict[tuple[str, str], str], sources: list[str]) -> str:
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- The allowed-collision list of wiki/trajectory_planning.md 4.2.",
        "",
        "     Derived once, by scripts/derive_collision_model.py, from:",
    ]
    head += [f"       {source}" for source in sources]
    head += [
        "     and from the primitive fit that same script produces, which is the",
        "     geometry this list filters. It is not read at runtime, because",
        "     crane_model is ROS-free and takes no file path, but it is the",
        "     checked-in artefact, and CraneModelCollision.AllowedPairsAreTheCheckedInList",
        "     asserts that the table compiled into src/model.cpp is exactly these pairs.",
        "",
        "     Reasons, as the MoveIt format uses them: Adjacent, no third shaped link",
        "     between the two; Never, no sampled configuration brought them together;",
        "     Always, every sampled configuration had them overlapping; Default, they",
        "     overlap at the neutral configuration. A pair is disabled only when every",
        "     tool that carries both links disabled it. -->",
        '<robot name="crane_model">',
    ]
    body = [
        f'    <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>'
        for (a, b), reason in sorted(disabled.items())
    ]
    return "\n".join(head + body + ["</robot>", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description", action="append", default=[], metavar="TOOL=PATH", required=True
    )
    parser.add_argument("--package", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--write-srdf", type=Path, default=None)
    arguments = parser.parse_args()

    packages = {}
    for entry in arguments.package:
        name, _, directory = entry.partition("=")
        packages[name] = Path(directory)

    fits: dict[str, dict[str, tuple[Primitive, dict]]] = {}
    matrices: dict[str, dict[tuple[str, str], str]] = {}
    carried: dict[tuple[str, str], set[str]] = {}
    sources = []
    for entry in arguments.description:
        tool, _, path = entry.partition("=")
        description = Description(Path(path), packages)
        sources.append(f"{tool}: {path}")
        shapes: dict[str, Primitive] = {}
        fits[tool] = {}
        for link, elements in description.collisions().items():
            points = np.vstack(
                [
                    element["points"] @ rpy_matrix(*element["origin_rpy"]).T
                    + element["origin_xyz"]
                    for element in elements
                ]
            )
            shapes[link] = fit_primitive(points)
            fits[tool][link] = (shapes[link], elements[0])
        matrices[tool] = derive_matrix(
            description, shapes, arguments.samples, arguments.seed
        )
        for first_index, first in enumerate(sorted(shapes)):
            for second in sorted(shapes)[first_index + 1 :]:
                carried.setdefault((first, second), set()).add(tool)

    # Disabled only where every tool that carries both links disabled it: an
    # extra checked pair costs a distance query, a wrongly dropped one is a
    # collision nobody looks for.
    merged = {}
    for pair, tools in carried.items():
        reasons = {matrices[tool].get(pair) for tool in tools}
        if None not in reasons:
            merged[pair] = sorted(reasons)[0]

    print("// --- generated by scripts/derive_collision_model.py, do not hand-edit ---")
    print("// link primitives")
    print(cxx_table(fits))
    print("// allowed-collision pairs")
    print(cxx_pairs(merged))
    for tool in sorted(matrices):
        print(
            f"// {tool}: {len(fits[tool])} shaped links, "
            f"{len(matrices[tool])} disabled pairs",
            file=sys.stderr,
        )
    if arguments.write_srdf is not None:
        arguments.write_srdf.write_text(srdf(merged, sources))
        print(f"// wrote {arguments.write_srdf}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
