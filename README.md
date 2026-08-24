# crane_model

ROS-free shared model boundary for the concrete-block crane architecture.
The public C++17/Eigen API fixes the generalized, actuated, passive, state,
input, tool, frame, status, dynamics, transmission, collision, and symbolic
graph contracts.

## What has a backend

| Call | Backend |
|---|---|
| `cylinder_jacobian`, `transmission`, `cylinder_force` | closed-form cylinder geometry of `wiki/hydraulics.md` §2–§4 |
| `forward_kinematics`, `jacobian` | Pinocchio, on the `robot_description_xml` the caller supplies |
| `collision_query`, `collision_queries` | Coal, over geometry fitted to the same description |
| everything else | `BackendUnavailable` at runtime |

`Model::create` parses the description with Pinocchio and maps the canonical
eight coordinates of contract §2 onto it by their URDF joint names, which keep
their legacy spelling (`wiki/implementation/ros2_interfaces.md` §3.1). A
description that will not parse is `InvalidRobotDescription`, one that does not
carry a canonical joint is `MissingJoint` — including the tool-dependent `q8`,
so the wrong tool against the wrong description is caught at construction.

Pinocchio and Coal are *private* implementation details: they appear in no
public header, and a consumer links the shared libraries without ever seeing
them (contract §6). CasADi is still absent.

### Frames

`Frame` maps onto URDF link names once, in `src/model.cpp`. The contract's own
enum comments abbreviate those links in *coordinate* numbering (`K5_tip`,
`K6_tilt`, `K7_rotator`); the descriptions number their links K0…K8 in *link*
numbering, which `wiki/robot_model.md` §0.1 warns is not the same sequence.
`Frame::World` and `Frame::ToolContact` are not in every description —
`tool_contact_point` is a 7040 link and no crane description carries `world` —
and a frame the description does not carry returns `FrameUnavailable` rather
than a substituted pose.

`forward_kinematics(q, from, to)` returns the pose of `to` expressed in `from`,
and labels it `expressed_in = from`. `jacobian(q, frame)` returns the 6×8
Jacobian of `frame` expressed in `frame` itself. All eight columns are real: the
passive tip and tilt coordinates move the tool like the actuated ones, and the
telescope column carries both stages, because the description mimics
`q5_small_telescope` onto `q4_big_telescope`.

Both calls are allocation-free after construction and are asserted to be, but
they write into the `pinocchio::Data` workspace the model owns, so one `Model`
must not be called from two threads at once.

## Collision

One model, one library, one URDF (`wiki/trajectory_planning.md` §4.2). Pinocchio
places the links of the description the caller supplied and Coal answers the
queries, against the `CollisionScene` the caller supplies **and** against the
crane itself.

### Where the geometry comes from

The descriptions carry their link collision geometry as `package://` STL meshes.
This library is ROS-free and `ModelConfig` carries the description XML and
nothing else, so there is no package index to resolve a URI against and no path
to read a mesh from. The convex primitives are therefore fitted **once, offline**
by `scripts/derive_collision_model.py`, and the fit is checked in as
`src/collision_model.hpp`.

What still comes from the description at runtime is every *placement*: each
primitive is attached to the link frame Pinocchio parsed, and each carries the
`<collision>` element it was fitted to, so `PrimitivesWereFittedToThisDescription`
fails when a description moves that element rather than checking against stale
geometry. A description that does not carry all fifteen shaped links — the
joints-only fixtures other packages build models from, for instance — is a
perfectly good model for everything else and returns `CollisionBackendFailure`
here, naming the link it is missing.

§4.2 asks for capsules. The fit takes the smaller of an enclosing capsule and an
enclosing box per link, which returns a capsule where the link really is a beam
and a box where it is not: the enclosing capsule around `K0_mounting_base` has a
0.98 m radius, and a model that refuses motions clearing the base by a metre is
not conservative, it is unusable. Both candidates contain the mesh, so the
smaller is safe in the direction that matters and only ever tighter.

### Self-collision and the allowed-collision list

Self-collision is checked. The allowed-collision list is derived once, by the
same script, from these descriptions and from the fit it produces — adjacent
links, links no sampled configuration brought together, links every sampled
configuration had overlapping, and links overlapping at the neutral
configuration. It is checked in twice on purpose: as
`config/allowed_collisions.srdf` in the SRDF spelling
`wiki/implementation/libraries.md` keeps MoveIt's format for, and as the table
the queries use; `AllowedPairsAreTheCheckedInList` asserts the two are the same
list. The SRDF in `epsilon_crane_moveit` was the starting point for the pairs
and is not read by anything here.

### What the calls return

`collision_queries` returns one result per scene primitive, in scene order, and
then one final result for the crane against itself. A scene result names the
primitive in `other_id`; the self result names the closest checked link pair as
`first|second`. `collision_query` is the worst of the same set: the smallest
`minimum_distance_m` found anywhere, with its witness points and its `other_id`.

Distances are signed, so `minimum_distance_m` stays a real number on both sides
of contact and `collision` is exactly `minimum_distance_m < 0`. Witness points
are in `K0_mounting_base`.

`collision_queries` returns a dynamically sized container and is therefore **not
an RT API call** (contract §10); neither is `collision_query`, which shares its
implementation. The allocation guard in the contract test covers the five calls
that are RT — the three hydraulic ones, `forward_kinematics` and `jacobian` —
and `CollisionQueriesIsNotARealTimeCall` demonstrates that collision allocates,
so the guard's silence about it is a fact rather than an oversight.

### The scene arrives in `K0_mounting_base`

Contract §8 and ROS 2 Interfaces §4: the assembly planner and the world model are
the only two places `world` is converted, and `/crane/collision_scene` is
published already converted. This library has no `world` frame to convert from
and does not convert. A primitive that is not usable — empty or repeated id,
non-finite pose or extent, a non-positive extent, a pose whose rotation is not a
rotation, a shape outside the enum — returns `InvalidScene` for the whole scene,
never a silently shortened one.

`dimensions_m` is the primitive's **extent along each axis of its own frame**:
a box carries its three side lengths, a cylinder `(2r, 2r, length)` and a sphere
its diameter three times. The contract does not fix this and something has to; a
cylinder or sphere whose extents disagree is `InvalidScene`, so a caller who put
a radius in the first entry is told rather than silently reinterpreted.

### A carried payload

The frozen collision signature takes `q` and a scene and no `Payload`, so a
carried payload reaches the model the way it can: as a scene primitive with the
reserved id `payload`, at the pose the caller reads out of `forward_kinematics`.
The caller can compute that pose exactly, because it is the caller who supplies
`q`. What the model owes it is the half a caller cannot supply — the payload is
**not** checked against the links that carry it, which is everything the rotator
joint moves, and is checked against everything else. That is what "attached to
`K8`" means for a collision model.

Attaching it inside the model instead — geometry the model owns and places from
`q` — needs the payload in the call or in `ModelConfig`, which is a signature
change and therefore an architecture change under contract §12.

### Sway, and who owns it

`wiki/trajectory_planning.md` §4.3 asks that collision be checked for a tool that
swings, with the envelope
$\Delta_\text{sway} = \ell_\text{tool}\sin(\max\lvert q_\text{u}^{+}\rvert)$ and
the same bound `mpc` §3 imposes.

**The model does not inflate geometry, and the planner owns the envelope.** The
frozen header has no inflation parameter, and adding one would be an
architecture change under contract §12. It does not need one: the model takes
`q` — all eight coordinates, the two passive ones included — so a query *at* the
sway bound is a real query and not an approximation, and it returns a true
signed minimum distance, so requiring `minimum_distance_m > Δ_sway` at the
nominal pendulum angle is exact rather than conservative-by-inflation. Either
route is available to the caller and neither needs anything from here.

That places the envelope where `crane_planning` already owns the clearance
policy, and keeps the two consistent by construction for the reason §4.3 gives:
the bound is `mpc` §3's state constraint, and the planner is the side that knows
it. §4.3 prescribes inflating the geometry; this note is a deliberate departure,
because inflation cannot be expressed through the frozen API and the two
alternatives above are exactly as tight.

## Test fixtures

The mock target is exported only when `BUILD_TESTING=ON`. Deployment builds
must use `BUILD_TESTING=OFF` and must not install or depend on test fixtures.
The production public API must remain free of ROS and backend-specific types.

`test/description/` holds the two expanded machine descriptions the contract
test builds real models from, one per tool. They are generated, never
hand-edited; each file's header carries the `xacro` command that produced it.
Regenerate them when `epsilon_crane_description` or a tool description changes —
`LinkagePlacementsAgreeWithTheCompiledConstants` and
`CylinderTransmissionFollowsTheDescriptionsGeometry` then say whether the
hydraulic subset's compiled-in geometry still agrees with the description, and
`PrimitivesWereFittedToThisDescription` says the same for the collision fit.
When it fails, re-run the derivation rather than editing the fit:

```bash
./scripts/derive_collision_model.py \
  --description pzs100=test/description/pzs100.urdf \
  --description 7040=test/description/epsilon_7040.urdf \
  --package epsilon_crane_description=../../src/epsilon_crane_description \
  --package pzs100_description=../../src/crane_tools_description/pzs100 \
  --package epsilon_7040_description=../../src/crane_tools_description/7040 \
  --write-srdf config/allowed_collisions.srdf
```

It prints the two tables of `src/collision_model.hpp` on stdout and rewrites the
SRDF; it needs `numpy` and the `coal` Python bindings, both already in the image,
and it is not part of the build.

Build this package in the integration workspace so Eigen and retained
dependencies are available:

```bash
colcon build --packages-select crane_model --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select crane_model
colcon test-result --verbose
```

The normative API, validation, real-time boundary, and change-control rules
live in `wiki/implementation/model_api_contract.md` in the integration
workspace.
