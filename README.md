# crane_model

ROS-free shared model boundary for the concrete-block crane architecture.
The public C++17/Eigen API fixes the generalized, actuated, passive, state,
input, tool, frame, status, dynamics, transmission and collision contracts. It
is **numeric only**; the crane's symbolic model is `scripts/crane_symbolic.py`,
in Python (model API contract §9).

## Updating the checked-in URDF descriptions

The files in `test/description` are expanded fixtures. Generate both of them
from the current xacros by running this from the workspace root:

```bash
source install/setup.bash
ros2 launch crane_model update_test_descriptions.launch.py
```

The launch file expands
`epsilon_crane_description/urdf/crane_description.urdf.xacro` once for each
tool and writes `pzs100.urdf` and `epsilon_7040.urdf` to
`test/description`. To write to another directory, pass
`output_dir:=/path/to/output`.

## What has a backend

| Call | Backend |
|---|---|
| `cylinder_jacobian`, `transmission`, `cylinder_force` | closed-form cylinder geometry of `wiki/hydraulics.md` §2–§4 |
| `forward_kinematics`, `jacobian` | Pinocchio, on the `robot_description_xml` the caller supplies |
| `full_dynamics`, `inverse_dynamics`, `reduced_actuated_dynamics` | Pinocchio `crba`, `nonLinearEffects` and `rnea` on the same model |
| `passive_equilibrium` | Newton on the passive rows of `rnea`, with Pinocchio's gravity Jacobian |
| `collision_query`, `collision_queries` | Coal, over geometry fitted to the same description |

`Model::create` parses the description with Pinocchio and maps the canonical
eight coordinates of contract §2 onto it by their URDF joint names, which keep
their legacy spelling (`wiki/implementation/ros2_interfaces.md` §3.1). A
description that will not parse is `InvalidRobotDescription`, one that does not
carry a canonical joint is `MissingJoint` — including the tool-dependent `q8`,
so the wrong tool against the wrong description is caught at construction.

Pinocchio and Coal are *private* implementation details: they appear in no
public header, and a consumer links the shared libraries without ever seeing
them (contract §6). There is no CasADi in the C++ at all — see
**The Python symbolic model** below.

### `config/hydraulics.yaml` — the constants the description does not carry

The URDF is the ground truth for rigid-body dynamics and cannot drift. Four
things live outside it, and they are in one file rather than in this source:

- the chamber areas, `r_gear` and `V_m` of `wiki/hydraulics.md` §6.1, and the
  boom and arm linkage geometry of §6.2 — the numbers `cylinder_jacobian`,
  `transmission` and `cylinder_force` are computed from;
- the 7040 jaw four-bar of §2.6, whose equations the wiki carries and whose
  numbers it does not; each entry names the deployed file it was ported from;
- the per-joint **damping override** table, which is how the telescope entry
  comes out zero (see **`D`** below);
- the canonical joint-name map of contract §2, and the two smoothing constants
  `wiki/mpc.md` §3.1 asks of the symbolic model.

They are a file because a Python export of this same model — `pinocchio.casadi`
over the same URDF — needs every one of them and would otherwise be *told* them
a second time. A constant compiled in here and typed in there is drift nothing
would catch, so nothing above appears as a literal in any `.hpp` or `.cpp` of
this package. `test_contract.cpp` restates `wiki/hydraulics.md` §6.1 and §6.2
and asserts the file against them, so an edit to the file that contradicts the
wiki fails the build.

`Model::create` reads it first, before it touches the description — the joint
map is in there, so there is nothing to walk until it has been read. A missing
file, a missing key, a key that is not a number and a damping override with no
stated reason are each `InvalidArgument` naming the key: never a default,
because a hydraulic constant that quietly fell back to zero is a force limit
that quietly fell back to zero.

The path is compiled in, the installed copy first and the source tree second.
This package is ROS-free, so there is no ament index to ask, and `ModelConfig`
is frozen, so a caller cannot pass a third. yaml-cpp does the parsing
(`wiki/implementation/libraries.md`); it brings no ROS with it.

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

## Rigid-body dynamics

`wiki/robot_model.md` §1, evaluated in the canonical eight coordinates:

$$\mat{M}(q)\,\ddot q + \vec h(q,\dot q) + \mat D\,\dot q = \vec\tau$$

`full_dynamics` returns `M`, the bias `h + D dq` that contract §7 asks for, and
an inverse-dynamics torque; `inverse_dynamics` returns all eight rows for a given
`ddq`; `reduced_actuated_dynamics` returns the effective inertia and residual of
§3.4. All three are allocation-free after construction and are covered by the
guard in the contract test.

### The mimic joints are counted, not dropped

Pinocchio drops `<mimic>`, so the two mimicked joints — the second telescope
stage `q5_small_telescope` and, on the PZS100, the mirrored `q11_right_rail_joint`
— are ordinary joints in the parsed model. Their velocity rows are constant
multiples of the coordinate that drives them, so with that constant projection
$\mat P$ the model reports $\mat P^{\mathsf T}\mat M_\text{full}\mat P$ and
$\mat P^{\mathsf T}\vec h_\text{full}$. Their inertia therefore lands on `q4` and
on `q8` rather than being lost. `MassMatrixIsTheKineticEnergyOfTheSameDescription`
is what says so: it checks $\tfrac12 \dot q^{\mathsf T}\mat M\dot q$ against
Pinocchio's kinetic energy of the same description, an algorithm that never forms
a mass matrix.

Everything else the description carries and this API does not — the four cylinder
sub-chains, and the 7040's driven inner jaw — stays at its neutral configuration
with zero velocity, exactly as it does for forward kinematics. Those bodies do
carry mass (about 166 kg of boom cylinder and 118 kg of arm cylinder), so they
contribute inertia at a placement the closed linkage would not put them at. The
description closes those loops only in Gazebo; that is a property of the
description, not a choice made here, and it is one of the things the
recorded-trajectory parity campaign is for.

### Where each term comes from

- **`h`** is Pinocchio's `nonLinearEffects` — Coriolis, centrifugal and gravity.
  It is kept whole: `wiki/robot_model.md` §3.1 warns that the centrifugal
  coupling from slewing into the sway grows with radius and rate, and
  `CentrifugalCouplingFromSlewingReachesThePassiveRows` shows that term arriving
  in the passive rows and scaling as the square of the slewing rate.
- **Gravity** is `ModelConfig::gravity_m_s2`, written into `model.gravity` at
  construction. Nothing here compiles in a 9.81.
- **`D`** is read from the selected description's `<dynamics damping>`, per
  `wiki/implementation/parameters.md` §1 and §5 — which is also what makes the
  passive damping tool-dependent without a table in this file, since the two
  descriptions carry §5's hand-tuned per-tool values. Two deliberate departures:
  - **the telescope entry is zero**, and it is zero because
    `config/hydraulics.yaml` says so. `damping_overrides` is a table of
    `{joint, d, reason}`; `parse` applies whatever is on it and knows nothing
    about which joint that is. The row's own reason is §5's: the URDF's `3.4e4`
    is a simulation stability hack an order of magnitude above the identified
    value and must not enter the model, and no identified value is recorded
    anywhere in the vault, so the entry is zero rather than a guess.
  - **the mimicked joints' own damping is not added.** §5 tabulates damping per
    machine axis, and the mirrored rail's entry is the same simulation number
    duplicated for Gazebo; adding it would silently double the PZS100 tool axis
    and leave the 7040 alone.
  Coulomb friction, which the description also carries, is not applied: the
  equation of motion of §1 has a viscous term and nothing else.

### What `FullDynamics::inverse_dynamics_tau` is evaluated at

The frozen struct carries an inverse-dynamics torque but `full_dynamics` takes no
acceleration. The model reports the torque at the **consistent** acceleration of
§3.1 with $\ddot q_\text{a}=\vec 0$ — the torque holding the actuated axes still
while the pendulum swings freely. Its passive rows vanish by construction (§3.3)
and its actuated rows are exactly `bias_eff`. The alternative reading, $\tau$ at
$\ddot q=\vec 0$, would make the field a copy of `bias`.

`reduced_actuated_dynamics` takes a `ddq_a` that does not enter either output:
$\bar{\mat M}$ and $\bar{\vec h}$ are properties of $(q,\dot q,\text{payload})$
alone and $\tau_\text{a}=\bar{\mat M}\ddot q_\text{a}+\bar{\vec h}$ is the
caller's product to form. The argument is validated for finiteness and is
otherwise unused; it stays in the signature because the contract froze it there.

### The payload

`Payload` is the body of `wiki/robot_model.md` §5, attached rigidly to
`K8_rotator_lower_part`. The contract leaves two things open and this library
fixes them, because something has to:

- **`inertia_k8_kg_m2` is about the payload's own centre of mass**, with the axes
  of `K8` — the URDF `<inertial>` convention, so the same numbers a description
  would carry for the same body. It is not the inertia about the `K8` origin.
  `ThePayloadIsTheBodyRobotModelSaysItIs` pins this: it checks what the payload
  adds to `M` and to the bias against the frame's own public Jacobian, and the
  König split it uses has no parallel-axis term in it.
- **`valid == false` is refused**, with `InvalidPayload`. Contract §4 calls it an
  explicit unknown payload and says it is not permission to use a zero-mass one,
  and the mock refuses it, so answering an unknown with the dynamics of an empty
  gripper would be exactly the substitution that rule forbids. **An empty gripper
  is a declared payload of zero mass** — `valid = true`, `mass_kg = 0` — which is
  accepted, and which is a different statement: a zero-mass body still carrying a
  tensor is a different mass matrix from no body at all.

A payload with negative or non-finite mass, non-finite geometry, an asymmetric
inertia, or an inertia that is not positive semi-definite is `InvalidPayload`.
Zero inertia is admissible: a point mass is a payload.

Pinocchio has no per-call payload, so the body is written into the inertia of the
joint that carries `K8_rotator_lower_part` for the length of one call and restored
afterwards. That is fixed-size spatial arithmetic rather than an allocation, but
it means a dynamics call briefly mutates the model as well as its `Data` — one
more reason a `Model` must not be called from two threads at once.

## Passive equilibrium

`passive_equilibrium(q_a, payload)` is `wiki/robot_model.md` §2.3: the pose the
two passive joints settle at once the actuated ones are held and the tool has
stopped swinging, i.e. the $q_\text{u}$ solving

$$\vec h_\text{u}(q_\text{a}, q_\text{eq}, \vec 0) = \vec 0$$

At $\dot q = \vec 0$ every velocity-dependent term of §1 drops out of $\vec h$ —
Coriolis, centrifugal and $\mat D\dot q$ alike — so the condition says the passive
rows of the *gravity* torque vanish, which makes $q_\text{eq}$ a critical point of
the potential energy over those two coordinates. The solve is Newton on exactly
that residual, taken from `rnea`, so the quantity it drives to zero is literally
the passive rows of `inverse_dynamics(q, 0, 0, payload)` — the invariant contract
§7 states. `ThePassiveRowsVanishAtTheReturnedPose` is the acceptance test and it
asserts it through that public call, not against the solver.

**The residual bound is `1e-8` N m.** The passive rows carry order `1e3` N m of
individual gravity terms that cancel at the equilibrium, so double precision
leaves a noise floor near `1e-13` N m; against a restoring stiffness of `2e3`
N m/rad the bound is an angle error below `1e-11` rad. Measured on both
descriptions the returned pose comes out between `2e-13` and `7e-9` N m.

### Hanging is not the same as solving the equation

A two-hinge pendulum has four critical points, and two of them are the tool
standing *up*. They satisfy $\vec h_\text{u} = \vec 0$ exactly as well as the
hanging one does, and returning one would be silently wrong three ways over:
`mpc` §2 damps the sway toward this pose, `trajectory_planning` §6 makes it a
trajectory endpoint, and issue 033's payload estimate inverts it. What separates
them is the sign of the passive stiffness $\partial\vec h_\text{u}/\partial
q_\text{u}$, which is the Hessian of that same potential, so **the returned pose
is required to have a positive definite one** — checked at every iterate rather
than only at the end.

That still leaves which well to start in, and the description answers it: the
solve samples the range the two passive joints are given there — $\pm\pi/2$ of tip
and a quarter turn either side of $\pi/2$ of tilt, in both machine descriptions —
on a 5×5 grid, and starts Newton at the sample with the smallest residual *among
those with a positive definite stiffness*. Which critical point this library
returns is therefore the description's statement and not one made here. A
description that leaves a passive joint unbounded falls back to a full turn.

`TheReturnedPoseIsTheOneTheToolHangsAt` checks the result against the minimum of
the potential energy of the same description — `computePotentialEnergy`, which
this library never calls — so the branch, and not only the equation, has an
independent oracle.

### An offset grasp

`wiki/robot_model.md` §5.1: the passive joints hold the tool so that the
*combined* centre of mass hangs under the passive pivot, so a grasp that is not
centred tilts the tool at rest. `AnOffsetGraspCarriesTheLoadBackUnderThePivot`
asserts the direction and not only that something moved: 5 cm of lateral offset in
`K8` displaces the payload's centre of mass by 5 cm if the tool is held at the
centred equilibrium, and letting it settle takes most of that back — toward the
pivot, not past it. It does not take all of it back, because the crane's own tool
hangs centred and is worth a couple of hundred kilograms; ten times the block
leaves proportionally less over, which is what says the mechanism is the mass
ratio §5.1 implies.

### When there is nowhere to hang

`SingularConfiguration`, which is the code `full_dynamics` and
`reduced_actuated_dynamics` already use when the passive rows do not resolve at a
configuration — one condition, one code. It covers three cases and none of them
returns a pose:

- the crane folded far enough that the free hanging pose needs more tip travel
  than the description gives that joint, so the tool would come to rest against a
  stop instead of hanging. The arm at its own upper limit is one;
- nothing in the passive range has a restoring stiffness at all — gravity turned
  the other way up is the test case, and there the *unstable* pose still solves
  the equation, so a solver without the stiffness test would answer with it;
- the iteration budget exhausted, or the description unable to produce finite
  numbers at that configuration.

A solver that always reported success would remove the only signal a caller has,
and all three of `mpc` §2, `trajectory_planning` §6 and issue 033 would read the
invented pose as a tuning problem. An unknown payload is `InvalidPayload` and a
non-finite `q_a` is `NonFiniteInput`, as everywhere else here.

### What it costs

The call **allocates nothing** after construction — the gravity-Jacobian
workspace is sized once in `create()` like every other buffer — so contract §10
does not put it behind a prepared workspace, and
`RealTimeCallsAllocateNothingAfterConstruction` covers it alongside the other
eight. That is a statement about allocation and not about cost: one call is up to
25 gravity-Jacobian passes for the seed grid plus a handful of Newton steps at two
Pinocchio passes each, so it is **tens of `rnea` evaluations, not one**, and its
worst case is bounded by the grid and by the 32-step iteration budget. Treat it as
a per-plan or per-grasp quantity — which is how `trajectory_planning` §6 and issue
033 use it — rather than something to recompute every control cycle.

### Failure

A non-finite `q`, `dq`, `ddq` or `ddq_a` is `NonFiniteInput`. A configuration at
which the description cannot produce finite numbers, or at which $\mat M_{uu}$ is
not positive definite so the passive rows do not solve, is
`SingularConfiguration` — never a matrix of infinities. The real descriptions do
not reach either state inside their joint ranges; the telescope run out to
$10^{200}\,$m does, which is what the test uses.

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
implementation. The allocation guard in the contract test covers the nine calls
that allocate nothing — the three hydraulic ones, `forward_kinematics`,
`jacobian`, the three rigid-body dynamics calls and `passive_equilibrium` — and
`CollisionQueriesIsNotARealTimeCall` demonstrates that collision allocates, so the
guard's silence about it is a fact rather than an oversight.

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

## The Python symbolic model

`scripts/crane_symbolic.py` states the crane's dynamics symbolically, in Python.
It exists because both OCPs moved to `acados_template`
(`docs/features/cbs-ocp-python/grill.md` D1), and the cpin model build, the mimic
projection, the transmission and the output map are common to both — so D8 puts
them in one importable module rather than in each export script.

It is now the **only** statement of the crane's symbolic model. It arrived
beside the C++ `crane_model::casadi_graph` so that the OCP port did not have to
debug whether the dynamics were right at the same time; with `crane_mpc` and
`crane_planning` both ported, that target, its `symbolic/` include root,
`src/symbolic_graph.{hpp,cpp}` and `test_symbolic_graph` are gone, and the
package declares no CasADi C++ dependency. Contract §9 records the arrangement
that replaced them.

It is a library. No `main`, no writes, and it imports neither acados nor ROS —
`export_model_fixture.py --check` asserts that last one, because a parity test
that needed a solver installed to say whether the dynamics are right would be
testing the wrong thing.

### The dimensions are the OCPs' and the coordinates are the model's

Both, and it says which everywhere. `q` and `dq` are the canonical eight of
contract §2; `x` and `u` are what the OCPs plan after issue 068 took the tool
coordinate out of them — **`nx = 14`, `nu = 5`**. The tool arrives in the
parameter vector `p` together with the payload body:

| `p` | what |
|---|---|
| `p[0]` | the tool coordinate the low-level controller is holding the gripper at |
| `p[1]` | payload mass |
| `p[2:5]` | payload centre of mass in K8 |
| `p[5:11]` | $\Theta_\text{L}$, the six independent entries, in `(xx, xy, xz, yy, yz, zz)` order |

The payload is a **symbol**, not a constant: what it is bound to is the OCP's
business. That is the difference from the retired `symbolic_graph.cpp`, whose
frozen signature baked it in.

The output map stays **six** axes wide although the OCPs plan five. The tool
cylinder is still in the model, and reporting what it carries at the pinned
configuration is how "the transmission did not leave with the coordinate" stays
something a test can check.

### The parity test is the deliverable

`test/test_symbolic_parity.cpp`. Its oracle is `crane_model`'s **numeric** public
API and never a second copy of the algebra — the numeric model survives the
port because planner IK, collision and `passive_equilibrium` need it, so the
oracle does not move.

| Python model | Oracle | Agreement |
|---|---|---|
| `mass`, `bias` | `full_dynamics` | `1e-9` relative |
| $\ddot q_\text{u}$ from `xdot` | passive rows of `inverse_dynamics` at that $\ddot q$ vanish | `1e-6` N m |
| $\mat M_\text{uu}$, $\mat M_\text{ua}$, $\vec h_\text{u}$ | the same rows of `full_dynamics` | `1e-9` relative |
| $\vec\tau_\text{a}$ from `z` | $\bar{\mat M}\vec u + \bar{\vec h}$ of `reduced_actuated_dynamics` | `1e-9` relative |
| $\vec v$ from `z` | `transmission().cylinder_velocity` | `1e-9` relative |
| $\vec Q$ from `z` | `transmission().pump_flow` | the derived smoothing bound of `wiki/mpc.md` §3.1 |
| `cylinder_jacobian` | `Model::cylinder_jacobian` | `1e-9` relative |
| `chamber_force` | `Model::cylinder_force` | `1e-9` relative |
| $\vec h_\text{u}$ at rest | zero at `passive_equilibrium`'s pose | `1e-6` N m |

Two of its assertions are not about the dynamics and are there because a
disagreement in them cancels everywhere else:

* **the mimic projection, asserted directly.** Pinocchio drops `<mimic>`, so
  `q5_small_telescope` and `q11_right_rail_joint` have to be reconstructed by
  both sides, and a projection error can cancel in a mass matrix. The export
  writes one function per mimic the description declares, *named after the joint*
  — so a description that lost one is a missing symbol at link time — and the
  test compares each against the `<mimic>` element it came from.
* **the damping override reaches the Python model.** The telescope row must be
  `0.0` and not the description's `3.4e4`, for the reason `parameters.md` §5
  gives; every other row must be the description's own number. A faithful URDF
  read is the *wrong* model here, and it is the one failure that leaves every
  other quantity plausible.

The spread of points covers configurations, velocities and payloads, and includes
one per axis where a cylinder geometry is close to degenerate: `q2 = 2.38` is
0.018 rad from where the boom four-bar stops closing, `q3 = 1.84` is where the
arm ratio passes through zero, and the 7040's tool coordinate is pinned 0.0023
rad from the jaw transmission's reversal. One state has every axis at rest, which
is where mpc §3.1's smoothing is the whole of the difference in $\vec Q$.

### Regenerating the fixture

The Python model reaches the test as **C**. `scripts/export_model_fixture.py`
code-generates it and the result is checked in, following
`scripts/derive_collision_model.py` and `test/derive_recorded_parity.py`; the
test therefore needs no Python interpreter, no acados and not even the CasADi C++
library — it links `crane_model` and libm.

```bash
./scripts/export_model_fixture.py            # rewrite test/generated/
./scripts/export_model_fixture.py --check    # what the suite runs
```

`--check` regenerates into a temporary directory and compares byte for byte, and
`export_model_fixture_is_current` is that check in the suite. CasADi's code
generation is **version-sensitive** and the image's CasADi is not pinned, so when
it fails it prints both CasADi version strings before anything else, and "the
model changed" is the second explanation to reach for. (The C++ and Python
version strings are legitimately different on this image and neither is a second
install; issue 067's notes are the long form.)

`test/generated/` holds two generated files and one hand-written one. The
generated pair is spelled `.inc` rather than `.c`/`.h` on purpose: every C and
C++ file here goes through `ament_cpplint` and `ament_cppcheck`, machine output
fails them by thousands, and reformatting it to pass would make the byte-for-byte
check compare against whatever the formatter last did. Do not pass those two
files to `pre-commit` for the same reason. `crane_symbolic_fixture.c` is three
lines and is what the build compiles.

## Test fixtures

The mock target is exported only when `BUILD_TESTING=ON`. Deployment builds
must use `BUILD_TESTING=OFF` and must not install or depend on test fixtures.
The production public API must remain free of ROS and backend-specific types.

`test/description/` holds the two expanded machine descriptions the contract
test builds real models from, one per tool. They are generated, never
hand-edited; each file's header carries the `xacro` command that produced it.
Regenerate them when `epsilon_crane_description` or a tool description changes —
`LinkagePlacementsAgreeWithTheCompiledConstants` and
`CylinderTransmissionFollowsTheDescriptionsGeometry` then say whether
`config/hydraulics.yaml`'s linkage geometry still agrees with the description, and
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

The dynamics tests do not check a mass matrix against a second mass matrix. Their
oracle is the **energy** of the same description — `computeKineticEnergy` and
`computePotentialEnergy`, which the library never calls — plus the public
Jacobian. $T$ checks `M`, $\partial U/\partial q$ checks the gravity part of the
bias, the minimum of $U$ over the two passive coordinates locates the equilibrium
of §2.3 — and now also checks the one `passive_equilibrium` returns — without any
torque algorithm being asked, and the reduction of §3.4 is recovered from
`inverse_dynamics` alone by solving its passive rows to zero. If one of them
fails, the disagreement is with the description, not with a transcription of the
same algebra.

### Parity with the generated model

`test/test_recorded_parity.cpp` compares this backend against the Maple/MATLAB
model of `src/matlab_codegen/mp_crane` at 64 configurations taken out of the
2026-08-19 machine recordings. It is PRD user story 65's evidence for retiring
that model, and the full write-up — the residuals, the two defects it found and
what the result licenses — is `wiki/robot_model.md` §6.

The fixture is `test/recorded_parity_fixture.txt`: the 64 configurations, and
the generated model's answers beside them, with the bag and topic of every
column in its header. `test/derive_recorded_parity.py` produced it once, by
hand, and `test/mp_crane_reference.cpp` is the evaluator that script compiles
and runs. **Neither is part of the build**: nothing in `CMakeLists.txt` or
`package.xml` names `mp_crane`, and the test links only `crane_model`. That is
deliberate — the generated model reaches this directory as a checked-in table of
numbers and by no other route, which is what lets the slice-4 retirement guard
assert its absence from the `hardware` closure while the validation against it
survives. Regenerating the fixture needs the recordings mounted and the retained
stack built, and is not something CI can or should do:

```bash
python3 test/derive_recorded_parity.py \
  --description test/description/pzs100.urdf \
  --recordings /home/vscode/Documents/control_recordings \
  --output test/recorded_parity_fixture.txt
```

The comparison is a ladder, not a tolerance. Forward kinematics and the passive
equilibrium agree to machine precision. The passive rows of `M` and the bias do
not, by 50 % — and the test evaluates this model three times, on the description
as checked in, on the description without the 200 kg rail gripper the generated
model does not carry, and on that with the tool frame moved down by the 48.7 mm
the generated model drops, where the two agree to 1e-4. Each rung names one
defect in the retired model; nothing is rounded away, and a change that made
this model agree with the generated one on the description as written fails the
first test in the file.

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
