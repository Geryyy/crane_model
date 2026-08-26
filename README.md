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
| `full_dynamics`, `inverse_dynamics`, `reduced_actuated_dynamics` | Pinocchio `crba`, `nonLinearEffects` and `rnea` on the same model |
| `passive_equilibrium` | Newton on the passive rows of `rnea`, with Pinocchio's gravity Jacobian |
| `collision_query`, `collision_queries` | Coal, over geometry fitted to the same description |
| `symbolic_graph` | CasADi, through Pinocchio's own algorithms over `casadi::SX` |

`Model::create` parses the description with Pinocchio and maps the canonical
eight coordinates of contract §2 onto it by their URDF joint names, which keep
their legacy spelling (`wiki/implementation/ros2_interfaces.md` §3.1). A
description that will not parse is `InvalidRobotDescription`, one that does not
carry a canonical joint is `MissingJoint` — including the tool-dependent `q8`,
so the wrong tool against the wrong description is caught at construction.

Pinocchio, Coal and CasADi are *private* implementation details: they appear in
no public header, and a consumer links the shared libraries without ever seeing
them (contract §6). The one exception is deliberate and is opt-in — see
**The symbolic graph** below.

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
  - **the telescope entry is zero.** §5 calls the URDF's `3.4e4` a simulation
    stability hack an order of magnitude above the identified value and says it
    must not enter the model. No identified value is recorded anywhere in the
    vault, so the entry is zero rather than a guess.
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

## The symbolic graph

`symbolic_graph(spec, payload)` is contract §9. `wiki/mpc.md` §5.2 exports the
acados model from it and `wiki/trajectory_planning.md` §5.2 builds the timing
OCP from it, "so there is one dynamics implementation in the system rather than
two". That sentence is the whole requirement, and this is how it is met.

### It is not a second implementation

The graph does not restate the equations of motion. It runs **the same two
Pinocchio algorithms** — `crba` and `nonLinearEffects` — over `casadi::SX`
instead of `double`, on the same parsed description, through the same
projection $\mat P$, with the same damping and the same payload body. The
cylinder transmission is one function template (`src/cylinder_geometry.hpp`)
that `cylinder_jacobian` and the graph each instantiate. What `src/symbolic_graph.cpp`
writes out is only the assembly: which rows go where, and the Schur complement
of `wiki/robot_model.md` §3.1.

`test_symbolic_graph` is the acceptance test, and it checks the graph against
public *numeric* calls at a spread of configurations, velocities and payloads —
never against a second copy of the algebra:

| Graph | Oracle | Agreement |
|---|---|---|
| $\ddot q_\text{u}$ from `f` | passive rows of `inverse_dynamics` at that $\ddot q$ vanish | `1e-6` N m, the bound contract §7's invariant uses |
| $\vec\tau_\text{a}$ from `z` | $\bar{\mat M}\vec u + \bar{\vec h}$ of `reduced_actuated_dynamics` | `1e-9` relative |
| $\mat M_\text{uu}$, $\mat M_\text{ua}$, $\vec h_\text{u}$ | the same rows of `full_dynamics` | `1e-9` relative |
| $\vec v$ from `z` | `transmission().cylinder_velocity` | exact |
| $\vec Q$ from `z` | `transmission().pump_flow` | the smoothing bound below |

The relative bounds are round-off: the two run the same double arithmetic and
differ only in the order the projection sums are accumulated.

### What the output map carries

`SymbolicGraphSpec::include_output_map` asks for $\vec z$, the algebraic output
of `wiki/nomenclature.md` §10, in four blocks of six: $\vec\tau_\text{a}$,
$\vec F_\text{cyl}$, $\vec v$ and $\vec Q$. That is what `mpc` §2 costs and what
§3 constrains, so a consumer does not rebuild the transmission inside its own
solver — constraint 6 becomes the box $\abs{F_i}\le F_i^{\max}$ and constraint 7
the sum $\sum_i Q_i \le Q_\text{P}^{\max}$. The limits themselves are the
consumer's; nothing here compiles in an $F_i^{\max}$.

$\vec Q$ is **smoothed**, and must be: `mpc` §3.1 requires both non-smooth
pieces of constraint 7 to be smoothed for a gradient-based solver, so the graph
carries $A_i^{\pm}(v)\sqrt{v^2+\eps^2}$ with a `tanh` of width $\eps_v$ in place
of the step at $v=0$. `Model::transmission` keeps the physical step, because it
is not being differentiated. The two therefore differ by

$$\abs{\Delta Q_i} \le A_i\eps + \tfrac12\abs{A_i^{+}-A_i^{-}}\left(1-\tanh\frac{\abs{v_i}}{\eps_v}\right)\abs{v_i}$$

with $\eps = 10^{-6}$ and $\eps_v = 10^{-3}$ m/s compiled in — `SymbolicGraphSpec`
is frozen and has nowhere to carry them. That is around `1e-8` m³/s once an axis
is moving at more than a few millimetres per second, and it is the bound the
test asserts rather than a tolerance picked to pass.

### The CasADi handles live in a separate target

Contract §9 keeps `casadi::SX` and `casadi::MX` out of the public header, and
`test/consumer_public_only.cpp` enforces it rather than documenting it: that
fixture links `crane_model` alone, includes only `crane_model/model.hpp`, and
fails to compile if a CasADi include guard or `casadi::SX` reached it. It does
*not* claim CasADi is unreachable in the absolute — CasADi is installed under
`/usr/local/include`, a default system include directory, so no target in this
image can be denied it.

A consumer that wants the functions links **`crane_model::casadi_graph`**, which
is installed under its own include root (`include/crane_model_casadi_graph/`) so
that linking `crane_model` alone does not put it on the path:

```cmake
find_package(crane_model REQUIRED)
target_link_libraries(crane_mpc PRIVATE crane_model::casadi_graph)
```

```cpp
#include "crane_model/symbolic/casadi_graph.hpp"

const auto graph = crane_model::symbolic::casadi_graph(config, spec, payload);
// graph.value()->f, ->F_disc, ->z, ->passive_rows
```

`f`, `z` and `passive_rows` are `SXFunction`s, which is what acados exports
from; `F_disc` is the one `MXFunction`, an ERK4 step of `sample_time_s` that
calls `f` four times instead of carrying four copies of its expression graph.
`mpc` §5.2 says an explicit integrator is enough — with the ideal inner loop the
model is not stiff.

It takes a `ModelConfig` and not a `Model` because it has to. `Model` keeps its
parse behind a private pimpl, the public header is frozen, and `Model` has no
friends — so there is no route from a `const Model&` to what it parsed and no
member to add one to. Pass the same `ModelConfig` that built the `Model`. The
description is parsed a second time, by the same `detail::parse` `Model::create`
uses, so it is one implementation invoked twice and not two.

### CasADi is pinned, and this package is where that is checked

**3.7.2**, pinned in `.devcontainer/Dockerfile.vscode` as the `CASADI_VERSION`
build arg and asserted by `test/test_casadi_version.cpp` against the environment
the tests are running in. It is checked here because this is where CasADi enters
the CBS stack: `crane_mpc` and `crane_planning` reach it through
`crane_model::casadi_graph` and not on their own.

The pin exists for the OCP port. That port's acceptance criterion is that its
export script regenerates the checked-in C byte-identically on a no-op run, and
CasADi's code generation is version-sensitive, so an unpinned clone of `master`
made the criterion unenforceable — a clean checkout on a freshly built image
produced a dirty diff that said nothing about the model.

The pin has to be a **release tag** rather than a commit, for a reason worth
knowing before reading a version off this image:

- `casadi.__version__` is not `casadi.CasadiMeta.version()`. Off a release tag
  CasADi sets `CASADI_IS_RELEASE=0` and appends `+` to the version string, and
  `casadi/__init__.py` then substitutes `git describe --first-parent`, which on
  `master` names the last tag *reachable from it* — not the version that was
  built.
- So the pre-pin image, which had exactly one CasADi, reported `3.7.2+` through
  the C++ `CASADI_VERSION_STRING` and `3.6.3-1247.f0e2e2b22` through
  `casadi.__version__`, out of one `libcasadi.so.3.7`. That looks like two
  installs and is not one. `CASADI_GIT_REVISION` is the string that settles it:
  it is unique to a build, where a release number is not.

`test_casadi_version` therefore separates the two failures, because their
remedies differ:

| Test | Asserts | Remedy when it fails |
|---|---|---|
| `TheCppAndPythonHalvesAreOneCasadiBuild` | the header, the linked library and `import casadi` share a version *and a git revision* | remove the second install; a rebuild will not help |
| `BothSidesReportThePinnedReleaseTag` | both halves report the pinned tag | rebuild the devcontainer image |

The second is **skipped, loudly, while `CASADI_IS_RELEASE` is 0** — that bit
identifies an unpinned `master` build exactly, and rebuilding the image is a
human action that cannot be taken from inside the container. It prints every
reading it took before skipping, because this image's gtest drops the message of
a `GTEST_SKIP()` from both the console and the JUnit XML. Once the image is
rebuilt on the pin the skip stops and the assertion is live.

### It is not a real-time call

Contract §10 lists symbolic-graph creation as non-real-time and this is one.
Building a graph copies the model, parses (on the `casadi_graph` route), builds
a few thousand expression nodes and evaluates them once — about 30 ms and
thousands of allocations for either machine description.
`BuildingAGraphIsNotARealTimeCall` demonstrates the allocation rather than
asserting the absence of it. **Nothing on the 100 Hz path constructs one**: the
controller receives a graph that was built during configuration, or does not use
one at all.

### Failure

The payload is baked into the graph as a constant, so an invalid payload is
`InvalidPayload` and a description without `K8_rotator_lower_part` is
`FrameUnavailable`, both before CasADi is reached. A `sample_time_s` that is not
finite and positive is `InvalidArgument`: it is not a discretization step.

`SymbolicBackendFailure` is what a backend that cannot produce a usable graph
returns, and it returns **no graph at all** — never a handle whose dimensions
are zero. Two things reach it: an exception out of CasADi or Pinocchio, and a
graph that builds but does not evaluate to finite numbers at the description's
own neutral configuration. The second is a real case and is what the test uses:
a description with no mass anywhere has a singular $\mat M_\text{uu}$ at every
configuration, so the passive rows are $0/0$, and a consumer has to be told that
here rather than by its first solve. The dimensions the public handle reports
are read back off the built functions for the same reason.

### What is deliberately not here

No optimal control problem: cost, constraints, horizon, solver and tuning belong
to the planner and the MPC. No acados. No code generation as a build step —
CasADi's `Function::generate` is available to a consumer that wants it, and
nothing here writes C to disk. The payload is a constant in the graph, not a
parameter, which is the signature contract §9 froze; issue 033's streamed
payload estimate will want it as a `casadi::SX` parameter instead, and that is a
contract change under §12.

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
