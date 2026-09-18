# crane_model

Python shared model boundary for the concrete-block crane architecture.
The public Python API provides the numeric model (forward kinematics, Jacobians,
dynamics, collision queries) and the symbolic model (`crane_model/symbolic.py`)
used by planning and control. Both operate on the same URDF description and
hydraulics configuration.

This page carries what the API left open and this library had to fix, and the
measurements behind it. `crane_model/model.py` is the numeric backend;

## Backends

| Call | Backend |
|---|---|
| `cylinder_jacobian`, `transmission`, `cylinder_force` | closed-form cylinder geometry |
| `forward_kinematics`, `jacobian` | Pinocchio, on the `robot_description_xml` the caller supplies |
| `full_dynamics`, `inverse_dynamics`, `reduced_actuated_dynamics` | Pinocchio `crba`, `nonLinearEffects` and `rnea` on the same model |
| `passive_equilibrium` | Newton on the passive rows of `rnea`, with Pinocchio's gravity Jacobian |
| `collision_query`, `collision_queries` | Coal, over geometry fitted to the same description |

`CraneModel` constructor maps the canonical eight coordinates onto the parsed
description by their URDF joint names, which keep their legacy spelling. A
description that will not parse raises `CraneModelError` with code
`ErrorCode.INVALID_ROBOT_DESCRIPTION`; one that does not carry a canonical joint
is `ErrorCode.MISSING_JOINT` — including the tool's `q8`, so a description without
the tool axis is caught at construction.

### `config/hydraulics.yaml`

The constants the description does not carry, with their own header explaining
why each is there rather than compiled in. The `CraneModel` constructor reads it
**before** the description, because the canonical joint map is in it. A missing
file, a missing key, a key that is not a number or a damping override with no stated
reason raises `CraneModelError` with code `ErrorCode.INVALID_ARGUMENT` naming the
key — never a default, because a hydraulic constant that quietly fell back to zero
is a force limit that quietly fell back to zero.

The `pump` block and `system_pressure_pa` are not read by the model at all:
they are `crane_mpc`'s and `crane_planning`'s, imported through
`hydraulic_limits()`, which returns the three flat-named constants both declare
as parameters. They live here so neither package repeats a number the other
also needs.

`config/control_safe_limits.yaml` is here for a different reason — nothing in
this package reads it. It is the control-safe joint box that `crane_mpc`
enforces and `crane_planning` intersects its description-read limits with, and
it lives here because those two are the packages that both depend on this one.

### Frames

`Frame` enum maps onto URDF link names. The enum abbreviates them in
*coordinate* numbering (`K5_TIP`, `K6_TILT`, `K7_ROTATOR`); the descriptions number
their links K0…K8 in *link* numbering, which is not the same sequence.
`Frame.WORLD` is in no crane description, and a frame the description does not
carry raises `CraneModelError` with code `ErrorCode.FRAME_UNAVAILABLE`.

`forward_kinematics(q, from_frame, to_frame)` returns the pose of `to_frame`
expressed in `from_frame`. `jacobian(q, frame)` returns the 6×8 Jacobian of
`frame` in `frame` itself. All eight columns are real: the passive tip and tilt
coordinates move the tool like the actuated ones, and the telescope column carries
both stages, because the description mimics `q5_small_telescope` onto
`q4_big_telescope`.

## Rigid-body dynamics

In the canonical eight coordinates:

$$\mat{M}(q)\,\ddot q + \vec h(q,\dot q) + \mat D\,\dot q = \vec\tau$$

`full_dynamics` returns `M`, the bias `h + D dq` and an inverse-dynamics torque;
`inverse_dynamics` returns all eight rows for a given `ddq`;
`reduced_actuated_dynamics` returns the effective inertia and residual of the
actuated reduction.

### The mimic joints are counted, not dropped

Pinocchio drops `<mimic>`, so `q5_small_telescope` and the PZS100's mirrored
`q11_right_rail_joint` are ordinary joints in the parsed model. Their velocity
rows are constant multiples of the coordinate that drives them, so with that
projection the model reports the projected mass matrix and bias vector: their
inertia lands on `q4` and `q8` rather than being lost.

The four cylinder sub-chains stay at their neutral configuration with zero
velocity. They carry mass — about 166 kg of boom cylinder, 118 kg of arm
cylinder — so they contribute inertia at a placement the closed linkage would
not put them at. The description closes those loops only in Gazebo; that is a
property of the description, and one of the things the recorded-trajectory
parity campaign is for.

Frozen, they would also *render* detached. There is still only one model: before
each frame `_pose_cylinders` writes the closed linkage's offset over the neutral
`body_pos`/`body_quat` MuJoCo welded them in at -- a rotation about the joint
axis, a slide along it -- and welds them back before anything steps again, since
a link's placement feeds the composite inertia. The viewer therefore renders the
very data the plant integrates, and ctrl-drag lands on the right body with no
remapping.

### Where each term comes from

- **`h`** is `nonLinearEffects`, kept whole: the centrifugal coupling from
  slewing into the sway grows with radius and rate, and that term has to reach
  the passive rows.
- **Gravity** is `ModelConfig::gravity_m_s2`. Nothing here compiles in a 9.81.
- **`D`** is the selected description's `<dynamics damping>`. Two departures:
  - **the override table is empty** since 2026-09-07 (issue 127). Its one row
    forced the telescope to zero because the URDF's `3.4e4` was a simulation
    stability hack and no identified value existed; the description has carried
    the C3 fit's `5.640e4` since `epsilon_crane_description` `f28942e`. That
    fit's `±5 %` bracket on that axis is 200 times wide — the weakest of the
    five `d`, and still better than a zero.
  - **the mimicked joints' own damping is not added.** The mirrored rail's entry
    is the same simulation number duplicated for Gazebo; adding it would
    silently double the PZS100 tool axis.
  Coulomb friction, which the description also carries, is not applied: the
  equation of motion has a viscous term and nothing else.

### What `FullDynamics::inverse_dynamics_tau` is evaluated at

The frozen struct carries a torque but `full_dynamics` takes no acceleration.
The model reports it at the **consistent** acceleration with
$\ddot q_\text{a}=\vec 0$ — the torque holding the actuated axes still while the
pendulum swings freely. Its passive rows vanish by construction and its actuated
rows are exactly `bias_eff`. The alternative reading, $\tau$ at
$\ddot q=\vec 0$, would make the field a copy of `bias`.

`reduced_actuated_dynamics` takes a `ddq_a` that enters neither output —
$\bar{\mat M}$ and $\bar{\vec h}$ are properties of $(q,\dot q,\text{payload})$
alone. It is validated for finiteness and otherwise unused; it stays in the
signature because the API froze it there.

### The payload

`Payload` is a body attached rigidly to `Frame.ROTATOR_LOWER_PART`. Two things the
API leaves open:

- **`inertia_k8_kg_m2` is about the payload's own centre of mass**, with the axes
  of `K8` — the URDF `<inertial>` convention, so the same numbers a description
  would carry for the same body. Not the inertia about the `K8` origin.
- **`valid == False` is refused** with `CraneModelError` code `ErrorCode.INVALID_PAYLOAD`.
  It is an explicit unknown, not permission to use a zero-mass body. **An empty gripper
  is a declared payload of zero mass** — `valid = True`, `mass_kg = 0` — which is
  accepted and is a different statement: a zero-mass body still carrying a
  tensor is a different mass matrix from no body at all.

Negative or non-finite mass, non-finite geometry, an asymmetric inertia or one
that is not positive semi-definite raises the same error. Zero inertia is
admissible: a point mass is a payload.

## Passive equilibrium

`passive_equilibrium(q_a, payload)` returns the pose the two passive joints
settle at once the actuated ones are held and the tool has stopped swinging: the
$q_\text{u}$ solving $\vec h_\text{u}(q_\text{a}, q_\text{eq}, \vec 0) = \vec 0$.
At $\dot q = \vec 0$ every velocity-dependent term drops out, so the condition
says the passive rows of the *gravity* torque vanish and $q_\text{eq}$ is a
critical point of the potential. Newton runs on exactly that residual out of
`rnea`, so what it drives to zero is literally the passive rows of
`inverse_dynamics(q, 0, 0, payload)`.

**The residual bound is `1e-8` N m.** The passive rows carry order `1e3` N m of
gravity terms that cancel there, so double precision leaves a noise floor near
`1e-13` N m; against a restoring stiffness of `2e3` N m/rad the bound is an angle
error below `1e-11` rad. Measured on both descriptions the returned pose comes
out between `2e-13` and `7e-9` N m.

### Hanging is not the same as solving the equation

A two-hinge pendulum has four critical points and two of them are the tool
standing *up*. They satisfy $\vec h_\text{u} = \vec 0$ as well as the hanging one
does, and returning one would be wrong three ways over: the MPC damps the sway
toward this pose, the planner makes it a trajectory endpoint, and the payload
estimate of issue 033 inverts it. What separates them is the sign of
$\partial\vec h_\text{u}/\partial q_\text{u}$, so **the returned pose is required
to have a positive definite one**, checked at every iterate.

Which well to start in is the description's statement, not one made here: the
solve samples the range the two passive joints are given there on a 5×5 grid and
starts Newton at the sample with the smallest residual *among those with a
positive definite stiffness*. A description that leaves a passive joint unbounded
falls back to a full turn.

An offset grasp therefore tilts the tool at rest: the *combined* centre of mass
hangs under the pivot, and how much of a lateral offset is taken back is the mass
ratio between tool and payload.

### When there is nowhere to hang

`SingularConfiguration` — the code `full_dynamics` and
`reduced_actuated_dynamics` already use when the passive rows do not resolve.
Three cases, none of which returns a pose:

- the crane folded far enough that the free hanging pose needs more tip travel
  than the description gives that joint, so the tool would rest against a stop
  instead of hanging. The arm at its own upper limit is one;
- nothing in the passive range has a restoring stiffness — gravity turned the
  other way up is the test case, and there the *unstable* pose still solves the
  equation, so a solver without the stiffness test would answer with it;
- the iteration budget exhausted, or non-finite numbers at that configuration.

A solver that always reported success would remove the only signal a caller has,
and all three consumers would read the invented pose as a tuning problem.

### What it costs

The call allocates nothing after construction, so it needs no prepared
workspace. That is a statement about allocation and not about cost: one call is
up to 25 gravity-Jacobian passes for the seed grid plus Newton steps at two
Pinocchio passes each, bounded by the grid and the 32-step budget. **Tens of
`rnea` evaluations, not one** — a per-plan or per-grasp quantity, not something
to recompute every control cycle.

## Collision

One model, one library, one URDF. Pinocchio places the links and Coal answers the
queries, against the `CollisionScene` the caller supplies **and** against the
crane itself.

### Where the geometry comes from

The descriptions carry link collision geometry as `package://` STL meshes. The convex
primitives are fitted **once, offline** by `scripts/derive_collision_model.py`
and checked in as `config/collision_model.yaml` for the Python model, which
`test/test_python_collision.py` validates.

Every *placement* still comes from the description at runtime, and each
primitive records the `<collision>` element it was fitted to. Nothing asserts
that the two still agree since the test-surface strip of 104, so re-run the
derivation after a description change rather than waiting to be told. A
description that does not carry all fifteen shaped links — the joints-only
fixtures other packages build models from, for instance — returns
`CollisionBackendFailure` naming the link it is missing.

The fit takes the smaller of an enclosing capsule and an enclosing box per link:
the enclosing capsule around `K0_mounting_base` has a 0.98 m radius, and a model
that refuses motions clearing the base by a metre is not conservative, it is
unusable. Both candidates contain the mesh, so the smaller is only ever tighter.

The allowed-collision list is derived by the same script from the same
descriptions — adjacent links, links no sampled configuration brought together,
links every sampled configuration had overlapping, links overlapping at the
neutral pose — and is checked in as `config/allowed_collisions.srdf` as well, in
the MoveIt spelling. The SRDF in `epsilon_crane_moveit` was the starting point
and is read by nothing here.

### The self pairs are bounded before they are asked

49 pairs survive the allowed list, and an exact Coal distance for each of them was
the whole cost of a query against an empty scene. Each body carries a bounding
sphere (`LinkGeometry.bounds`, built with it), so

    |c_i - c_j| - (r_i + r_j)

is a lower bound on the pair's distance. Pairs are tried cheapest-bound first and
the loop stops once the running best is under the next bound -- everything after it
in that order is at least as far. **87% of pairs never reach Coal** on the planner's
bench corpus, which is 3.7x on the query and 1.45x on a whole `crane_planning` call.

The answer is the one exhaustive evaluation would have given, distance and witness
pair alike; ties break on pair order, not on bound order.
`test_the_broad_phase_answers_exactly_what_checking_every_pair_would` holds it to
that by inflating the radii until nothing can cull, which forces the exhaustive path
through the same code. The cull is sound only while that expression is a *lower*
bound, so a radius measured about anything but the shape's own AABB centre breaks
it silently -- there is a second test for exactly that.

Scene primitives are still checked exhaustively: one query each, and a scene is
short where the self list is not.

### What the calls return

`collision_queries` returns one result per scene primitive in scene order, then
one for the crane against itself; the self result names the closest checked link
pair as `first|second`. `collision_query` is the worst of the same set.

Distances are signed, so `minimum_distance_m` stays a real number on both sides
of contact and `collision` is exactly `minimum_distance_m < 0`. Witness points
are in `K0_mounting_base`. Both calls return a dynamically sized container and
are therefore **not RT calls**; the other nine allocate nothing after
construction.

### The scene arrives in `K0_mounting_base`

`/crane/collision_scene` is published already converted, and this library has no
`world` frame to convert from. An unusable primitive — empty or repeated id,
non-finite or non-positive extent, a rotation that is not one, a shape outside
the enum — returns `InvalidScene` for the whole scene, never a silently
shortened one.

`dimensions_m` is the primitive's **extent along each axis of its own frame**: a
box carries its three side lengths, a cylinder `(2r, 2r, length)`, a sphere its
diameter three times. A cylinder or sphere whose extents disagree is
`InvalidScene`, so a caller who put a radius in the first entry is told rather
than silently reinterpreted.

### A carried payload, and the sway envelope

The frozen collision signature takes `q` and a scene and no `Payload`, so a
carried payload arrives as a scene primitive with the reserved id `payload`, at
the pose the caller reads out of `forward_kinematics`. What the model owes it is
the half a caller cannot supply: the payload is **not** checked against the links
the rotator joint moves, and is checked against everything else.

**The model does not inflate geometry, and the planner owns the sway envelope.**
The frozen header has no inflation parameter and adding one is an architecture
change. It does not need one — the model takes all eight coordinates, so a query
*at* the sway bound is exact rather than conservative-by-inflation, and so is
requiring `minimum_distance_m > Δ_sway` at the nominal angle. This keeps the
envelope where `crane_planning` already owns the clearance policy.

## The Python symbolic model

`crane_model/symbolic.py`, the only statement of the crane's symbolic model
since the C++ `casadi_graph` was retired. Its module docstring carries the
dimensions, the two coordinate sets and why there are two; the short version is
that `q`/`dq` are the canonical eight and `x`/`u` are what the OCPs plan, rigid
at `NX_RIGID = 14`, `NU = 5` and `NX = 25` once C3's actuator states are on. The
payload is a **symbol** in the parameter vector, not a constant baked into a
frozen signature.

The symbolic parity gtest went with the test-surface strip of 104. It had proven
`mass`/`bias`, the passive rows of `M` and `h_u`, the reduced actuated dynamics,
cylinder velocity, pump flow, `cylinder_jacobian` and `chamber_force` against the
numeric API at `1e-9` relative, and the passive acceleration out of `xdot`
against the passive rows of `inverse_dynamics` vanishing at `1e-6` N m. It also
asserted two things that are not dynamics, because a disagreement in them cancels
everywhere else: the mimic projection Pinocchio drops, and that the damping
override table reaches the Python model at all. A disagreement now surfaces as
whichever OCP it breaks.

## The actuation system and the inner loop

Three files, and between them the whole path from a planned velocity to a force
on the rigid body. They exist so a gain can be validated against the plant
without a machine, and so nobody writes the numbers down twice.

| file | what it is |
|---|---|
| `crane_model/actuator.py` | C3 integrated: dead time, PT1 lag, force state |
| `crane_model/velocity_loop.py` | the JTC's `PidTrajectoryPlugin` law |
| `config/velocity_loop.yaml` | the shipped gains, and Psi's identified domain |
| `crane_model/mujoco_plant.py` `main` | the three above over `MujocoPlant`, at 100 Hz |

`actuator.py` is the same C3 as `symbolic.py`, integrated instead of symbolic —
`test_actuator.py` pins the two together against each other and against the ARX2
closed form the fit was scored with. The one asymmetry is block 1: the 60 ms
transport delay is in `actuator.py` and deliberately absent from `symbolic.py`,
where the node's predictor already carries it. A plant has to answer late; a
predictor that also did would double-count.

Damping is **not** in either. It is in the description as joint damping,
byte-identical to the C3 fit's `d` column, which is where it belongs and where
Pinocchio and MuJoCo both already read it.

`velocity_loop.yaml` is the authoritative copy of the gains; the deployed
`crane_controller_hydraulic_a2b_jtc_pid_pzs100.ros2_control.yaml` carries the
same numbers because `controller_manager` cannot import Python, and
`test_velocity_loop.py::test_mirrors_ros2_control` is the only thing keeping the
two honest. `p` is K_I and `d` is K_P — read the file's header before touching a
gain, and read `wiki/controller_design.md` 4.1 before changing one.

Psi is not modelled anywhere here. It is an input linearization that exists only
on the machine, and everything in this package assumes it exact.

## Build, test and regenerate

```bash
colcon build --packages-select crane_model
colcon test --packages-select crane_model
colcon test-result --verbose
```

`test/description/pzs100.urdf` is an expanded fixture, generated and never
hand-edited. Regenerate it when `epsilon_crane_description` or a tool
description changes:

```bash
source install/setup.bash
ros2 launch crane_model update_test_descriptions.launch.py   # output_dir:= to redirect
```

Then re-derive the collision fit, the YAML spelling in one run — validated by
`test_python_collision.py`:

```bash
./scripts/derive_collision_model.py \
  --description pzs100=test/description/pzs100.urdf \
  --package epsilon_crane_description=../../epsilon_crane_description \
  --package pzs100_description=../../crane_tools_description/pzs100 \
  --write-yaml config/collision_model.yaml \
  --write-srdf config/allowed_collisions.srdf
```

### Parity with the generated model

`test/test_python_parity.py` compares this Python backend against the Maple/MATLAB
model of `src/matlab_codegen/mp_crane` at 64 configurations out of the
2026-08-19 recordings. It is the evidence for retiring that model.

The fixture, `test/recorded_parity_fixture.txt`, is **frozen**: the script that
derived it and the evaluator went with the test-surface strip of 104, and `mp_crane`
is retired, so its answers will not be recomputed. The generated model
reaches this directory as a checked-in table of numbers only.

The comparison is a ladder, not a tolerance. Forward kinematics and the passive
equilibrium agree to machine precision. The passive rows of `M` and the bias do
not, by 50 % — and the test evaluates this model three times: on the description
as checked in, without the 200 kg rail gripper the generated model does not
carry, and with the tool frame moved down by the 48.7 mm it drops, where the two
agree to `1e-4`. Each rung names one defect in the retired model, and a change
that made this model agree on the description as written would fail the first test.
