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
| everything else | `BackendUnavailable` at runtime |

`Model::create` parses the description with Pinocchio and maps the canonical
eight coordinates of contract §2 onto it by their URDF joint names, which keep
their legacy spelling (`wiki/implementation/ros2_interfaces.md` §3.1). A
description that will not parse is `InvalidRobotDescription`, one that does not
carry a canonical joint is `MissingJoint` — including the tool-dependent `q8`,
so the wrong tool against the wrong description is caught at construction.

Pinocchio is a *private* implementation detail: it appears in no public header,
and a consumer links the shared libraries without ever seeing them (contract
§6). Coal and CasADi are still absent.

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
hydraulic subset's compiled-in geometry still agrees with the description.

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
