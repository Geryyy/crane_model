# crane_model

ROS-free shared model boundary for the concrete-block crane architecture.
The public C++17/Eigen API fixes the generalized, actuated, passive, state,
input, tool, frame, status, dynamics, transmission, collision, and symbolic
graph contracts. The first slice provides explicit production-backend failure
and a deterministic test-only mock; Pinocchio, Coal, and CasADi backends are
not implemented yet.

The mock target is exported only when `BUILD_TESTING=ON`. Deployment builds
must use `BUILD_TESTING=OFF` and must not install or depend on test fixtures.
The production public API must remain free of ROS and backend-specific types.

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
