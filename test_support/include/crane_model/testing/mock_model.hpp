#ifndef CRANE_MODEL__TESTING__MOCK_MODEL_HPP_
#define CRANE_MODEL__TESTING__MOCK_MODEL_HPP_

#include <string>
#include <cstddef>
#include <vector>

#include "crane_model/model.hpp"

namespace crane_model::testing
{

class MockModel
{
public:
  struct CallCounters
  {
    std::size_t forbidden_non_rt_calls{0};
  };

  static Result<MockModel> create(Tool tool);

  explicit MockModel(Tool tool);

  [[nodiscard]] Tool tool() const noexcept;
  [[nodiscard]] const std::array<std::string, 8>& urdf_joint_names() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  void reset_call_counters() const noexcept;
  [[nodiscard]] CallCounters call_counters() const noexcept;

  Result<Pose> forward_kinematics(const Q& q, Frame from, Frame to) const;
  Result<Jacobian6x8> jacobian(const Q& q, Frame frame) const;
  Result<QU> passive_equilibrium(const QA& q_a, const Payload& payload) const;
  Result<FullDynamics> full_dynamics(
    const Q& q, const DQ& dq, const Payload& payload) const;
  Result<ReducedDynamics> reduced_actuated_dynamics(
    const Q& q, const DQ& dq, const Input& ddq_a, const Payload& payload) const;
  Result<DQ> inverse_dynamics(
    const Q& q, const DQ& dq, const DQ& ddq, const Payload& payload) const;

  Result<ActuatedJacobian> cylinder_jacobian(const Q& q) const;
  Result<CylinderTransmission> transmission(
    const Q& q, const DQA& dq_a, const ChamberPressure& pressure) const;
  Result<Vector6> cylinder_force(const ChamberPressure& pressure) const;

  Result<CollisionResult> collision_query(
    const Q& q, const CollisionScene& scene) const;
  Result<std::vector<CollisionResult>> collision_queries(
    const Q& q, const CollisionScene& scene) const;

  Result<SymbolicGraph> symbolic_graph(
    const SymbolicGraphSpec& spec, const Payload& payload) const;

private:
  Tool tool_;
  std::array<std::string, 8> names_;
  mutable CallCounters call_counters_{};
};

}  // namespace crane_model::testing

#endif  // CRANE_MODEL__TESTING__MOCK_MODEL_HPP_
