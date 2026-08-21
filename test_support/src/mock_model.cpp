#include "crane_model/testing/mock_model.hpp"

#include <cmath>
#include <unordered_set>

namespace crane_model::testing
{
namespace
{

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

template<typename Derived>
bool finite(const Eigen::MatrixBase<Derived>& value)
{
  return value.array().isFinite().all();
}
bool finite(const ChamberPressure& value)
{
  return value.p_a_pa.array().isFinite().all() && value.p_b_pa.array().isFinite().all();
}

bool nonnegative(const ChamberPressure& value)
{
  return (value.p_a_pa.array() >= 0.0).all() && (value.p_b_pa.array() >= 0.0).all();
}

bool valid_payload(const Payload& payload)
{
  return payload.valid && std::isfinite(payload.mass_kg) && payload.mass_kg >= 0.0 &&
         payload.center_of_mass_k8_m.array().isFinite().all() &&
         payload.inertia_k8_kg_m2.array().isFinite().all();
}

bool valid_scene(const CollisionScene& scene)
{
  std::unordered_set<std::string> ids;
  for (const auto& primitive : scene.primitives) {
    if (primitive.id.empty() || !ids.insert(primitive.id).second ||
        !primitive.pose_in_mounting_base.matrix().array().isFinite().all() ||
        !primitive.dimensions_m.array().isFinite().all()) {
      return false;
    }
  }
  return true;
}

std::array<std::string, 8> joint_names(Tool tool)
{
  return {
    "theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint",
    "q4_big_telescope", "theta6_tip_joint", "theta7_tilt_joint",
    "theta8_rotator_joint",
    tool == Tool::Pzs100 ? "q9_left_rail_joint" : "theta10_outer_jaw_joint"};
}

template<typename Vector>
Status check_finite(const Vector& value, const char* name)
{
  if (!value.array().isFinite().all()) {
    return failure(ErrorCode::NonFiniteInput, std::string(name) + " contains non-finite data");
  }
  return {};
}

}  // namespace

Result<MockModel> MockModel::create(Tool tool)
{
  if (tool != Tool::Pzs100 && tool != Tool::Epsilon7040) {
    return Result<MockModel>::failure(
      failure(ErrorCode::UnsupportedTool, "tool is not supported"));
  }
  return Result<MockModel>::success(MockModel(tool));
}

MockModel::MockModel(Tool tool)
: tool_(tool), names_(joint_names(tool))
{
}

Tool MockModel::tool() const noexcept { return tool_; }

const std::array<std::string, 8>& MockModel::urdf_joint_names() const noexcept
{
  return names_;
}

bool MockModel::ready() const noexcept { return true; }

void MockModel::reset_call_counters() const noexcept
{
  call_counters_ = {};
}

MockModel::CallCounters MockModel::call_counters() const noexcept
{
  return call_counters_;
}

Result<Pose> MockModel::forward_kinematics(const Q& q, Frame from, Frame to) const
{
  const auto status = check_finite(q, "q");
  if (!status.ok()) {
    return Result<Pose>::failure(status);
  }
  Pose pose;
  pose.expressed_in = to;
  pose.position_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  pose.orientation = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  if (from == to) {
    pose.position_m = Eigen::Vector3d::Zero();
  }
  return Result<Pose>::success(std::move(pose));
}

Result<Jacobian6x8> MockModel::jacobian(const Q& q, Frame frame) const
{
  const auto status = check_finite(q, "q");
  if (!status.ok()) {
    return Result<Jacobian6x8>::failure(status);
  }
  Jacobian6x8 result;
  result.expressed_in = frame;
  for (Eigen::Index index = 0; index < 6; ++index) {
    result.value(index, index) = static_cast<double>(index + 1);
  }
  return Result<Jacobian6x8>::success(std::move(result));
}

Result<QU> MockModel::passive_equilibrium(const QA& q_a, const Payload& payload) const
{
  const auto status = check_finite(q_a, "q_a");
  if (!status.ok()) {
    return Result<QU>::failure(status);
  }
  if (!valid_payload(payload)) {
    return Result<QU>::failure(failure(ErrorCode::InvalidPayload, "payload is invalid"));
  }
  QU result;
  result << 0.12, -0.08;
  return Result<QU>::success(std::move(result));
}

Result<FullDynamics> MockModel::full_dynamics(
  const Q& q, const DQ& dq, const Payload& payload) const
{
  const auto q_status = check_finite(q, "q");
  const auto dq_status = check_finite(dq, "dq");
  if (!q_status.ok()) {
    return Result<FullDynamics>::failure(q_status);
  }
  if (!dq_status.ok()) {
    return Result<FullDynamics>::failure(dq_status);
  }
  if (!valid_payload(payload)) {
    return Result<FullDynamics>::failure(
      failure(ErrorCode::InvalidPayload, "payload is invalid"));
  }
  FullDynamics result;
  for (Eigen::Index index = 0; index < 8; ++index) {
    result.mass(index, index) = static_cast<double>(index + 1);
    result.bias(index) = 0.1 * static_cast<double>(index + 1);
    result.inverse_dynamics_tau(index) = 0.2 * static_cast<double>(index + 1);
  }
  result.inverse_dynamics_tau(4) = 0.0;
  result.inverse_dynamics_tau(5) = 0.0;
  return Result<FullDynamics>::success(std::move(result));
}

Result<ReducedDynamics> MockModel::reduced_actuated_dynamics(
  const Q& q, const DQ& dq, const Input& ddq_a, const Payload& payload) const
{
  const auto q_status = check_finite(q, "q");
  const auto dq_status = check_finite(dq, "dq");
  const auto input_status = check_finite(ddq_a, "ddq_a");
  if (!q_status.ok()) {
    return Result<ReducedDynamics>::failure(q_status);
  }
  if (!dq_status.ok()) {
    return Result<ReducedDynamics>::failure(dq_status);
  }
  if (!input_status.ok()) {
    return Result<ReducedDynamics>::failure(input_status);
  }
  if (!valid_payload(payload)) {
    return Result<ReducedDynamics>::failure(
      failure(ErrorCode::InvalidPayload, "payload is invalid"));
  }
  ReducedDynamics result;
  for (Eigen::Index index = 0; index < 6; ++index) {
    result.mass_eff(index, index) = 2.0 + static_cast<double>(index);
    result.bias_eff(index) = 0.3 * static_cast<double>(index + 1);
  }
  return Result<ReducedDynamics>::success(std::move(result));
}

Result<DQ> MockModel::inverse_dynamics(
  const Q& q, const DQ& dq, const DQ& ddq, const Payload& payload) const
{
  const auto q_status = check_finite(q, "q");
  const auto dq_status = check_finite(dq, "dq");
  const auto ddq_status = check_finite(ddq, "ddq");
  if (!q_status.ok()) {
    return Result<DQ>::failure(q_status);
  }
  if (!dq_status.ok()) {
    return Result<DQ>::failure(dq_status);
  }
  if (!ddq_status.ok()) {
    return Result<DQ>::failure(ddq_status);
  }
  if (!valid_payload(payload)) {
    return Result<DQ>::failure(failure(ErrorCode::InvalidPayload, "payload is invalid"));
  }
  DQ result;
  for (Eigen::Index index = 0; index < 8; ++index) {
    result(index) = 0.4 * static_cast<double>(index + 1);
  }
  result(4) = 0.0;
  result(5) = 0.0;
  return Result<DQ>::success(std::move(result));
}

Result<ActuatedJacobian> MockModel::cylinder_jacobian(const Q& q) const
{
  const auto status = check_finite(q, "q");
  if (!status.ok()) {
    return Result<ActuatedJacobian>::failure(status);
  }
  ActuatedJacobian result = ActuatedJacobian::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    result(index, index) = 0.5 + static_cast<double>(index) * 0.1;
  }
  return Result<ActuatedJacobian>::success(std::move(result));
}

Result<CylinderTransmission> MockModel::transmission(
  const Q& q, const DQA& dq_a, const ChamberPressure& pressure) const
{
  const auto q_status = check_finite(q, "q");
  const auto dq_status = check_finite(dq_a, "dq_a");
  if (!q_status.ok()) {
    return Result<CylinderTransmission>::failure(q_status);
  }
  if (!dq_status.ok()) {
    return Result<CylinderTransmission>::failure(dq_status);
  }
  if (!finite(pressure)) {
    return Result<CylinderTransmission>::failure(
      failure(ErrorCode::NonFiniteInput, "pressure contains non-finite data"));
  }
  if (!nonnegative(pressure)) {
    return Result<CylinderTransmission>::failure(
      failure(ErrorCode::InvalidArgument, "pressure must be non-negative"));
  }
  CylinderTransmission result;
  result.joint_to_cylinder = ActuatedJacobian::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    const double scale = 0.5 + static_cast<double>(index) * 0.1;
    result.joint_to_cylinder(index, index) = scale;
    result.cylinder_velocity(index) = scale * dq_a(index);
    result.pressure_force(index) =
      0.001 * (pressure.p_a_pa(index) - pressure.p_b_pa(index));
    result.pump_flow(index) = std::abs(result.cylinder_velocity(index)) *
                              (0.002 + static_cast<double>(index) * 0.0001);
  }
  return Result<CylinderTransmission>::success(std::move(result));
}

Result<Vector6> MockModel::cylinder_force(const ChamberPressure& pressure) const
{
  if (!finite(pressure)) {
    return Result<Vector6>::failure(
      failure(ErrorCode::NonFiniteInput, "pressure contains non-finite data"));
  }
  if (!nonnegative(pressure)) {
    return Result<Vector6>::failure(
      failure(ErrorCode::InvalidArgument, "pressure must be non-negative"));
  }
  Vector6 result;
  for (Eigen::Index index = 0; index < 6; ++index) {
    result(index) = 0.001 * (pressure.p_a_pa(index) - pressure.p_b_pa(index));
  }
  return Result<Vector6>::success(std::move(result));
}

Result<CollisionResult> MockModel::collision_query(
  const Q& q, const CollisionScene& scene) const
{
  ++call_counters_.forbidden_non_rt_calls;
  const auto q_status = check_finite(q, "q");
  if (!q_status.ok()) {
    return Result<CollisionResult>::failure(q_status);
  }
  if (!valid_scene(scene)) {
    return Result<CollisionResult>::failure(
      failure(ErrorCode::InvalidScene, "collision scene has invalid or duplicate IDs"));
  }
  CollisionResult result;
  result.collision = false;
  result.minimum_distance_m = 0.25;
  result.other_id = scene.primitives.empty() ? "fixture_clearance" :
                    scene.primitives.front().id;
  result.witness_on_robot_m = Eigen::Vector3d(0.1, 0.2, 0.3);
  result.witness_on_other_m = Eigen::Vector3d(0.4, 0.5, 0.6);
  return Result<CollisionResult>::success(std::move(result));
}

Result<std::vector<CollisionResult>> MockModel::collision_queries(
  const Q& q, const CollisionScene& scene) const
{
  auto result = collision_query(q, scene);
  if (!result.ok()) {
    return Result<std::vector<CollisionResult>>::failure(result.status());
  }
  std::vector<CollisionResult> values;
  values.push_back(std::move(result).value());
  return Result<std::vector<CollisionResult>>::success(std::move(values));
}

Result<SymbolicGraph> MockModel::symbolic_graph(
  const SymbolicGraphSpec& spec, const Payload& payload) const
{
  ++call_counters_.forbidden_non_rt_calls;
  if (!std::isfinite(spec.sample_time_s) || spec.sample_time_s <= 0.0) {
    return Result<SymbolicGraph>::failure(
      failure(ErrorCode::InvalidArgument, "sample_time_s must be positive and finite"));
  }
  if (!valid_payload(payload)) {
    return Result<SymbolicGraph>::failure(
      failure(ErrorCode::InvalidPayload, "payload is invalid"));
  }
  return Result<SymbolicGraph>::success(
    SymbolicGraph(kStateDof, kInputDof, spec.include_output_map));
}

}  // namespace crane_model::testing
