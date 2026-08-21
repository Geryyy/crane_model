#include "crane_model/model.hpp"

#include <cmath>

namespace crane_model
{
namespace
{

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

bool finite(const Eigen::Vector3d& value)
{
  return value.array().isFinite().all();
}

std::array<std::string, 8> joint_names(Tool tool)
{
  return {
    "theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint",
    "q4_big_telescope", "theta6_tip_joint", "theta7_tilt_joint",
    "theta8_rotator_joint",
    tool == Tool::Pzs100 ? "q9_left_rail_joint" : "theta10_outer_jaw_joint"};
}

bool contains(const std::string& text, const std::string& needle)
{
  return text.find(needle) != std::string::npos;
}

bool valid_tool(Tool tool)
{
  return tool == Tool::Pzs100 || tool == Tool::Epsilon7040;
}

}  // namespace

struct SymbolicGraph::Impl
{
  std::size_t state_dimension{0};
  std::size_t input_dimension{0};
  bool has_output_map{false};
};

SymbolicGraph::SymbolicGraph(std::unique_ptr<Impl> impl) noexcept
: impl_(std::move(impl))
{
}

SymbolicGraph::SymbolicGraph(
  std::size_t state_dimension, std::size_t input_dimension,
  bool has_output_map)
: impl_(std::make_unique<Impl>(
    Impl{state_dimension, input_dimension, has_output_map}))
{
}

SymbolicGraph::SymbolicGraph(SymbolicGraph&&) noexcept = default;
SymbolicGraph& SymbolicGraph::operator=(SymbolicGraph&&) noexcept = default;
SymbolicGraph::~SymbolicGraph() = default;

std::size_t SymbolicGraph::state_dimension() const noexcept
{
  return impl_ ? impl_->state_dimension : 0;
}

std::size_t SymbolicGraph::input_dimension() const noexcept
{
  return impl_ ? impl_->input_dimension : 0;
}

bool SymbolicGraph::has_output_map() const noexcept
{
  return impl_ && impl_->has_output_map;
}

struct Model::Impl
{
  Tool tool{Tool::Pzs100};
  std::array<std::string, 8> names{};
};

Model::Model(std::unique_ptr<Impl> impl) noexcept
: impl_(std::move(impl))
{
}

Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

Result<Model> Model::create(const ModelConfig& config)
{
  if (config.robot_description_xml.empty()) {
    return Result<Model>::failure(
      failure(ErrorCode::InvalidRobotDescription, "robot_description_xml is empty"));
  }
  if (!valid_tool(config.tool)) {
    return Result<Model>::failure(
      failure(ErrorCode::UnsupportedTool, "tool is not supported"));
  }
  if (!contains(config.robot_description_xml, "<robot") ||
    config.robot_description_xml.find('>') == std::string::npos)
  {
    return Result<Model>::failure(
      failure(ErrorCode::InvalidRobotDescription, "robot_description_xml is malformed"));
  }
  if (!finite(config.gravity_m_s2) || config.gravity_m_s2.norm() <= 0.0) {
    return Result<Model>::failure(
      failure(ErrorCode::InvalidArgument, "gravity_m_s2 must be finite and non-zero"));
  }

  const auto names = joint_names(config.tool);
  for (const auto& name : names) {
    if (!contains(config.robot_description_xml, name)) {
      return Result<Model>::failure(
        failure(ErrorCode::MissingJoint, "robot description is missing joint " + name));
    }
  }

  // The first slice intentionally has no Pinocchio/Coal backend available.
  // Returning this explicit status prevents a plausible zero model from
  // reaching a controller.
  return Result<Model>::failure(
    failure(ErrorCode::BackendUnavailable,
            "Pinocchio/Coal production backend is not linked in this slice"));
}

Tool Model::tool() const noexcept
{
  return impl_ ? impl_->tool : Tool::Pzs100;
}

const std::array<std::string, 8>& Model::urdf_joint_names() const noexcept
{
  static const auto fallback = joint_names(Tool::Pzs100);
  return impl_ ? impl_->names : fallback;
}

bool Model::ready() const noexcept
{
  return static_cast<bool>(impl_);
}

#define CRANE_MODEL_UNAVAILABLE(type, name) \
  Result<type> Model::name \
  { \
    return Result<type>::failure( \
      failure(ErrorCode::BackendUnavailable, "production model backend is unavailable")); \
  }

CRANE_MODEL_UNAVAILABLE(Pose, forward_kinematics(const Q&, Frame, Frame) const)
CRANE_MODEL_UNAVAILABLE(Jacobian6x8, jacobian(const Q&, Frame) const)
CRANE_MODEL_UNAVAILABLE(QU, passive_equilibrium(const QA&, const Payload&) const)
CRANE_MODEL_UNAVAILABLE(FullDynamics, full_dynamics(const Q&, const DQ&, const Payload&) const)
CRANE_MODEL_UNAVAILABLE(
  ReducedDynamics,
  reduced_actuated_dynamics(const Q&, const DQ&, const Input&, const Payload&) const)
CRANE_MODEL_UNAVAILABLE(
  DQ, inverse_dynamics(const Q&, const DQ&, const DQ&, const Payload&) const)
CRANE_MODEL_UNAVAILABLE(ActuatedJacobian, cylinder_jacobian(const Q&) const)
CRANE_MODEL_UNAVAILABLE(
  CylinderTransmission, transmission(const Q&, const DQA&, const ChamberPressure&) const)
CRANE_MODEL_UNAVAILABLE(Vector6, cylinder_force(const ChamberPressure&) const)
CRANE_MODEL_UNAVAILABLE(
  CollisionResult, collision_query(const Q&, const CollisionScene&) const)
CRANE_MODEL_UNAVAILABLE(
  std::vector<CollisionResult>, collision_queries(const Q&, const CollisionScene&) const)
CRANE_MODEL_UNAVAILABLE(
  SymbolicGraph, symbolic_graph(const SymbolicGraphSpec&, const Payload&) const)

#undef CRANE_MODEL_UNAVAILABLE

}  // namespace crane_model
