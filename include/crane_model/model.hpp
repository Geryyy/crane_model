#ifndef CRANE_MODEL__MODEL_HPP_
#define CRANE_MODEL__MODEL_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace crane_model
{

namespace testing
{
class MockModel;
}

inline constexpr std::size_t kGeneralizedDof = 8;
inline constexpr std::size_t kActuatedDof = 6;
inline constexpr std::size_t kPassiveDof = 2;
inline constexpr std::size_t kStateDof = 16;
inline constexpr std::size_t kInputDof = 6;
inline constexpr std::size_t kCylinderAxisCount = 6;

using Q = Eigen::Matrix<double, 8, 1>;
using DQ = Eigen::Matrix<double, 8, 1>;
using QA = Eigen::Matrix<double, 6, 1>;
using DQA = Eigen::Matrix<double, 6, 1>;
using QU = Eigen::Matrix<double, 2, 1>;
using DQU = Eigen::Matrix<double, 2, 1>;
using State = Eigen::Matrix<double, 16, 1>;  // [q_a, q_u, dq_a, dq_u]
using Input = Eigen::Matrix<double, 6, 1>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using ActuatedJacobian = Eigen::Matrix<double, 6, 6>;
using FullMass = Eigen::Matrix<double, 8, 8>;
using PassiveMass = Eigen::Matrix<double, 2, 2>;
using ReducedMass = Eigen::Matrix<double, 6, 6>;

enum class Tool : std::uint8_t { Pzs100, Epsilon7040 };

enum class Frame : std::uint8_t {
  World,
  MountingBase,
  SlewingColumn,
  Boom,
  Arm,
  BigTelescope,
  Tip,
  Tilt,
  Rotator,
  RotatorLowerPart,
  Tcp,
  ToolContact,
};

struct Payload
{
  double mass_kg{};
  Eigen::Vector3d center_of_mass_k8_m{};
  Eigen::Matrix3d inertia_k8_kg_m2{};
  bool valid{false};
};

enum class CollisionShape : std::uint8_t { Box, Cylinder, Sphere };

struct CollisionPrimitive
{
  std::string id;
  CollisionShape shape{CollisionShape::Box};
  Eigen::Isometry3d pose_in_mounting_base{Eigen::Isometry3d::Identity()};
  Eigen::Vector3d dimensions_m{};
  bool structural{false};
};

struct CollisionScene
{
  std::vector<CollisionPrimitive> primitives;
};

struct Pose
{
  Frame expressed_in{Frame::MountingBase};
  Eigen::Vector3d position_m{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

struct PassiveState
{
  QU q_u{};
  DQU dq_u{};
};

enum class ErrorCode : std::uint16_t {
  Ok = 0,
  InvalidArgument,
  NonFiniteInput,
  InvalidPayload,
  InvalidScene,
  MissingJoint,
  UnsupportedTool,
  InvalidRobotDescription,
  FrameUnavailable,
  SingularConfiguration,
  BackendUnavailable,
  CollisionBackendFailure,
  SymbolicBackendFailure,
  NotReady,
};

struct Status
{
  ErrorCode code{ErrorCode::Ok};
  std::string message;

  [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::Ok; }
};

template<typename T>
class Result
{
public:
  [[nodiscard]] static Result success(T value)
  {
    Result result;
    result.value_.emplace(std::move(value));
    return result;
  }

  [[nodiscard]] static Result failure(Status status)
  {
    Result result;
    result.status_ = normalize(std::move(status));
    return result;
  }

  Result(Result&&) noexcept = default;
  Result& operator=(Result&&) noexcept = default;
  Result(const Result&) = delete;
  Result& operator=(const Result&) = delete;

  [[nodiscard]] bool ok() const noexcept
  {
    return status_.ok() && value_.has_value();
  }

  [[nodiscard]] const Status& status() const noexcept { return status_; }
  [[nodiscard]] const T& value() const & { return value_.value(); }
  [[nodiscard]] T&& value() && { return std::move(value_.value()); }

private:
  Result() = default;
  static Status normalize(Status status)
  {
    if (status.code == ErrorCode::Ok) {
      status.code = ErrorCode::InvalidArgument;
    }
    if (status.message.empty()) {
      status.message = "crane_model operation failed";
    }
    return status;
  }

  Status status_{};
  std::optional<T> value_;
};

template<>
class Result<void>
{
public:
  [[nodiscard]] static Result success() { return Result{}; }

  [[nodiscard]] static Result failure(Status status)
  {
    Result result;
    if (status.code == ErrorCode::Ok) {
      status.code = ErrorCode::InvalidArgument;
    }
    if (status.message.empty()) {
      status.message = "crane_model operation failed";
    }
    result.status_ = std::move(status);
    return result;
  }

  [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] const Status& status() const noexcept { return status_; }

private:
  Status status_{};
};

struct ModelConfig
{
  std::string robot_description_xml;
  Tool tool{Tool::Pzs100};
  Eigen::Vector3d gravity_m_s2{0.0, 0.0, -9.81};
};

struct Jacobian6x8
{
  Eigen::Matrix<double, 6, 8> value{};
  Frame expressed_in{Frame::MountingBase};
};

struct FullDynamics
{
  FullMass mass{};
  DQ bias{};
  DQ inverse_dynamics_tau{};
};

struct ReducedDynamics
{
  ReducedMass mass_eff{};
  Eigen::Matrix<double, 6, 1> bias_eff{};
};

struct CylinderTransmission
{
  ActuatedJacobian joint_to_cylinder{};
  Eigen::Matrix<double, 6, 1> cylinder_velocity{};
  Eigen::Matrix<double, 6, 1> pressure_force{};
  Eigen::Matrix<double, 6, 1> pump_flow{};
};

struct ChamberPressure
{
  Eigen::Matrix<double, 6, 1> p_a_pa{};
  Eigen::Matrix<double, 6, 1> p_b_pa{};
};

struct CollisionResult
{
  bool collision{false};
  double minimum_distance_m{};
  std::string other_id;
  Eigen::Vector3d witness_on_robot_m{};
  Eigen::Vector3d witness_on_other_m{};
};

struct SymbolicGraphSpec
{
  double sample_time_s{0.04};
  bool include_output_map{true};
};

class SymbolicGraph
{
public:
  SymbolicGraph(SymbolicGraph&&) noexcept;
  SymbolicGraph& operator=(SymbolicGraph&&) noexcept;
  SymbolicGraph(const SymbolicGraph&) = delete;
  SymbolicGraph& operator=(const SymbolicGraph&) = delete;
  ~SymbolicGraph();

  [[nodiscard]] std::size_t state_dimension() const noexcept;
  [[nodiscard]] std::size_t input_dimension() const noexcept;
  [[nodiscard]] bool has_output_map() const noexcept;

private:
  struct Impl;
  explicit SymbolicGraph(std::unique_ptr<Impl> impl) noexcept;
  SymbolicGraph(
    std::size_t state_dimension, std::size_t input_dimension,
    bool has_output_map);
  std::unique_ptr<Impl> impl_;

  friend class Model;
  friend class testing::MockModel;
};

class Model
{
public:
  static Result<Model> create(const ModelConfig& config);

  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();

  [[nodiscard]] Tool tool() const noexcept;
  [[nodiscard]] const std::array<std::string, 8>& urdf_joint_names() const noexcept;
  [[nodiscard]] bool ready() const noexcept;

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
  Result<Vector6> cylinder_force(
    const ChamberPressure& pressure) const;

  Result<CollisionResult> collision_query(
    const Q& q, const CollisionScene& scene) const;
  Result<std::vector<CollisionResult>> collision_queries(
    const Q& q, const CollisionScene& scene) const;

  Result<SymbolicGraph> symbolic_graph(
    const SymbolicGraphSpec& spec, const Payload& payload) const;

private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace crane_model

#endif  // CRANE_MODEL__MODEL_HPP_
