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

private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

/// What one cylinder can push and pull at the relief pressure, newtons.
/**
 * Two numbers and not one, because a differential cylinder is not symmetric:
 * `hydraulics.md` §4's `F_i = A_A p_A - A_B p_B` with one chamber at the relief
 * setting and the other at tank gives a larger extending force than retracting
 * one. A consumer that needs a symmetric bound takes the smaller of the two.
 */
struct CylinderForceLimit
{
  double extend{};
  double retract{};

  /// The symmetric bound: the weaker of the two directions.
  [[nodiscard]] double symmetric() const noexcept
  {
    return extend < retract ? extend : retract;
  }
};

/// `F_i^max` per actuated axis, from the model's own chamber areas and a pressure.
/**
 * This is a **model** question and lives here for that reason: the chamber areas
 * are the description's plus `config/hydraulics.yaml`, both of which this package
 * already holds, so `Model::cylinder_force` is asked rather than a table copied
 * into a planner or a controller where it would drift.
 *
 * `system_pressure_pa` is the one number that cannot be derived and is not
 * measured -- `wiki/implementation/parameters.md` §7 lists the pressure constants
 * among its gaps -- so it is an argument and every deployment says where its own
 * value came from.
 */
[[nodiscard]] Result<std::array<CylinderForceLimit, kActuatedDof>>
derive_cylinder_force_limits(const Model& model, double system_pressure_pa);

}  // namespace crane_model

#endif  // CRANE_MODEL__MODEL_HPP_
