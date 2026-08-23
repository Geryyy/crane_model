#include "crane_model/model.hpp"

#include <cmath>
#include <utility>

namespace crane_model
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

// The six actuated axes in the canonical projection [0, 1, 2, 3, 6, 7] of the
// model API contract §2, i.e. the hardware axis codes SW, HA, KA, SA, RO, GR.
enum AxisIndex : std::size_t
{
  kSlewingAxis = 0,
  kBoomAxis = 1,
  kArmAxis = 2,
  kTelescopeAxis = 3,
  kRotatorAxis = 4,
  kToolAxis = 5,
};

// Machine constants, wiki/hydraulics.md §6. They are verified there against
// parameter_def.cpp and the URDF; symbols follow wiki/nomenclature.md §7.
constexpr double kGearRadius = 0.1;               // r_gear, m
constexpr double kMotorDisplacement = 1.4324e-4;  // V_m, m^3/rad

constexpr double kSlewingPistonArea = 6.362e-3;      // A_A, q1
constexpr double kBoomPistonArea = 1.5394e-2;        // A_A, q2
constexpr double kBoomAnnulusArea = 9.032e-3;        // A_B, q2
constexpr double kArmPistonArea = 6.362e-3;          // A_A, q3
constexpr double kArmAnnulusArea = 2.5133e-3;        // A_B, q3
constexpr double kTelescopePistonArea = 3.8485e-3;   // A_A, q4
constexpr double kTelescopeAnnulusArea = 2.592e-3;   // A_B, q4
constexpr double kToolPistonArea = 7.854e-3;         // A_A, q8
constexpr double kToolAnnulusArea = 4.737e-3;        // A_B, q8

// Boom four-bar, wiki/hydraulics.md §2.2 with the values of §6.2.
constexpr double kBoomFootX = 0.433;             // p_S0
constexpr double kBoomFootY = -1.7682;
constexpr double kBoomPivotX = -0.12;            // p_S1
constexpr double kBoomPivotY = -0.07;
constexpr double kBoomLinkX = 3.49288 - 3.039;   // a_2 + p_S2x
constexpr double kBoomLinkY = -0.036034;         // p_S2y
constexpr double kDrawbarLength = 0.57;          // r_13, Zugstange
constexpr double kPushbarLength = 0.124;         // r_23, Druckstange

// Arm cylinder, wiki/hydraulics.md §2.3 with the values of §6.2. The lateral
// term is p_S4y - p_S3z and not p_S4z - p_S3z because the two attachment
// points live in links with different CAD-to-model rotations; see the callout
// in §2.3 before "normalising" it.
constexpr double kArmFootX = -1.6802;              // p_S3x
constexpr double kArmFootY = -0.0485;              // p_S3y
constexpr double kArmLinkX = -0.3925 + 0.274489;   // a_3 + p_S4x
constexpr double kArmLinkY = -0.468;               // -p_S4z
constexpr double kArmLateralOffset = 0.224 - 0.224;  // p_S4y - p_S3z

// 7040 jaw four-bar, wiki/hydraulics.md §2.6. §6.2 carries none of its numbers,
// so every constant below is ported from the deployed model under src/ and
// names the file it came from. None of them is a §6.2 value.
//
// The fitted mirror law phi_sim,9(q8), which §2.6 says is the variant the
// deployed model uses, in the Horner form of
// timber_crane_model_cpp/include/timber_crane_model_cpp/maple/
// comp_eq_four_bar.hpp:7, with jaw_linkage_p of
// crane_tools_description/7040/config/gripper_parameter.yaml:13.
constexpr double kJawMirrorQuadratic = -0.121220;  // p_0
constexpr double kJawMirrorLinear = 1.400122;      // p_1
constexpr double kJawMirrorConstant = -0.227867;   // p_2

// The two jaw pivots on the pincer frame and the two jaw arm lengths. The
// deployed model reads them from the URDF DH transforms in
// timber_crane_parameter/src/parameter_def.cpp:196-206; the joint origins are
// in crane_tools_description/7040/urdf/links/. Note a_9 and a_11 are the y and
// a_10 and a_12 the x coordinate of their origin, per the comment there.
constexpr double kOuterJawPivotOffset = 0.328;  // a_9, -dh_trans9 y
constexpr double kOuterJawArmLength = 0.8126;   // a_10, dh_trans10 x
constexpr double kInnerJawPivotOffset = 0.336;  // a_11, dh_trans11 y
constexpr double kInnerJawArmLength = 0.8172;   // a_12, dh_trans12 x

// The jaw cylinder's two attachment points. The barrel end sits on the outer
// jaw, joint pincer_cylinder_mounting_outer_jaw_joint of crane_tools_description
// /7040/urdf/joints/hydraulic/gripper_cylinder_joints.urdf.xacro:15. The rod end
// sits on the inner jaw and is hard-coded in gripper_parameter.yaml:5-7 because
// the URDF closes that loop only in Gazebo.
constexpr double kJawCylinderOuterX = -0.8971;    // p_S7x
constexpr double kJawCylinderOuterY = 0.01754;    // p_S7y
constexpr double kJawCylinderInnerX = -0.908397;  // p_S8x
constexpr double kJawCylinderInnerY = 0.015289;   // p_S8y

// Effective areas per axis. a_a and a_b are the force-producing areas of §4,
// a_eff_pos and a_eff_neg the direction-dependent pump draw of §3. The rotator
// carries V_m in all four, in m^3/rad.
struct AxisAreas
{
  double a_a{};
  double a_b{};
  double a_eff_pos{};
  double a_eff_neg{};
};

std::array<AxisAreas, kActuatedDof> axis_areas()
{
  std::array<AxisAreas, kActuatedDof> areas{};
  // q1 slewing: two cylinders, symmetric circuit, so both chambers see 2 A_A.
  areas[kSlewingAxis] = {
    2.0 * kSlewingPistonArea, 2.0 * kSlewingPistonArea,
    2.0 * kSlewingPistonArea, 2.0 * kSlewingPistonArea};
  // q2 boom: a single differential cylinder.
  areas[kBoomAxis] = {
    kBoomPistonArea, kBoomAnnulusArea, kBoomPistonArea, kBoomAnnulusArea};
  // q3 arm: two differential cylinders, one expression (§2.3).
  areas[kArmAxis] = {
    2.0 * kArmPistonArea, 2.0 * kArmAnnulusArea,
    2.0 * kArmPistonArea, 2.0 * kArmAnnulusArea};
  // q4 telescope: regenerative on extension. Rod-side oil is fed back to the
  // piston side, so only the annulus difference is drawn from the pump. The
  // force still follows the physical areas, which is why a_a and a_eff_pos
  // differ on this axis alone.
  areas[kTelescopeAxis] = {
    kTelescopePistonArea, kTelescopeAnnulusArea,
    kTelescopePistonArea - kTelescopeAnnulusArea, kTelescopeAnnulusArea};
  // q7 rotator: a motor, where V_m takes the role the piston area plays
  // elsewhere (§2.5). Symmetric in both directions.
  areas[kRotatorAxis] = {
    kMotorDisplacement, kMotorDisplacement, kMotorDisplacement, kMotorDisplacement};
  // q8 tool: a cylinder axis on the same pump, areas per §6.1.
  areas[kToolAxis] = {
    kToolPistonArea, kToolAnnulusArea, kToolPistonArea, kToolAnnulusArea};
  return areas;
}

Eigen::Matrix2d planar_rotation(double angle)
{
  Eigen::Matrix2d rotation;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  rotation << cosine, -sine, sine, cosine;
  return rotation;
}

// S_perp, the planar 90 degree rotation of wiki/nomenclature.md §7.
Eigen::Matrix2d perpendicular()
{
  Eigen::Matrix2d s_perp;
  s_perp << 0.0, -1.0, 1.0, 0.0;
  return s_perp;
}

// Piston displacement s_i of one axis together with its transmission ratio
// ds_i/dq_i, which is the diagonal entry of J_cyl. The frozen API exposes only
// the ratio today; s_i is carried because the stroke limits of §6.3 are an
// independent constraint on the two nonlinear axes and will need it.
struct CylinderStroke
{
  double stroke_m{};
  double ratio{};
  bool valid{false};
};

// wiki/hydraulics.md §2.2. The cylinder drives the coupler point p_J of a
// four-bar, not the boom directly, so s_2 is the distance from the cylinder
// foot p_S0 to that coupler point. The derivative is the analytic chain rule
// through p_J(d^2(q2)); it is exact, not a difference quotient.
CylinderStroke boom_stroke(double q2)
{
  const Eigen::Vector2d p_s0(kBoomFootX, kBoomFootY);
  const Eigen::Vector2d p_s1(kBoomPivotX, kBoomPivotY);
  const Eigen::Matrix2d s_perp = perpendicular();

  const Eigen::Vector2d p_s2 = planar_rotation(q2) * Eigen::Vector2d(kBoomLinkX, kBoomLinkY);
  const Eigen::Vector2d d_pivot = p_s2 - p_s1;
  const Eigen::Vector2d d_pivot_rate = s_perp * p_s2;  // d(p_S2)/dq2

  const double d_squared = d_pivot.squaredNorm();
  if (!(d_squared > 0.0)) {
    return CylinderStroke{};
  }
  const double d_squared_rate = 2.0 * d_pivot.dot(d_pivot_rate);

  const double reach_sum = kDrawbarLength + kPushbarLength;
  const double reach_difference = kDrawbarLength - kPushbarLength;
  const double delta = (reach_sum * reach_sum - d_squared) *
    (d_squared - reach_difference * reach_difference);
  if (!(delta > 0.0)) {
    // The triangle inequality on d, r_13 and r_23 fails: the linkage cannot
    // close at this q2, so there is no piston displacement to report.
    return CylinderStroke{};
  }

  const double link_difference =
    kDrawbarLength * kDrawbarLength - kPushbarLength * kPushbarLength;
  const double alpha = (link_difference + d_squared) / (2.0 * d_squared);
  const double alpha_rate = -link_difference / (2.0 * d_squared * d_squared) * d_squared_rate;

  const double root = std::sqrt(delta);
  const double delta_gradient =
    reach_sum * reach_sum + reach_difference * reach_difference - 2.0 * d_squared;
  // The branch is fixed by the derivation: the intersection whose local y is
  // negative in the frame with p_S1 at the origin and p_S2 on +x (§2.2). Since
  // S_perp d points along local +y, that is the negative root, and it is the
  // branch whose stroke spans the s_2 limits of §6.3.
  const double beta = -root / (2.0 * d_squared);
  const double beta_rate =
    -(delta_gradient * d_squared / root - 2.0 * root) /
    (4.0 * d_squared * d_squared) * d_squared_rate;

  const Eigen::Vector2d p_j = p_s1 + alpha * d_pivot + beta * (s_perp * d_pivot);
  const Eigen::Vector2d p_j_rate = alpha_rate * d_pivot + alpha * d_pivot_rate +
    beta_rate * (s_perp * d_pivot) + beta * (s_perp * d_pivot_rate);

  const Eigen::Vector2d c_cyl = p_j - p_s0;
  const double stroke = c_cyl.norm();
  if (!(stroke > 0.0)) {
    return CylinderStroke{};
  }
  return CylinderStroke{stroke, c_cyl.dot(p_j_rate) / stroke, true};
}

// wiki/hydraulics.md §2.3. A direct cylinder: the in-plane part rotates with
// q3 while the out-of-plane offset stays constant.
CylinderStroke arm_stroke(double q3)
{
  const Eigen::Vector2d moving = planar_rotation(q3) * Eigen::Vector2d(kArmLinkX, kArmLinkY);
  const Eigen::Vector2d in_plane = moving - Eigen::Vector2d(kArmFootX, kArmFootY);
  const Eigen::Vector2d in_plane_rate = perpendicular() * moving;

  const double stroke =
    std::sqrt(in_plane.squaredNorm() + kArmLateralOffset * kArmLateralOffset);
  if (!(stroke > 0.0)) {
    return CylinderStroke{};
  }
  return CylinderStroke{stroke, in_plane.dot(in_plane_rate) / stroke, true};
}

// wiki/hydraulics.md §2.6 for the 7040, composed the way the deployed model
// composes it. The commanded outer-jaw angle q8 drives the inner jaw through
// the quadratic fit phi_sim,9(q8), and the cylinder spans the two jaws, so its
// length depends on both angles. Frames 19 and 20 of the deployed
// comp_transform_8_19.hpp and comp_transform_8_20.hpp place the two pins: the
// outer jaw hangs off -a_9 with its arm mirrored, the inner jaw off +a_11.
// Planar, exactly as the deployed comp_transform_8_24.hpp is -- the two pins
// are p_S8z + p_S7z = 24.25 mm apart out of plane, which the deployed model
// drops and which is worth at most 0.6 mm of length over the jaw range.
CylinderStroke jaw_stroke(double q8)
{
  const double mirror_angle =
    (kJawMirrorQuadratic * q8 + kJawMirrorLinear) * q8 + kJawMirrorConstant;
  const double mirror_rate = 2.0 * kJawMirrorQuadratic * q8 + kJawMirrorLinear;

  const Eigen::Matrix2d s_perp = perpendicular();
  Eigen::Matrix2d mirror;
  mirror << -1.0, 0.0, 0.0, 1.0;

  const Eigen::Vector2d outer_arm = planar_rotation(q8) *
    Eigen::Vector2d(kOuterJawArmLength + kJawCylinderOuterX, kJawCylinderOuterY);
  const Eigen::Vector2d inner_arm = planar_rotation(mirror_angle) *
    Eigen::Vector2d(kInnerJawArmLength + kJawCylinderInnerX, kJawCylinderInnerY);

  const Eigen::Vector2d c_cyl = inner_arm - mirror * outer_arm +
    Eigen::Vector2d(kInnerJawPivotOffset + kOuterJawPivotOffset, 0.0);
  const Eigen::Vector2d c_cyl_rate =
    mirror_rate * (s_perp * inner_arm) - mirror * (s_perp * outer_arm);

  const double stroke = c_cyl.norm();
  if (!(stroke > 0.0)) {
    return CylinderStroke{};
  }
  return CylinderStroke{stroke, c_cyl.dot(c_cyl_rate) / stroke, true};
}

// J_cyl of wiki/nomenclature.md §7: diagonal by construction, because the
// geometry does not couple the axes at all (§5.7).
Status fill_cylinder_jacobian(Tool tool, const Q& q, ActuatedJacobian& jacobian)
{
  if (!finite(q)) {
    return failure(ErrorCode::NonFiniteInput, "q is not finite");
  }

  jacobian.setZero();
  // q1 slewing: rack and pinion, the only constant transmission (§2.1).
  jacobian(kSlewingAxis, kSlewingAxis) = kGearRadius;

  const CylinderStroke boom = boom_stroke(q[1]);
  if (!boom.valid) {
    return failure(
      ErrorCode::SingularConfiguration, "the boom four-bar does not close at this q2");
  }
  jacobian(kBoomAxis, kBoomAxis) = boom.ratio;

  const CylinderStroke arm = arm_stroke(q[2]);
  if (!arm.valid) {
    return failure(
      ErrorCode::SingularConfiguration, "the arm cylinder is degenerate at this q3");
  }
  jacobian(kArmAxis, kArmAxis) = arm.ratio;

  // q4 telescope: the cylinder is one-to-one with the joint coordinate; the
  // factor of two sits between q4 and the tip travel, not here (§2.4).
  jacobian(kTelescopeAxis, kTelescopeAxis) = 1.0;
  // q7 rotator: a motor, so the motor angle is the joint coordinate (§2.5).
  jacobian(kRotatorAxis, kRotatorAxis) = 1.0;
  // q8 tool (§2.6). The PZS100 rail cylinder is one-to-one with q8 and there is
  // no linkage to solve; the 7040 jaw is a four-bar and its cylinder spans both
  // jaws, so its ratio is configuration-dependent like the boom's.
  if (tool == Tool::Pzs100) {
    jacobian(kToolAxis, kToolAxis) = 1.0;
    return Status{};
  }

  const CylinderStroke jaw = jaw_stroke(q[7]);
  if (!jaw.valid) {
    return failure(
      ErrorCode::SingularConfiguration, "the 7040 jaw cylinder is degenerate at this q8");
  }
  jacobian(kToolAxis, kToolAxis) = jaw.ratio;
  return Status{};
}

Status check_pressure(const ChamberPressure& pressure)
{
  if (!finite(pressure.p_a_pa) || !finite(pressure.p_b_pa)) {
    return failure(ErrorCode::NonFiniteInput, "chamber pressure is not finite");
  }
  if ((pressure.p_a_pa.array() < 0.0).any() || (pressure.p_b_pa.array() < 0.0).any()) {
    return failure(ErrorCode::InvalidArgument, "chamber pressure must be non-negative");
  }
  return Status{};
}

// F_i = A_A p_A - A_B p_B (§4). For the rotator both areas are V_m, so the
// entry is a torque in N m rather than a force in N.
double chamber_force(
  const AxisAreas& areas, const ChamberPressure& pressure, std::size_t axis)
{
  const Eigen::Index index = static_cast<Eigen::Index>(axis);
  return areas.a_a * pressure.p_a_pa[index] - areas.a_b * pressure.p_b_pa[index];
}

Status not_ready()
{
  return failure(ErrorCode::NotReady, "model has no implementation state");
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
  std::array<AxisAreas, kActuatedDof> areas{};
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

  // Everything the hydraulic subset evaluates is built here, once. The three
  // subset calls only read this state afterwards; the rigid-body, collision
  // and symbolic backends are still absent and say so at call time.
  auto impl = std::make_unique<Impl>();
  impl->tool = config.tool;
  impl->names = names;
  impl->areas = axis_areas();
  return Result<Model>::success(Model(std::move(impl)));
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

Result<ActuatedJacobian> Model::cylinder_jacobian(const Q& q) const
{
  if (!impl_) {
    return Result<ActuatedJacobian>::failure(not_ready());
  }
  ActuatedJacobian jacobian;
  Status status = fill_cylinder_jacobian(impl_->tool, q, jacobian);
  if (!status.ok()) {
    return Result<ActuatedJacobian>::failure(std::move(status));
  }
  return Result<ActuatedJacobian>::success(jacobian);
}

Result<CylinderTransmission> Model::transmission(
  const Q& q, const DQA& dq_a, const ChamberPressure& pressure) const
{
  if (!impl_) {
    return Result<CylinderTransmission>::failure(not_ready());
  }
  if (!finite(dq_a)) {
    return Result<CylinderTransmission>::failure(
      failure(ErrorCode::NonFiniteInput, "dq_a is not finite"));
  }
  Status status = check_pressure(pressure);
  if (!status.ok()) {
    return Result<CylinderTransmission>::failure(std::move(status));
  }

  CylinderTransmission result;
  status = fill_cylinder_jacobian(impl_->tool, q, result.joint_to_cylinder);
  if (!status.ok()) {
    return Result<CylinderTransmission>::failure(std::move(status));
  }

  result.cylinder_velocity.noalias() = result.joint_to_cylinder * dq_a;
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    const Eigen::Index index = static_cast<Eigen::Index>(axis);
    const AxisAreas& areas = impl_->areas[axis];
    const double velocity = result.cylinder_velocity[index];
    // The area switch of §3 is a step at v_i = 0: the pump draw per unit of
    // piston travel is not the same extending and retracting.
    const double effective_area = velocity > 0.0 ? areas.a_eff_pos : areas.a_eff_neg;
    result.pump_flow[index] = effective_area * std::abs(velocity);
    result.pressure_force[index] = chamber_force(areas, pressure, axis);
  }
  return Result<CylinderTransmission>::success(result);
}

Result<Vector6> Model::cylinder_force(const ChamberPressure& pressure) const
{
  if (!impl_) {
    return Result<Vector6>::failure(not_ready());
  }
  Status status = check_pressure(pressure);
  if (!status.ok()) {
    return Result<Vector6>::failure(std::move(status));
  }

  Vector6 force;
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    force[static_cast<Eigen::Index>(axis)] = chamber_force(impl_->areas[axis], pressure, axis);
  }
  return Result<Vector6>::success(force);
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
CRANE_MODEL_UNAVAILABLE(
  CollisionResult, collision_query(const Q&, const CollisionScene&) const)
CRANE_MODEL_UNAVAILABLE(
  std::vector<CollisionResult>, collision_queries(const Q&, const CollisionScene&) const)
CRANE_MODEL_UNAVAILABLE(
  SymbolicGraph, symbolic_graph(const SymbolicGraphSpec&, const Payload&) const)

#undef CRANE_MODEL_UNAVAILABLE

}  // namespace crane_model
