#include "crane_model/model.hpp"

#include <urdf_parser/urdf_parser.h>

#include <coal/collision_object.h>
#include <coal/distance.h>
#include <coal/shape/geometric_shapes.h>

#include <Eigen/Cholesky>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

#include "collision_model.hpp"
#include "cylinder_geometry.hpp"
#include "model_detail.hpp"

#if !defined(CRANE_MODEL_HYDRAULICS_YAML_INSTALLED) || \
  !defined(CRANE_MODEL_HYDRAULICS_YAML_SOURCE)
#error "the build must say where config/hydraulics.yaml is; see CMakeLists.txt"
#endif

namespace crane_model
{

// --- config/hydraulics.yaml --------------------------------------------------
//
// The constants of `hydraulics::Constants`, read out of the file instead of
// compiled in, so the Python export that builds these same dynamics with
// `pinocchio.casadi` reads the numbers rather than a copy of them. The parsing
// is yaml-cpp's (wiki/implementation/libraries.md); nothing here re-implements
// a format.
//
// Every read goes through the reader below, so a key that is absent, empty, of
// the wrong shape or not a number is `InvalidArgument` naming the file and the
// key. Nothing in here has a default: style guide §4, and a hydraulic constant
// that quietly fell back to zero is a force limit that quietly fell back to
// zero.
namespace hydraulics
{
namespace
{

// yaml-cpp's `Node` is a handle into the document, and `node = node[key]`
// *writes* through that handle. Every descent below therefore goes through
// `reset`, which rebinds it instead.
class Reader
{
public:
  Reader(const YAML::Node& root, std::string path)
  : root_(root), path_(std::move(path))
  {
  }

  [[nodiscard]] const Status& status() const { return status_; }

  void fail(const std::string& key, const char * why)
  {
    if (status_.ok()) {
      status_ = Status{ErrorCode::InvalidArgument, path_ + ": " + key + " " + why};
    }
  }

  // The node at a dotted key, or an undefined node once the first descent has
  // failed. Only mappings are descended into; a sequence is asked for by name.
  YAML::Node at(const std::string& key)
  {
    YAML::Node node(root_);
    std::size_t start = 0;
    while (true) {
      const std::size_t dot = key.find('.', start);
      const std::string part = key.substr(
        start, dot == std::string::npos ? std::string::npos : dot - start);
      if (!node.IsMap() || !node[part]) {
        fail(key, "is missing");
        return YAML::Node(YAML::NodeType::Undefined);
      }
      node.reset(node[part]);
      if (dot == std::string::npos) {
        return node;
      }
      start = dot + 1;
    }
  }

  double number(const std::string& key)
  {
    const YAML::Node node = at(key);
    double value = 0.0;
    if (!node.IsDefined()) {
      return value;
    }
    if (!YAML::convert<double>::decode(node, value) || !std::isfinite(value)) {
      fail(key, "is not a finite number");
      return 0.0;
    }
    return value;
  }

  std::string text(const std::string& key)
  {
    const YAML::Node node = at(key);
    std::string value;
    if (!node.IsDefined()) {
      return value;
    }
    if (!YAML::convert<std::string>::decode(node, value) || value.empty()) {
      fail(key, "is not a non-empty string");
      return std::string{};
    }
    return value;
  }

  YAML::Node sequence(const std::string& key, std::size_t size)
  {
    const YAML::Node node = at(key);
    if (!node.IsDefined()) {
      return node;
    }
    if (!node.IsSequence() || (size != 0 && node.size() != size)) {
      fail(key, "is not a sequence of the expected length");
      return YAML::Node(YAML::NodeType::Undefined);
    }
    return node;
  }

  // One entry of a sequence of mappings, which `at` cannot reach by name.
  std::string entry_text(const YAML::Node& entry, const std::string& key, const char * name)
  {
    std::string value;
    if (!entry.IsMap() || !entry[name] ||
      !YAML::convert<std::string>::decode(entry[name], value) || value.empty())
    {
      fail(key, "carries an entry whose fields are missing or empty");
      return std::string{};
    }
    return value;
  }

  double entry_number(const YAML::Node& entry, const std::string& key, const char * name)
  {
    double value = 0.0;
    if (!entry.IsMap() || !entry[name] ||
      !YAML::convert<double>::decode(entry[name], value) || !std::isfinite(value))
    {
      fail(key, "carries an entry whose fields are missing or not finite");
      return 0.0;
    }
    return value;
  }

private:
  YAML::Node root_;
  std::string path_;
  Status status_{};
};

}  // namespace

std::string default_path()
{
  // The installed copy is what a deployed build has; the source tree is what a
  // build tree has before `colcon` has installed anything. Both are compiled in
  // because `crane_model` is ROS-free (contract §6) and there is no ament index
  // here to ask, and `ModelConfig` is frozen so a caller cannot pass a third.
  const std::ifstream installed(CRANE_MODEL_HYDRAULICS_YAML_INSTALLED);
  if (installed.good()) {
    return CRANE_MODEL_HYDRAULICS_YAML_INSTALLED;
  }
  return CRANE_MODEL_HYDRAULICS_YAML_SOURCE;
}

Status load(const std::string& path, Constants& out)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception& error) {
    return Status{
      ErrorCode::InvalidArgument,
      "the hydraulic constants could not be read from " + path + ": " + error.what()};
  }
  if (!root.IsMap()) {
    return Status{ErrorCode::InvalidArgument, path + ": the document is not a mapping"};
  }

  Reader reader(root, path);
  Constants value;
  value.r_gear = reader.number("r_gear");
  value.v_m = reader.number("V_m");

  value.slewing_a_a = reader.number("areas.q1_slewing.A_A");
  value.boom_a_a = reader.number("areas.q2_boom.A_A");
  value.boom_a_b = reader.number("areas.q2_boom.A_B");
  value.arm_a_a = reader.number("areas.q3_arm.A_A");
  value.arm_a_b = reader.number("areas.q3_arm.A_B");
  value.telescope_a_a = reader.number("areas.q4_telescope.A_A");
  value.telescope_a_b = reader.number("areas.q4_telescope.A_B");
  value.tool_a_a = reader.number("areas.q8_tool.A_A");
  value.tool_a_b = reader.number("areas.q8_tool.A_B");

  value.boom_ps0_x = reader.number("boom.pS0.x");
  value.boom_ps0_y = reader.number("boom.pS0.y");
  value.boom_ps1_x = reader.number("boom.pS1.x");
  value.boom_ps1_y = reader.number("boom.pS1.y");
  value.boom_a2 = reader.number("boom.a2");
  value.boom_ps2_x = reader.number("boom.pS2.x");
  value.boom_ps2_y = reader.number("boom.pS2.y");
  value.r13 = reader.number("boom.r13");
  value.r23 = reader.number("boom.r23");

  value.arm_ps3_x = reader.number("arm.pS3.x");
  value.arm_ps3_y = reader.number("arm.pS3.y");
  value.arm_ps3_z = reader.number("arm.pS3.z");
  value.arm_a3 = reader.number("arm.a3");
  value.arm_ps4_x = reader.number("arm.pS4.x");
  value.arm_ps4_y = reader.number("arm.pS4.y");
  value.arm_ps4_z = reader.number("arm.pS4.z");

  value.jaw_p0 = reader.number("jaw.mirror.p0");
  value.jaw_p1 = reader.number("jaw.mirror.p1");
  value.jaw_p2 = reader.number("jaw.mirror.p2");
  value.jaw_a9 = reader.number("jaw.a9");
  value.jaw_a10 = reader.number("jaw.a10");
  value.jaw_a11 = reader.number("jaw.a11");
  value.jaw_a12 = reader.number("jaw.a12");
  value.jaw_ps7_x = reader.number("jaw.pS7.x");
  value.jaw_ps7_y = reader.number("jaw.pS7.y");
  value.jaw_ps8_x = reader.number("jaw.pS8.x");
  value.jaw_ps8_y = reader.number("jaw.pS8.y");

  value.eps_abs = reader.number("smoothing.eps_abs");
  value.eps_v = reader.number("smoothing.eps_v");

  // The canonical eight of contract §2. The first seven are shared and the
  // eighth is the tool joint, which is the whole of the map's tool dependence.
  const YAML::Node shared = reader.sequence("joints.shared", kGeneralizedDof - 1);
  for (std::size_t index = 0; index + 1 < kGeneralizedDof && shared.IsDefined(); ++index) {
    std::string name;
    if (!YAML::convert<std::string>::decode(shared[index], name) || name.empty()) {
      reader.fail("joints.shared", "carries an entry that is not a joint name");
      break;
    }
    value.pzs100_joints[index] = name;
    value.epsilon7040_joints[index] = name;
  }
  value.pzs100_joints[kGeneralizedDof - 1] = reader.text("joints.tool.pzs100");
  value.epsilon7040_joints[kGeneralizedDof - 1] = reader.text("joints.tool.epsilon7040");

  // The damping table. A reason is mandatory: an override with no recorded
  // reason is indistinguishable from a typo, and this is the one place in the
  // model where the description is deliberately contradicted.
  const YAML::Node overrides = reader.sequence("damping_overrides", 0);
  if (overrides.IsDefined()) {
    for (const YAML::Node& entry : overrides) {
      DampingOverride record;
      record.joint = reader.entry_text(entry, "damping_overrides", "joint");
      record.d = reader.entry_number(entry, "damping_overrides", "d");
      record.reason = reader.entry_text(entry, "damping_overrides", "reason");
      value.damping_overrides.push_back(std::move(record));
    }
  }

  if (!reader.status().ok()) {
    return reader.status();
  }
  out = std::move(value);
  return Status{};
}

Status load(Constants& out)
{
  return load(default_path(), out);
}

}  // namespace hydraulics

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

bool valid_tool(Tool tool)
{
  return tool == Tool::Pzs100 || tool == Tool::Epsilon7040;
}

// The Frame -> URDF link map of the model API contract §3, written down once.
// The contract's own enum comments abbreviate the links in *coordinate*
// numbering (`K5_tip`, `K6_tilt`, `K7_rotator`); the descriptions number their
// links K0..K8 in *link* numbering, which robot_model §0.1 warns is not the
// same sequence. The spellings below are the ones the URDF actually carries.
//
// `world` and `tool_contact_point` are not in every description: only the 7040
// gripper defines a contact point, and the world-to-K0 step is the world-model
// boundary of contract §8 and stays outside this library. Both resolve to
// FrameUnavailable rather than to a substituted pose.
constexpr std::size_t kFrameCount = 12;

struct FrameLink
{
  Frame frame;
  const char * link;
};

constexpr std::array<FrameLink, kFrameCount> kFrameLinks{{
  {Frame::World, "world"},
  {Frame::MountingBase, "K0_mounting_base"},
  {Frame::SlewingColumn, "K1_slewing_column"},
  {Frame::Boom, "K2_boom"},
  {Frame::Arm, "K3_arm"},
  {Frame::BigTelescope, "K4_outer_telescope"},
  {Frame::Tip, "K5_inner_telescope"},
  {Frame::Tilt, "K6_double_joint_link"},
  {Frame::Rotator, "K7_rotator_upper_part"},
  {Frame::RotatorLowerPart, "K8_rotator_lower_part"},
  {Frame::Tcp, "K8_tool_center_point"},
  {Frame::ToolContact, "tool_contact_point"},
}};

constexpr bool frame_table_is_ordered()
{
  for (std::size_t index = 0; index < kFrameCount; ++index) {
    if (static_cast<std::size_t>(kFrameLinks[index].frame) != index) {
      return false;
    }
  }
  return true;
}
static_assert(frame_table_is_ordered(), "kFrameLinks must be indexable by Frame");

bool valid_frame(Frame frame)
{
  return static_cast<std::size_t>(frame) < kFrameCount;
}

// The canonical projections of contract §2 and the cylinder transmission of
// wiki/hydraulics.md §2--§4 are shared with the symbolic graph rather than
// written twice. `cylinder_geometry.hpp` carries the numeric transmission
// geometry, while the Python code generator reads the same YAML constants.
using cylinder::AxisAreas;
using cylinder::kArmAxis;
using cylinder::kBoomAxis;
using cylinder::kRotatorAxis;
using cylinder::kSlewingAxis;
using cylinder::kTelescopeAxis;
using cylinder::kToolAxis;
using detail::CoupledJoint;
using detail::Drive;
using detail::JointSlot;
using detail::kActuatedRows;
using detail::kMaxDrives;
using detail::kPassiveRows;

// J_cyl of wiki/nomenclature.md §7: diagonal by construction, because the
// geometry does not couple the axes at all (§5.7).
//
// A configuration at which a linkage does not close leaves its ratio
// non-finite rather than raising a flag, because the symbolic instantiation has
// no configuration to raise one at. Here that non-finite ratio is this
// configuration's `SingularConfiguration`, named per axis; the three constant
// axes cannot reach it.
Status fill_cylinder_jacobian(
  const hydraulics::Constants& constants, Tool tool, const Q& q, ActuatedJacobian& jacobian)
{
  if (!finite(q)) {
    return failure(ErrorCode::NonFiniteInput, "q is not finite");
  }
  static constexpr std::array<const char *, kActuatedDof> kDegenerate{{
    "the slewing rack is degenerate",
    "the boom four-bar does not close at this q2",
    "the arm cylinder is degenerate at this q3",
    "the telescope cylinder is degenerate",
    "the rotator motor is degenerate",
    "the 7040 jaw cylinder is degenerate at this q8"}};

  const std::array<double, kActuatedDof> diagonal =
    cylinder::jacobian_diagonal(constants, tool, q[1], q[2], q[7]);
  jacobian.setZero();
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    if (!std::isfinite(diagonal[axis])) {
      return failure(ErrorCode::SingularConfiguration, kDegenerate[axis]);
    }
    const Eigen::Index index = static_cast<Eigen::Index>(axis);
    jacobian(index, index) = diagonal[axis];
  }
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

// --- the payload of wiki/robot_model.md §5 -----------------------------------
//
// A rigid body attached to K8_rotator_lower_part, given by m_L, r_L^(8) and
// Theta_L. Two things the contract leaves open, fixed here and stated in the
// README because something has to fix them:
//
//   * `inertia_k8_kg_m2` is the rotational inertia **about the payload centre of
//     mass**, with the axes of K8 -- the URDF `<inertial>` convention, so the
//     same numbers a description would carry for the same body. It is not the
//     inertia about the K8 origin.
//   * `valid == false` is not a payload. Contract §4 calls it an explicit
//     unknown, and an unknown payload is refused rather than silently replaced
//     by a zero-mass one; a caller carrying nothing declares a valid payload of
//     zero mass, which is a different statement and is accepted.
//
// A rotation-invariant floor on the symmetry and definiteness checks. The
// entries are kg m^2 and a message round trip loses a few bits, so this is
// relative to the tensor's own norm rather than absolute.
constexpr double kInertiaTolerance = 1.0e-9;

Status check_payload(const Payload& payload)
{
  if (!payload.valid) {
    return failure(
      ErrorCode::InvalidPayload,
      "payload is not declared valid; an unknown payload is not a zero-mass payload");
  }
  if (!std::isfinite(payload.mass_kg) || !finite(payload.center_of_mass_k8_m) ||
    !finite(payload.inertia_k8_kg_m2))
  {
    return failure(ErrorCode::InvalidPayload, "payload carries non-finite data");
  }
  if (payload.mass_kg < 0.0) {
    return failure(ErrorCode::InvalidPayload, "payload mass must be non-negative");
  }
  const Eigen::Matrix3d& inertia = payload.inertia_k8_kg_m2;
  const double scale = std::max(1.0, inertia.norm());
  if ((inertia - inertia.transpose()).norm() > kInertiaTolerance * scale) {
    return failure(ErrorCode::InvalidPayload, "payload inertia is not symmetric");
  }
  // Positive *semi*-definite: a point mass carries no rotational inertia at all
  // and is a perfectly good payload, so zero is admissible and negative is not.
  Eigen::LDLT<Eigen::Matrix3d> factorisation(inertia);
  if (factorisation.info() != Eigen::Success || !factorisation.isPositive()) {
    return failure(ErrorCode::InvalidPayload, "payload inertia is not positive semi-definite");
  }
  return Status{};
}

// --- the partitioned dynamics of wiki/robot_model.md §1 ----------------------

// M and h split by the actuated/passive partition. The blocks are gathered by
// index rather than by a block expression, because I_a is not contiguous.
struct Partition
{
  Eigen::Matrix<double, 6, 6> M_aa{};
  Eigen::Matrix<double, 6, 2> M_au{};
  Eigen::Matrix<double, 2, 6> M_ua{};
  PassiveMass M_uu{};
  Eigen::Matrix<double, 6, 1> h_a{};
  QU h_u{};
};

Partition partition(const FullMass& mass, const DQ& bias)
{
  Partition split;
  for (std::size_t row = 0; row < kActuatedDof; ++row) {
    split.h_a[static_cast<Eigen::Index>(row)] = bias[kActuatedRows[row]];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      split.M_aa(
        static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(column)) = mass(kActuatedRows[row], kActuatedRows[column]);
    }
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      split.M_au(
        static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(column)) = mass(kActuatedRows[row], kPassiveRows[column]);
    }
  }
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    split.h_u[static_cast<Eigen::Index>(row)] = bias[kPassiveRows[row]];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      split.M_ua(
        static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(column)) = mass(kPassiveRows[row], kActuatedRows[column]);
    }
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      split.M_uu(
        static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(column)) = mass(kPassiveRows[row], kPassiveRows[column]);
    }
  }
  return split;
}

// The Cholesky of M_uu, which robot_model §3.1 says is what the sway solve is.
// A configuration at which it is not positive definite is one where the passive
// accelerations are not determined, and contract §5 says so rather than
// returning a matrix of infinities.
Status factorise_passive_mass(const PassiveMass& M_uu, Eigen::LLT<PassiveMass>& factorisation)
{
  factorisation.compute(M_uu);
  if (factorisation.info() != Eigen::Success) {
    return failure(
      ErrorCode::SingularConfiguration,
      "M_uu is not positive definite at this configuration, so the passive rows do not solve");
  }
  return Status{};
}

// --- the passive equilibrium of wiki/robot_model.md §2.3 ---------------------
//
// h_u(q_a, q_eq, 0) = 0. At dq = 0 every velocity-dependent term of §1 drops out
// of h -- Coriolis, centrifugal and D dq alike -- so the condition says that the
// passive rows of the *gravity* torque vanish, which is to say that q_eq is a
// critical point of the potential energy over the two passive coordinates.
// Newton converges quadratically on it, because the exact derivative is at hand:
// dh_u/dq_u is the passive block of Pinocchio's gravity Jacobian, i.e. the
// Hessian of that same potential, and it is symmetric for that reason.
//
// A two-hinge pendulum has four critical points on the torus and two of them are
// the tool standing *up*. They satisfy the equation exactly as well as the
// hanging one does, so the solve may not return one: robot_model §2.3 asks for
// where the tool settles, mpc §2 uses it as the point the sway cost damps
// toward, and trajectory_planning §6 makes it the endpoint the tool must arrive
// at without swinging. The condition that distinguishes them is the sign of the
// stiffness, so the returned pose is required to have a positive definite one --
// checked at every iterate, so a step that leaves the hanging well is refused
// where it happens rather than at the end.

// The residual h_u is driven to, in N m. The passive rows carry of order 1e3 N m
// of individual gravity terms that cancel at the equilibrium, so double
// precision leaves a noise floor near 1e-13 N m; against a restoring stiffness of
// 2e3 N m/rad this tolerance is an angle error below 1e-11 rad, and it is
// reported in the README as the model's own residual bound. It is two decades
// above the achieved residual on both descriptions and five below the 1e-6 N m
// the invariant of contract §7 is asserted at.
constexpr double kEquilibriumResidual = 1.0e-8;

// Newton from inside the well reaches that in single digits of iterations. The
// budget is a runaway guard, not a working range: a solve that has not settled
// within it has not found the well and says so.
constexpr int kEquilibriumIterations = 32;

// Samples per passive axis in the seed search, endpoints included. Five over the
// pi of tip range puts a sample within an eighth of a turn of anywhere in it,
// which is well inside the quarter turn that separates the hanging well from the
// saddles either side of it. Twenty-five samples is also what makes this call
// cost tens of `rnea` evaluations rather than one: it allocates nothing, so
// contract §10 does not exclude it, but the README says plainly that its cost is
// a per-plan one and not a per-cycle one.
constexpr int kEquilibriumSamples = 5;

// A canonical joint must exist and must be a single degree of freedom. The
// contract calls a description that does not carry it `MissingJoint`; a
// description that carries the name on something that is not a one-DoF joint is
// not a joint map at all, and says so as `InvalidRobotDescription`.
Status bind_joint(const pinocchio::Model& model, const std::string& name, JointSlot& slot)
{
  if (!model.existJointName(name)) {
    return failure(ErrorCode::MissingJoint, "robot description is missing joint " + name);
  }
  const pinocchio::JointIndex id = model.getJointId(name);
  const auto& joint = model.joints[id];
  if (joint.nv() != 1 || (joint.nq() != 1 && joint.nq() != 2)) {
    return failure(
      ErrorCode::InvalidRobotDescription, "joint " + name + " is not a single degree of freedom");
  }
  slot.config_index = static_cast<Eigen::Index>(joint.idx_q());
  slot.velocity_index = static_cast<Eigen::Index>(joint.idx_v());
  slot.unbounded = joint.nq() == 2;
  return Status{};
}

// --- the collision model -----------------------------------------------------
//
// wiki/trajectory_planning.md §4.2: one model, one library, one URDF. Pinocchio
// places the links and Coal answers the queries, over the geometry
// src/collision_model.hpp carries and over the `CollisionScene` the caller
// supplies. Contract §8 puts the world-to-K0 conversion at the world-model
// boundary, so the scene arrives in `K0_mounting_base` and every pose and
// witness point below is in that frame.

using collision_model::kAllowedSelfPairs;
using collision_model::kLinkPrimitives;
using collision_model::LinkPrimitive;
using collision_model::LinkShape;

// The reserved scene id of a payload the tool is carrying. The frozen collision
// signature takes `q` and a scene and no `Payload`, so a carried payload reaches
// the model the only way it can: as a scene primitive, at the pose the caller
// reads out of forward_kinematics. What the model owes it is the half a caller
// cannot supply -- the primitive is not checked against the links that hold it,
// which is what "attached to K8" means for a collision model. README says so.
constexpr const char * kPayloadId = "payload";

// How a self-collision names the other side in `CollisionResult::other_id`.
constexpr char kSelfPairSeparator = '|';

// `dimensions_m` is the primitive's extent along each axis of its own frame, so
// a box carries its side lengths, a cylinder (2r, 2r, length) and a sphere its
// diameter three times. The contract does not fix this; the model does, and it
// checks it: a cylinder or sphere whose extents disagree is `InvalidScene` and
// not a silently reinterpreted radius.
constexpr double kExtentTolerance = 1.0e-9;
// A scene rotation has to be a rotation. Message round-trips lose a few bits,
// so this is loose enough for float, tight enough to catch a scaled basis.
constexpr double kRotationTolerance = 1.0e-6;

std::shared_ptr<coal::CollisionGeometry> link_geometry(const LinkPrimitive& entry)
{
  if (entry.shape == LinkShape::Capsule) {
    return std::make_shared<coal::Capsule>(entry.extents_m[0], 2.0 * entry.extents_m[1]);
  }
  return std::make_shared<coal::Box>(
    2.0 * entry.extents_m[0], 2.0 * entry.extents_m[1], 2.0 * entry.extents_m[2]);
}

pinocchio::SE3 primitive_placement(const LinkPrimitive& entry)
{
  const Eigen::Quaterniond orientation(
    entry.orientation[3], entry.orientation[0], entry.orientation[1], entry.orientation[2]);
  return pinocchio::SE3(
    orientation.normalized().toRotationMatrix(),
    Eigen::Vector3d(entry.origin_m[0], entry.origin_m[1], entry.origin_m[2]));
}

coal::Transform3s coal_transform(const pinocchio::SE3& placement)
{
  return coal::Transform3s(placement.rotation(), placement.translation());
}

bool self_pair_is_allowed(const std::string& first, const std::string& second)
{
  return std::any_of(
    kAllowedSelfPairs.begin(), kAllowedSelfPairs.end(),
    [&first, &second](const collision_model::LinkPair& pair) {
      return (first == pair.first && second == pair.second) ||
             (first == pair.second && second == pair.first);
    });
}

// One scene primitive, validated and turned into geometry Coal can answer for.
struct SceneBody
{
  const CollisionPrimitive * primitive{nullptr};
  std::shared_ptr<coal::CollisionGeometry> geometry;
  coal::Transform3s pose;
  bool payload{false};
};

Status scene_geometry(
  const CollisionPrimitive& primitive, std::shared_ptr<coal::CollisionGeometry>& geometry)
{
  const Eigen::Vector3d& extents = primitive.dimensions_m;
  if ((extents.array() <= 0.0).any()) {
    return failure(
      ErrorCode::InvalidScene,
      "collision primitive " + primitive.id + " has a non-positive extent");
  }
  switch (primitive.shape) {
    case CollisionShape::Box:
      geometry = std::make_shared<coal::Box>(extents.x(), extents.y(), extents.z());
      return Status{};
    case CollisionShape::Cylinder:
      if (std::abs(extents.x() - extents.y()) > kExtentTolerance * extents.x()) {
        return failure(
          ErrorCode::InvalidScene,
          "cylinder " + primitive.id + " needs equal x and y extents, both its diameter");
      }
      geometry = std::make_shared<coal::Cylinder>(0.5 * extents.x(), extents.z());
      return Status{};
    case CollisionShape::Sphere:
      if (std::abs(extents.x() - extents.y()) > kExtentTolerance * extents.x() ||
        std::abs(extents.x() - extents.z()) > kExtentTolerance * extents.x())
      {
        return failure(
          ErrorCode::InvalidScene,
          "sphere " + primitive.id + " needs three equal extents, all its diameter");
      }
      geometry = std::make_shared<coal::Sphere>(0.5 * extents.x());
      return Status{};
    default:
      break;
  }
  return failure(
    ErrorCode::InvalidScene,
    "collision primitive " + primitive.id + " has a shape that is not a CollisionShape value");
}

// Contract §4: an unusable primitive is refused, never skipped, so one bad entry
// fails the whole scene rather than quietly shrinking it.
Status build_scene(const CollisionScene& scene, std::vector<SceneBody>& bodies)
{
  bodies.clear();
  bodies.reserve(scene.primitives.size());
  for (const CollisionPrimitive& primitive : scene.primitives) {
    if (primitive.id.empty()) {
      return failure(ErrorCode::InvalidScene, "a collision primitive has an empty id");
    }
    const bool duplicate = std::any_of(
      bodies.begin(), bodies.end(),
      [&primitive](const SceneBody& body) {return body.primitive->id == primitive.id;});
    if (duplicate) {
      return failure(
        ErrorCode::InvalidScene, "collision primitive id " + primitive.id + " is used twice");
    }
    if (!finite(primitive.pose_in_mounting_base.matrix()) || !finite(primitive.dimensions_m)) {
      return failure(
        ErrorCode::InvalidScene,
        "collision primitive " + primitive.id + " has a non-finite pose or extent");
    }
    const Eigen::Matrix3d rotation = primitive.pose_in_mounting_base.linear();
    const double orthonormal =
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm();
    if (orthonormal > kRotationTolerance || rotation.determinant() < 0.0) {
      return failure(
        ErrorCode::InvalidScene,
        "collision primitive " + primitive.id + " carries a pose that is not a rotation");
    }

    SceneBody body;
    body.primitive = &primitive;
    body.payload = primitive.id == kPayloadId;
    Status status = scene_geometry(primitive, body.geometry);
    if (!status.ok()) {
      return status;
    }
    body.pose = coal::Transform3s(rotation, primitive.pose_in_mounting_base.translation());
    bodies.push_back(std::move(body));
  }
  return Status{};
}

// One Coal query, in K0. `enable_signed_distance` is what makes the result
// usable on both sides of contact: the distance stays a real number when the two
// overlap, and the witness points stay the pair that realises it.
Status pair_distance(
  const coal::CollisionGeometry * robot, const coal::Transform3s& robot_pose,
  const coal::CollisionGeometry * other, const coal::Transform3s& other_pose,
  std::string other_id, CollisionResult& out)
{
  coal::DistanceRequest request;
  request.enable_signed_distance = true;
  coal::DistanceResult result;
  try {
    coal::distance(robot, robot_pose, other, other_pose, request, result);
  } catch (const std::exception& error) {
    return failure(
      ErrorCode::CollisionBackendFailure,
      "Coal failed the query against " + other_id + ": " + error.what());
  }
  if (!std::isfinite(result.min_distance) || !finite(result.nearest_points[0]) ||
    !finite(result.nearest_points[1]))
  {
    return failure(
      ErrorCode::CollisionBackendFailure,
      "Coal returned no usable distance for " + other_id);
  }
  out.collision = result.min_distance < 0.0;
  out.minimum_distance_m = result.min_distance;
  out.other_id = std::move(other_id);
  out.witness_on_robot_m = result.nearest_points[0];
  out.witness_on_other_m = result.nearest_points[1];
  return Status{};
}

}  // namespace

namespace detail
{

Status attach_payload(
  pinocchio::Model& model, const PayloadMount& mount, const Payload& payload)
{
  Status status = check_payload(payload);
  if (!status.ok()) {
    return status;
  }
  if (!mount.attachable) {
    return failure(
      ErrorCode::FrameUnavailable,
      std::string("the robot description carries no link ") +
      kFrameLinks[static_cast<std::size_t>(Frame::RotatorLowerPart)].link +
      ", so a payload cannot be attached");
  }
  // Theta_L is about the payload's own centre of mass with the axes of K8,
  // which is the URDF `<inertial>` convention; act() carries the whole body
  // from K8 into the frame of the joint that moves it.
  const pinocchio::Inertia body(
    payload.mass_kg, payload.center_of_mass_k8_m, payload.inertia_k8_kg_m2);
  model.inertias[mount.joint] = mount.bare_inertia + mount.placement.act(body);
  return Status{};
}

// The description, parsed and mapped onto the canonical eight coordinates.
// `Model::create` builds its implementation state out of what this returns and
// the joint map, the mimic projection P, the damping and the payload mount are
// kept in one parsed numeric model.
Status parse(const ModelConfig& config, ParsedModel& out)
{
  if (config.robot_description_xml.empty()) {
    return failure(ErrorCode::InvalidRobotDescription, "robot_description_xml is empty");
  }
  if (!valid_tool(config.tool)) {
    return failure(ErrorCode::UnsupportedTool, "tool is not supported");
  }
  if (!finite(config.gravity_m_s2) || config.gravity_m_s2.norm() <= 0.0) {
    return failure(ErrorCode::InvalidArgument, "gravity_m_s2 must be finite and non-zero");
  }

  // Everything the description does not carry, before anything that does: the
  // canonical joint map is in there, so the description cannot even be walked
  // until the file has been read. A missing or malformed file fails here, which
  // is `Model::create` failing, which is what a wrong force limit deserves.
  Status constants = hydraulics::load(out.constants);
  if (!constants.ok()) {
    return constants;
  }

  // The description is parsed twice on purpose. Pinocchio builds the kinematic
  // tree and drops `<mimic>` -- with mimic parsing on it refuses this
  // description outright, because the PZS100 declares the right rail as a mimic
  // of a joint that comes *later* in its own depth-first order. urdfdom, which
  // is Pinocchio's own URDF front end, still carries the mimic declarations, so
  // the coupled joints are read from there rather than assumed.
  ::urdf::ModelInterfaceSharedPtr tree;
  try {
    tree = ::urdf::parseURDF(config.robot_description_xml);
    if (!tree) {
      return failure(
        ErrorCode::InvalidRobotDescription, "robot_description_xml is not valid URDF");
    }
    pinocchio::urdf::buildModelFromXML(config.robot_description_xml, out.model);
  } catch (const std::exception& error) {
    return failure(
      ErrorCode::InvalidRobotDescription,
      std::string("robot_description_xml could not be parsed: ") + error.what());
  }
  out.model.gravity.linear() = config.gravity_m_s2;
  out.neutral = pinocchio::neutral(out.model);
  out.tool = config.tool;

  const std::array<std::string, kGeneralizedDof>& canonical = out.constants.joints(config.tool);
  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    Status status = bind_joint(out.model, canonical[index], out.joints[index]);
    if (!status.ok()) {
      return status;
    }
  }

  for (const auto& entry : tree->joints_) {
    const auto& joint = entry.second;
    if (!joint || !joint->mimic) {
      continue;
    }
    const auto driver = std::find(canonical.begin(), canonical.end(), joint->mimic->joint_name);
    if (driver == canonical.end() || !out.model.existJointName(joint->name)) {
      continue;
    }
    CoupledJoint coupled;
    Status status = bind_joint(out.model, joint->name, coupled.slot);
    if (!status.ok()) {
      return status;
    }
    coupled.source = static_cast<std::size_t>(std::distance(canonical.begin(), driver));
    coupled.multiplier = joint->mimic->multiplier;
    coupled.offset = joint->mimic->offset;
    out.coupled.push_back(coupled);
  }

  // P, one column per canonical coordinate: the coordinate's own row plus the
  // rows of the mimics that follow it. Built once, so no dynamics call and no
  // symbolic assembly has to walk the mimic list again.
  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    out.drives[index][0] = Drive{out.joints[index].velocity_index, 1.0};
    out.drive_count[index] = 1;
  }
  for (const CoupledJoint& coupled : out.coupled) {
    std::size_t& count = out.drive_count[coupled.source];
    if (count == kMaxDrives) {
      return failure(
        ErrorCode::InvalidRobotDescription,
        "a canonical joint is mimicked by more joints than this model carries rows for");
    }
    out.drives[coupled.source][count] = Drive{coupled.slot.velocity_index, coupled.multiplier};
    ++count;
  }

  // D of robot_model §1. The description is its source
  // (wiki/implementation/parameters.md §1), and §5 says those entries are the
  // identified values on the actuated axes and the hand-tuned per-tool values on
  // the two passive ones -- which is why reading them from the selected
  // description gets the tool dependence right without a table here.
  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    const ::urdf::JointConstSharedPtr joint = tree->getJoint(canonical[index]);
    if (joint && joint->dynamics) {
      out.damping[static_cast<Eigen::Index>(index)] = joint->dynamics->damping;
    }
  }
  // Except where `config/hydraulics.yaml` says otherwise. The override table
  // is data carrying its own reason, so this loop has no idea that the joint on
  // it today is the telescope: it applies whatever the file lists, and a table
  // naming a joint that is not one of the canonical eight is a failure rather
  // than a line that quietly does nothing.
  for (const hydraulics::DampingOverride& record : out.constants.damping_overrides) {
    const auto slot = std::find(canonical.begin(), canonical.end(), record.joint);
    if (slot == canonical.end()) {
      return failure(
        ErrorCode::InvalidArgument,
        "the damping override for " + record.joint +
        " names no canonical coordinate of this description");
    }
    out.damping[std::distance(canonical.begin(), slot)] = record.d;
  }

  // Where a payload attaches (robot_model §5): the joint that carries
  // K8_rotator_lower_part, and the fixed placement of that link within it.
  const char * const rotator_lower_part =
    kFrameLinks[static_cast<std::size_t>(Frame::RotatorLowerPart)].link;
  if (out.model.existFrame(rotator_lower_part)) {
    const pinocchio::Frame& frame = out.model.frames[out.model.getFrameId(rotator_lower_part)];
    out.mount.joint = frame.parentJoint;
    out.mount.placement = frame.placement;
    out.mount.bare_inertia = out.model.inertias[out.mount.joint];
    out.mount.attachable = true;
  }
  return Status{};
}

}  // namespace detail

struct Model::Impl
{
  explicit Impl(pinocchio::Model parsed)
  : model(std::move(parsed)),
    data(model),
    configuration(pinocchio::neutral(model)),
    neutral(configuration),
    joint_jacobian(pinocchio::Data::Matrix6x::Zero(6, model.nv)),
    velocity(Eigen::VectorXd::Zero(model.nv)),
    acceleration(Eigen::VectorXd::Zero(model.nv)),
    gravity_jacobian(Eigen::MatrixXd::Zero(model.nv, model.nv))
  {
  }

  Tool tool{Tool::Pzs100};
  std::array<std::string, 8> names{};
  // `config/hydraulics.yaml` as `parse` read it. Held rather than consulted
  // again, because every transmission call below is a real-time one
  // (contract §10) and reading a file is not.
  hydraulics::Constants constants{};
  std::array<AxisAreas, kActuatedDof> areas{};

  // Built once in create(); every kinematic call below only reads the model and
  // writes into the two workspaces, so contract §10 holds without allocating.
  pinocchio::Model model;
  pinocchio::Data data;
  Eigen::VectorXd configuration;
  Eigen::VectorXd neutral;
  pinocchio::Data::Matrix6x joint_jacobian;
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
  // dg/dq of the parsed model, the workspace the equilibrium solve reads its
  // passive stiffness block out of. Sized once here, like every other buffer.
  Eigen::MatrixXd gravity_jacobian;
  std::array<JointSlot, kGeneralizedDof> joints{};
  std::vector<CoupledJoint> coupled;
  std::array<pinocchio::FrameIndex, kFrameCount> frames{};
  std::array<bool, kFrameCount> frame_present{};

  // P of the parsed model, one column per canonical coordinate: the
  // coordinate's own row plus the rows of the `<mimic>` joints that follow it.
  std::array<std::array<Drive, kMaxDrives>, kGeneralizedDof> drives{};
  std::array<std::size_t, kGeneralizedDof> drive_count{};

  // D of robot_model §1, diagonal and kept out of Pinocchio: neither `rnea` nor
  // `nonLinearEffects` applies `model.damping`, which is what makes the split of
  // §1 -- h from the description, D a fitted parameter -- hold by construction.
  Q damping{Q::Zero()};

  // The range the description gives the two passive joints, which is where the
  // equilibrium solve of §2.3 looks for the well. The description is the only
  // statement in the workspace of where the double hinge is allowed to be, and
  // the hanging equilibrium is the critical point inside it -- the standing ones
  // are half a turn away, outside both ranges. A joint the description leaves
  // unbounded falls back to a full turn, which is the whole torus and still
  // contains exactly one hanging well.
  QU passive_lower{QU::Constant(-M_PI)};
  QU passive_upper{QU::Constant(M_PI)};

  [[nodiscard]] double sample_passive(Eigen::Index row, int index) const
  {
    const double span = passive_upper[row] - passive_lower[row];
    return passive_lower[row] + span * index / (kEquilibriumSamples - 1);
  }

  // The payload body of robot_model §5. Pinocchio has no per-call payload, so it
  // is written into the inertia of the joint that carries K8_rotator_lower_part
  // for the duration of one call and restored afterwards. That is fixed-size
  // spatial arithmetic, not an allocation, and it is the same reason a Model
  // must not be called from two threads at once.
  detail::PayloadMount mount;

  // One fitted primitive, bound to the link frame of the parsed description.
  struct Body
  {
    std::string link;
    pinocchio::FrameIndex frame{0};
    std::shared_ptr<coal::CollisionGeometry> geometry;
    pinocchio::SE3 placement{pinocchio::SE3::Identity()};
    bool tool_side{false};  // distal to the rotator, so it is what holds a payload
  };

  std::vector<Body> bodies;
  std::vector<std::pair<std::size_t, std::size_t>> self_pairs;
  std::vector<std::string> unshaped_links;

  Status require_frame(Frame frame) const
  {
    const std::size_t index = static_cast<std::size_t>(frame);
    if (!frame_present[index]) {
      return failure(
        ErrorCode::FrameUnavailable,
        std::string("the robot description carries no link ") + kFrameLinks[index].link);
    }
    return Status{};
  }

  void write_slot(const JointSlot& slot, double value)
  {
    if (slot.unbounded) {
      configuration[slot.config_index] = std::cos(value);
      configuration[slot.config_index + 1] = std::sin(value);
    } else {
      configuration[slot.config_index] = value;
    }
  }

  // Joints outside the canonical eight keep their neutral value: the four
  // cylinder sub-chains and the 7040's driven inner jaw are loops the
  // description closes only in Gazebo, so their configuration is not a function
  // of q and does not move any frame this API exposes.
  void write_configuration(const Q& q)
  {
    configuration = neutral;
    for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
      write_slot(joints[index], q[static_cast<Eigen::Index>(index)]);
    }
    for (const CoupledJoint& joint : coupled) {
      write_slot(
        joint.slot,
        joint.multiplier * q[static_cast<Eigen::Index>(joint.source)] + joint.offset);
    }
  }

  // dq or ddq of the canonical eight, spread over the rows of the parsed model.
  // Every joint outside the eight and outside the mimics keeps zero, which is
  // consistent with write_configuration leaving it at its neutral value.
  void expand(const DQ& value, Eigen::VectorXd& out) const
  {
    out.setZero();
    for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
      const double coordinate = value[static_cast<Eigen::Index>(index)];
      for (std::size_t entry = 0; entry < drive_count[index]; ++entry) {
        const Drive& drive = drives[index][entry];
        out[drive.velocity_index] = drive.weight * coordinate;
      }
    }
  }

  // P^T applied to a generalized force or a bias vector of the parsed model.
  void project(const Eigen::VectorXd& full, DQ& out) const
  {
    for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
      double sum = 0.0;
      for (std::size_t entry = 0; entry < drive_count[index]; ++entry) {
        const Drive& drive = drives[index][entry];
        sum += drive.weight * full[drive.velocity_index];
      }
      out[static_cast<Eigen::Index>(index)] = sum;
    }
  }

  // `crba` fills the upper triangle of data.M only, so the lower half is read
  // back transposed rather than mirrored into place -- mirroring would be an
  // aliased assignment on a dynamically sized matrix, and this is on the RT path.
  [[nodiscard]] double mass_entry(Eigen::Index row, Eigen::Index column) const
  {
    return row <= column ? data.M(row, column) : data.M(column, row);
  }

  // P^T M_full P, symmetric by construction.
  void project_mass(FullMass& out) const
  {
    for (std::size_t row = 0; row < kGeneralizedDof; ++row) {
      for (std::size_t column = row; column < kGeneralizedDof; ++column) {
        double sum = 0.0;
        for (std::size_t left = 0; left < drive_count[row]; ++left) {
          for (std::size_t right = 0; right < drive_count[column]; ++right) {
            sum += drives[row][left].weight * drives[column][right].weight *
              mass_entry(drives[row][left].velocity_index, drives[column][right].velocity_index);
          }
        }
        out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = sum;
        out(static_cast<Eigen::Index>(column), static_cast<Eigen::Index>(row)) = sum;
      }
    }
  }

  Status attach_payload(const Payload& payload)
  {
    return detail::attach_payload(model, mount, payload);
  }

  void detach_payload()
  {
    if (mount.attachable) {
      model.inertias[mount.joint] = mount.bare_inertia;
    }
  }

  // Puts the joint inertia back however the call leaves, because a model that
  // kept a payload after a failed evaluation would answer the next caller with a
  // load it is not carrying.
  class PayloadGuard
  {
public:
    explicit PayloadGuard(Impl& owner) noexcept
    : owner_(owner)
    {
    }
    ~PayloadGuard() {owner_.detach_payload();}
    PayloadGuard(const PayloadGuard&) = delete;
    PayloadGuard& operator=(const PayloadGuard&) = delete;

private:
    Impl& owner_;
  };

  // M and h + D dq for one (q, dq, payload): the equation of motion of
  // robot_model §1, in the canonical eight coordinates.
  Status evaluate(const Q& q, const DQ& dq, const Payload& payload, FullMass& mass, DQ& bias)
  {
    if (!finite(q)) {
      return failure(ErrorCode::NonFiniteInput, "q is not finite");
    }
    if (!finite(dq)) {
      return failure(ErrorCode::NonFiniteInput, "dq is not finite");
    }
    Status status = attach_payload(payload);
    if (!status.ok()) {
      return status;
    }
    const PayloadGuard guard(*this);

    write_configuration(q);
    expand(dq, velocity);
    pinocchio::crba(model, data, configuration);
    pinocchio::nonLinearEffects(model, data, configuration, velocity);
    project_mass(mass);
    project(data.nle, bias);

    bias += damping.cwiseProduct(dq);
    if (!finite(mass) || !finite(bias)) {
      return failure(
        ErrorCode::SingularConfiguration,
        "the mass matrix or the bias is not finite at this configuration");
    }
    return Status{};
  }

  // robot_model §3.3: tau = M ddq + h + D dq, a single `rnea` and the damping
  // term the description does not carry.
  Status evaluate_torque(
    const Q& q, const DQ& dq, const DQ& ddq, const Payload& payload, DQ& tau)
  {
    if (!finite(q)) {
      return failure(ErrorCode::NonFiniteInput, "q is not finite");
    }
    if (!finite(dq)) {
      return failure(ErrorCode::NonFiniteInput, "dq is not finite");
    }
    if (!finite(ddq)) {
      return failure(ErrorCode::NonFiniteInput, "ddq is not finite");
    }
    Status status = attach_payload(payload);
    if (!status.ok()) {
      return status;
    }
    const PayloadGuard guard(*this);

    write_configuration(q);
    expand(dq, velocity);
    expand(ddq, acceleration);
    pinocchio::rnea(model, data, configuration, velocity, acceleration);
    project(data.tau, tau);

    tau += damping.cwiseProduct(dq);
    if (!finite(tau)) {
      return failure(
        ErrorCode::SingularConfiguration,
        "the inverse-dynamics torque is not finite at this configuration");
    }
    return Status{};
  }

  // The passive rows of the gravity torque at one (q_a, q_u) with dq = 0, and the
  // stiffness dg_u/dq_u that goes with them, from a single pass: Pinocchio's
  // gravity-derivative algorithm reports dg/dq into the workspace and g itself
  // into `data.g`, so a seed search costs one pass per sample and not two. The
  // payload is expected to be attached already -- the whole solve runs under one
  // guard -- and the configuration it writes is left in place for
  // `passive_torque` to reuse.
  Status passive_gravity(Q& q, const QU& passive, QU& gravity, PassiveMass& stiffness)
  {
    for (std::size_t row = 0; row < kPassiveDof; ++row) {
      q[kPassiveRows[row]] = passive[static_cast<Eigen::Index>(row)];
    }
    write_configuration(q);
    // The passive coordinates are neither a `<mimic>` nor mimicked, so neither
    // the two rows nor the two columns below need the projection P: it is the
    // identity on both.
    gravity_jacobian.setZero();
    pinocchio::computeGeneralizedGravityDerivatives(
      model, data, configuration, gravity_jacobian);
    DQ generalized;
    project(data.g, generalized);
    for (std::size_t row = 0; row < kPassiveDof; ++row) {
      const Eigen::Index index = static_cast<Eigen::Index>(row);
      gravity[index] = generalized[kPassiveRows[row]];
      for (std::size_t column = 0; column < kPassiveDof; ++column) {
        stiffness(index, static_cast<Eigen::Index>(column)) = gravity_jacobian(
          joints[static_cast<std::size_t>(kPassiveRows[row])].velocity_index,
          joints[static_cast<std::size_t>(kPassiveRows[column])].velocity_index);
      }
    }
    if (!finite(gravity) || !finite(stiffness)) {
      return failure(
        ErrorCode::SingularConfiguration,
        "the passive gravity torque is not finite at this configuration");
    }
    // The Hessian of a potential is symmetric; the two off-diagonal entries come
    // out of different sweeps of the same algorithm, so they are averaged rather
    // than one of them being picked.
    const double coupling = 0.5 * (stiffness(0, 1) + stiffness(1, 0));
    stiffness(0, 1) = coupling;
    stiffness(1, 0) = coupling;
    return Status{};
  }

  // The same two rows taken from `rnea` at rest instead, on the configuration
  // `passive_gravity` has just written. This is the exact quantity
  // `inverse_dynamics(q, 0, 0, payload)` returns -- one algorithm, one call, the
  // damping term vanishing with dq -- so the residual the Newton iteration drives
  // to zero is the one contract §7 states the invariant on, and not a second
  // computation of it that could agree to fewer digits.
  Status passive_torque(QU& residual)
  {
    velocity.setZero();
    acceleration.setZero();
    pinocchio::rnea(model, data, configuration, velocity, acceleration);
    DQ tau;
    project(data.tau, tau);
    for (std::size_t row = 0; row < kPassiveDof; ++row) {
      residual[static_cast<Eigen::Index>(row)] = tau[kPassiveRows[row]];
    }
    if (!finite(residual)) {
      return failure(
        ErrorCode::SingularConfiguration,
        "the passive rows of the inverse dynamics are not finite at this configuration");
    }
    return Status{};
  }

  // What the solve says when it started in a hanging well inside the passive
  // range and the iteration did not settle in one. Every observed instance is the
  // same physical case: the crane folded far enough that the free hanging pose
  // needs more tip travel than the description gives that joint, so the tool
  // would come to rest against a stop instead of hanging. That is not a pose
  // robot_model §2.3 defines, and a plausible-looking q_u would be worse than
  // none -- mpc §2 would damp the sway toward it and trajectory_planning §6 would
  // make it an endpoint.
  static Status unreachable_equilibrium()
  {
    return failure(
      ErrorCode::SingularConfiguration,
      "the passive joints reach no hanging pose from inside the range the description gives "
      "them at this q_a");
  }

  // wiki/robot_model.md §2.3. The payload is attached once for the whole solve
  // rather than once per iterate, so the iteration is the one equation the whole
  // time.
  //
  // Two stages, and the first is the one that picks the branch. Newton alone is
  // local and h_u is flat a quarter turn from the well, so a fixed start inside
  // the passive range is not enough -- at the crane's own zero the tool hangs at
  // the far end of the tip range and a start in the middle of it sits on the
  // saddle between the two wells. So the range the description gives the two
  // passive joints is sampled first, and the sample the Newton starts from is the
  // one with the smallest residual *among those with a positive definite
  // stiffness*: that condition is what separates hanging from standing, and it is
  // applied before the iteration rather than after it.
  Status solve_equilibrium(const QA& q_a, const Payload& payload, QU& q_equilibrium)
  {
    if (!finite(q_a)) {
      return failure(ErrorCode::NonFiniteInput, "q_a is not finite");
    }
    Status status = attach_payload(payload);
    if (!status.ok()) {
      return status;
    }
    const PayloadGuard guard(*this);

    Q q = Q::Zero();
    for (std::size_t row = 0; row < kActuatedDof; ++row) {
      q[kActuatedRows[row]] = q_a[static_cast<Eigen::Index>(row)];
    }
    QU gravity;
    QU residual;
    PassiveMass stiffness;

    QU passive = QU::Zero();
    double best = std::numeric_limits<double>::infinity();
    for (int tip = 0; tip < kEquilibriumSamples; ++tip) {
      for (int tilt = 0; tilt < kEquilibriumSamples; ++tilt) {
        QU sample;
        sample[0] = sample_passive(0, tip);
        sample[1] = sample_passive(1, tilt);
        if (!passive_gravity(q, sample, gravity, stiffness).ok()) {
          continue;
        }
        if (Eigen::LLT<PassiveMass>(stiffness).info() != Eigen::Success) {
          continue;
        }
        if (gravity.norm() < best) {
          best = gravity.norm();
          passive = sample;
        }
      }
    }
    if (!(best < std::numeric_limits<double>::infinity())) {
      return failure(
        ErrorCode::SingularConfiguration,
        "no pose in the passive joint range has a restoring stiffness at this q_a, so there is "
        "nowhere for the tool to hang");
    }

    for (int iteration = 0; iteration < kEquilibriumIterations; ++iteration) {
      status = passive_gravity(q, passive, gravity, stiffness);
      if (status.ok()) {
        status = passive_torque(residual);
      }
      if (!status.ok()) {
        return status;
      }
      Eigen::LLT<PassiveMass> factorisation(stiffness);
      if (factorisation.info() != Eigen::Success) {
        return unreachable_equilibrium();
      }
      if (residual.norm() <= kEquilibriumResidual) {
        q_equilibrium = passive;
        return Status{};
      }
      passive -= factorisation.solve(residual);
      if (!finite(passive)) {
        return unreachable_equilibrium();
      }
    }
    return unreachable_equilibrium();
  }

  // The fit is for this machine. A description that does not carry every link it
  // was fitted to would answer with a crane that is missing parts, which is
  // worse than not answering; contract §5 says so rather than substituting.
  Status require_collision_model() const
  {
    if (!unshaped_links.empty()) {
      return failure(
        ErrorCode::CollisionBackendFailure,
        "the robot description carries no link " + unshaped_links.front() +
        ", which the collision model is fitted to");
    }
    if (bodies.empty() || self_pairs.empty()) {
      return failure(
        ErrorCode::CollisionBackendFailure, "the collision model carries nothing to check");
    }
    return Status{};
  }

  // Every body's primitive, placed in K0_mounting_base. The scene is already in
  // that frame (contract §8), so this is the only transform the query needs.
  Status place_bodies(const Q& q, std::vector<coal::Transform3s>& poses)
  {
    Status status = require_frame(Frame::MountingBase);
    if (!status.ok()) {
      return status;
    }
    const pinocchio::FrameIndex base = frames[static_cast<std::size_t>(Frame::MountingBase)];
    write_configuration(q);
    pinocchio::forwardKinematics(model, data, configuration);
    pinocchio::updateFramePlacement(model, data, base);
    poses.clear();
    poses.reserve(bodies.size());
    for (const Body& body : bodies) {
      pinocchio::updateFramePlacement(model, data, body.frame);
      poses.push_back(
        coal_transform(data.oMf[base].actInv(data.oMf[body.frame]) * body.placement));
    }
    return Status{};
  }

  // One result per scene primitive, in scene order, and then one for the crane
  // against itself. Dynamically sized, so contract §10 keeps it off the RT path.
  Status collide(const Q& q, const CollisionScene& scene, std::vector<CollisionResult>& results)
  {
    if (!finite(q)) {
      return failure(ErrorCode::NonFiniteInput, "q is not finite");
    }
    Status status = require_collision_model();
    if (!status.ok()) {
      return status;
    }
    std::vector<SceneBody> scene_bodies;
    status = build_scene(scene, scene_bodies);
    if (!status.ok()) {
      return status;
    }
    std::vector<coal::Transform3s> poses;
    status = place_bodies(q, poses);
    if (!status.ok()) {
      return status;
    }

    results.clear();
    results.reserve(scene_bodies.size() + 1);
    CollisionResult candidate;
    for (const SceneBody& other : scene_bodies) {
      CollisionResult closest;
      bool found = false;
      for (std::size_t index = 0; index < bodies.size(); ++index) {
        // The payload is carried by the tool, so the links holding it are not
        // an obstacle to it. Everything else on the crane still is.
        if (other.payload && bodies[index].tool_side) {
          continue;
        }
        status = pair_distance(
          bodies[index].geometry.get(), poses[index], other.geometry.get(), other.pose,
          other.primitive->id, candidate);
        if (!status.ok()) {
          return status;
        }
        if (!found || candidate.minimum_distance_m < closest.minimum_distance_m) {
          closest = candidate;
          found = true;
        }
      }
      if (!found) {
        return failure(
          ErrorCode::InvalidScene,
          "primitive " + other.primitive->id + " is left with nothing to be checked against");
      }
      results.push_back(std::move(closest));
    }

    CollisionResult self;
    bool found = false;
    for (const std::pair<std::size_t, std::size_t>& pair : self_pairs) {
      status = pair_distance(
        bodies[pair.first].geometry.get(), poses[pair.first],
        bodies[pair.second].geometry.get(), poses[pair.second],
        bodies[pair.first].link + kSelfPairSeparator + bodies[pair.second].link, candidate);
      if (!status.ok()) {
        return status;
      }
      if (!found || candidate.minimum_distance_m < self.minimum_distance_m) {
        self = candidate;
        found = true;
      }
    }
    results.push_back(std::move(self));
    return Status{};
  }
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
  detail::ParsedModel source;
  Status status = detail::parse(config, source);
  if (!status.ok()) {
    return Result<Model>::failure(std::move(status));
  }

  // What `parse` produced, carried into the implementation state. Everything
  // below this point is what only a `Model` needs -- the passive range, the
  // frame table and the collision geometry -- and the symbolic graph does not.
  auto impl = std::make_unique<Impl>(std::move(source.model));
  impl->tool = source.tool;
  impl->names = source.constants.joints(source.tool);
  impl->constants = source.constants;
  impl->areas = cylinder::axis_areas(source.constants);
  impl->joints = source.joints;
  impl->coupled = std::move(source.coupled);
  impl->drives = source.drives;
  impl->drive_count = source.drive_count;
  impl->damping = source.damping;
  impl->mount = source.mount;

  // The passive range of robot_model §2.3, read out of the description rather
  // than written down: both machine descriptions bound the tip at +-pi/2 and the
  // tilt between pi/4 and 3pi/4. That range is where the equilibrium solve looks
  // for the hanging well, so which of the pendulum's critical points this library
  // returns is the description's statement and not one made here.
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    const JointSlot& slot = impl->joints[static_cast<std::size_t>(kPassiveRows[row])];
    if (slot.unbounded) {
      continue;
    }
    const double lower = impl->model.lowerPositionLimit[slot.config_index];
    const double upper = impl->model.upperPositionLimit[slot.config_index];
    if (std::isfinite(lower) && std::isfinite(upper) && lower < upper) {
      impl->passive_lower[static_cast<Eigen::Index>(row)] = lower;
      impl->passive_upper[static_cast<Eigen::Index>(row)] = upper;
    }
  }

  for (std::size_t index = 0; index < kFrameCount; ++index) {
    const bool present = impl->model.existFrame(kFrameLinks[index].link);
    impl->frame_present[index] = present;
    impl->frames[index] = present ? impl->model.getFrameId(kFrameLinks[index].link) : 0U;
  }

  // The collision model. Building it is construction work, not query work
  // (contract §10), and a description that carries none of these links is not
  // refused here -- only a collision call needs them, and a caller that never
  // makes one is entitled to a model.
  for (const LinkPrimitive& entry : kLinkPrimitives) {
    if (entry.tool != config.tool) {
      continue;
    }
    if (!impl->model.existFrame(entry.link)) {
      impl->unshaped_links.emplace_back(entry.link);
      continue;
    }
    Model::Impl::Body body;
    body.link = entry.link;
    body.frame = impl->model.getFrameId(entry.link);
    body.geometry = link_geometry(entry);
    body.placement = primitive_placement(entry);
    impl->bodies.push_back(std::move(body));
  }

  // Which bodies the tool carries a payload with, read out of the description
  // rather than listed: everything the rotator joint moves is on the tool side.
  if (impl->model.existJointName(impl->names[6])) {
    const pinocchio::JointIndex rotator = impl->model.getJointId(impl->names[6]);
    for (Model::Impl::Body& body : impl->bodies) {
      pinocchio::JointIndex joint = impl->model.frames[body.frame].parentJoint;
      while (joint != 0) {
        if (joint == rotator) {
          body.tool_side = true;
          break;
        }
        joint = impl->model.parents[joint];
      }
    }
  }

  for (std::size_t first = 0; first < impl->bodies.size(); ++first) {
    for (std::size_t second = first + 1; second < impl->bodies.size(); ++second) {
      if (!self_pair_is_allowed(impl->bodies[first].link, impl->bodies[second].link)) {
        impl->self_pairs.emplace_back(first, second);
      }
    }
  }

  return Result<Model>::success(Model(std::move(impl)));
}

Tool Model::tool() const noexcept
{
  return impl_ ? impl_->tool : Tool::Pzs100;
}

const std::array<std::string, 8>& Model::urdf_joint_names() const noexcept
{
  // A model with no implementation state never read the file, so it has no
  // joint map to report. Eight empty strings, rather than one tool's map
  // written down a second time here for a `Model` that cannot be used anyway.
  static const std::array<std::string, 8> unread{};
  return impl_ ? impl_->names : unread;
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
  Status status = fill_cylinder_jacobian(impl_->constants, impl_->tool, q, jacobian);
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
  status = fill_cylinder_jacobian(
    impl_->constants, impl_->tool, q, result.joint_to_cylinder);
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

Result<std::array<CylinderForceLimit, kActuatedDof>>
derive_cylinder_force_limits(const Model& model, double system_pressure_pa)
{
  using Limits = std::array<CylinderForceLimit, kActuatedDof>;
  if (!std::isfinite(system_pressure_pa) || !(system_pressure_pa > 0.0)) {
    return Result<Limits>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the hydraulic relief pressure must be finite and positive; it is the one number the "
        "cylinder force limit of robot_model 4.6 cannot be derived from the description and it is "
        "not measured on this machine (parameters.md 7)"));
  }

  ChamberPressure pushing;
  pushing.p_a_pa.setConstant(system_pressure_pa);
  pushing.p_b_pa.setZero();
  ChamberPressure pulling;
  pulling.p_a_pa.setZero();
  pulling.p_b_pa.setConstant(system_pressure_pa);

  auto push = model.cylinder_force(pushing);
  if (!push.ok()) {
    return Result<Limits>::failure(push.status());
  }
  auto pull = model.cylinder_force(pulling);
  if (!pull.ok()) {
    return Result<Limits>::failure(pull.status());
  }

  Limits limits{};
  for (std::size_t row = 0; row < kActuatedDof; ++row) {
    const double extend = std::abs(push.value()[static_cast<Eigen::Index>(row)]);
    const double retract = std::abs(pull.value()[static_cast<Eigen::Index>(row)]);
    if (!std::isfinite(extend) || !(extend > 0.0) ||
      !std::isfinite(retract) || !(retract > 0.0))
    {
      return Result<Limits>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "actuated row " + std::to_string(row) +
          " has no usable chamber area, so no cylinder force limit follows from the relief "
          "pressure and constraint 6 of mpc 3 would have no right-hand side there"));
    }
    limits[row] = CylinderForceLimit{extend, retract};
  }
  return Result<Limits>::success(limits);
}

// robot_model §2.1: a direct Pinocchio evaluation. `from` is the frame the
// answer is expressed in and `to` is the frame whose pose is asked for, so
// `forward_kinematics(q, MountingBase, Tcp)` is the tool pose in K0.
Result<Pose> Model::forward_kinematics(const Q& q, Frame from, Frame to) const
{
  if (!impl_) {
    return Result<Pose>::failure(not_ready());
  }
  if (!valid_frame(from) || !valid_frame(to)) {
    return Result<Pose>::failure(failure(ErrorCode::InvalidArgument, "frame is not a Frame value"));
  }
  if (!finite(q)) {
    return Result<Pose>::failure(failure(ErrorCode::NonFiniteInput, "q is not finite"));
  }
  Status status = impl_->require_frame(from);
  if (status.ok()) {
    status = impl_->require_frame(to);
  }
  if (!status.ok()) {
    return Result<Pose>::failure(std::move(status));
  }

  const pinocchio::FrameIndex source = impl_->frames[static_cast<std::size_t>(from)];
  const pinocchio::FrameIndex target = impl_->frames[static_cast<std::size_t>(to)];
  impl_->write_configuration(q);
  pinocchio::forwardKinematics(impl_->model, impl_->data, impl_->configuration);
  pinocchio::updateFramePlacement(impl_->model, impl_->data, source);
  pinocchio::updateFramePlacement(impl_->model, impl_->data, target);
  const pinocchio::SE3 relative = impl_->data.oMf[source].actInv(impl_->data.oMf[target]);

  Pose pose;
  pose.expressed_in = from;
  pose.position_m = relative.translation();
  pose.orientation = Eigen::Quaterniond(relative.rotation());
  return Result<Pose>::success(std::move(pose));
}

// The 6x8 Jacobian of `frame`, expressed in `frame` itself, which is what the
// result declares. All eight columns are real: the passive tip and tilt
// coordinates move the tool exactly as the actuated ones do (robot_model §2.1),
// and the telescope column carries both stages, because the description mimics
// the small telescope onto q4.
Result<Jacobian6x8> Model::jacobian(const Q& q, Frame frame) const
{
  if (!impl_) {
    return Result<Jacobian6x8>::failure(not_ready());
  }
  if (!valid_frame(frame)) {
    return Result<Jacobian6x8>::failure(
      failure(ErrorCode::InvalidArgument, "frame is not a Frame value"));
  }
  if (!finite(q)) {
    return Result<Jacobian6x8>::failure(failure(ErrorCode::NonFiniteInput, "q is not finite"));
  }
  Status status = impl_->require_frame(frame);
  if (!status.ok()) {
    return Result<Jacobian6x8>::failure(std::move(status));
  }

  const pinocchio::FrameIndex target = impl_->frames[static_cast<std::size_t>(frame)];
  impl_->write_configuration(q);
  pinocchio::computeJointJacobians(impl_->model, impl_->data, impl_->configuration);
  pinocchio::updateFramePlacement(impl_->model, impl_->data, target);
  impl_->joint_jacobian.setZero();
  pinocchio::getFrameJacobian(
    impl_->model, impl_->data, target, pinocchio::LOCAL, impl_->joint_jacobian);

  Jacobian6x8 result;
  result.expressed_in = frame;
  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    result.value.col(static_cast<Eigen::Index>(index)) =
      impl_->joint_jacobian.col(impl_->joints[index].velocity_index);
  }
  for (const CoupledJoint& coupled : impl_->coupled) {
    result.value.col(static_cast<Eigen::Index>(coupled.source)) +=
      coupled.multiplier * impl_->joint_jacobian.col(coupled.slot.velocity_index);
  }
  return Result<Jacobian6x8>::success(std::move(result));
}

// The whole scene and the crane itself, reduced to the one pair that matters:
// the smallest distance found anywhere, and whatever it was against.
Result<CollisionResult> Model::collision_query(const Q& q, const CollisionScene& scene) const
{
  if (!impl_) {
    return Result<CollisionResult>::failure(not_ready());
  }
  std::vector<CollisionResult> results;
  Status status = impl_->collide(q, scene, results);
  if (!status.ok()) {
    return Result<CollisionResult>::failure(std::move(status));
  }
  const auto worst = std::min_element(
    results.begin(), results.end(),
    [](const CollisionResult& left, const CollisionResult& right) {
      return left.minimum_distance_m < right.minimum_distance_m;
    });
  return Result<CollisionResult>::success(*worst);
}

// One result per scene primitive, in scene order, then one for the crane against
// itself. The container is dynamically sized, so this is not an RT API call
// (contract §10) and the allocation guard in the contract test does not cover it.
Result<std::vector<CollisionResult>> Model::collision_queries(
  const Q& q, const CollisionScene& scene) const
{
  if (!impl_) {
    return Result<std::vector<CollisionResult>>::failure(not_ready());
  }
  std::vector<CollisionResult> results;
  Status status = impl_->collide(q, scene, results);
  if (!status.ok()) {
    return Result<std::vector<CollisionResult>>::failure(std::move(status));
  }
  return Result<std::vector<CollisionResult>>::success(std::move(results));
}

// robot_model §1 evaluated whole: the 8x8 mass matrix, the bias
// h + D dq of contract §7, and the inverse-dynamics torque of §3.3.
//
// `inverse_dynamics_tau` needs an acceleration the signature does not carry, and
// there is exactly one the model can supply without inventing an input: the
// *consistent* one, ddq_a = 0 with ddq_u from §3.1. That is the torque holding
// the actuated axes still while the pendulum swings freely, its passive rows
// vanish by construction (§3.3), and its actuated rows are bias_eff of §3.4 at
// zero actuated acceleration. Reporting tau at ddq = 0 instead would make the
// field a copy of `bias`. README states this reading.
Result<FullDynamics> Model::full_dynamics(
  const Q& q, const DQ& dq, const Payload& payload) const
{
  if (!impl_) {
    return Result<FullDynamics>::failure(not_ready());
  }
  FullDynamics result;
  Status status = impl_->evaluate(q, dq, payload, result.mass, result.bias);
  if (!status.ok()) {
    return Result<FullDynamics>::failure(std::move(status));
  }

  const Partition split = partition(result.mass, result.bias);
  Eigen::LLT<PassiveMass> factorisation;
  status = factorise_passive_mass(split.M_uu, factorisation);
  if (!status.ok()) {
    return Result<FullDynamics>::failure(std::move(status));
  }

  DQ ddq = DQ::Zero();
  const QU ddq_u = -factorisation.solve(split.h_u);
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    ddq[kPassiveRows[row]] = ddq_u[static_cast<Eigen::Index>(row)];
  }
  result.inverse_dynamics_tau.noalias() = result.mass * ddq;
  result.inverse_dynamics_tau += result.bias;
  if (!finite(result.inverse_dynamics_tau)) {
    return Result<FullDynamics>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the consistent passive acceleration is not finite at this configuration"));
  }
  return Result<FullDynamics>::success(std::move(result));
}

// robot_model §3.3, all eight rows. The passive ones are only zero when ddq_u is
// the one §3.1 gives for this ddq_a; for any other acceleration they are the
// generalized force the passive joints would need and do not have, which is what
// makes the invariant of contract §7 a test and not a tautology.
Result<DQ> Model::inverse_dynamics(
  const Q& q, const DQ& dq, const DQ& ddq, const Payload& payload) const
{
  if (!impl_) {
    return Result<DQ>::failure(not_ready());
  }
  DQ tau;
  Status status = impl_->evaluate_torque(q, dq, ddq, payload, tau);
  if (!status.ok()) {
    return Result<DQ>::failure(std::move(status));
  }
  return Result<DQ>::success(std::move(tau));
}

// robot_model §3.4: the Schur complement that eliminates ddq_u, and the residual
// that comes with it. `ddq_a` is validated and does not enter either -- the
// reduction is a property of (q, dq, payload) alone, and tau_a = M_eff ddq_a +
// h_eff is the caller's product to form. It stays in the signature because the
// contract froze it there.
Result<ReducedDynamics> Model::reduced_actuated_dynamics(
  const Q& q, const DQ& dq, const Input& ddq_a, const Payload& payload) const
{
  if (!impl_) {
    return Result<ReducedDynamics>::failure(not_ready());
  }
  if (!finite(ddq_a)) {
    return Result<ReducedDynamics>::failure(
      failure(ErrorCode::NonFiniteInput, "ddq_a is not finite"));
  }
  FullMass mass;
  DQ bias;
  Status status = impl_->evaluate(q, dq, payload, mass, bias);
  if (!status.ok()) {
    return Result<ReducedDynamics>::failure(std::move(status));
  }

  const Partition split = partition(mass, bias);
  Eigen::LLT<PassiveMass> factorisation;
  status = factorise_passive_mass(split.M_uu, factorisation);
  if (!status.ok()) {
    return Result<ReducedDynamics>::failure(std::move(status));
  }

  // M_uu^{-1} M_ua and M_uu^{-1} h_u, the two solves both terms of §3.4 share.
  const Eigen::Matrix<double, 2, 6> solved_mass = factorisation.solve(split.M_ua);
  const QU solved_bias = factorisation.solve(split.h_u);

  ReducedDynamics result;
  result.mass_eff.noalias() = split.M_aa - split.M_au * solved_mass;
  result.bias_eff.noalias() = split.h_a - split.M_au * solved_bias;
  if (!finite(result.mass_eff) || !finite(result.bias_eff)) {
    return Result<ReducedDynamics>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the effective inertia is not finite at this configuration"));
  }
  return Result<ReducedDynamics>::success(std::move(result));
}

// wiki/robot_model.md §2.3: where the tool hangs once the actuated joints are
// held and it has stopped swinging. The signature carries no q_u and no initial
// guess, so the branch is chosen inside -- see `passive_seed` and the stiffness
// test in `solve_equilibrium`.
Result<QU> Model::passive_equilibrium(const QA& q_a, const Payload& payload) const
{
  if (!impl_) {
    return Result<QU>::failure(not_ready());
  }
  QU q_equilibrium;
  Status status = impl_->solve_equilibrium(q_a, payload, q_equilibrium);
  if (!status.ok()) {
    return Result<QU>::failure(std::move(status));
  }
  return Result<QU>::success(std::move(q_equilibrium));
}

}  // namespace crane_model
