#include <gtest/gtest.h>

#include <urdf_parser/urdf_parser.h>

#include <Eigen/Eigenvalues>

#include <limits>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "crane_model/model.hpp"
#include "crane_model/testing/mock_model.hpp"

// Private to the implementation; on this test's include path only, so the
// generated collision geometry and the allowed-collision list can be checked
// against the description and against config/allowed_collisions.srdf.
#include "collision_model.hpp"

static_assert(crane_model::Q::RowsAtCompileTime == 8);
static_assert(crane_model::DQ::RowsAtCompileTime == 8);
static_assert(crane_model::QA::RowsAtCompileTime == 6);
static_assert(crane_model::QU::RowsAtCompileTime == 2);
static_assert(crane_model::State::RowsAtCompileTime == 16);
static_assert(crane_model::ActuatedJacobian::RowsAtCompileTime == 6);
static_assert(crane_model::ActuatedJacobian::ColsAtCompileTime == 6);

namespace
{
std::atomic<bool> g_allocation_guard{false};
std::atomic<std::size_t> g_allocation_count{0};

void * guarded_allocate(std::size_t size, std::size_t alignment)
{
  if (g_allocation_guard.load(std::memory_order_relaxed)) {
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (size == 0) {
    size = 1;
  }
  void * result = nullptr;
  if (posix_memalign(&result, alignment, size) != 0) {
    throw std::bad_alloc();
  }
  return result;
}
}  // namespace

void * operator new(std::size_t size) { return guarded_allocate(size, alignof(std::max_align_t)); }
void * operator new[](std::size_t size)
{
  return guarded_allocate(size, alignof(std::max_align_t));
}
void * operator new(std::size_t size, std::align_val_t alignment)
{
  return guarded_allocate(size, static_cast<std::size_t>(alignment));
}
void * operator new[](std::size_t size, std::align_val_t alignment)
{
  return guarded_allocate(size, static_cast<std::size_t>(alignment));
}
void operator delete(void * pointer) noexcept { std::free(pointer); }
void operator delete[](void * pointer) noexcept { std::free(pointer); }
void operator delete(void * pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void * pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void * pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void * pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void * pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void * pointer, std::size_t, std::align_val_t) noexcept
{
  std::free(pointer);
}

namespace
{

crane_model::Payload valid_payload()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 42.0;
  payload.center_of_mass_k8_m = Eigen::Vector3d(0.1, -0.2, -0.3);
  payload.inertia_k8_kg_m2 = Eigen::Matrix3d::Identity() * 0.8;
  return payload;
}

crane_model::Q valid_q()
{
  crane_model::Q q;
  q << 0.1, 0.2, -0.3, 0.4, 0.05, -0.06, 0.7, 0.08;
  return q;
}

// The real machine description, expanded from the xacro of
// src/epsilon_crane_description and checked in beside this file. It is the
// whole point of the slice-4 tracer: the model is built from this, not from a
// string that merely mentions the joint names.
std::string read_description(const char * file)
{
  const std::string path = std::string(CRANE_MODEL_TEST_DESCRIPTION_DIR) + "/" + file;
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "cannot read " << path;
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

const std::string& description_for(crane_model::Tool tool)
{
  static const std::string rail = read_description("pzs100.urdf");
  static const std::string jaws = read_description("epsilon_7040.urdf");
  return tool == crane_model::Tool::Pzs100 ? rail : jaws;
}

// A description that parses but carries none of the canonical joints, so the
// joint map fails on its own terms rather than on the parser's.
const char * const kJointlessDescription =
  "<robot name=\"fixture\"><link name=\"K0_mounting_base\"/></robot>";

// Every Frame the contract defines, in enum order.
const std::array<crane_model::Frame, 12> kAllFrames{{
  crane_model::Frame::World,
  crane_model::Frame::MountingBase,
  crane_model::Frame::SlewingColumn,
  crane_model::Frame::Boom,
  crane_model::Frame::Arm,
  crane_model::Frame::BigTelescope,
  crane_model::Frame::Tip,
  crane_model::Frame::Tilt,
  crane_model::Frame::Rotator,
  crane_model::Frame::RotatorLowerPart,
  crane_model::Frame::Tcp,
  crane_model::Frame::ToolContact,
}};

Eigen::Isometry3d isometry(const crane_model::Pose& pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = pose.orientation.toRotationMatrix();
  transform.translation() = pose.position_m;
  return transform;
}

// The rotation vector of a rotation matrix, i.e. log(R) read as axis * angle.
Eigen::Vector3d rotation_vector(const Eigen::Matrix3d& rotation)
{
  const Eigen::AngleAxisd angle_axis(rotation);
  return angle_axis.angle() * angle_axis.axis();
}

crane_model::Result<crane_model::Model> production_model(crane_model::Tool tool)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description_for(tool);
  config.tool = tool;
  return crane_model::Model::create(config);
}

// Only q2 and q3 have a stroke-dependent transmission, so the hydraulic tests
// vary those two and leave the linear axes at zero.
crane_model::Q linkage_configuration(double q2, double q3)
{
  crane_model::Q q = crane_model::Q::Zero();
  q[1] = q2;
  q[2] = q3;
  return q;
}

// The 7040's tool axis is the only one whose ratio depends on q8, so the jaw
// tests vary that and leave every other axis at zero.
crane_model::Q jaw_configuration(double q8)
{
  crane_model::Q q = crane_model::Q::Zero();
  q[7] = q8;
  return q;
}

crane_model::ChamberPressure zero_pressure()
{
  crane_model::ChamberPressure pressure;
  pressure.p_a_pa.setZero();
  pressure.p_b_pa.setZero();
  return pressure;
}

// wiki/hydraulics.md §2.2, §2.3 and §6.2, restated here so the assertions
// check the model against the note rather than against itself. Only the
// piston displacements are restated; every transmission ratio below is a
// difference quotient of these, never a copy of the analytic derivative the
// implementation uses.
namespace reference
{

constexpr double kDrawbarLength = 0.57;   // r_13
constexpr double kPushbarLength = 0.124;  // r_23
constexpr double kStep = 1.0e-6;

Eigen::Vector2d rotate(double angle, double x, double y)
{
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return Eigen::Vector2d(cosine * x - sine * y, sine * x + cosine * y);
}

// The three placements of the boom four-bar. Their defaults are §6.2's numbers;
// the linkage cross-check below builds the same struct out of the placements
// Pinocchio reads from the real description instead, which is how a description
// that disagrees with the compiled-in geometry is caught.
struct BoomGeometry
{
  Eigen::Vector2d foot{0.433, -1.7682};              // p_S0
  Eigen::Vector2d pivot{-0.12, -0.07};               // p_S1
  Eigen::Vector2d link{3.49288 - 3.039, -0.036034};  // a_2 + p_S2x, p_S2y
};

Eigen::Vector2d boom_attachment(const BoomGeometry& geometry, double q2)
{
  return rotate(q2, geometry.link.x(), geometry.link.y());  // R(q2) (a_2 + p_S2x, p_S2y)
}

double boom_stroke_from(const BoomGeometry& geometry, double q2)
{
  const Eigen::Vector2d p_s0 = geometry.foot;
  const Eigen::Vector2d p_s1 = geometry.pivot;
  const Eigen::Vector2d d_pivot = boom_attachment(geometry, q2) - p_s1;
  const double d_squared = d_pivot.squaredNorm();
  const double sum = kDrawbarLength + kPushbarLength;
  const double difference = kDrawbarLength - kPushbarLength;
  const double delta = (sum * sum - d_squared) * (d_squared - difference * difference);
  if (delta <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const Eigen::Vector2d turned(-d_pivot.y(), d_pivot.x());  // S_perp d
  const double link = kDrawbarLength * kDrawbarLength - kPushbarLength * kPushbarLength;
  const Eigen::Vector2d p_j = p_s1 + (link + d_squared) / (2.0 * d_squared) * d_pivot -
    std::sqrt(delta) / (2.0 * d_squared) * turned;
  return (p_j - p_s0).norm();
}

double boom_stroke(double q2)
{
  return boom_stroke_from(BoomGeometry{}, q2);
}

// The same cylinder foot and the same boom-side attachment, but with the
// drawbar and pushbar deleted: what the stroke would be if the boom cylinder
// acted directly on the boom instead of through the four-bar coupler point.
double boom_direct_stroke(double q2)
{
  const BoomGeometry geometry;
  return (boom_attachment(geometry, q2) - geometry.foot).norm();
}

struct ArmGeometry
{
  Eigen::Vector2d foot{-1.6802, -0.0485};             // p_S3x, p_S3y
  Eigen::Vector2d link{-0.3925 + 0.274489, -0.468};   // a_3 + p_S4x, -p_S4z
  double lateral{0.224 - 0.224};                      // p_S4y - p_S3z
};

double arm_stroke_from(const ArmGeometry& geometry, double q3)
{
  const Eigen::Vector2d moving = rotate(q3, geometry.link.x(), geometry.link.y());
  const Eigen::Vector2d in_plane = moving - geometry.foot;
  return std::sqrt(in_plane.squaredNorm() + geometry.lateral * geometry.lateral);
}

double arm_stroke(double q3)
{
  return arm_stroke_from(ArmGeometry{}, q3);
}

// wiki/hydraulics.md §2.6 for the 7040, with the numbers ported from the
// deployed model. Unlike the two above, this is a port-fidelity restatement and
// not an independent one: it repeats the deployed fit's own coefficients,
// because the Freudenstein constants a, b, c and d that produced them are in no
// file of this workspace, so there is nothing here to derive phi_sim,9 from.
// What it does check is the composition — that the model drives the inner jaw
// through the fit and takes the cylinder length between the two attachment
// points, rather than treating the jaw as a one-to-one axis like the rail.
double jaw_mirror_angle(double q8)
{
  return (-0.121220 * q8 + 1.400122) * q8 - 0.227867;  // Horner, jaw_linkage_p
}

struct JawGeometry
{
  double outer_pivot{0.328};                       // a_9
  double outer_arm{0.8126};                        // a_10
  double inner_pivot{0.336};                       // a_11
  double inner_arm{0.8172};                        // a_12
  Eigen::Vector2d outer_pin{-0.8971, 0.01754};     // p_S7
  Eigen::Vector2d inner_pin{-0.908397, 0.015289};  // p_S8
};

double jaw_stroke_from(const JawGeometry& geometry, double q8)
{
  const Eigen::Vector2d outer = rotate(
    q8, geometry.outer_arm + geometry.outer_pin.x(), geometry.outer_pin.y());
  const Eigen::Vector2d inner = rotate(
    jaw_mirror_angle(q8), geometry.inner_arm + geometry.inner_pin.x(), geometry.inner_pin.y());
  const double ground = geometry.outer_pivot + geometry.inner_pivot;  // a_9 + a_11
  // The outer jaw's arm is mirrored against the inner jaw's, so the two arm
  // x components add rather than subtract.
  return std::hypot(inner.x() + outer.x() + ground, inner.y() - outer.y());
}

double jaw_stroke(double q8)
{
  return jaw_stroke_from(JawGeometry{}, q8);
}

// The same two attachment points, but with the inner jaw frozen at the mirror
// angle of q8 = 0: what the stroke would be if only the commanded jaw moved.
double jaw_stroke_with_frozen_inner_jaw(double q8)
{
  const Eigen::Vector2d outer = rotate(q8, 0.8126 - 0.8971, 0.01754);
  const Eigen::Vector2d inner = rotate(jaw_mirror_angle(0.0), 0.8172 - 0.908397, 0.015289);
  return std::hypot(inner.x() + outer.x() + 0.328 + 0.336, inner.y() - outer.y());
}

template<typename Stroke>
double transmission_ratio(Stroke stroke, double q)
{
  return (stroke(q + kStep) - stroke(q - kStep)) / (2.0 * kStep);
}

}  // namespace reference

// The same linkage, but with every length that the description actually carries
// read back out of it with Pinocchio instead of written down. What is left over
// is exactly the geometry the description does not carry -- the two boom bar
// lengths, the arm cylinder's rod-end attachment, the 7040's inner-jaw cylinder
// pin and its fitted mirror law -- and each of those is named where it is used.
namespace parsed
{

class Description
{
public:
  explicit Description(crane_model::Tool tool)
  : model_(build(tool)), data_(model_)
  {
    pinocchio::forwardKinematics(model_, data_, pinocchio::neutral(model_));
    pinocchio::updateFramePlacements(model_, data_);
  }

  // Translation of `child` relative to `parent` at the zero configuration.
  // Every pair below shares a parent joint, so the value is the fixed placement
  // the description declares and not a function of the configuration.
  Eigen::Vector3d offset(const std::string& parent, const std::string& child) const
  {
    EXPECT_TRUE(model_.existFrame(parent)) << "no frame " << parent;
    EXPECT_TRUE(model_.existFrame(child)) << "no frame " << child;
    return data_.oMf[model_.getFrameId(parent)]
           .actInv(data_.oMf[model_.getFrameId(child)]).translation();
  }

  Eigen::Vector2d planar(const std::string& parent, const std::string& child) const
  {
    return offset(parent, child).head<2>();
  }

private:
  static pinocchio::Model build(crane_model::Tool tool)
  {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(description_for(tool), model);
    return model;
  }

  pinocchio::Model model_;
  pinocchio::Data data_;
};

}  // namespace parsed

}  // namespace

TEST(CraneModelContract, DimensionsAndProjectionsAreFrozen)
{
  EXPECT_EQ(crane_model::kGeneralizedDof, 8U);
  EXPECT_EQ(crane_model::kActuatedDof, 6U);
  EXPECT_EQ(crane_model::kPassiveDof, 2U);
  EXPECT_EQ(crane_model::kStateDof, 16U);
  EXPECT_EQ(crane_model::kInputDof, 6U);

  const auto mock = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(mock.ok());
  const auto& names = mock.value().urdf_joint_names();
  EXPECT_EQ(names[0], "theta1_slewing_joint");
  EXPECT_EQ(names[3], "q4_big_telescope");
  EXPECT_EQ(names[4], "theta6_tip_joint");
  EXPECT_EQ(names[7], "q9_left_rail_joint");
}

TEST(CraneModelContract, BothToolMapsAreStable)
{
  const auto pzs = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  const auto jaws = crane_model::testing::MockModel::create(
    crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(pzs.ok());
  ASSERT_TRUE(jaws.ok());
  EXPECT_EQ(pzs.value().urdf_joint_names()[7], "q9_left_rail_joint");
  EXPECT_EQ(jaws.value().urdf_joint_names()[7], "theta10_outer_jaw_joint");
}

TEST(CraneModelContract, ProductionModelCreatesForBothTools)
{
  const auto pzs = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(pzs.ok());
  EXPECT_TRUE(pzs.value().ready());
  EXPECT_EQ(pzs.value().tool(), crane_model::Tool::Pzs100);
  EXPECT_EQ(pzs.value().urdf_joint_names()[7], "q9_left_rail_joint");

  const auto jaws = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(jaws.ok());
  EXPECT_TRUE(jaws.value().ready());
  EXPECT_EQ(jaws.value().tool(), crane_model::Tool::Epsilon7040);
  EXPECT_EQ(jaws.value().urdf_joint_names()[7], "theta10_outer_jaw_joint");
}

TEST(CraneModelContract, RobotDescriptionAndToolValidationIsExplicit)
{
  const auto invalid = crane_model::Model::create({});
  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code, crane_model::ErrorCode::InvalidRobotDescription);
  EXPECT_FALSE(invalid.status().message.empty());

  auto malformed = crane_model::ModelConfig{};
  malformed.robot_description_xml = "<robot";
  auto result = crane_model::Model::create(malformed);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::InvalidRobotDescription);

  auto missing_joint = crane_model::ModelConfig{};
  missing_joint.robot_description_xml = kJointlessDescription;
  result = crane_model::Model::create(missing_joint);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::MissingJoint);

  // The tool half of the joint map is checked against the description too: the
  // rail description does not carry the 7040's jaw joint (contract §2).
  auto wrong_tool = crane_model::ModelConfig{};
  wrong_tool.robot_description_xml = description_for(crane_model::Tool::Pzs100);
  wrong_tool.tool = crane_model::Tool::Epsilon7040;
  result = crane_model::Model::create(wrong_tool);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::MissingJoint);
  EXPECT_NE(result.status().message.find("theta10_outer_jaw_joint"), std::string::npos);

  auto invalid_tool = crane_model::ModelConfig{};
  invalid_tool.robot_description_xml = description_for(crane_model::Tool::Pzs100);
  invalid_tool.tool = static_cast<crane_model::Tool>(255);
  result = crane_model::Model::create(invalid_tool);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::UnsupportedTool);

  auto non_finite_gravity = crane_model::ModelConfig{};
  non_finite_gravity.robot_description_xml = description_for(crane_model::Tool::Pzs100);
  non_finite_gravity.gravity_m_s2[2] = std::numeric_limits<double>::quiet_NaN();
  result = crane_model::Model::create(non_finite_gravity);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::InvalidArgument);

  const auto mock = crane_model::testing::MockModel::create(
    static_cast<crane_model::Tool>(255));
  EXPECT_EQ(mock.status().code, crane_model::ErrorCode::UnsupportedTool);
}

TEST(CraneModelContract, MockRejectsInvalidPayloadAndNonFiniteInput)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto invalid_payload = model.value().passive_equilibrium(
    crane_model::QA::Ones(), crane_model::Payload{});
  EXPECT_FALSE(invalid_payload.ok());
  EXPECT_EQ(invalid_payload.status().code, crane_model::ErrorCode::InvalidPayload);

  auto q = valid_q();
  q[2] = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite = model.value().jacobian(q, crane_model::Frame::Tcp);
  EXPECT_FALSE(non_finite.ok());
  EXPECT_EQ(non_finite.status().code, crane_model::ErrorCode::NonFiniteInput);
}

TEST(CraneModelContract, ValidZeroMassPayloadIsNotSilentlyReclassified)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  auto payload = valid_payload();
  payload.mass_kg = 0.0;
  const auto equilibrium = model.value().passive_equilibrium(
    crane_model::QA::Zero(), payload);
  EXPECT_TRUE(equilibrium.ok());
}

TEST(CraneModelContract, MockProvidesNamedNonzeroKinematicsAndDynamics)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto payload = valid_payload();
  const auto q = valid_q();
  const auto fk = model.value().forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  ASSERT_TRUE(fk.ok());
  EXPECT_DOUBLE_EQ(fk.value().position_m[0], 1.0);
  EXPECT_DOUBLE_EQ(fk.value().position_m[2], 3.0);

  const auto dynamics = model.value().full_dynamics(q, crane_model::DQ::Ones(), payload);
  ASSERT_TRUE(dynamics.ok());
  EXPECT_GT(dynamics.value().mass.trace(), 0.0);
  EXPECT_NE(dynamics.value().bias.norm(), 0.0);
}

TEST(CraneModelContract, EveryMockModelMethodHasAValidFixturePath)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  EXPECT_EQ(model.value().tool(), crane_model::Tool::Pzs100);
  EXPECT_TRUE(model.value().ready());
  const auto q = valid_q();
  const auto dq = crane_model::DQ::Ones();
  const auto payload = valid_payload();
  crane_model::ChamberPressure pressure;
  pressure.p_a_pa.setZero();
  pressure.p_b_pa.setZero();
  const auto scene = crane_model::CollisionScene{};

  EXPECT_TRUE(model.value().forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp).ok());
  EXPECT_TRUE(model.value().jacobian(q, crane_model::Frame::Tcp).ok());
  EXPECT_TRUE(model.value().passive_equilibrium(crane_model::QA::Zero(), payload).ok());
  EXPECT_TRUE(model.value().full_dynamics(q, dq, payload).ok());
  EXPECT_TRUE(model.value().reduced_actuated_dynamics(
    q, dq, crane_model::Input::Zero(), payload).ok());
  EXPECT_TRUE(model.value().inverse_dynamics(q, dq, dq, payload).ok());
  EXPECT_TRUE(model.value().cylinder_jacobian(q).ok());
  EXPECT_TRUE(model.value().transmission(q, crane_model::DQA::Zero(), pressure).ok());
  EXPECT_TRUE(model.value().cylinder_force(pressure).ok());
  EXPECT_TRUE(model.value().collision_query(q, scene).ok());
  EXPECT_TRUE(model.value().collision_queries(q, scene).ok());
  EXPECT_TRUE(model.value().symbolic_graph({}, payload).ok());
}

TEST(CraneModelContract, NonFiniteInputsAreRejectedAcrossMockMethods)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  auto q = valid_q();
  q[0] = std::numeric_limits<double>::quiet_NaN();
  crane_model::DQ dq = crane_model::DQ::Zero();
  dq(0) = std::numeric_limits<double>::infinity();
  crane_model::ChamberPressure pressure;
  pressure.p_a_pa.setZero();
  pressure.p_b_pa.setZero();
  pressure.p_a_pa[0] = std::numeric_limits<double>::quiet_NaN();
  const auto payload = valid_payload();

  EXPECT_EQ(model.value().forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().jacobian(q, crane_model::Frame::Tcp).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().passive_equilibrium(
    crane_model::QA::Constant(std::numeric_limits<double>::quiet_NaN()), payload)
      .status().code, crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().full_dynamics(q, dq, payload).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().reduced_actuated_dynamics(
    q, dq, crane_model::Input::Zero(), payload).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().inverse_dynamics(q, dq, dq, payload).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().cylinder_jacobian(q).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().transmission(
    crane_model::Q::Zero(), crane_model::DQA::Zero(), pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().cylinder_force(pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().collision_query(
    q, crane_model::CollisionScene{}).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().collision_queries(
    q, crane_model::CollisionScene{}).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  crane_model::SymbolicGraphSpec bad_spec;
  bad_spec.sample_time_s = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(model.value().symbolic_graph(bad_spec, payload).status().code,
    crane_model::ErrorCode::InvalidArgument);
}

TEST(CraneModelContract, InvalidSceneAndPayloadInputsAreNotSilentlyAccepted)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  crane_model::CollisionPrimitive invalid_primitive;
  invalid_primitive.id = "";
  crane_model::CollisionScene invalid_scene;
  invalid_scene.primitives.push_back(invalid_primitive);
  EXPECT_EQ(model.value().collision_query(valid_q(), invalid_scene).status().code,
    crane_model::ErrorCode::InvalidScene);
  EXPECT_EQ(model.value().collision_queries(valid_q(), invalid_scene).status().code,
    crane_model::ErrorCode::InvalidScene);
  EXPECT_EQ(model.value().full_dynamics(
    valid_q(), crane_model::DQ::Zero(), crane_model::Payload{}).status().code,
    crane_model::ErrorCode::InvalidPayload);
  crane_model::SymbolicGraphSpec bad_spec;
  bad_spec.sample_time_s = 0.0;
  EXPECT_EQ(model.value().symbolic_graph(bad_spec, valid_payload()).status().code,
    crane_model::ErrorCode::InvalidArgument);

  crane_model::ChamberPressure negative_pressure;
  negative_pressure.p_a_pa.setZero();
  negative_pressure.p_b_pa.setZero();
  negative_pressure.p_a_pa(0) = -1.0;
  EXPECT_EQ(model.value().transmission(
    valid_q(), crane_model::DQA::Zero(), negative_pressure).status().code,
    crane_model::ErrorCode::InvalidArgument);
  EXPECT_EQ(model.value().cylinder_force(negative_pressure).status().code,
    crane_model::ErrorCode::InvalidArgument);
}

TEST(CraneModelContract, PreparedHydraulicPathAllocatesNoMemoryOrCallsNonRtMethods)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = valid_q();
  const auto payload = valid_payload();
  crane_model::ChamberPressure pressure;
  pressure.p_a_pa.setZero();
  pressure.p_b_pa.setZero();
  crane_model::DQA dq_a = crane_model::DQA::Zero();
  // Warm all code paths before observing allocations. This is a test of the
  // prepared fixed-size mock path, not a claim about the future backend.
  ASSERT_TRUE(model.value().cylinder_jacobian(q).ok());
  ASSERT_TRUE(model.value().transmission(q, dq_a, pressure).ok());
  ASSERT_TRUE(model.value().cylinder_force(pressure).ok());
  model.value().reset_call_counters();
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_allocation_guard.store(true, std::memory_order_relaxed);
  bool all_ok = true;
  for (int index = 0; index < 1000; ++index) {
    all_ok = all_ok && model.value().cylinder_jacobian(q).ok();
    all_ok = all_ok && model.value().transmission(q, dq_a, pressure).ok();
    all_ok = all_ok && model.value().cylinder_force(pressure).ok();
  }
  g_allocation_guard.store(false, std::memory_order_relaxed);
  EXPECT_TRUE(all_ok);
  EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(model.value().call_counters().forbidden_non_rt_calls, 0U);
}

TEST(CraneModelContract, PassiveInverseDynamicsAndCylinderJacobianInvariants)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto tau = model.value().inverse_dynamics(
    valid_q(), crane_model::DQ::Zero(), crane_model::DQ::Zero(), valid_payload());
  ASSERT_TRUE(tau.ok());
  EXPECT_DOUBLE_EQ(tau.value()[4], 0.0);
  EXPECT_DOUBLE_EQ(tau.value()[5], 0.0);

  const auto jacobian = model.value().cylinder_jacobian(valid_q());
  ASSERT_TRUE(jacobian.ok());
  for (Eigen::Index row = 0; row < 6; ++row) {
    for (Eigen::Index column = 0; column < 6; ++column) {
      if (row != column) {
        EXPECT_DOUBLE_EQ(jacobian.value()(row, column), 0.0);
      }
    }
  }
}

TEST(CraneModelContract, SymbolicGraphHasFrozenDimensions)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto graph = model.value().symbolic_graph({}, valid_payload());
  ASSERT_TRUE(graph.ok());
  EXPECT_EQ(graph.value().state_dimension(), crane_model::kStateDof);
  EXPECT_EQ(graph.value().input_dimension(), crane_model::kInputDof);
  EXPECT_TRUE(graph.value().has_output_map());
}

TEST(CraneModelContract, CollisionSceneRejectsDuplicateIds)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  crane_model::CollisionPrimitive first;
  first.id = "same";
  crane_model::CollisionPrimitive second = first;
  crane_model::CollisionScene scene;
  scene.primitives = {first, second};
  const auto result = model.value().collision_query(valid_q(), scene);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::InvalidScene);
}

// --- hydraulic transmission subset -----------------------------------------
//
// These assert the real backend of wiki/hydraulics.md §2--§4 against the real
// linkage dimensions of §6. They are the S3 extension of the PRD's Seams
// section; the rest of the model is still absent and is asserted to say so.

TEST(CraneModelHydraulicSubset, CylinderJacobianIsDiagonalAndRealValued)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto jacobian = model.value().cylinder_jacobian(linkage_configuration(0.0, 0.0));
  ASSERT_TRUE(jacobian.ok());

  for (Eigen::Index row = 0; row < 6; ++row) {
    for (Eigen::Index column = 0; column < 6; ++column) {
      if (row != column) {
        EXPECT_DOUBLE_EQ(jacobian.value()(row, column), 0.0);
      }
    }
    EXPECT_NE(jacobian.value()(row, row), 0.0);
  }

  // q1 rack and pinion, q4 telescope, q7 motor and q8 rail are one-to-one or
  // a constant radius; only q2 and q3 depend on the configuration.
  EXPECT_DOUBLE_EQ(jacobian.value()(0, 0), 0.1);
  EXPECT_DOUBLE_EQ(jacobian.value()(3, 3), 1.0);
  EXPECT_DOUBLE_EQ(jacobian.value()(4, 4), 1.0);
  EXPECT_DOUBLE_EQ(jacobian.value()(5, 5), 1.0);
}

TEST(CraneModelHydraulicSubset, BoomIsAFourBarAndNotADirectCylinder)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());

  // The whole stroke range of §6.3, s_2 in [1.253, 2.229] m.
  for (const double q2 : {-1.05, -0.9, -0.4, 0.0, 0.6, 1.2, 1.6}) {
    const auto jacobian = model.value().cylinder_jacobian(linkage_configuration(q2, 0.0));
    ASSERT_TRUE(jacobian.ok()) << "q2 = " << q2;
    EXPECT_NEAR(
      jacobian.value()(1, 1),
      reference::transmission_ratio(&reference::boom_stroke, q2), 1.0e-8) << "q2 = " << q2;
  }

  // A direct cylinder from the same foot to the same boom-side attachment is
  // a different transmission, so the four-bar is not being approximated away.
  const auto near_retracted = model.value().cylinder_jacobian(linkage_configuration(-1.05, 0.0));
  ASSERT_TRUE(near_retracted.ok());
  const double direct =
    reference::transmission_ratio(&reference::boom_direct_stroke, -1.05);
  EXPECT_NEAR(near_retracted.value()(1, 1), 0.127229860069, 1.0e-9);
  EXPECT_NEAR(direct, 0.120389763, 1.0e-6);
  EXPECT_GT(std::abs(near_retracted.value()(1, 1) - direct) / near_retracted.value()(1, 1), 0.05);

  // Below q2 = -1.184 the drawbar and pushbar no longer reach: d < r_13 - r_23
  // and the triangle inequality of §2.2 fails. A direct cylinder has no such
  // configuration, so this is only reachable through the linkage model.
  const auto unreachable = model.value().cylinder_jacobian(linkage_configuration(-1.3, 0.0));
  EXPECT_FALSE(unreachable.ok());
  EXPECT_EQ(unreachable.status().code, crane_model::ErrorCode::SingularConfiguration);
}

TEST(CraneModelHydraulicSubset, BoomAndArmTransmissionRatiosDependOnStroke)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto retracted = model.value().cylinder_jacobian(linkage_configuration(-0.9, -0.9));
  const auto extended = model.value().cylinder_jacobian(linkage_configuration(0.0, 0.0));
  ASSERT_TRUE(retracted.ok());
  ASSERT_TRUE(extended.ok());

  // The amount the geometry predicts, taken from the strokes of §2.2 and §2.3.
  EXPECT_NEAR(retracted.value()(1, 1), 0.210686543710, 1.0e-9);
  EXPECT_NEAR(extended.value()(1, 1), 0.451149447172, 1.0e-9);
  EXPECT_NEAR(retracted.value()(2, 2), 0.249852055264, 1.0e-9);
  EXPECT_NEAR(extended.value()(2, 2), 0.482592819169, 1.0e-9);

  EXPECT_NEAR(
    extended.value()(1, 1) - retracted.value()(1, 1),
    reference::transmission_ratio(&reference::boom_stroke, 0.0) -
    reference::transmission_ratio(&reference::boom_stroke, -0.9), 1.0e-8);
  EXPECT_NEAR(
    extended.value()(2, 2) - retracted.value()(2, 2),
    reference::transmission_ratio(&reference::arm_stroke, 0.0) -
    reference::transmission_ratio(&reference::arm_stroke, -0.9), 1.0e-8);

  // Same demanded joint velocity, different piston velocity: the ratio moves
  // by more than a factor of 1.9 over these two configurations.
  EXPECT_GT(extended.value()(1, 1) / retracted.value()(1, 1), 2.1);
  EXPECT_GT(extended.value()(2, 2) / retracted.value()(2, 2), 1.9);

  // The arm is a direct cylinder, so its ratio is the plain stroke derivative.
  for (const double q3 : {-0.9, -0.3, 0.0, 0.6, 1.4}) {
    const auto jacobian = model.value().cylinder_jacobian(linkage_configuration(0.0, q3));
    ASSERT_TRUE(jacobian.ok()) << "q3 = " << q3;
    EXPECT_NEAR(
      jacobian.value()(2, 2),
      reference::transmission_ratio(&reference::arm_stroke, q3), 1.0e-8) << "q3 = " << q3;
  }
}

TEST(CraneModelHydraulicSubset, ChamberAreaSwitchIsADirectionDependentStep)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.0, 0.0);
  const auto pressure = zero_pressure();

  // A_eff_pos / A_eff_neg per axis, wiki/hydraulics.md §3.
  const crane_model::Vector6 expected_ratio =
    (crane_model::Vector6() <<
      1.0,                            // q1, two cylinders, symmetric
      1.5394e-2 / 9.032e-3,           // q2, A_A / A_B
      (2.0 * 6.362e-3) / (2.0 * 2.5133e-3),   // q3, 2 A_A / 2 A_B
      (3.8485e-3 - 2.592e-3) / 2.592e-3,      // q4, regenerative on extension
      1.0,                            // q7, motor, symmetric
      7.854e-3 / 4.737e-3).finished();        // q8, A_A / A_B

  // The switch is at v_i = 0 and is a step, so an arbitrarily small speed
  // already sees the full asymmetry. Both magnitudes give the same ratio.
  for (const double speed : {1.0e-9, 0.2}) {
    const auto extending = model.value().transmission(
      q, crane_model::DQA::Constant(speed), pressure);
    const auto retracting = model.value().transmission(
      q, crane_model::DQA::Constant(-speed), pressure);
    ASSERT_TRUE(extending.ok());
    ASSERT_TRUE(retracting.ok());
    for (Eigen::Index axis = 0; axis < 6; ++axis) {
      EXPECT_NEAR(
        std::abs(extending.value().cylinder_velocity[axis]),
        std::abs(retracting.value().cylinder_velocity[axis]), 1.0e-15) << "axis " << axis;
      EXPECT_GT(retracting.value().pump_flow[axis], 0.0) << "axis " << axis;
      EXPECT_NEAR(
        extending.value().pump_flow[axis] / retracting.value().pump_flow[axis],
        expected_ratio[axis], 1.0e-9) << "axis " << axis << " speed " << speed;
    }
    // Extending and retracting the same axis at the same speed demand
    // different flows on every axis except the two symmetric circuits.
    EXPECT_NE(extending.value().pump_flow[1], retracting.value().pump_flow[1]);
    EXPECT_NE(extending.value().pump_flow[2], retracting.value().pump_flow[2]);
    EXPECT_NE(extending.value().pump_flow[3], retracting.value().pump_flow[3]);
    EXPECT_NE(extending.value().pump_flow[5], retracting.value().pump_flow[5]);
    EXPECT_DOUBLE_EQ(extending.value().pump_flow[0], retracting.value().pump_flow[0]);
    EXPECT_DOUBLE_EQ(extending.value().pump_flow[4], retracting.value().pump_flow[4]);
  }
}

TEST(CraneModelHydraulicSubset, TelescopeIsRegenerativeOnExtension)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.0, 0.0);
  const auto pressure = zero_pressure();
  const double speed = 0.05;

  crane_model::DQA extend = crane_model::DQA::Zero();
  extend[3] = speed;
  crane_model::DQA retract = crane_model::DQA::Zero();
  retract[3] = -speed;
  const auto extending = model.value().transmission(q, extend, pressure);
  const auto retracting = model.value().transmission(q, retract, pressure);
  ASSERT_TRUE(extending.ok());
  ASSERT_TRUE(retracting.ok());

  // s_4 = q_4, so the piston speed is the joint speed (§2.4).
  EXPECT_DOUBLE_EQ(extending.value().cylinder_velocity[3], speed);
  // Only the annulus difference A_A - A_B is drawn from the pump while
  // extending: it extends fast and weak, retracts slow and strong (§3).
  EXPECT_NEAR(extending.value().pump_flow[3], (3.8485e-3 - 2.592e-3) * speed, 1.0e-15);
  EXPECT_NEAR(retracting.value().pump_flow[3], 2.592e-3 * speed, 1.0e-15);
  EXPECT_NEAR(
    extending.value().pump_flow[3] / retracting.value().pump_flow[3],
    (3.8485e-3 - 2.592e-3) / 2.592e-3, 1.0e-12);
  EXPECT_LT(extending.value().pump_flow[3], retracting.value().pump_flow[3]);
}

TEST(CraneModelHydraulicSubset, PumpFlowIsPopulatedFromTheEffectiveAreas)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.0, 0.0);
  const auto pressure = zero_pressure();

  const auto standing = model.value().transmission(q, crane_model::DQA::Zero(), pressure);
  ASSERT_TRUE(standing.ok());
  EXPECT_DOUBLE_EQ(standing.value().pump_flow.norm(), 0.0);

  crane_model::DQA dq_a;
  dq_a << 0.2, 0.05, -0.05, 0.02, 0.3, 0.01;
  const auto moving = model.value().transmission(q, dq_a, pressure);
  ASSERT_TRUE(moving.ok());
  const crane_model::Vector6 velocity = moving.value().joint_to_cylinder * dq_a;
  EXPECT_TRUE(moving.value().cylinder_velocity.isApprox(velocity));

  const crane_model::Vector6 expected_flow =
    (crane_model::Vector6() <<
      2.0 * 6.362e-3 * std::abs(velocity[0]),
      1.5394e-2 * velocity[1],
      2.0 * 2.5133e-3 * std::abs(velocity[2]),
      (3.8485e-3 - 2.592e-3) * velocity[3],
      1.4324e-4 * velocity[4],
      7.854e-3 * velocity[5]).finished();
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    EXPECT_NEAR(moving.value().pump_flow[axis], expected_flow[axis], 1.0e-15) << "axis " << axis;
    EXPECT_GT(moving.value().pump_flow[axis], 0.0) << "axis " << axis;
  }
  // Q_P is the sum of the magnitudes and stays under the pump limit of §6.1.
  EXPECT_LT(moving.value().pump_flow.sum(), 1.4e-3);
}

TEST(CraneModelHydraulicSubset, CylinderForceUsesTheChamberAreas)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  crane_model::ChamberPressure pressure;
  pressure.p_a_pa.setConstant(1.0e7);  // 100 bar
  pressure.p_b_pa.setConstant(5.0e6);  // 50 bar

  const auto force = model.value().cylinder_force(pressure);
  ASSERT_TRUE(force.ok());
  const crane_model::Vector6 expected =
    (crane_model::Vector6() <<
      2.0 * 6.362e-3 * 1.0e7 - 2.0 * 6.362e-3 * 5.0e6,
      1.5394e-2 * 1.0e7 - 9.032e-3 * 5.0e6,
      2.0 * 6.362e-3 * 1.0e7 - 2.0 * 2.5133e-3 * 5.0e6,
      3.8485e-3 * 1.0e7 - 2.592e-3 * 5.0e6,
      1.4324e-4 * 1.0e7 - 1.4324e-4 * 5.0e6,  // N m: V_m takes the area's role
      7.854e-3 * 1.0e7 - 4.737e-3 * 5.0e6).finished();
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    EXPECT_NEAR(force.value()[axis], expected[axis], 1.0e-6) << "axis " << axis;
  }

  // The transmission reports the same forces, which is the measurement path
  // that makes tau observable without a torque sensor (§4).
  const auto transmission = model.value().transmission(
    linkage_configuration(0.0, 0.0), crane_model::DQA::Zero(), pressure);
  ASSERT_TRUE(transmission.ok());
  EXPECT_TRUE(transmission.value().pressure_force.isApprox(force.value()));

  // The areas are tool-independent, so the 7040 reports the same forces.
  const auto jaws = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(jaws.ok());
  const auto jaw_force = jaws.value().cylinder_force(pressure);
  ASSERT_TRUE(jaw_force.ok());
  EXPECT_TRUE(jaw_force.value().isApprox(force.value()));
}

TEST(CraneModelHydraulicSubset, InvalidHydraulicInputsAreRejected)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.0, 0.0);
  const auto pressure = zero_pressure();

  auto non_finite_q = q;
  non_finite_q[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(model.value().cylinder_jacobian(non_finite_q).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().transmission(
    non_finite_q, crane_model::DQA::Zero(), pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);

  crane_model::DQA non_finite_dq = crane_model::DQA::Zero();
  non_finite_dq[0] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(model.value().transmission(q, non_finite_dq, pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);

  auto non_finite_pressure = pressure;
  non_finite_pressure.p_b_pa[2] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(model.value().cylinder_force(non_finite_pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(model.value().transmission(
    q, crane_model::DQA::Zero(), non_finite_pressure).status().code,
    crane_model::ErrorCode::NonFiniteInput);

  auto negative_pressure = pressure;
  negative_pressure.p_a_pa[0] = -1.0;
  EXPECT_EQ(model.value().cylinder_force(negative_pressure).status().code,
    crane_model::ErrorCode::InvalidArgument);
  EXPECT_EQ(model.value().transmission(
    q, crane_model::DQA::Zero(), negative_pressure).status().code,
    crane_model::ErrorCode::InvalidArgument);
}

TEST(CraneModelHydraulicSubset, SevenThousandFortyJawUsesTheDeployedFourBarFit)
{
  const auto model = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(model.ok());

  // PORT FIDELITY, NOT INDEPENDENT VALIDATION. reference::jaw_stroke restates
  // §2.6's composition, but its mirror law repeats the deployed quadratic's own
  // coefficients: the Freudenstein constants a, b, c and d that produced that
  // quadratic are in no file of this workspace, so there is no second source to
  // check the fit itself against. What is checked here is everything around the
  // fit — that the model drives the inner jaw through it and takes the cylinder
  // length between the two attachment points.
  for (const double q8 : {0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}) {
    const auto jacobian = model.value().cylinder_jacobian(jaw_configuration(q8));
    ASSERT_TRUE(jacobian.ok()) << "q8 = " << q8;
    const double ratio = jacobian.value()(5, 5);
    EXPECT_NEAR(
      ratio, reference::transmission_ratio(&reference::jaw_stroke, q8), 1.0e-8) << "q8 = " << q8;
    // Not the rail's one-to-one, and not a single-jaw cylinder either: freezing
    // the inner jaw changes the ratio by a third or more at every q8.
    EXPECT_NE(ratio, 1.0) << "q8 = " << q8;
    EXPECT_GT(
      std::abs(ratio - reference::transmission_ratio(
        &reference::jaw_stroke_with_frozen_inner_jaw, q8)) / std::abs(ratio), 0.3) << "q8 = " << q8;
  }

  // The amounts the ported geometry predicts, inside the jaw's hydraulic travel
  // (s_8 spans 0.537 to 0.829 m over q8 in 0.944 to 2.997 rad, the limits of
  // pincer_cylinder_piston_in_barrel_linear_joint in the 7040 URDF).
  const auto opened = model.value().cylinder_jacobian(jaw_configuration(1.0));
  const auto closed = model.value().cylinder_jacobian(jaw_configuration(3.0));
  ASSERT_TRUE(opened.ok());
  ASSERT_TRUE(closed.ok());
  EXPECT_NEAR(opened.value()(5, 5), 0.144581325689, 1.0e-9);
  EXPECT_NEAR(closed.value()(5, 5), 0.055268972307, 1.0e-9);

  // The linkage toggles at q8 = 0.2623 rad, where the ratio changes sign. A
  // one-to-one axis has no such configuration, so this is only reachable
  // through the four-bar; it sits below the reachable stroke.
  const auto below = model.value().cylinder_jacobian(jaw_configuration(0.1));
  const auto above = model.value().cylinder_jacobian(jaw_configuration(0.5));
  ASSERT_TRUE(below.ok());
  ASSERT_TRUE(above.ok());
  EXPECT_LT(below.value()(5, 5), 0.0);
  EXPECT_GT(above.value()(5, 5), 0.0);

  // Only the tool axis is tool-dependent, and the Jacobian stays diagonal.
  const auto rail = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(rail.ok());
  const auto rail_jacobian = rail.value().cylinder_jacobian(jaw_configuration(1.5));
  const auto jaw_jacobian = model.value().cylinder_jacobian(jaw_configuration(1.5));
  ASSERT_TRUE(rail_jacobian.ok());
  ASSERT_TRUE(jaw_jacobian.ok());
  for (Eigen::Index row = 0; row < 6; ++row) {
    for (Eigen::Index column = 0; column < 6; ++column) {
      if (row != column) {
        EXPECT_DOUBLE_EQ(jaw_jacobian.value()(row, column), 0.0);
      }
    }
    if (row != 5) {
      EXPECT_DOUBLE_EQ(jaw_jacobian.value()(row, row), rail_jacobian.value()(row, row));
    }
  }
  EXPECT_NE(jaw_jacobian.value()(5, 5), rail_jacobian.value()(5, 5));

  // transmission() carries the same ratio, so the GR axis now produces a
  // cylinder velocity and a pump flow like the other five.
  const double jaw_speed = 0.4;
  crane_model::DQA dq_a = crane_model::DQA::Zero();
  dq_a[5] = jaw_speed;
  const auto transmission =
    model.value().transmission(jaw_configuration(1.5), dq_a, zero_pressure());
  ASSERT_TRUE(transmission.ok());
  EXPECT_NEAR(transmission.value().joint_to_cylinder(5, 5), 0.177829106844, 1.0e-9);
  EXPECT_NEAR(transmission.value().cylinder_velocity[5], 0.177829106844 * jaw_speed, 1.0e-12);
  EXPECT_NEAR(
    transmission.value().pump_flow[5],
    7.854e-3 * transmission.value().cylinder_velocity[5], 1.0e-18);
}

TEST(CraneModelHydraulicSubset, NoCallReportsBackendUnavailableAnyMore)
{
  // Both tools. Slice 4 arrived in pieces -- forward kinematics, the Jacobian,
  // collision, the rigid-body dynamics, the passive equilibrium, and now the
  // symbolic graph -- and the graph was the last of them. `BackendUnavailable`
  // is therefore no longer an answer this library gives: whatever each call
  // reports at a given argument, it does not report "no backend".
  //
  // `test_symbolic_graph` is what says the graph is the *same* equations. This
  // only says it exists.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto payload = valid_payload();
    const auto unavailable = crane_model::ErrorCode::BackendUnavailable;
    const auto q = valid_q();

    const auto graph = model.value().symbolic_graph({}, payload);
    ASSERT_TRUE(graph.ok()) << graph.status().message;
    EXPECT_EQ(graph.value().state_dimension(), crane_model::kStateDof);
    EXPECT_EQ(graph.value().input_dimension(), crane_model::kInputDof);
    EXPECT_TRUE(graph.value().has_output_map());

    EXPECT_NE(model.value().passive_equilibrium(
      crane_model::QA::Zero(), payload).status().code, unavailable);
    EXPECT_NE(model.value().cylinder_jacobian(q).status().code, unavailable);
    EXPECT_NE(model.value().full_dynamics(
      q, crane_model::DQ::Zero(), payload).status().code, unavailable);
    EXPECT_NE(model.value().collision_query(
      q, crane_model::CollisionScene{}).status().code, unavailable);
  }
}

TEST(CraneModelHydraulicSubset, RealTimeCallsAllocateNothingAfterConstruction)
{
  // Both tools, because the 7040's jaw four-bar is on the same RT path, and now
  // also forward kinematics, the Jacobian, the three rigid-body dynamics calls
  // and the passive equilibrium: contract §10 says a call that cannot be shown
  // allocation-free is not an RT API call, and all of these are on the control
  // path. The Pinocchio model, its Data workspace, the two velocity buffers and
  // the gravity-Jacobian workspace are built once, in create(), which is not
  // real-time; the payload is written into the model's own inertia for the length
  // of a call, which is fixed-size arithmetic.
  //
  // These nine are the whole of what this guard claims. Collision is not among
  // them and is not meant to be; CollisionQueriesIsNotARealTimeCall shows it
  // allocating, so the omission here is a statement and not a gap.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    auto q = linkage_configuration(0.2, -0.3);
    q[7] = 1.5;
    const auto pressure = zero_pressure();
    crane_model::DQA dq_a;
    dq_a << 0.3, 0.1, -0.1, 0.02, 0.4, 0.01;
    const crane_model::DQ dq = crane_model::DQ::Constant(0.15);
    const auto ddq_a = crane_model::Input::Constant(0.2);
    const auto payload = valid_payload();
    const auto base = crane_model::Frame::MountingBase;
    const auto tcp = crane_model::Frame::Tcp;
    // The equilibrium solve needs a q_a the tool can actually hang at, because a
    // failing call would allocate its own diagnostic and prove nothing. This is
    // the actuated half of the `loaded_configuration` the dynamics tests below
    // use, written out because those helpers come later in the file.
    crane_model::QA q_a;
    q_a << 0.4, 0.35, 0.9, 0.8, 0.6, 0.3;
    // Warm every code path before observing allocations.
    ASSERT_TRUE(model.value().cylinder_jacobian(q).ok());
    ASSERT_TRUE(model.value().transmission(q, dq_a, pressure).ok());
    ASSERT_TRUE(model.value().cylinder_force(pressure).ok());
    ASSERT_TRUE(model.value().forward_kinematics(q, base, tcp).ok());
    ASSERT_TRUE(model.value().jacobian(q, tcp).ok());
    ASSERT_TRUE(model.value().full_dynamics(q, dq, payload).ok());
    ASSERT_TRUE(model.value().inverse_dynamics(q, dq, dq, payload).ok());
    ASSERT_TRUE(model.value().reduced_actuated_dynamics(q, dq, ddq_a, payload).ok());
    ASSERT_TRUE(model.value().passive_equilibrium(q_a, payload).ok());

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_allocation_guard.store(true, std::memory_order_relaxed);
    bool all_ok = true;
    for (int index = 0; index < 1000; ++index) {
      all_ok = all_ok && model.value().cylinder_jacobian(q).ok();
      all_ok = all_ok && model.value().transmission(q, dq_a, pressure).ok();
      all_ok = all_ok && model.value().cylinder_force(pressure).ok();
      all_ok = all_ok && model.value().forward_kinematics(q, base, tcp).ok();
      all_ok = all_ok && model.value().jacobian(q, tcp).ok();
      all_ok = all_ok && model.value().full_dynamics(q, dq, payload).ok();
      all_ok = all_ok && model.value().inverse_dynamics(q, dq, dq, payload).ok();
      all_ok = all_ok && model.value().reduced_actuated_dynamics(q, dq, ddq_a, payload).ok();
    }
    // The equilibrium is a bounded iteration over the same fixed-size workspaces
    // and it allocates nothing, but one call already walks the whole seed grid
    // and the whole Newton iteration -- tens of `rnea` evaluations, not one. What
    // is being counted is allocations per call, so it is exercised a handful of
    // times rather than a thousand. The cost is in the README; contract §10's
    // test is allocation, and it passes it.
    for (int index = 0; index < 5; ++index) {
      all_ok = all_ok && model.value().passive_equilibrium(q_a, payload).ok();
    }
    g_allocation_guard.store(false, std::memory_order_relaxed);
    EXPECT_TRUE(all_ok);
    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
  }
}

// --- the rigid-body description ---------------------------------------------
//
// Slice 4's tracer bullet: one call end to end through Pinocchio -- the real
// description in, a parsed model, a pose out.

TEST(CraneModelDescription, JointMapComesFromTheRealDescriptionForBothTools)
{
  const auto rail = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(rail.ok()) << rail.status().message;
  const auto jaws = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(jaws.ok()) << jaws.status().message;

  // The canonical order of contract §2, spelled as ROS 2 Interfaces §3.1
  // spells it, which is the legacy URDF spelling and not the q_i symbols.
  const std::array<const char *, 7> shared{{
    "theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope",
    "theta6_tip_joint", "theta7_tilt_joint", "theta8_rotator_joint"}};
  for (std::size_t index = 0; index < shared.size(); ++index) {
    EXPECT_EQ(rail.value().urdf_joint_names()[index], shared[index]);
    EXPECT_EQ(jaws.value().urdf_joint_names()[index], shared[index]);
  }
  EXPECT_EQ(rail.value().urdf_joint_names()[7], "q9_left_rail_joint");
  EXPECT_EQ(jaws.value().urdf_joint_names()[7], "theta10_outer_jaw_joint");
}

TEST(CraneModelDescription, ForwardKinematicsAnswersForEveryFramePair)
{
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto q = valid_q();

    for (const auto from : kAllFrames) {
      for (const auto to : kAllFrames) {
        const auto pose = model.value().forward_kinematics(q, from, to);
        if (!pose.ok()) {
          // The only permitted refusal is a frame this description does not
          // carry. It is never a wrong pose and never a stub.
          EXPECT_EQ(pose.status().code, crane_model::ErrorCode::FrameUnavailable)
            << static_cast<int>(from) << " -> " << static_cast<int>(to);
          continue;
        }
        EXPECT_EQ(pose.value().expressed_in, from);
        EXPECT_TRUE(pose.value().position_m.allFinite());
        EXPECT_NEAR(pose.value().orientation.norm(), 1.0, 1.0e-12);
        if (from == to) {
          EXPECT_LT(pose.value().position_m.norm(), 1.0e-15);
          EXPECT_LT(rotation_vector(isometry(pose.value()).linear()).norm(), 1.0e-15);
        }
      }
    }

    // Chained through a third frame, and inverted: both are properties of a
    // real transform tree and neither holds for a fixture.
    const auto base_to_tip = model.value().forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
    const auto tip_to_tcp = model.value().forward_kinematics(
      q, crane_model::Frame::Tip, crane_model::Frame::Tcp);
    const auto base_to_tcp = model.value().forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    const auto tcp_to_base = model.value().forward_kinematics(
      q, crane_model::Frame::Tcp, crane_model::Frame::MountingBase);
    ASSERT_TRUE(base_to_tip.ok());
    ASSERT_TRUE(tip_to_tcp.ok());
    ASSERT_TRUE(base_to_tcp.ok());
    ASSERT_TRUE(tcp_to_base.ok());
    const Eigen::Isometry3d chained = isometry(base_to_tip.value()) * isometry(tip_to_tcp.value());
    EXPECT_TRUE(chained.isApprox(isometry(base_to_tcp.value()), 1.0e-12));
    EXPECT_TRUE(
      isometry(tcp_to_base.value()).isApprox(isometry(base_to_tcp.value()).inverse(), 1.0e-12));
  }
}

TEST(CraneModelDescription, FramesTheDescriptionDoesNotCarryAreRefusedNotFaked)
{
  const auto rail = production_model(crane_model::Tool::Pzs100);
  const auto jaws = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(rail.ok());
  ASSERT_TRUE(jaws.ok());
  const auto q = valid_q();
  const auto base = crane_model::Frame::MountingBase;

  // `world` is the world-model boundary of contract §8 and no crane description
  // carries it, so both tools refuse it in either direction.
  for (const auto * model : {&rail.value(), &jaws.value()}) {
    EXPECT_EQ(
      model->forward_kinematics(q, base, crane_model::Frame::World).status().code,
      crane_model::ErrorCode::FrameUnavailable);
    EXPECT_EQ(
      model->forward_kinematics(q, crane_model::Frame::World, base).status().code,
      crane_model::ErrorCode::FrameUnavailable);
    EXPECT_EQ(
      model->jacobian(q, crane_model::Frame::World).status().code,
      crane_model::ErrorCode::FrameUnavailable);
  }

  // `tool_contact_point` is a 7040 link only. Same enum value, same call, two
  // different and both correct answers.
  EXPECT_EQ(
    rail.value().forward_kinematics(q, base, crane_model::Frame::ToolContact).status().code,
    crane_model::ErrorCode::FrameUnavailable);
  const auto contact =
    jaws.value().forward_kinematics(q, base, crane_model::Frame::ToolContact);
  ASSERT_TRUE(contact.ok());
  const auto tcp = jaws.value().forward_kinematics(q, base, crane_model::Frame::Tcp);
  ASSERT_TRUE(tcp.ok());
  EXPECT_GT((contact.value().position_m - tcp.value().position_m).norm(), 0.01);

  // A value that is not a Frame is an argument error, not a frame lookup.
  EXPECT_EQ(
    rail.value().forward_kinematics(q, base, static_cast<crane_model::Frame>(200)).status().code,
    crane_model::ErrorCode::InvalidArgument);
  auto non_finite = q;
  non_finite[5] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    rail.value().forward_kinematics(non_finite, base, crane_model::Frame::Tcp).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(
    rail.value().jacobian(non_finite, crane_model::Frame::Tcp).status().code,
    crane_model::ErrorCode::NonFiniteInput);
}

TEST(CraneModelDescription, EveryCoordinateMovesTheToolIncludingTheTelescopeTwice)
{
  const auto model = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(model.ok());
  const auto base = crane_model::Frame::MountingBase;
  const auto tcp = crane_model::Frame::Tcp;
  const auto reference = model.value().forward_kinematics(crane_model::Q::Zero(), base, tcp);
  ASSERT_TRUE(reference.ok());

  for (Eigen::Index index = 0; index < 8; ++index) {
    crane_model::Q q = crane_model::Q::Zero();
    q[index] = 0.2;
    const auto moved = model.value().forward_kinematics(q, base, tcp);
    ASSERT_TRUE(moved.ok()) << "coordinate " << index;
    const Eigen::Isometry3d delta =
      isometry(reference.value()).inverse() * isometry(moved.value());
    const double motion =
      delta.translation().norm() + rotation_vector(delta.linear()).norm();
    // q8 is the gripper, which does not move the tool centre point; the other
    // seven all do, the two passive ones (indices 4 and 5) included.
    if (index == 7) {
      EXPECT_LT(motion, 1.0e-15) << "coordinate " << index;
    } else {
      EXPECT_GT(motion, 1.0e-3) << "coordinate " << index;
    }
  }

  // robot_model §0: both telescope stages advance by the same q4, because the
  // description mimics `q5_small_telescope` onto `q4_big_telescope`. The tool
  // therefore travels twice the commanded extension, which is the doubling of
  // hydraulics §2.4 and is a property of the description, not a constant here.
  crane_model::Q extended = crane_model::Q::Zero();
  extended[3] = 0.4;
  const auto out = model.value().forward_kinematics(extended, base, tcp);
  ASSERT_TRUE(out.ok());
  EXPECT_NEAR(
    (out.value().position_m - reference.value().position_m).norm(), 2.0 * 0.4, 1.0e-12);
}

TEST(CraneModelDescription, JacobianIsTheDerivativeOfTheForwardKinematics)
{
  const double step = 1.0e-6;
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto q = valid_q();

    for (const auto frame : {crane_model::Frame::Tcp, crane_model::Frame::Boom,
        crane_model::Frame::Tip})
    {
      const auto jacobian = model.value().jacobian(q, frame);
      ASSERT_TRUE(jacobian.ok());
      EXPECT_EQ(jacobian.value().expressed_in, frame);
      EXPECT_EQ(jacobian.value().value.rows(), 6);
      EXPECT_EQ(jacobian.value().value.cols(), 8);

      const auto centre = model.value().forward_kinematics(
        q, crane_model::Frame::MountingBase, frame);
      ASSERT_TRUE(centre.ok());
      const Eigen::Isometry3d origin = isometry(centre.value());

      for (Eigen::Index index = 0; index < 8; ++index) {
        crane_model::Q forward = q;
        crane_model::Q backward = q;
        forward[index] += step;
        backward[index] -= step;
        const auto ahead = model.value().forward_kinematics(
          forward, crane_model::Frame::MountingBase, frame);
        const auto behind = model.value().forward_kinematics(
          backward, crane_model::Frame::MountingBase, frame);
        ASSERT_TRUE(ahead.ok());
        ASSERT_TRUE(behind.ok());
        const Eigen::Isometry3d plus = origin.inverse() * isometry(ahead.value());
        const Eigen::Isometry3d minus = origin.inverse() * isometry(behind.value());
        Eigen::Matrix<double, 6, 1> difference;
        difference.head<3>() = (plus.translation() - minus.translation()) / (2.0 * step);
        difference.tail<3>() =
          (rotation_vector(plus.linear()) - rotation_vector(minus.linear())) / (2.0 * step);
        EXPECT_LT(
          (difference - jacobian.value().value.col(index)).norm(), 1.0e-6)
          << "tool " << static_cast<int>(tool) << " frame " << static_cast<int>(frame)
          << " column " << index;
      }
    }
  }
}

TEST(CraneModelDescription, PassiveCoordinatesAreColumnsOfTheToolJacobian)
{
  const auto model = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(model.ok());
  const auto jacobian = model.value().jacobian(valid_q(), crane_model::Frame::Tcp);
  ASSERT_TRUE(jacobian.ok());

  // robot_model §2.1: the tool pose depends on all eight coordinates, the two
  // passive ones included, so no column of the tool Jacobian is zero except the
  // gripper's, which moves the jaws and not the tool centre point.
  for (Eigen::Index index = 0; index < 8; ++index) {
    if (index == 7) {
      EXPECT_LT(jacobian.value().value.col(index).norm(), 1.0e-15) << "column " << index;
    } else {
      EXPECT_GT(jacobian.value().value.col(index).norm(), 1.0e-3) << "column " << index;
    }
  }
  // Named explicitly, because a Jacobian that zeroed them would still look
  // plausible: q5 is the tip and q6 the tilt of the passive pendulum.
  EXPECT_GT(jacobian.value().value.col(4).norm(), 0.5);
  EXPECT_GT(jacobian.value().value.col(5).norm(), 0.5);

  // The root does not move, so its Jacobian is zero and says so without
  // pretending to be unavailable.
  const auto root = model.value().jacobian(valid_q(), crane_model::Frame::MountingBase);
  ASSERT_TRUE(root.ok());
  EXPECT_DOUBLE_EQ(root.value().value.norm(), 0.0);
}

// --- the constants and the description are cross-checked ---------------------
//
// Issue 005's notes closed on this gap: the hydraulic subset compiles §6.2's
// linkage dimensions in and the description was validated by substring, so a
// description carrying different lengths was accepted without complaint. The
// two tests below close it from both ends -- first that each placement the
// description declares equals the compiled-in number, then that the
// transmission the subset produces is still the one that geometry implies.
//
// Tolerance: 1e-5 m. Every placement agrees exactly except a_2, where §6.2
// rounds the description's 3.49288333 m to 3.49288 m.
namespace
{
constexpr double kLinkageTolerance = 1.0e-5;

void expect_offset(
  const parsed::Description& description, const std::string& parent, const std::string& child,
  const Eigen::Vector3d& expected)
{
  const Eigen::Vector3d actual = description.offset(parent, child);
  EXPECT_LT((actual - expected).norm(), kLinkageTolerance)
    << parent << " -> " << child << " is " << actual.transpose()
    << " in the description but " << expected.transpose() << " in the model";
}
}  // namespace

TEST(CraneModelDescription, LinkagePlacementsAgreeWithTheCompiledConstants)
{
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const parsed::Description description(tool);
    const reference::BoomGeometry boom;
    const reference::ArmGeometry arm;

    // Boom four-bar, wiki/hydraulics.md §2.2. p_S0 appears twice in the
    // description -- once as the cylinder joint and once as the suspension
    // frame that exists so the point is visible without sim_hydraulics.
    expect_offset(
      description, "K1_slewing_column", "K1_boom_cylinder_suspension",
      Eigen::Vector3d(boom.foot.x(), boom.foot.y(), 0.0));
    expect_offset(
      description, "K1_slewing_column", "boom_cylinder_mounting_on_slewing_column",
      Eigen::Vector3d(boom.foot.x(), boom.foot.y(), 0.0));
    expect_offset(
      description, "K1_slewing_column", "boom_cylinder_linkage_big_mounting_on_slewing_column",
      Eigen::Vector3d(boom.pivot.x(), boom.pivot.y(), 0.0));
    // a_2 and p_S2 separately, then their sum, which is what the model rotates
    // with q2.
    expect_offset(description, "theta2_boom_joint", "K2_boom", Eigen::Vector3d(3.49288, 0.0, 0.0));
    expect_offset(
      description, "K2_boom", "boom_cylinder_linkage_small_mounting_on_boom",
      Eigen::Vector3d(-3.039, boom.link.y(), 0.0));
    expect_offset(
      description, "theta2_boom_joint", "boom_cylinder_linkage_small_mounting_on_boom",
      Eigen::Vector3d(boom.link.x(), boom.link.y(), 0.0));

    // Arm cylinder, §2.3. p_S3z is the out-of-plane half of the pair of
    // cylinders; the right one carries +0.224.
    expect_offset(
      description, "K2_boom", "K2_arm_cylinder_suspension",
      Eigen::Vector3d(arm.foot.x(), arm.foot.y(), 0.0));
    expect_offset(
      description, "K2_boom", "arm_cylinder_mounting_on_boom_right",
      Eigen::Vector3d(arm.foot.x(), arm.foot.y(), 0.224));
    expect_offset(
      description, "theta3_arm_joint", "K3_arm", Eigen::Vector3d(-0.3925, 0.0, 0.0));

    if (tool != crane_model::Tool::Epsilon7040) {
      continue;
    }
    // 7040 jaw four-bar, §2.6. a_9 and a_11 are the y and a_10 and a_12 the x
    // coordinate of their dh transform, exactly as the deployed model reads
    // them.
    const reference::JawGeometry jaw;
    expect_offset(
      description, "K8_tool_center_point", "K9", Eigen::Vector3d(0.0, -jaw.outer_pivot, 0.0));
    expect_offset(
      description, "K8_tool_center_point", "K11", Eigen::Vector3d(0.0, jaw.inner_pivot, 0.0));
    expect_offset(
      description, "theta10_outer_jaw_joint", "K10_outer_jaw",
      Eigen::Vector3d(jaw.outer_arm, 0.0, 0.0));
    expect_offset(
      description, "theta12_inner_jaw_joint", "K12_inner_jaw",
      Eigen::Vector3d(jaw.inner_arm, 0.0, 0.0));
    // The cylinder's barrel end. Its z is -0.015 m, which the planar four-bar
    // of §2.6 drops on purpose; only x and y are the model's p_S7.
    EXPECT_LT(
      (description.planar("K10_outer_jaw", "pincer_cylinder_mounting_outer_jaw_joint") -
      jaw.outer_pin).norm(), kLinkageTolerance);
  }
}

TEST(CraneModelDescription, CylinderTransmissionFollowsTheDescriptionsGeometry)
{
  const parsed::Description rail(crane_model::Tool::Pzs100);
  const parsed::Description jaws(crane_model::Tool::Epsilon7040);

  // Rebuild the two crane linkages out of the description's own placements.
  // What is left as a literal is exactly what the description does not carry:
  // the drawbar and pushbar lengths, whose loop the description closes only in
  // Gazebo, and the arm cylinder's rod-end attachment p_S4, which the Gazebo
  // closure places at the K3_arm origin rather than at its real pin.
  reference::BoomGeometry boom;
  boom.foot = rail.planar("K1_slewing_column", "K1_boom_cylinder_suspension");
  boom.pivot =
    rail.planar("K1_slewing_column", "boom_cylinder_linkage_big_mounting_on_slewing_column");
  boom.link =
    rail.planar("theta2_boom_joint", "boom_cylinder_linkage_small_mounting_on_boom");

  reference::ArmGeometry arm;
  const Eigen::Vector3d arm_foot = rail.offset("K2_boom", "arm_cylinder_mounting_on_boom_right");
  arm.foot = arm_foot.head<2>();
  arm.link.x() = rail.offset("theta3_arm_joint", "K3_arm").x() + 0.274489;  // a_3 + p_S4x
  arm.lateral = 0.224 - arm_foot.z();  // p_S4y - p_S3z

  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  // The boom is the one axis with a residual, and all of it is §6.2's rounding
  // of a_2: 3.49288 m against the description's 3.49288333 m moves the ratio by
  // up to 5e-6. That is the 1e-5 m placement tolerance above, carried through
  // the four-bar. Every other axis agrees to the difference quotient's own
  // error, so they are held to 1e-8.
  constexpr double kBoomRatioTolerance = 1.0e-5;
  constexpr double kExactRatioTolerance = 1.0e-8;
  for (const double q2 : {-1.0, -0.4, 0.0, 0.8, 1.5}) {
    const auto jacobian = model.value().cylinder_jacobian(linkage_configuration(q2, 0.0));
    ASSERT_TRUE(jacobian.ok()) << "q2 = " << q2;
    EXPECT_NEAR(
      jacobian.value()(1, 1),
      reference::transmission_ratio(
        [&boom](double angle) {return reference::boom_stroke_from(boom, angle);}, q2),
      kBoomRatioTolerance) << "q2 = " << q2;
  }
  for (const double q3 : {-0.9, 0.0, 0.7, 1.5}) {
    const auto jacobian = model.value().cylinder_jacobian(linkage_configuration(0.0, q3));
    ASSERT_TRUE(jacobian.ok()) << "q3 = " << q3;
    EXPECT_NEAR(
      jacobian.value()(2, 2),
      reference::transmission_ratio(
        [&arm](double angle) {return reference::arm_stroke_from(arm, angle);}, q3),
      kExactRatioTolerance) << "q3 = " << q3;
  }

  // The 7040 jaw. Its pivots, arms and barrel-end pin come from the description;
  // the rod-end pin p_S8 and the fitted mirror law do not exist in any file of
  // this workspace and stay the ported literals of issue 014.
  reference::JawGeometry jaw;
  jaw.outer_pivot = -jaws.offset("K8_tool_center_point", "K9").y();
  jaw.inner_pivot = jaws.offset("K8_tool_center_point", "K11").y();
  jaw.outer_arm = jaws.offset("theta10_outer_jaw_joint", "K10_outer_jaw").x();
  jaw.inner_arm = jaws.offset("theta12_inner_jaw_joint", "K12_inner_jaw").x();
  jaw.outer_pin = jaws.planar("K10_outer_jaw", "pincer_cylinder_mounting_outer_jaw_joint");

  const auto jaw_model = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(jaw_model.ok());
  for (const double q8 : {0.5, 1.0, 2.0, 3.0}) {
    const auto jacobian = jaw_model.value().cylinder_jacobian(jaw_configuration(q8));
    ASSERT_TRUE(jacobian.ok()) << "q8 = " << q8;
    EXPECT_NEAR(
      jacobian.value()(5, 5),
      reference::transmission_ratio(
        [&jaw](double angle) {return reference::jaw_stroke_from(jaw, angle);}, q8),
      kExactRatioTolerance) << "q8 = " << q8;
  }
}

// --- the rigid-body dynamics -------------------------------------------------
//
// wiki/robot_model.md §1 and §3. The oracle here is deliberately not a second
// mass matrix: it is the *energy* of the same description, evaluated by
// Pinocchio algorithms the model under test never calls. T(q, dq) is what
// (1/2) dq^T M dq has to be, U(q) is what the gravity part of h has to be the
// gradient of, and the minimum of U over the two passive coordinates is the
// equilibrium of §2.3 without any torque algorithm being asked.

namespace
{
namespace energy
{

// The same description, parsed a second time, with the canonical map and the
// `<mimic>` couplings read out of urdfdom rather than out of the model under
// test.
class Reference
{
public:
  Reference(crane_model::Tool tool, const std::array<std::string, 8>& names)
  : model_(build(tool)), data_(model_), neutral_(pinocchio::neutral(model_))
  {
    for (std::size_t index = 0; index < 8; ++index) {
      map_[index].push_back(slot(names[index], 1.0, 0.0));
    }
    const ::urdf::ModelInterfaceSharedPtr tree = ::urdf::parseURDF(description_for(tool));
    EXPECT_TRUE(static_cast<bool>(tree));
    for (const auto& entry : tree->joints_) {
      const auto& joint = entry.second;
      if (!joint || !joint->mimic) {
        continue;
      }
      const auto source = std::find(names.begin(), names.end(), joint->mimic->joint_name);
      if (source == names.end() || !model_.existJointName(joint->name)) {
        continue;
      }
      map_[static_cast<std::size_t>(std::distance(names.begin(), source))].push_back(
        slot(joint->name, joint->mimic->multiplier, joint->mimic->offset));
    }
  }

  void set_gravity(const Eigen::Vector3d& gravity) { model_.gravity.linear() = gravity; }

  // The payload of robot_model §5, appended the other way round: as a body on
  // the joint that carries K8_rotator_lower_part, at that link's own placement.
  void add_payload(const crane_model::Payload& payload)
  {
    const pinocchio::FrameIndex frame = model_.getFrameId("K8_rotator_lower_part");
    model_.appendBodyToJoint(
      model_.frames[frame].parentJoint,
      pinocchio::Inertia(
        payload.mass_kg, payload.center_of_mass_k8_m, payload.inertia_k8_kg_m2),
      model_.frames[frame].placement);
    data_ = pinocchio::Data(model_);
  }

  double kinetic(const crane_model::Q& q, const crane_model::DQ& dq)
  {
    return pinocchio::computeKineticEnergy(model_, data_, configuration(q), velocity(dq));
  }

  double potential(const crane_model::Q& q)
  {
    return pinocchio::computePotentialEnergy(model_, data_, configuration(q));
  }

private:
  struct Slot
  {
    Eigen::Index idx_q{0};
    Eigen::Index idx_v{0};
    bool unbounded{false};
    double multiplier{1.0};
    double offset{0.0};
  };

  static pinocchio::Model build(crane_model::Tool tool)
  {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(description_for(tool), model);
    return model;
  }

  Slot slot(const std::string& name, double multiplier, double offset) const
  {
    EXPECT_TRUE(model_.existJointName(name)) << "no joint " << name;
    const auto& joint = model_.joints[model_.getJointId(name)];
    Slot value;
    value.idx_q = static_cast<Eigen::Index>(joint.idx_q());
    value.idx_v = static_cast<Eigen::Index>(joint.idx_v());
    value.unbounded = joint.nq() == 2;  // the `continuous` rotator, stored as (cos, sin)
    value.multiplier = multiplier;
    value.offset = offset;
    return value;
  }

  Eigen::VectorXd configuration(const crane_model::Q& q) const
  {
    Eigen::VectorXd value = neutral_;
    for (std::size_t index = 0; index < 8; ++index) {
      for (const Slot& entry : map_[index]) {
        const double coordinate =
          entry.multiplier * q[static_cast<Eigen::Index>(index)] + entry.offset;
        if (entry.unbounded) {
          value[entry.idx_q] = std::cos(coordinate);
          value[entry.idx_q + 1] = std::sin(coordinate);
        } else {
          value[entry.idx_q] = coordinate;
        }
      }
    }
    return value;
  }

  Eigen::VectorXd velocity(const crane_model::DQ& dq) const
  {
    Eigen::VectorXd value = Eigen::VectorXd::Zero(model_.nv);
    for (std::size_t index = 0; index < 8; ++index) {
      for (const Slot& entry : map_[index]) {
        value[entry.idx_v] = entry.multiplier * dq[static_cast<Eigen::Index>(index)];
      }
    }
    return value;
  }

  pinocchio::Model model_;
  pinocchio::Data data_;
  Eigen::VectorXd neutral_;
  std::array<std::vector<Slot>, 8> map_;
};

// Central differences of U. The step is the cube root of the double-precision
// resolution of an energy of order 1e5 J, which is where rounding and truncation
// meet; the residual error is under 1e-5 N m on torques of order 1e4.
constexpr double kEnergyStep = 5.0e-6;

crane_model::DQ gravity_gradient(Reference& reference, const crane_model::Q& q)
{
  crane_model::DQ gradient;
  for (Eigen::Index index = 0; index < 8; ++index) {
    crane_model::Q forward = q;
    crane_model::Q backward = q;
    forward[index] += kEnergyStep;
    backward[index] -= kEnergyStep;
    gradient[index] =
      (reference.potential(forward) - reference.potential(backward)) / (2.0 * kEnergyStep);
  }
  return gradient;
}

// The passive equilibrium of robot_model §2.3, located as the minimum of U over
// the two passive coordinates: a coarse sweep to pick the well, then Newton on
// the numerical gradient to sit down in it. The Hessian uses a much larger step
// than the gradient because a second difference amplifies rounding by 1/h^2 and
// only sets the search *direction* -- the fixed point is where the gradient
// vanishes, so it is the gradient's accuracy that ends up in the residual.
crane_model::QU equilibrium(Reference& reference, const crane_model::Q& q_a)
{
  const auto potential_at = [&reference, &q_a](const crane_model::QU& passive) {
      crane_model::Q probe = q_a;
      probe[4] = passive[0];
      probe[5] = passive[1];
      return reference.potential(probe);
    };

  constexpr int kSweep = 48;
  crane_model::QU best;
  best << 0.0, M_PI_2;
  double lowest = std::numeric_limits<double>::infinity();
  for (int tip = 0; tip <= kSweep; ++tip) {
    for (int tilt = 0; tilt <= kSweep; ++tilt) {
      crane_model::QU candidate;
      candidate << -M_PI_2 + M_PI * tip / kSweep, 0.4 + 2.0 * tilt / kSweep;
      const double value = potential_at(candidate);
      if (value < lowest) {
        lowest = value;
        best = candidate;
      }
    }
  }

  constexpr double kCurvatureStep = 1.0e-3;
  for (int iteration = 0; iteration < 50; ++iteration) {
    Eigen::Vector2d gradient;
    Eigen::Matrix2d curvature;
    const double centre = potential_at(best);
    for (int axis = 0; axis < 2; ++axis) {
      crane_model::QU forward = best;
      crane_model::QU backward = best;
      forward[axis] += kEnergyStep;
      backward[axis] -= kEnergyStep;
      gradient[axis] = (potential_at(forward) - potential_at(backward)) / (2.0 * kEnergyStep);
      forward[axis] = best[axis] + kCurvatureStep;
      backward[axis] = best[axis] - kCurvatureStep;
      curvature(axis, axis) = (potential_at(forward) - 2.0 * centre + potential_at(backward)) /
        (kCurvatureStep * kCurvatureStep);
    }
    crane_model::QU corners = best;
    corners[0] += kCurvatureStep;
    corners[1] += kCurvatureStep;
    const double plus_plus = potential_at(corners);
    corners[1] = best[1] - kCurvatureStep;
    const double plus_minus = potential_at(corners);
    corners[0] = best[0] - kCurvatureStep;
    const double minus_minus = potential_at(corners);
    corners[1] = best[1] + kCurvatureStep;
    const double minus_plus = potential_at(corners);
    curvature(0, 1) = (plus_plus - plus_minus - minus_plus + minus_minus) /
      (4.0 * kCurvatureStep * kCurvatureStep);
    curvature(1, 0) = curvature(0, 1);

    const Eigen::Vector2d step = curvature.ldlt().solve(gradient);
    best -= step;
    if (step.norm() < 1.0e-13) {
      break;
    }
  }
  return best;
}

}  // namespace energy

// A crane holding the tool out at radius, well away from any joint zero, so
// every coupling term in M and h is populated. The two passive coordinates are
// deliberately off their equilibrium: the invariant tests put them back.
crane_model::Q loaded_configuration()
{
  crane_model::Q q;
  q << 0.4, 0.35, 0.9, 0.8, 0.12, 1.4, 0.6, 0.3;
  return q;
}

crane_model::DQ loaded_velocity()
{
  crane_model::DQ dq;
  dq << 0.30, 0.12, -0.20, 0.08, 0.25, -0.18, 0.40, 0.05;
  return dq;
}

// A payload that is a payload: the concrete block of the PZS100, offset from the
// tool axis and carrying a real tensor about its own centre of mass.
crane_model::Payload block_payload()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 220.0;
  payload.center_of_mass_k8_m = Eigen::Vector3d(0.04, -0.03, 0.55);
  payload.inertia_k8_kg_m2 = Eigen::Vector3d(18.0, 22.0, 9.0).asDiagonal();
  return payload;
}

// The empty gripper: contract §4 spells it as a *declared* payload of zero mass,
// which is not the same statement as `valid == false`.
crane_model::Payload empty_gripper()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 0.0;
  payload.center_of_mass_k8_m.setZero();
  payload.inertia_k8_kg_m2.setZero();
  return payload;
}

crane_model::Result<crane_model::Model> model_with_gravity(
  crane_model::Tool tool, const Eigen::Vector3d& gravity)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description_for(tool);
  config.tool = tool;
  config.gravity_m_s2 = gravity;
  return crane_model::Model::create(config);
}

const std::array<Eigen::Index, 6> kActuatedRows{{0, 1, 2, 3, 6, 7}};
const std::array<Eigen::Index, 2> kPassiveRows{{4, 5}};

crane_model::QU passive_rows(const crane_model::DQ& vector)
{
  crane_model::QU rows;
  rows << vector[kPassiveRows[0]], vector[kPassiveRows[1]];
  return rows;
}

crane_model::Vector6 actuated_rows(const crane_model::DQ& vector)
{
  crane_model::Vector6 rows;
  for (Eigen::Index index = 0; index < 6; ++index) {
    rows[index] = vector[kActuatedRows[static_cast<std::size_t>(index)]];
  }
  return rows;
}

crane_model::DQ full_acceleration(const crane_model::Input& ddq_a, const crane_model::QU& ddq_u)
{
  crane_model::DQ ddq = crane_model::DQ::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    ddq[kActuatedRows[static_cast<std::size_t>(index)]] = ddq_a[index];
  }
  ddq[kPassiveRows[0]] = ddq_u[0];
  ddq[kPassiveRows[1]] = ddq_u[1];
  return ddq;
}

// tau_a for one ddq_a, taken out of the *full* dynamics and nothing else: the
// two passive rows of inverse_dynamics are affine in ddq_u, so three calls
// determine them exactly, a fourth evaluates the torque at the ddq_u that makes
// them vanish, and the actuated rows of that call are tau_a by definition. No
// Schur complement appears anywhere in here, which is what makes it a check on
// reduced_actuated_dynamics rather than a copy of it.
crane_model::Vector6 consistent_actuated_torque(
  const crane_model::Model& model, const crane_model::Q& q, const crane_model::DQ& dq,
  const crane_model::Input& ddq_a, const crane_model::Payload& payload,
  crane_model::QU * ddq_u_out = nullptr)
{
  const auto torque_at = [&](const crane_model::QU& ddq_u) {
      const auto result = model.inverse_dynamics(q, dq, full_acceleration(ddq_a, ddq_u), payload);
      EXPECT_TRUE(result.ok()) << result.status().message;
      return result.ok() ? result.value() : crane_model::DQ::Zero().eval();
    };

  const crane_model::QU zero = crane_model::QU::Zero();
  const crane_model::QU offset = passive_rows(torque_at(zero));
  Eigen::Matrix2d passive_mass;
  for (Eigen::Index column = 0; column < 2; ++column) {
    crane_model::QU probe = crane_model::QU::Zero();
    probe[column] = 1.0;
    passive_mass.col(column) = passive_rows(torque_at(probe)) - offset;
  }

  const crane_model::QU ddq_u = passive_mass.partialPivLu().solve(-offset);
  if (ddq_u_out != nullptr) {
    *ddq_u_out = ddq_u;
  }
  const crane_model::DQ tau = torque_at(ddq_u);
  EXPECT_LT(passive_rows(tau).norm(), 1.0e-6) << "the consistent acceleration left a passive row";
  return actuated_rows(tau);
}

}  // namespace

TEST(CraneModelDynamics, MassMatrixIsTheKineticEnergyOfTheSameDescription)
{
  // (1/2) dq^T M dq is the kinetic energy of the description, so M is checked
  // against an algorithm that never forms a mass matrix. This is also what
  // catches the two `<mimic>` joints: the second telescope stage and the
  // PZS100's mirrored rail carry inertia, and their rows have to land on the
  // coordinate that drives them.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    energy::Reference bare(tool, model.value().urdf_joint_names());
    energy::Reference loaded(tool, model.value().urdf_joint_names());
    loaded.add_payload(block_payload());

    const auto q = loaded_configuration();
    for (const double scale : {0.3, 1.0, 2.5}) {
      const crane_model::DQ dq = scale * loaded_velocity();
      for (const auto& carried : {empty_gripper(), block_payload()}) {
        const auto dynamics = model.value().full_dynamics(q, dq, carried);
        ASSERT_TRUE(dynamics.ok()) << dynamics.status().message;
        const double expected =
          carried.mass_kg > 0.0 ? loaded.kinetic(q, dq) : bare.kinetic(q, dq);
        EXPECT_NEAR(0.5 * dq.dot(dynamics.value().mass * dq), expected, 1.0e-8 * expected);

        // M is symmetric and positive definite, which is what makes the
        // Cholesky of robot_model §3.1 and the Schur complement of §3.4 legal.
        EXPECT_LT(
          (dynamics.value().mass - dynamics.value().mass.transpose()).norm(),
          1.0e-12 * dynamics.value().mass.norm());
        const Eigen::LLT<crane_model::FullMass> factorisation(dynamics.value().mass);
        EXPECT_EQ(factorisation.info(), Eigen::Success);
      }
    }
  }
}

TEST(CraneModelDynamics, BiasIsTheGradientOfThePotentialEnergyAtRest)
{
  // At rest h is the gravity vector alone, and the gravity vector is the
  // gradient of the potential energy -- again an energy, not a torque algorithm.
  // The tolerance is the central difference's own error, not the model's.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    energy::Reference reference(tool, model.value().urdf_joint_names());
    reference.add_payload(block_payload());

    const auto q = loaded_configuration();
    const auto dynamics = model.value().full_dynamics(q, crane_model::DQ::Zero(), block_payload());
    ASSERT_TRUE(dynamics.ok()) << dynamics.status().message;
    const crane_model::DQ expected = energy::gravity_gradient(reference, q);
    for (Eigen::Index index = 0; index < 8; ++index) {
      EXPECT_NEAR(dynamics.value().bias[index], expected[index], 1.0e-4) << "row " << index;
    }
    EXPECT_GT(expected.norm(), 1.0e3) << "the fixture is not carrying the crane's weight";
  }
}

TEST(CraneModelDynamics, GravityComesFromTheConfigurationNotFromACompiledConstant)
{
  // ModelConfig::gravity_m_s2 is the only source. At rest the bias is linear in
  // it, so half a gravity is exactly half a bias and a sideways gravity is a
  // different vector entirely -- neither of which a compiled-in 9.81 could do.
  const Eigen::Vector3d earth(0.0, 0.0, -9.81);
  const auto standard = model_with_gravity(crane_model::Tool::Pzs100, earth);
  const auto halved = model_with_gravity(crane_model::Tool::Pzs100, 0.5 * earth);
  const auto sideways = model_with_gravity(crane_model::Tool::Pzs100, Eigen::Vector3d(0, -9.81, 0));
  ASSERT_TRUE(standard.ok());
  ASSERT_TRUE(halved.ok());
  ASSERT_TRUE(sideways.ok());

  const auto q = loaded_configuration();
  const auto rest = crane_model::DQ::Zero();
  const auto payload = block_payload();
  const auto full = standard.value().full_dynamics(q, rest, payload);
  const auto half = halved.value().full_dynamics(q, rest, payload);
  const auto lateral = sideways.value().full_dynamics(q, rest, payload);
  ASSERT_TRUE(full.ok());
  ASSERT_TRUE(half.ok());
  ASSERT_TRUE(lateral.ok());

  EXPECT_LT(
    (half.value().bias - 0.5 * full.value().bias).norm(), 1.0e-9 * full.value().bias.norm());
  EXPECT_GT((lateral.value().bias - full.value().bias).norm(), 0.1 * full.value().bias.norm());
  // The mass matrix is not a function of gravity, and must not become one.
  EXPECT_LT((half.value().mass - full.value().mass).norm(), 1.0e-12 * full.value().mass.norm());
}

TEST(CraneModelDynamics, DampingIsTheDescriptionsExceptTheTelescope)
{
  // h is quadratic in dq and the gravity term does not depend on it at all, so
  // the odd part of the bias in dq is exactly D dq. That extracts D through the
  // public API without the test knowing how the model stores it.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    const auto q = loaded_configuration();
    const auto dq = loaded_velocity();
    const auto payload = block_payload();
    const auto forward = model.value().full_dynamics(q, dq, payload);
    const auto backward = model.value().full_dynamics(q, -dq, payload);
    ASSERT_TRUE(forward.ok());
    ASSERT_TRUE(backward.ok());
    const crane_model::DQ damping_term = 0.5 * (forward.value().bias - backward.value().bias);

    const ::urdf::ModelInterfaceSharedPtr tree = ::urdf::parseURDF(description_for(tool));
    ASSERT_TRUE(static_cast<bool>(tree));
    const auto& names = model.value().urdf_joint_names();
    for (Eigen::Index index = 0; index < 8; ++index) {
      const auto joint = tree->getJoint(names[static_cast<std::size_t>(index)]);
      ASSERT_TRUE(static_cast<bool>(joint));
      ASSERT_TRUE(static_cast<bool>(joint->dynamics));
      // wiki/implementation/parameters.md §5: every axis takes the
      // description's damping except q4, whose entry that note calls a
      // simulation stability hack that must not enter the model.
      const double expected = index == 3 ? 0.0 : joint->dynamics->damping * dq[index];
      EXPECT_NEAR(damping_term[index], expected, 1.0e-6 * std::max(1.0, std::abs(expected)))
        << "row " << index;
    }
    // And the carve-out is a real one: the description does carry a telescope
    // damping, and it is large enough that leaving it in would show.
    EXPECT_GT(tree->getJoint(names[3])->dynamics->damping, 1.0e4);
  }
}

TEST(CraneModelDynamics, CentrifugalCouplingFromSlewingReachesThePassiveRows)
{
  // robot_model §3.1 warns against dropping the Coriolis terms because the
  // centrifugal coupling from slewing into the sway grows with radius and rate.
  // The coupling is exactly quadratic in the slewing rate, and the passive rows
  // see no damping at all here because dq_u is zero, so the whole increment is
  // the term the warning is about.
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = loaded_configuration();
  const auto payload = block_payload();

  const auto at_rate = [&](double rate) {
      crane_model::DQ dq = crane_model::DQ::Zero();
      dq[0] = rate;
      const auto dynamics = model.value().full_dynamics(q, dq, payload);
      EXPECT_TRUE(dynamics.ok());
      return dynamics.ok() ? passive_rows(dynamics.value().bias) :
             crane_model::QU::Zero().eval();
    };

  const crane_model::QU still = at_rate(0.0);
  const crane_model::QU single = at_rate(0.4) - still;
  const crane_model::QU doubled = at_rate(0.8) - still;
  EXPECT_GT(single.norm(), 1.0) << "slewing does not reach the passive rows at all";
  EXPECT_LT((doubled - 4.0 * single).norm(), 1.0e-9 * doubled.norm());
}

TEST(CraneModelDynamics, PassiveRowsVanishAtTheEquilibrium)
{
  // The invariant of contract §7 and robot_model §3.3. The equilibrium is
  // located by minimising the potential energy of the same description, which is
  // a criterion the model under test never evaluates, so this is a check on the
  // whole chain -- joint map, mimics, payload attachment, gravity -- and not a
  // tautology.
  //
  // The tolerance is the search's, not the model's. Near the minimum the passive
  // gravity torque is the curvature of U times the offset of the located point,
  // and that offset is set by the noise of a central difference on an energy of
  // order 1e5 J, which predicts a residual of order 1e-6 N m. Measured on these
  // two descriptions it is 3e-7 to 7e-7 N m, against a restoring torque of 32 to
  // 80 N m one degree off the same equilibrium -- so the rows are zero to about
  // 1e-8 of the scale they are zero against. 1e-4 N m is two decades of margin
  // over the measured value and is the approved tolerance.
  constexpr double kPassiveResidual = 1.0e-4;
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    for (const auto& payload : {empty_gripper(), block_payload()}) {
      energy::Reference reference(tool, model.value().urdf_joint_names());
      reference.add_payload(payload);

      crane_model::Q q = loaded_configuration();
      const crane_model::QU settled = energy::equilibrium(reference, q);
      q[4] = settled[0];
      q[5] = settled[1];

      const auto tau = model.value().inverse_dynamics(
        q, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
      ASSERT_TRUE(tau.ok()) << tau.status().message;
      EXPECT_LT(passive_rows(tau.value()).norm(), kPassiveResidual)
        << "tau_u = " << passive_rows(tau.value()).transpose();

      // Not vacuously zero: a degree away from the equilibrium the same rows
      // carry the restoring torque the invariant is zero against, which is
      // eight decades above the residual above.
      crane_model::Q tilted = q;
      tilted[5] += M_PI / 180.0;
      const auto off = model.value().inverse_dynamics(
        tilted, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
      ASSERT_TRUE(off.ok());
      EXPECT_GT(passive_rows(off.value()).norm(), 1.0e1);

      // And the actuated rows are not zero at the same point: holding the crane
      // up against gravity is what they are for.
      EXPECT_GT(actuated_rows(tau.value()).norm(), 1.0e3);
    }
  }
}

TEST(CraneModelDynamics, AnOffsetGraspMovesThePassiveEquilibrium)
{
  // robot_model §5.1. The passive joints hold the tool so the combined centre of
  // mass hangs under the pivot, so a grasp that is not centred tilts the tool at
  // rest. This is the property issue 033's payload estimate is observable
  // through, which is why it is asserted here rather than assumed there.
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto& names = model.value().urdf_joint_names();

  const auto settled_with = [&](const Eigen::Vector3d& centre) {
      crane_model::Payload payload = block_payload();
      payload.center_of_mass_k8_m = centre;
      energy::Reference reference(crane_model::Tool::Pzs100, names);
      reference.add_payload(payload);
      const crane_model::QU settled = energy::equilibrium(reference, loaded_configuration());

      // The model has to agree that this is an equilibrium, which is what ties
      // the shift to the payload attachment rather than to the reference alone.
      crane_model::Q q = loaded_configuration();
      q[4] = settled[0];
      q[5] = settled[1];
      const auto tau = model.value().inverse_dynamics(
        q, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
      EXPECT_TRUE(tau.ok());
      EXPECT_LT(passive_rows(tau.value()).norm(), 1.0e-4);
      return settled;
    };

  const Eigen::Vector3d centred(0.0, 0.0, 0.55);
  const crane_model::QU nominal = settled_with(centred);
  const crane_model::QU small = settled_with(centred + Eigen::Vector3d(0.05, 0.0, 0.0));
  const crane_model::QU large = settled_with(centred + Eigen::Vector3d(0.10, 0.0, 0.0));
  const crane_model::QU across = settled_with(centred + Eigen::Vector3d(0.0, 0.05, 0.0));

  // 5 cm of offset is worth degrees, not arcseconds: §5.1 puts it at
  // arctan(d / l_tool) ~ 2.9 deg for a metre of effective pendulum, and calls
  // that the whole placement budget.
  const double shift = (small - nominal).norm();
  EXPECT_GT(shift, 1.0 * M_PI / 180.0) << "an offset grasp barely moved the equilibrium";
  EXPECT_LT(shift, 10.0 * M_PI / 180.0);
  // Twice the offset is close to twice the tilt, and the two axes move
  // differently: the double hinge is not a single pendulum.
  EXPECT_NEAR((large - nominal).norm() / shift, 2.0, 0.2);
  EXPECT_GT((across - small).norm(), 0.5 * shift);
}

// --- the passive equilibrium -------------------------------------------------
//
// wiki/robot_model.md §2.3: q_eq(q_a, payload) solves h_u(q_a, q_eq, 0) = 0, and
// for a two-joint pendulum that is the tool hanging along gravity. The two tests
// above already located it from the potential energy and showed the invariant
// holding there; these ask the same of the API call that now returns it.

namespace
{

// A configuration built from an actuated vector and the passive pose the model
// returned for it -- the round trip the invariant of contract §7 is stated on.
crane_model::Q settled_configuration(const crane_model::QA& q_a, const crane_model::QU& q_u)
{
  crane_model::Q q = crane_model::Q::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    q[kActuatedRows[static_cast<std::size_t>(index)]] = q_a[index];
  }
  q[kPassiveRows[0]] = q_u[0];
  q[kPassiveRows[1]] = q_u[1];
  return q;
}

}  // namespace

TEST(CraneModelEquilibrium, ThePassiveRowsVanishAtTheReturnedPose)
{
  // The acceptance test of the whole call, and it is against the *dynamics* and
  // not against the solver: the returned q_u goes back in through
  // inverse_dynamics with zero passive velocity and acceleration, and the two
  // passive rows have to be zero. That is the invariant contract §7 states and
  // the one issue 031 established -- the same assertion
  // PassiveRowsVanishAtTheEquilibrium makes about a pose found by minimising the
  // potential energy, now made about the pose this API returns instead.
  //
  // dq is zero in all eight rows and not only the passive two. robot_model §3.1
  // is explicit that the centrifugal coupling from slewing reaches the passive
  // rows, so a nonzero actuated rate would put a real Coriolis term there and the
  // invariant would not hold -- the equilibrium condition is stated at rest.
  //
  // 1e-6 N m against a restoring torque of order 1e2 N m one degree away. The
  // solve's own bound is 1e-8 N m, stated in the README; the margin here is for
  // the difference between a residual the model drove to zero and the same
  // quantity recomputed through the public call.
  constexpr double kPassiveResidual = 1.0e-6;
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    const crane_model::QA q_a = actuated_rows(loaded_configuration());
    // With a payload and without one: contract §4's declared zero-mass gripper is
    // a payload, and the two give different equilibria.
    for (const auto& payload : {empty_gripper(), block_payload()}) {
      const auto settled = model.value().passive_equilibrium(q_a, payload);
      ASSERT_TRUE(settled.ok()) << settled.status().message;
      const crane_model::Q q = settled_configuration(q_a, settled.value());

      const auto tau = model.value().inverse_dynamics(
        q, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
      ASSERT_TRUE(tau.ok()) << tau.status().message;
      EXPECT_LT(passive_rows(tau.value()).norm(), kPassiveResidual)
        << "tau_u = " << passive_rows(tau.value()).transpose();

      // Not vacuously zero, exactly as at the energy minimum: a degree off the
      // returned pose the same two rows carry the restoring torque, and the
      // actuated rows are busy holding the crane up.
      crane_model::Q tilted = q;
      tilted[kPassiveRows[1]] += M_PI / 180.0;
      const auto off = model.value().inverse_dynamics(
        tilted, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
      ASSERT_TRUE(off.ok());
      EXPECT_GT(passive_rows(off.value()).norm(), 1.0e1);
      EXPECT_GT(actuated_rows(tau.value()).norm(), 1.0e3);
    }
  }
}

TEST(CraneModelEquilibrium, TheReturnedPoseIsTheOneTheToolHangsAt)
{
  // A critical point of the potential is not yet an equilibrium the tool settles
  // at: the two-hinge pendulum has four of them and two are the tool standing up.
  // They satisfy h_u = 0 exactly as well, so the test that the returned pose is
  // the hanging one is against the *minimum* of the same potential energy the
  // model never evaluates -- the oracle of PassiveRowsVanishAtTheEquilibrium,
  // asked for the same q_a.
  //
  // 1e-6 rad is the search's accuracy, not the model's: the oracle sits down in
  // the well by Newton on a central difference of an energy of order 1e5 J.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    const crane_model::QA q_a = actuated_rows(loaded_configuration());
    for (const auto& payload : {empty_gripper(), block_payload()}) {
      energy::Reference reference(tool, model.value().urdf_joint_names());
      reference.add_payload(payload);
      const crane_model::QU expected =
        energy::equilibrium(reference, settled_configuration(q_a, crane_model::QU::Zero()));

      const auto settled = model.value().passive_equilibrium(q_a, payload);
      ASSERT_TRUE(settled.ok()) << settled.status().message;
      EXPECT_LT((settled.value() - expected).norm(), 1.0e-6)
        << "q_eq = " << settled.value().transpose() << ", energy minimum = "
        << expected.transpose();

      // And it is a pose the description lets the double hinge reach. Both
      // machine descriptions bound the tilt to a quarter turn around pi/2, which
      // is what makes "hanging" a statement about a reachable pose at all.
      EXPECT_GT(settled.value()[0], -M_PI_2);
      EXPECT_LT(settled.value()[0], M_PI_2);
      EXPECT_GT(settled.value()[1], 0.785398);
      EXPECT_LT(settled.value()[1], 2.35619);
    }
  }
}

TEST(CraneModelEquilibrium, AnOffsetGraspCarriesTheLoadBackUnderThePivot)
{
  // robot_model §5.1, and the direction it predicts rather than only the
  // magnitude: the passive joints hold the tool so that the payload hangs under
  // the passive pivot, so a grasp offset in K8 does not displace the load -- it
  // tilts the tool until the offset is taken up. This is the property issue 033's
  // payload estimate inverts, which is why it is asserted here and not there.
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const crane_model::QA q_a = actuated_rows(loaded_configuration());
  const Eigen::Vector3d centred(0.0, 0.0, 0.55);
  const double offset = 0.05;

  // Where the payload's centre of mass ends up, horizontally, relative to the
  // pivot the two passive joints hang the tool from -- K5_inner_telescope, the
  // parent of the tip hinge. Both come out of the public forward kinematics.
  const auto horizontal_offset =
    [&](const crane_model::QU& q_u, const Eigen::Vector3d& centre_of_mass) {
      const crane_model::Q q = settled_configuration(q_a, q_u);
      const auto pivot = model.value().forward_kinematics(
        q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
      const auto tool = model.value().forward_kinematics(
        q, crane_model::Frame::MountingBase, crane_model::Frame::RotatorLowerPart);
      EXPECT_TRUE(pivot.ok());
      EXPECT_TRUE(tool.ok());
      const Eigen::Vector3d load =
        tool.value().position_m + tool.value().orientation.toRotationMatrix() * centre_of_mass;
      return (load - pivot.value().position_m).head<2>().eval();
    };

  const auto settled_with = [&](const Eigen::Vector3d& centre, double mass_kg) {
      crane_model::Payload payload = block_payload();
      payload.center_of_mass_k8_m = centre;
      payload.mass_kg = mass_kg;
      const auto settled = model.value().passive_equilibrium(q_a, payload);
      EXPECT_TRUE(settled.ok()) << settled.status().message;
      return settled.ok() ? settled.value() : crane_model::QU::Zero().eval();
    };

  const Eigen::Vector3d offset_centre = centred + Eigen::Vector3d(offset, 0.0, 0.0);
  const crane_model::QU nominal = settled_with(centred, block_payload().mass_kg);
  const crane_model::QU shifted = settled_with(offset_centre, block_payload().mass_kg);

  // Held at the centred equilibrium, an offset grasp puts the load off to the
  // side by very nearly the offset itself: that is the error §5.1 calls the whole
  // placement budget.
  const Eigen::Vector2d unmoved = horizontal_offset(nominal, centred);
  const Eigen::Vector2d displaced = horizontal_offset(nominal, offset_centre) - unmoved;
  EXPECT_NEAR(displaced.norm(), offset, 0.1 * offset);

  // Letting the tool settle takes most of that back. The equilibrium moved in the
  // direction that carries the offset load toward the pivot -- toward it, not
  // past it, which is what the positive projection says and what a mere "it
  // moves" test would not catch.
  const Eigen::Vector2d taken_up = horizontal_offset(shifted, offset_centre) - unmoved;
  EXPECT_GT(taken_up.dot(displaced), 0.0) << "the tilt carried the load past the pivot";
  EXPECT_LT(taken_up.norm(), 0.7 * displaced.norm())
    << "the equilibrium did not carry the offset grasp back toward the pivot";
  EXPECT_GT((shifted - nominal).norm(), 1.0 * M_PI / 180.0)
    << "an offset grasp barely moved the equilibrium";

  // What is left is the crane's own tool, which hangs centred and is worth a
  // couple of hundred kilograms itself, so it is the *combined* centre of mass
  // that ends up under the pivot and the payload's own stops short by the mass
  // ratio §5.1 implies. Ten times the block -- a payload the model has no opinion
  // about -- and what is left over shrinks with that ratio rather than staying
  // put, which is what says the mechanism is the one §5.1 describes and not a
  // coincidence at one mass.
  const crane_model::QU heavy = settled_with(offset_centre, 10.0 * block_payload().mass_kg);
  const Eigen::Vector2d heavy_taken_up = horizontal_offset(heavy, offset_centre) - unmoved;
  EXPECT_LT(heavy_taken_up.norm(), 0.3 * taken_up.norm())
    << "a ten times heavier grasp left the same offset behind";
}

TEST(CraneModelEquilibrium, APoseTheToolCannotHangAtIsAFailureNotAPlausibleAnswer)
{
  // Contract §5: a failed solve is a non-Ok Status, never a pose. The code is
  // `SingularConfiguration` and it is the same code `full_dynamics` and
  // `reduced_actuated_dynamics` already use when the passive rows do not resolve
  // at a configuration -- one condition, one code, so a caller that already
  // branches on it does not learn a second one. A solver that always reported
  // success would leave the caller no signal at all: mpc §2 would damp the sway
  // toward the invented pose and trajectory_planning §6 would make it an
  // endpoint, and both would look like a tuning problem.
  const auto payload = block_payload();
  const auto singular = crane_model::ErrorCode::SingularConfiguration;
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());

  // The arm folded to the description's own upper limit. The tool then needs
  // more tip travel than the description gives that joint before it points down,
  // so it would come to rest against a stop rather than hang, and there is no
  // pose to return.
  crane_model::Q folded = loaded_configuration();
  folded[2] = 4.6;
  const auto stowed = model.value().passive_equilibrium(actuated_rows(folded), payload);
  EXPECT_FALSE(stowed.ok());
  EXPECT_EQ(stowed.status().code, singular);
  EXPECT_FALSE(stowed.status().message.empty());

  // The same crane with gravity the other way up. Every pose the double hinge
  // reaches is then a pose the tool is being pushed away from, so the solve has
  // no well to settle in -- and, crucially, it does not answer with the pose it
  // would have hung at, which still solves h_u = 0 and is now the unstable one.
  const auto inverted = model_with_gravity(
    crane_model::Tool::Pzs100, Eigen::Vector3d(0.0, 0.0, 9.81));
  ASSERT_TRUE(inverted.ok());
  const auto upside_down =
    inverted.value().passive_equilibrium(actuated_rows(loaded_configuration()), payload);
  EXPECT_FALSE(upside_down.ok());
  EXPECT_EQ(upside_down.status().code, singular);

  // And the ordinary input validation, on the production model rather than on the
  // mock: an unknown payload is not a zero-mass one, and a q_a that is not finite
  // is refused before anything is evaluated.
  EXPECT_EQ(
    model.value().passive_equilibrium(
      actuated_rows(loaded_configuration()), crane_model::Payload{}).status().code,
    crane_model::ErrorCode::InvalidPayload);
  EXPECT_EQ(
    model.value().passive_equilibrium(
      crane_model::QA::Constant(std::numeric_limits<double>::quiet_NaN()), payload).status().code,
    crane_model::ErrorCode::NonFiniteInput);
}

TEST(CraneModelDynamics, ThePayloadIsTheBodyRobotModelSaysItIs)
{
  // robot_model §5: a rigid body attached to K8_rotator_lower_part, with m_L,
  // r_L^(8) and Theta_L in that frame. What it adds to M and to the gravity
  // term is then fixed by the frame's own Jacobian, which the public API
  // already returns -- so this checks the attachment point, the frame the two
  // geometric quantities are read in, and the convention that Theta_L is about
  // the payload's own centre of mass, with no second model anywhere.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    const auto q = loaded_configuration();
    const auto dq = crane_model::DQ::Zero();
    const auto payload = block_payload();

    const auto jacobian = model.value().jacobian(q, crane_model::Frame::RotatorLowerPart);
    ASSERT_TRUE(jacobian.ok()) << jacobian.status().message;
    const auto pose = model.value().forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::RotatorLowerPart);
    ASSERT_TRUE(pose.ok());

    // The Jacobian of the payload's centre of mass, in K8: a point rigidly
    // attached to the frame moves with v - r x omega.
    const Eigen::Matrix<double, 3, 8> linear = jacobian.value().value.topRows<3>();
    const Eigen::Matrix<double, 3, 8> angular = jacobian.value().value.bottomRows<3>();
    Eigen::Matrix3d lever;
    const Eigen::Vector3d& r = payload.center_of_mass_k8_m;
    lever << 0.0, -r.z(), r.y(), r.z(), 0.0, -r.x(), -r.y(), r.x(), 0.0;
    const Eigen::Matrix<double, 3, 8> centre_of_mass = linear - lever * angular;

    const auto bare = model.value().full_dynamics(q, dq, empty_gripper());
    const auto loaded = model.value().full_dynamics(q, dq, payload);
    ASSERT_TRUE(bare.ok());
    ASSERT_TRUE(loaded.ok());

    // Koenig: the payload's kinetic energy is its centre of mass plus its
    // rotation about it, so Theta_L never picks up a parallel-axis term.
    const crane_model::FullMass expected_mass =
      payload.mass_kg * centre_of_mass.transpose() * centre_of_mass +
      angular.transpose() * payload.inertia_k8_kg_m2 * angular;
    const crane_model::FullMass mass_increment = loaded.value().mass - bare.value().mass;
    EXPECT_LT((mass_increment - expected_mass).norm(), 1.0e-9 * expected_mass.norm())
      << "the payload does not add the inertia of a body at K8";

    // And the gravity term it adds is the weight of that same point, carried
    // back through the same Jacobian. K8's rotation turns g into the frame the
    // Jacobian is expressed in.
    const Eigen::Vector3d gravity_in_k8 =
      pose.value().orientation.toRotationMatrix().transpose() * Eigen::Vector3d(0.0, 0.0, -9.81);
    const crane_model::DQ expected_bias =
      -payload.mass_kg * centre_of_mass.transpose() * gravity_in_k8;
    const crane_model::DQ bias_increment = loaded.value().bias - bare.value().bias;
    EXPECT_LT((bias_increment - expected_bias).norm(), 1.0e-9 * expected_bias.norm())
      << "the payload does not hang where robot_model §5 puts it";
    EXPECT_GT(expected_bias.norm(), 1.0e2);
  }
}

TEST(CraneModelDynamics, ThePayloadIsValidatedAndNotRepaired)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = loaded_configuration();
  const auto dq = crane_model::DQ::Zero();
  const auto invalid = crane_model::ErrorCode::InvalidPayload;

  const auto refuses = [&](const crane_model::Payload& payload, const char * why) {
      EXPECT_EQ(model.value().full_dynamics(q, dq, payload).status().code, invalid) << why;
      EXPECT_EQ(model.value().inverse_dynamics(q, dq, dq, payload).status().code, invalid) << why;
      EXPECT_EQ(
        model.value().reduced_actuated_dynamics(
          q, dq, crane_model::Input::Zero(), payload).status().code, invalid) << why;
    };

  // Contract §4: an undeclared payload is an explicit unknown, and an unknown is
  // refused rather than silently answered as if the gripper were empty.
  refuses(crane_model::Payload{}, "valid == false");

  crane_model::Payload negative = block_payload();
  negative.mass_kg = -1.0;
  refuses(negative, "negative mass");

  crane_model::Payload infinite = block_payload();
  infinite.mass_kg = std::numeric_limits<double>::infinity();
  refuses(infinite, "non-finite mass");

  crane_model::Payload not_a_number = block_payload();
  not_a_number.center_of_mass_k8_m[1] = std::numeric_limits<double>::quiet_NaN();
  refuses(not_a_number, "non-finite centre of mass");

  crane_model::Payload asymmetric = block_payload();
  asymmetric.inertia_k8_kg_m2(0, 1) = 1.0;
  refuses(asymmetric, "asymmetric inertia");

  crane_model::Payload indefinite = block_payload();
  indefinite.inertia_k8_kg_m2(2, 2) = -9.0;
  refuses(indefinite, "inertia that is not positive semi-definite");

  // A declared payload of zero mass is a statement and is accepted, including
  // the point-mass case where Theta_L is zero.
  EXPECT_TRUE(model.value().full_dynamics(q, dq, empty_gripper()).ok());
  crane_model::Payload point_mass = block_payload();
  point_mass.inertia_k8_kg_m2.setZero();
  EXPECT_TRUE(model.value().full_dynamics(q, dq, point_mass).ok());

  // And it is not the same object as an empty gripper: a zero-mass body still
  // carrying a tensor is a different mass matrix, so nothing here is collapsing
  // a payload to its mass.
  crane_model::Payload weightless = empty_gripper();
  weightless.inertia_k8_kg_m2 = Eigen::Vector3d(18.0, 22.0, 9.0).asDiagonal();
  const auto empty = model.value().full_dynamics(q, dq, empty_gripper());
  const auto spinning = model.value().full_dynamics(q, dq, weightless);
  ASSERT_TRUE(empty.ok());
  ASSERT_TRUE(spinning.ok());
  EXPECT_GT((spinning.value().mass - empty.value().mass).norm(), 1.0);
}

TEST(CraneModelDynamics, ReducedActuatedDynamicsIsTheReductionOfTheFullDynamics)
{
  // robot_model §3.4, proved against the full dynamics rather than against a
  // second Schur complement. tau_a(ddq_a) is recovered from inverse_dynamics
  // alone, by solving the passive rows to zero; it is affine in ddq_a, so its
  // value at zero is h_eff and its six increments are the columns of M_eff.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok()) << model.status().message;
    const auto q = loaded_configuration();
    const auto dq = loaded_velocity();
    const auto payload = block_payload();

    const crane_model::Vector6 at_rest = consistent_actuated_torque(
      model.value(), q, dq, crane_model::Input::Zero(), payload);
    crane_model::ReducedMass expected_mass;
    for (Eigen::Index column = 0; column < 6; ++column) {
      crane_model::Input ddq_a = crane_model::Input::Zero();
      ddq_a[column] = 1.0;
      expected_mass.col(column) =
        consistent_actuated_torque(model.value(), q, dq, ddq_a, payload) - at_rest;
    }

    const auto reduced = model.value().reduced_actuated_dynamics(
      q, dq, crane_model::Input::Zero(), payload);
    ASSERT_TRUE(reduced.ok()) << reduced.status().message;
    EXPECT_LT(
      (reduced.value().mass_eff - expected_mass).norm(), 1.0e-8 * expected_mass.norm());
    EXPECT_LT((reduced.value().bias_eff - at_rest).norm(), 1.0e-8 * at_rest.norm());

    // ddq_a is validated and does not change the reduction: M_eff and h_eff are
    // properties of (q, dq, payload), and tau_a = M_eff ddq_a + h_eff is the
    // caller's product to form.
    const auto accelerating = model.value().reduced_actuated_dynamics(
      q, dq, crane_model::Input::Constant(0.7), payload);
    ASSERT_TRUE(accelerating.ok());
    EXPECT_EQ(accelerating.value().mass_eff, reduced.value().mass_eff);
    EXPECT_EQ(accelerating.value().bias_eff, reduced.value().bias_eff);

    // §3.4: the effective inertia is strictly smaller than M_aa, because the
    // pendulum absorbs part of the acceleration instead of transmitting it.
    const auto full = model.value().full_dynamics(q, dq, payload);
    ASSERT_TRUE(full.ok());
    crane_model::ReducedMass actuated_block;
    for (Eigen::Index row = 0; row < 6; ++row) {
      for (Eigen::Index column = 0; column < 6; ++column) {
        actuated_block(row, column) = full.value().mass(
          kActuatedRows[static_cast<std::size_t>(row)],
          kActuatedRows[static_cast<std::size_t>(column)]);
      }
    }
    const crane_model::ReducedMass absorbed = actuated_block - reduced.value().mass_eff;
    const Eigen::SelfAdjointEigenSolver<crane_model::ReducedMass> spectrum(absorbed);
    EXPECT_GT(spectrum.eigenvalues().minCoeff(), -1.0e-9 * absorbed.norm());
    EXPECT_GT(spectrum.eigenvalues().maxCoeff(), 1.0);
  }
}

TEST(CraneModelDynamics, FullDynamicsCarriesTheConsistentInverseDynamicsTorque)
{
  // The frozen struct carries an inverse-dynamics torque but no acceleration to
  // evaluate it at. The one the model supplies is the consistent one of §3.1
  // with ddq_a = 0 -- the torque holding the actuated axes still while the
  // pendulum swings freely -- so its passive rows vanish and its actuated rows
  // are h_eff. Reporting tau at ddq = 0 instead would make the field a copy of
  // `bias`, and this is what says which of the two it is.
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = loaded_configuration();
  const auto dq = loaded_velocity();
  const auto payload = block_payload();

  const auto full = model.value().full_dynamics(q, dq, payload);
  ASSERT_TRUE(full.ok());
  const auto reduced = model.value().reduced_actuated_dynamics(
    q, dq, crane_model::Input::Zero(), payload);
  ASSERT_TRUE(reduced.ok());

  EXPECT_LT(passive_rows(full.value().inverse_dynamics_tau).norm(), 1.0e-9);
  EXPECT_LT(
    (actuated_rows(full.value().inverse_dynamics_tau) - reduced.value().bias_eff).norm(),
    1.0e-9 * reduced.value().bias_eff.norm());
  EXPECT_GT(full.value().inverse_dynamics_tau.norm(), 1.0e3);

  // It is a torque and not the bias: the pendulum's free acceleration is a real
  // term, so the two differ by more than rounding.
  EXPECT_GT(
    (full.value().inverse_dynamics_tau - full.value().bias).norm(),
    1.0e-3 * full.value().bias.norm());

  // And the same number comes back out of inverse_dynamics at the acceleration
  // the reduction eliminated.
  crane_model::QU ddq_u = crane_model::QU::Zero();
  const crane_model::Vector6 tau_a = consistent_actuated_torque(
    model.value(), q, dq, crane_model::Input::Zero(), payload, &ddq_u);
  EXPECT_LT((tau_a - actuated_rows(full.value().inverse_dynamics_tau)).norm(), 1.0e-8);
  EXPECT_GT(ddq_u.norm(), 1.0e-3) << "the pendulum is not accelerating at all in this fixture";
}

TEST(CraneModelDynamics, NonFiniteAndUnusableConfigurationsAreRefused)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto payload = block_payload();
  const auto q = loaded_configuration();
  const auto dq = loaded_velocity();
  const auto non_finite = crane_model::ErrorCode::NonFiniteInput;

  crane_model::Q bad_q = q;
  bad_q[2] = std::numeric_limits<double>::quiet_NaN();
  crane_model::DQ bad_dq = dq;
  bad_dq[5] = std::numeric_limits<double>::infinity();

  EXPECT_EQ(model.value().full_dynamics(bad_q, dq, payload).status().code, non_finite);
  EXPECT_EQ(model.value().full_dynamics(q, bad_dq, payload).status().code, non_finite);
  EXPECT_EQ(model.value().inverse_dynamics(bad_q, dq, dq, payload).status().code, non_finite);
  EXPECT_EQ(model.value().inverse_dynamics(q, bad_dq, dq, payload).status().code, non_finite);
  EXPECT_EQ(model.value().inverse_dynamics(q, dq, bad_dq, payload).status().code, non_finite);
  EXPECT_EQ(
    model.value().reduced_actuated_dynamics(bad_q, dq, crane_model::Input::Zero(), payload)
      .status().code, non_finite);
  EXPECT_EQ(
    model.value().reduced_actuated_dynamics(
      q, dq, crane_model::Input::Constant(std::numeric_limits<double>::quiet_NaN()), payload)
      .status().code, non_finite);

  // A configuration that is finite but at which the description cannot produce
  // finite numbers -- the telescope run out to 1e200 m -- is reported as such
  // rather than answered with a matrix of infinities.
  crane_model::Q overflowing = q;
  overflowing[3] = 1.0e200;
  const auto singular = crane_model::ErrorCode::SingularConfiguration;
  EXPECT_EQ(model.value().full_dynamics(overflowing, dq, payload).status().code, singular);
  EXPECT_EQ(model.value().inverse_dynamics(overflowing, dq, dq, payload).status().code, singular);
  EXPECT_EQ(
    model.value().reduced_actuated_dynamics(
      overflowing, dq, crane_model::Input::Zero(), payload).status().code, singular);
}

// --- collision ---------------------------------------------------------------
//
// wiki/trajectory_planning.md §4.2: one model, one library, one URDF, giving
// forward kinematics, dynamics, distances and collision. Pinocchio places the
// links of the description and Coal answers over the geometry
// src/collision_model.hpp carries -- fitted once, offline, to the `<collision>`
// elements of these same two descriptions, because a ROS-free library with no
// path in its config cannot resolve a `package://` mesh at runtime.

namespace
{

// The compiled table's `source` field, taken apart again. It is a string so the
// generated header stays readable; the comparison below is numeric, so a
// formatting change in the derivation cannot make this test pass or fail.
struct GeometrySource
{
  std::string kind;             // "mesh" or "box"
  std::string name;             // the mesh URI, empty for a box
  Eigen::Vector3d numbers{};    // mesh scale, or box side lengths
};

GeometrySource parse_source(const std::string& source)
{
  GeometrySource parsed;
  const std::size_t colon = source.find(':');
  parsed.kind = source.substr(0, colon);
  const std::size_t at = source.rfind('@');
  const std::size_t start = colon + 1;
  const std::size_t stop = parsed.kind == "mesh" ? at : source.size();
  parsed.name = parsed.kind == "mesh" ? source.substr(start, stop - start) : std::string{};
  std::string numbers = source.substr(parsed.kind == "mesh" ? at + 1 : start);
  std::replace(numbers.begin(), numbers.end(), ',', ' ');
  std::istringstream stream(numbers);
  stream >> parsed.numbers[0] >> parsed.numbers[1] >> parsed.numbers[2];
  return parsed;
}

// The `disable_collisions` rows of the checked-in SRDF, unordered.
std::set<std::pair<std::string, std::string>> checked_in_allowed_pairs()
{
  const std::string path = std::string(CRANE_MODEL_CONFIG_DIR) + "/allowed_collisions.srdf";
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "cannot read " << path;
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  const std::string text = buffer.str();

  std::set<std::pair<std::string, std::string>> pairs;
  for (std::size_t at = text.find("<disable_collisions"); at != std::string::npos;
    at = text.find("<disable_collisions", at + 1))
  {
    const auto attribute = [&text, at](const char * name) {
        const std::size_t key = text.find(name, at);
        const std::size_t open = text.find('"', key) + 1;
        return text.substr(open, text.find('"', open) - open);
      };
    pairs.emplace(attribute("link1=\""), attribute("link2=\""));
  }
  return pairs;
}

crane_model::CollisionPrimitive box_primitive(
  const std::string& id, const Eigen::Vector3d& position, const Eigen::Vector3d& sides)
{
  crane_model::CollisionPrimitive primitive;
  primitive.id = id;
  primitive.shape = crane_model::CollisionShape::Box;
  primitive.pose_in_mounting_base = Eigen::Isometry3d::Identity();
  primitive.pose_in_mounting_base.translation() = position;
  primitive.dimensions_m = sides;
  return primitive;
}

// Where the tool is, in K0, so a test can put an obstacle in its way without
// writing down a number that the description could move underneath it.
Eigen::Vector3d tool_position(const crane_model::Model& model, const crane_model::Q& q)
{
  const auto pose = model.forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  EXPECT_TRUE(pose.ok());
  return pose.ok() ? pose.value().position_m : Eigen::Vector3d::Zero();
}

// The crane reaching out: boom and arm up, telescope extended, so the tool is
// clear of the column and an obstacle beside it is an obstacle to the tool.
crane_model::Q reaching_configuration()
{
  crane_model::Q q = crane_model::Q::Zero();
  q[1] = 0.6;
  q[2] = 1.2;
  q[3] = 1.0;
  return q;
}

// theta3_arm_joint runs to 4.6 rad in both descriptions, far past the working
// range: at the stop the arm has folded back over the boom.
crane_model::Q folded_configuration()
{
  crane_model::Q q = crane_model::Q::Zero();
  q[2] = 4.6;
  return q;
}

const crane_model::CollisionResult& scene_entry(
  const std::vector<crane_model::CollisionResult>& results, const std::string& id)
{
  const auto found = std::find_if(
    results.begin(), results.end(),
    [&id](const crane_model::CollisionResult& result) {return result.other_id == id;});
  EXPECT_NE(found, results.end()) << "no result for " << id;
  return found == results.end() ? results.front() : *found;
}

bool names_a_self_pair(const crane_model::CollisionResult& result)
{
  return result.other_id.find('|') != std::string::npos;
}

}  // namespace

TEST(CraneModelCollision, PrimitivesWereFittedToThisDescription)
{
  // The fit is compiled in, so the description is where it can rot. Every entry
  // names the `<collision>` element it was fitted to; this asserts that element
  // is still there, still that shape, still at that pose. A description that
  // moves one fails here instead of being checked against stale geometry.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const ::urdf::ModelInterfaceSharedPtr tree = ::urdf::parseURDF(description_for(tool));
    ASSERT_TRUE(tree);
    std::size_t checked = 0;
    for (const auto& entry : crane_model::collision_model::kLinkPrimitives) {
      if (entry.tool != tool) {
        continue;
      }
      ++checked;
      const auto link = tree->getLink(entry.link);
      ASSERT_TRUE(link) << entry.link << " is gone from the description";
      ASSERT_TRUE(link->collision) << entry.link << " no longer carries collision geometry";
      const auto& collision = *link->collision;

      const GeometrySource source = parse_source(entry.source);
      if (source.kind == "mesh") {
        const auto mesh = std::dynamic_pointer_cast<::urdf::Mesh>(collision.geometry);
        ASSERT_TRUE(mesh) << entry.link << " no longer carries a mesh";
        EXPECT_EQ(mesh->filename, source.name) << entry.link;
        EXPECT_NEAR(mesh->scale.x, source.numbers[0], 1.0e-9) << entry.link;
        EXPECT_NEAR(mesh->scale.y, source.numbers[1], 1.0e-9) << entry.link;
        EXPECT_NEAR(mesh->scale.z, source.numbers[2], 1.0e-9) << entry.link;
      } else {
        const auto box = std::dynamic_pointer_cast<::urdf::Box>(collision.geometry);
        ASSERT_TRUE(box) << entry.link << " no longer carries a box";
        EXPECT_NEAR(box->dim.x, source.numbers[0], 1.0e-9) << entry.link;
        EXPECT_NEAR(box->dim.y, source.numbers[1], 1.0e-9) << entry.link;
        EXPECT_NEAR(box->dim.z, source.numbers[2], 1.0e-9) << entry.link;
      }

      const Eigen::Vector3d translation(
        collision.origin.position.x, collision.origin.position.y, collision.origin.position.z);
      EXPECT_LT(
        (translation - Eigen::Vector3d(
          entry.source_pose[0], entry.source_pose[1], entry.source_pose[2])).norm(), 1.0e-6)
        << entry.link << " moved its collision origin";
      const Eigen::Matrix3d declared = Eigen::Quaterniond(
        collision.origin.rotation.w, collision.origin.rotation.x,
        collision.origin.rotation.y, collision.origin.rotation.z).toRotationMatrix();
      // The table stores the element's roll-pitch-yaw; urdfdom has already
      // turned it into a quaternion, so the two meet as rotations.
      const Eigen::Matrix3d fitted =
        (Eigen::AngleAxisd(entry.source_pose[5], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(entry.source_pose[4], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(entry.source_pose[3], Eigen::Vector3d::UnitX())).toRotationMatrix();
      EXPECT_LT(rotation_vector(declared.transpose() * fitted).norm(), 1.0e-6)
        << entry.link << " turned its collision origin";

      // A primitive with no size is not a primitive.
      EXPECT_GT(entry.extents_m[0], 0.0) << entry.link;
      EXPECT_GT(entry.extents_m[1], 0.0) << entry.link;
      EXPECT_NEAR(
        Eigen::Vector4d(
          entry.orientation[0], entry.orientation[1],
          entry.orientation[2], entry.orientation[3]).norm(), 1.0, 1.0e-9) << entry.link;
    }
    EXPECT_EQ(checked, 15U) << "the description has fifteen shaped links per tool";
  }
}

TEST(CraneModelCollision, AllowedPairsAreTheCheckedInList)
{
  // The list the library filters with and the list a human reads are the same
  // list. `config/allowed_collisions.srdf` is the checked-in artefact of
  // `scripts/derive_collision_model.py`; the table in `src/collision_model.hpp`
  // is what the queries use.
  const auto checked_in = checked_in_allowed_pairs();
  ASSERT_FALSE(checked_in.empty());

  std::set<std::pair<std::string, std::string>> compiled;
  for (const auto& pair : crane_model::collision_model::kAllowedSelfPairs) {
    compiled.emplace(pair.first, pair.second);
  }
  EXPECT_EQ(compiled, checked_in);
  EXPECT_EQ(compiled.size(), crane_model::collision_model::kAllowedSelfPairCount);
}

TEST(CraneModelCollision, SceneQueriesReturnRealResultsAgainstTheSuppliedScene)
{
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto q = reaching_configuration();
    const Eigen::Vector3d tool_at = tool_position(model.value(), q);

    // A block two metres below the tool, and one the tool is standing in.
    crane_model::CollisionScene scene;
    scene.primitives.push_back(
      box_primitive("clear", tool_at - Eigen::Vector3d(0.0, 0.0, 2.0),
      Eigen::Vector3d(0.4, 0.4, 0.4)));
    scene.primitives.push_back(
      box_primitive("hit", tool_at, Eigen::Vector3d(0.4, 0.4, 0.4)));

    const auto results = model.value().collision_queries(q, scene);
    ASSERT_TRUE(results.ok()) << results.status().message;
    // One per scene primitive, in scene order, then the crane against itself.
    ASSERT_EQ(results.value().size(), 3U);
    EXPECT_EQ(results.value()[0].other_id, "clear");
    EXPECT_EQ(results.value()[1].other_id, "hit");
    EXPECT_TRUE(names_a_self_pair(results.value()[2]));

    const auto& clear = results.value()[0];
    EXPECT_FALSE(clear.collision);
    EXPECT_GT(clear.minimum_distance_m, 0.0);
    // Witness points are real points, in K0, and they realise the distance.
    EXPECT_TRUE(clear.witness_on_robot_m.allFinite());
    EXPECT_TRUE(clear.witness_on_other_m.allFinite());
    EXPECT_NEAR(
      (clear.witness_on_robot_m - clear.witness_on_other_m).norm(),
      clear.minimum_distance_m, 1.0e-6);
    // The one on the obstacle is on the obstacle: inside that 0.4 m box.
    const Eigen::Vector3d local =
      clear.witness_on_other_m - (tool_at - Eigen::Vector3d(0.0, 0.0, 2.0));
    EXPECT_LT(local.cwiseAbs().maxCoeff(), 0.2 + 1.0e-6);

    const auto& hit = results.value()[1];
    EXPECT_TRUE(hit.collision);
    EXPECT_LT(hit.minimum_distance_m, 0.0);

    // The single-result call is the worst of them, whichever it was.
    const auto worst = model.value().collision_query(q, scene);
    ASSERT_TRUE(worst.ok());
    double smallest = std::numeric_limits<double>::max();
    for (const auto& result : results.value()) {
      smallest = std::min(smallest, result.minimum_distance_m);
    }
    EXPECT_DOUBLE_EQ(worst.value().minimum_distance_m, smallest);
    EXPECT_TRUE(worst.value().collision);
    EXPECT_EQ(worst.value().other_id, "hit");

    // An empty scene is not an error: the crane is still checked against itself.
    const auto alone = model.value().collision_queries(q, crane_model::CollisionScene{});
    ASSERT_TRUE(alone.ok());
    ASSERT_EQ(alone.value().size(), 1U);
    EXPECT_TRUE(names_a_self_pair(alone.value().front()));
  }
}

TEST(CraneModelCollision, SelfCollisionIsCheckedAndTheAllowedPairsAreNot)
{
  const auto allowed = checked_in_allowed_pairs();
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto empty = crane_model::CollisionScene{};

    // Both directions, as the issue asks. First: a configuration the joint
    // limits allow and the geometry does not. theta3_arm_joint reaches 4.6 rad,
    // which folds the arm and both telescope stages back over the boom.
    const auto folded = model.value().collision_query(folded_configuration(), empty);
    ASSERT_TRUE(folded.ok()) << folded.status().message;
    EXPECT_TRUE(folded.value().collision);
    EXPECT_LT(folded.value().minimum_distance_m, 0.0);
    EXPECT_TRUE(names_a_self_pair(folded.value())) << folded.value().other_id;
    EXPECT_NEAR(
      (folded.value().witness_on_robot_m - folded.value().witness_on_other_m).norm(),
      -folded.value().minimum_distance_m, 1.0e-6);

    // Second: the pairs on the list never appear, at that same configuration or
    // at any of a spread of others. The adjacent links touch by construction --
    // K2_boom and K3_arm share a joint -- so an unfiltered check would report
    // them at every configuration and report nothing else usefully.
    const std::array<crane_model::Q, 4> configurations{{
      crane_model::Q::Zero(), reaching_configuration(), folded_configuration(), valid_q()}};
    for (const auto& q : configurations) {
      const auto results = model.value().collision_queries(q, empty);
      ASSERT_TRUE(results.ok()) << results.status().message;
      for (const auto& result : results.value()) {
        const std::size_t bar = result.other_id.find('|');
        ASSERT_NE(bar, std::string::npos);
        const std::string first = result.other_id.substr(0, bar);
        const std::string second = result.other_id.substr(bar + 1);
        EXPECT_EQ(allowed.count({first, second}) + allowed.count({second, first}), 0U)
          << first << " and " << second << " are on the allowed-collision list";
      }
    }

    // And the neutral configuration is clear, so a collision reported anywhere
    // else is the configuration's and not the geometry's.
    const auto neutral = model.value().collision_query(crane_model::Q::Zero(), empty);
    ASSERT_TRUE(neutral.ok());
    EXPECT_FALSE(neutral.value().collision) << neutral.value().other_id;
    EXPECT_GT(neutral.value().minimum_distance_m, 0.0);
  }
}

TEST(CraneModelCollision, ToolGeometryIsToolDependent)
{
  // Same crane, same configuration, same scene; the two tools are different
  // shapes and the model says so. The rail frame and the pincer frame are both
  // called K8_tool_center_point and they are not the same primitive.
  const auto rail = production_model(crane_model::Tool::Pzs100);
  const auto jaws = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(rail.ok());
  ASSERT_TRUE(jaws.ok());

  std::map<std::string, Eigen::Vector3d> extents;
  for (const auto& entry : crane_model::collision_model::kLinkPrimitives) {
    const std::string key = std::string(entry.tool == crane_model::Tool::Pzs100 ? "p:" : "e:") +
      entry.link;
    extents[key] =
      Eigen::Vector3d(entry.extents_m[0], entry.extents_m[1], entry.extents_m[2]);
  }
  EXPECT_GT((extents.at("p:K8_tool_center_point") - extents.at("e:K8_tool_center_point")).norm(),
    0.01);
  EXPECT_EQ(extents.count("p:K10_left_rail"), 1U);
  EXPECT_EQ(extents.count("e:K10_outer_jaw"), 1U);
  EXPECT_EQ(extents.count("p:K10_outer_jaw"), 0U);
  EXPECT_EQ(extents.count("e:K10_left_rail"), 0U);

  const auto q = reaching_configuration();
  crane_model::CollisionScene scene;
  scene.primitives.push_back(
    box_primitive(
      "post", tool_position(rail.value(), q) - Eigen::Vector3d(0.0, 0.0, 1.0),
      Eigen::Vector3d(0.3, 0.3, 2.0)));

  const auto rail_result = rail.value().collision_queries(q, scene);
  const auto jaw_result = jaws.value().collision_queries(q, scene);
  ASSERT_TRUE(rail_result.ok());
  ASSERT_TRUE(jaw_result.ok());
  EXPECT_NE(
    scene_entry(rail_result.value(), "post").minimum_distance_m,
    scene_entry(jaw_result.value(), "post").minimum_distance_m);
}

TEST(CraneModelCollision, ACarriedPayloadIsIncludedButNotAgainstTheToolCarryingIt)
{
  // The frozen collision signature takes no `Payload`, so a carried payload is a
  // scene primitive with the reserved id, placed at the pose the caller reads
  // out of forward_kinematics. What the model owes it is the half a caller
  // cannot supply: the links that hold it are not obstacles to it.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto q = reaching_configuration();
    const Eigen::Vector3d gripper = tool_position(model.value(), q);

    crane_model::CollisionScene carried;
    carried.primitives.push_back(
      box_primitive("payload", gripper, Eigen::Vector3d(0.6, 0.6, 0.6)));
    const auto held = model.value().collision_queries(q, carried);
    ASSERT_TRUE(held.ok()) << held.status().message;
    const auto& payload = scene_entry(held.value(), "payload");
    EXPECT_FALSE(payload.collision)
      << "the payload collided with " << payload.other_id;

    // The same block under any other id is exactly what it looks like: the tool
    // standing inside an obstacle.
    crane_model::CollisionScene loose;
    loose.primitives.push_back(
      box_primitive("block", gripper, Eigen::Vector3d(0.6, 0.6, 0.6)));
    const auto dropped = model.value().collision_queries(q, loose);
    ASSERT_TRUE(dropped.ok());
    EXPECT_TRUE(scene_entry(dropped.value(), "block").collision);

    // Carried is not ignored: it is still checked against the rest of the crane,
    // and it collides with the boom when the tool is folded back against it.
    crane_model::CollisionScene big;
    big.primitives.push_back(
      box_primitive("payload", gripper, Eigen::Vector3d(6.0, 6.0, 6.0)));
    const auto swept = model.value().collision_queries(q, big);
    ASSERT_TRUE(swept.ok());
    EXPECT_TRUE(scene_entry(swept.value(), "payload").collision);
  }
}

TEST(CraneModelCollision, SceneIsConsumedInMountingBaseAndValidated)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = reaching_configuration();

  // Consumed in K0, not converted: a primitive at the tool's K0 position is hit,
  // and slewing the crane away from it leaves the same primitive where it was.
  crane_model::CollisionScene scene;
  scene.primitives.push_back(
    box_primitive("here", tool_position(model.value(), q), Eigen::Vector3d(0.3, 0.3, 0.3)));
  const auto at_tool = model.value().collision_queries(q, scene);
  ASSERT_TRUE(at_tool.ok());
  EXPECT_TRUE(scene_entry(at_tool.value(), "here").collision);

  auto slewed = q;
  slewed[0] = 1.6;  // rad, most of a quarter turn away
  const auto turned = model.value().collision_queries(slewed, scene);
  ASSERT_TRUE(turned.ok());
  const auto& missed = scene_entry(turned.value(), "here");
  EXPECT_FALSE(missed.collision);
  EXPECT_GT(missed.minimum_distance_m, 1.0);
  // World is not a frame this library has, so nothing here could have converted.
  EXPECT_EQ(
    model.value().forward_kinematics(
      q, crane_model::Frame::World, crane_model::Frame::MountingBase).status().code,
    crane_model::ErrorCode::FrameUnavailable);

  // A primitive that is not usable is refused, and the whole scene with it.
  const Eigen::Vector3d somewhere(3.0, 0.0, 0.0);
  const Eigen::Vector3d cube(0.5, 0.5, 0.5);
  const auto refuses = [&model, &q](const crane_model::CollisionPrimitive& primitive) {
      crane_model::CollisionScene bad;
      bad.primitives.push_back(primitive);
      EXPECT_EQ(
        model.value().collision_query(q, bad).status().code,
        crane_model::ErrorCode::InvalidScene) << primitive.id;
      EXPECT_EQ(
        model.value().collision_queries(q, bad).status().code,
        crane_model::ErrorCode::InvalidScene) << primitive.id;
    };

  refuses(box_primitive("", somewhere, cube));
  refuses(box_primitive("flat", somewhere, Eigen::Vector3d(0.5, 0.0, 0.5)));
  refuses(box_primitive("inside out", somewhere, Eigen::Vector3d(0.5, -0.5, 0.5)));
  refuses(
    box_primitive(
      "not finite", somewhere,
      Eigen::Vector3d(0.5, std::numeric_limits<double>::quiet_NaN(), 0.5)));

  auto skewed = box_primitive("skewed", somewhere, cube);
  skewed.pose_in_mounting_base.linear() *= 2.0;
  refuses(skewed);

  auto unknown = box_primitive("unknown", somewhere, cube);
  unknown.shape = static_cast<crane_model::CollisionShape>(200);
  refuses(unknown);

  // `dimensions_m` is the primitive's extent on each axis of its own frame, so a
  // cylinder carries its diameter twice and a sphere three times. A caller who
  // meant a radius in the first entry gets told, not silently reinterpreted.
  auto cylinder = box_primitive("cylinder", somewhere, Eigen::Vector3d(0.4, 0.9, 1.0));
  cylinder.shape = crane_model::CollisionShape::Cylinder;
  refuses(cylinder);
  auto sphere = box_primitive("sphere", somewhere, Eigen::Vector3d(0.4, 0.4, 0.9));
  sphere.shape = crane_model::CollisionShape::Sphere;
  refuses(sphere);

  crane_model::CollisionScene twice;
  twice.primitives.push_back(box_primitive("same", somewhere, cube));
  twice.primitives.push_back(box_primitive("same", -somewhere, cube));
  EXPECT_EQ(
    model.value().collision_query(q, twice).status().code,
    crane_model::ErrorCode::InvalidScene);

  // The well formed versions of the two round shapes are accepted.
  crane_model::CollisionScene round;
  auto good_cylinder = box_primitive("cylinder", somewhere, Eigen::Vector3d(0.4, 0.4, 1.0));
  good_cylinder.shape = crane_model::CollisionShape::Cylinder;
  auto good_sphere = box_primitive("sphere", -somewhere, Eigen::Vector3d(0.4, 0.4, 0.4));
  good_sphere.shape = crane_model::CollisionShape::Sphere;
  round.primitives = {good_cylinder, good_sphere};
  const auto accepted = model.value().collision_queries(q, round);
  ASSERT_TRUE(accepted.ok()) << accepted.status().message;
  EXPECT_EQ(accepted.value().size(), 3U);

  auto non_finite_q = q;
  non_finite_q[4] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    model.value().collision_query(non_finite_q, round).status().code,
    crane_model::ErrorCode::NonFiniteInput);
  EXPECT_EQ(
    model.value().collision_queries(non_finite_q, round).status().code,
    crane_model::ErrorCode::NonFiniteInput);
}

TEST(CraneModelCollision, ADescriptionWithoutTheShapedLinksIsRefusedNotAnswered)
{
  // The fit is for this machine. A description that parses and carries the eight
  // joints is a usable model for everything else -- `crane_control` builds one
  // from a joints-only fixture -- but it is not a crane to check for collision,
  // and the model says which link it is missing rather than answering for a
  // crane with parts missing.
  crane_model::ModelConfig config;
  config.robot_description_xml =
    "<robot name=\"fixture\">"
    "<link name=\"K0_mounting_base\"/><link name=\"a\"/><link name=\"b\"/><link name=\"c\"/>"
    "<link name=\"d\"/><link name=\"e\"/><link name=\"f\"/><link name=\"g\"/><link name=\"h\"/>"
    "<joint name=\"theta1_slewing_joint\" type=\"revolute\">"
    "<parent link=\"K0_mounting_base\"/><child link=\"a\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"theta2_boom_joint\" type=\"revolute\"><parent link=\"a\"/><child link=\"b\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"theta3_arm_joint\" type=\"revolute\"><parent link=\"b\"/><child link=\"c\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"q4_big_telescope\" type=\"prismatic\"><parent link=\"c\"/><child link=\"d\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"0\" upper=\"2\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"theta6_tip_joint\" type=\"revolute\"><parent link=\"d\"/><child link=\"e\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"theta7_tilt_joint\" type=\"revolute\"><parent link=\"e\"/><child link=\"f\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/></joint>"
    "<joint name=\"theta8_rotator_joint\" type=\"continuous\">"
    "<parent link=\"f\"/><child link=\"g\"/><axis xyz=\"0 0 1\"/></joint>"
    "<joint name=\"q9_left_rail_joint\" type=\"prismatic\">"
    "<parent link=\"g\"/><child link=\"h\"/>"
    "<axis xyz=\"0 0 1\"/><limit lower=\"0\" upper=\"1\" effort=\"1\" velocity=\"1\"/></joint>"
    "</robot>";
  const auto model = crane_model::Model::create(config);
  ASSERT_TRUE(model.ok()) << model.status().message;
  EXPECT_TRUE(model.value().forward_kinematics(
    crane_model::Q::Zero(), crane_model::Frame::MountingBase,
    crane_model::Frame::MountingBase).ok());

  const auto refused =
    model.value().collision_query(crane_model::Q::Zero(), crane_model::CollisionScene{});
  EXPECT_EQ(refused.status().code, crane_model::ErrorCode::CollisionBackendFailure);
  const bool names_a_link = std::any_of(
    crane_model::collision_model::kLinkPrimitives.begin(),
    crane_model::collision_model::kLinkPrimitives.end(),
    [&refused](const crane_model::collision_model::LinkPrimitive& entry) {
      return entry.tool == crane_model::Tool::Pzs100 &&
      refused.status().message.find(entry.link) != std::string::npos;
    });
  EXPECT_TRUE(names_a_link) << refused.status().message;
}

TEST(CraneModelCollision, CollisionQueriesIsNotARealTimeCall)
{
  // Contract §10: a call returning a dynamically sized container is not an RT
  // API call. The guard in RealTimeCallsAllocateNothingAfterConstruction covers
  // the five that are; this is the demonstration that collision is not one of
  // them, so the guard's silence about it is a fact and not an oversight.
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = reaching_configuration();
  crane_model::CollisionScene scene;
  scene.primitives.push_back(
    box_primitive("post", Eigen::Vector3d(4.0, 0.0, 0.0), Eigen::Vector3d(0.3, 0.3, 2.0)));
  ASSERT_TRUE(model.value().collision_queries(q, scene).ok());

  g_allocation_count.store(0, std::memory_order_relaxed);
  g_allocation_guard.store(true, std::memory_order_relaxed);
  const bool ok = model.value().collision_queries(q, scene).ok();
  g_allocation_guard.store(false, std::memory_order_relaxed);
  EXPECT_TRUE(ok);
  EXPECT_GT(g_allocation_count.load(std::memory_order_relaxed), 0U);
}
