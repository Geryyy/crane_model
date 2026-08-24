#include <gtest/gtest.h>

#include <limits>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "crane_model/model.hpp"
#include "crane_model/testing/mock_model.hpp"

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

TEST(CraneModelHydraulicSubset, EveryCallOutsideTheSubsetStillReportsBackendUnavailable)
{
  // Both tools. Slice 4 arrives in pieces: forward kinematics and the Jacobian
  // have a backend now, the dynamics, passive equilibrium, collision and
  // symbolic graph do not, and each of them says so call by call rather than
  // handing a consumer a stub.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    const auto q = valid_q();
    const auto dq = crane_model::DQ::Ones();
    const auto payload = valid_payload();
    const auto scene = crane_model::CollisionScene{};
    const auto unavailable = crane_model::ErrorCode::BackendUnavailable;

    EXPECT_EQ(model.value().passive_equilibrium(
      crane_model::QA::Zero(), payload).status().code, unavailable);
    EXPECT_EQ(model.value().full_dynamics(q, dq, payload).status().code, unavailable);
    EXPECT_EQ(model.value().reduced_actuated_dynamics(
      q, dq, crane_model::Input::Zero(), payload).status().code, unavailable);
    EXPECT_EQ(model.value().inverse_dynamics(q, dq, dq, payload).status().code, unavailable);
    EXPECT_EQ(model.value().collision_query(q, scene).status().code, unavailable);
    EXPECT_EQ(model.value().collision_queries(q, scene).status().code, unavailable);
    EXPECT_EQ(model.value().symbolic_graph({}, payload).status().code, unavailable);
  }
}

TEST(CraneModelHydraulicSubset, RealTimeCallsAllocateNothingAfterConstruction)
{
  // Both tools, because the 7040's jaw four-bar is on the same RT path, and now
  // also forward kinematics and the Jacobian: contract §10 says a call that
  // cannot be shown allocation-free is not an RT API call, and both of these
  // are on the control path. The Pinocchio model and its Data workspace are
  // built once, in create(), which is not real-time.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = production_model(tool);
    ASSERT_TRUE(model.ok());
    auto q = linkage_configuration(0.2, -0.3);
    q[7] = 1.5;
    const auto pressure = zero_pressure();
    crane_model::DQA dq_a;
    dq_a << 0.3, 0.1, -0.1, 0.02, 0.4, 0.01;
    const auto base = crane_model::Frame::MountingBase;
    const auto tcp = crane_model::Frame::Tcp;
    // Warm every code path before observing allocations.
    ASSERT_TRUE(model.value().cylinder_jacobian(q).ok());
    ASSERT_TRUE(model.value().transmission(q, dq_a, pressure).ok());
    ASSERT_TRUE(model.value().cylinder_force(pressure).ok());
    ASSERT_TRUE(model.value().forward_kinematics(q, base, tcp).ok());
    ASSERT_TRUE(model.value().jacobian(q, tcp).ok());

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_allocation_guard.store(true, std::memory_order_relaxed);
    bool all_ok = true;
    for (int index = 0; index < 1000; ++index) {
      all_ok = all_ok && model.value().cylinder_jacobian(q).ok();
      all_ok = all_ok && model.value().transmission(q, dq_a, pressure).ok();
      all_ok = all_ok && model.value().cylinder_force(pressure).ok();
      all_ok = all_ok && model.value().forward_kinematics(q, base, tcp).ok();
      all_ok = all_ok && model.value().jacobian(q, tcp).ok();
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
