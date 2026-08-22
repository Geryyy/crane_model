#include <gtest/gtest.h>

#include <limits>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <string>

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

std::string description_for(crane_model::Tool tool)
{
  const std::string common = "<robot name=\"fixture\"> "
    "theta1_slewing_joint theta2_boom_joint theta3_arm_joint "
    "q4_big_telescope theta6_tip_joint theta7_tilt_joint theta8_rotator_joint ";
  return common + (tool == crane_model::Tool::Pzs100 ? "q9_left_rail_joint" :
                                                        "theta10_outer_jaw_joint") + " </robot>";
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

Eigen::Vector2d boom_attachment(double q2)
{
  return rotate(q2, 3.49288 - 3.039, -0.036034);  // R(q2) (a_2 + p_S2x, p_S2y)
}

double boom_stroke(double q2)
{
  const Eigen::Vector2d p_s0(0.433, -1.7682);
  const Eigen::Vector2d p_s1(-0.12, -0.07);
  const Eigen::Vector2d d_pivot = boom_attachment(q2) - p_s1;
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

// The same cylinder foot and the same boom-side attachment, but with the
// drawbar and pushbar deleted: what the stroke would be if the boom cylinder
// acted directly on the boom instead of through the four-bar coupler point.
double boom_direct_stroke(double q2)
{
  return (boom_attachment(q2) - Eigen::Vector2d(0.433, -1.7682)).norm();
}

double arm_stroke(double q3)
{
  const Eigen::Vector2d moving = rotate(q3, -0.3925 + 0.274489, -0.468);
  const Eigen::Vector2d in_plane = moving - Eigen::Vector2d(-1.6802, -0.0485);
  const double lateral = 0.224 - 0.224;  // p_S4y - p_S3z
  return std::sqrt(in_plane.squaredNorm() + lateral * lateral);
}

double transmission_ratio(double (* stroke)(double), double q)
{
  return (stroke(q + kStep) - stroke(q - kStep)) / (2.0 * kStep);
}

}  // namespace reference

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
  missing_joint.robot_description_xml = "<robot name=\"fixture\"></robot>";
  result = crane_model::Model::create(missing_joint);
  EXPECT_EQ(result.status().code, crane_model::ErrorCode::MissingJoint);

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

TEST(CraneModelHydraulicSubset, SevenThousandFortyJawLinkageIsNotBackedYet)
{
  const auto model = production_model(crane_model::Tool::Epsilon7040);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.0, 0.0);
  // §2.6 names the jaw four-bar and its fitted mirror law but §6 carries no
  // pivot geometry for it, so the transmission of the GR axis is not derivable
  // here. It says so instead of guessing a one-to-one jaw cylinder.
  EXPECT_EQ(model.value().cylinder_jacobian(q).status().code,
    crane_model::ErrorCode::BackendUnavailable);
  EXPECT_EQ(model.value().transmission(
    q, crane_model::DQA::Zero(), zero_pressure()).status().code,
    crane_model::ErrorCode::BackendUnavailable);
}

TEST(CraneModelHydraulicSubset, EveryCallOutsideTheSubsetStillReportsBackendUnavailable)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = valid_q();
  const auto dq = crane_model::DQ::Ones();
  const auto payload = valid_payload();
  const auto scene = crane_model::CollisionScene{};
  const auto unavailable = crane_model::ErrorCode::BackendUnavailable;

  EXPECT_EQ(model.value().forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp).status().code, unavailable);
  EXPECT_EQ(model.value().jacobian(q, crane_model::Frame::Tcp).status().code, unavailable);
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

TEST(CraneModelHydraulicSubset, SubsetCallsAllocateNothingAfterConstruction)
{
  const auto model = production_model(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto q = linkage_configuration(0.2, -0.3);
  const auto pressure = zero_pressure();
  crane_model::DQA dq_a;
  dq_a << 0.3, 0.1, -0.1, 0.02, 0.4, 0.01;
  // Warm every code path before observing allocations; the model itself is
  // built once, in create(), which is not real-time.
  ASSERT_TRUE(model.value().cylinder_jacobian(q).ok());
  ASSERT_TRUE(model.value().transmission(q, dq_a, pressure).ok());
  ASSERT_TRUE(model.value().cylinder_force(pressure).ok());

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
}
