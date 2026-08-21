#include <gtest/gtest.h>

#include <limits>
#include <atomic>
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

TEST(CraneModelContract, ProductionBackendNeverReturnsPlausibleZeroModel)
{
  const auto invalid = crane_model::Model::create({});
  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code, crane_model::ErrorCode::InvalidRobotDescription);

  const auto unavailable = crane_model::Model::create({description_for(
    crane_model::Tool::Pzs100), crane_model::Tool::Pzs100});
  EXPECT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code, crane_model::ErrorCode::BackendUnavailable);
  EXPECT_FALSE(unavailable.status().message.empty());
}

TEST(CraneModelContract, RobotDescriptionAndToolValidationIsExplicit)
{
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
