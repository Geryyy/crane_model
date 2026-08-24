// The symbolic graph of contract §9, and the consumer that opts into the CasADi
// handles.
//
// The claim this file exists to test is the one `wiki/mpc.md` §5.2 and
// `wiki/trajectory_planning.md` §5.2 both rest on: that there is *one* dynamics
// implementation in the system, and that the graph the MPC exports its model
// from and the graph the planner carries the passive rows out of is the same
// equations as the numeric model the rest of the package is tested against.
//
// So nothing here checks the graph against a restatement of the equations of
// motion. Every assertion is against a public numeric call --
// `inverse_dynamics`, `reduced_actuated_dynamics`, `full_dynamics`,
// `transmission` -- at the same argument, and the tolerances are round-off and
// the documented smoothing of mpc §3.1, not numbers chosen to make it pass.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include <casadi/casadi.hpp>

#include "crane_model/model.hpp"
#include "crane_model/symbolic/casadi_graph.hpp"

// Private to the implementation; on this test's include path only, so the
// smoothing bound below can be written in the same effective areas the model
// uses rather than in a magic epsilon.
#include "cylinder_geometry.hpp"

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

std::string read_description(const char * file)
{
  const std::string path = std::string(CRANE_MODEL_TEST_DESCRIPTION_DIR) + "/" + file;
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "cannot read " << path;
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

crane_model::ModelConfig configuration_for(crane_model::Tool tool)
{
  static const std::string rail = read_description("pzs100.urdf");
  static const std::string jaws = read_description("epsilon_7040.urdf");
  crane_model::ModelConfig config;
  config.robot_description_xml = tool == crane_model::Tool::Pzs100 ? rail : jaws;
  config.tool = tool;
  return config;
}

// The eight canonical coordinates and their rates, out of the 16-state of
// contract §2. The graph is given `x` and the numeric calls are given what this
// returns, so any disagreement about the state layout shows up as a
// disagreement about the dynamics rather than being hidden by a shared helper.
struct Canonical
{
  crane_model::Q q{crane_model::Q::Zero()};
  crane_model::DQ dq{crane_model::DQ::Zero()};
};

Canonical canonical(const crane_model::State& x)
{
  constexpr std::array<Eigen::Index, 6> actuated{{0, 1, 2, 3, 6, 7}};
  Canonical out;
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    out.q[actuated[static_cast<std::size_t>(axis)]] = x[axis];
    out.dq[actuated[static_cast<std::size_t>(axis)]] = x[8 + axis];
  }
  out.q[4] = x[6];
  out.q[5] = x[7];
  out.dq[4] = x[14];
  out.dq[5] = x[15];
  return out;
}

casadi::DM column(const Eigen::Ref<const Eigen::VectorXd>& value)
{
  casadi::DM out = casadi::DM::zeros(static_cast<casadi_int>(value.size()));
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    out(static_cast<casadi_int>(index)) = value[index];
  }
  return out;
}

double entry(const casadi::DM& value, std::size_t index)
{
  return static_cast<double>(casadi::DM(value(static_cast<casadi_int>(index))));
}

double entry(const casadi::DM& value, std::size_t row, std::size_t column_index)
{
  return static_cast<double>(
    casadi::DM(value(static_cast<casadi_int>(row), static_cast<casadi_int>(column_index))));
}

// Six configurations and rates across the working range: folded, reaching,
// slewing fast, the telescope out, the tool axis moving, and the crane's own
// zero. Every actuated rate is well away from zero, so the axes are all drawing
// flow and the effort term of mpc §2 is not evaluated at a trivial point.
std::vector<crane_model::State> states()
{
  std::vector<crane_model::State> out;
  crane_model::State x;
  x << 0.4, 0.35, 0.9, 0.8, 0.6, 0.3, 0.12, 1.4,
    0.3, 0.4, -0.3, 0.2, 0.4, 0.25, 0.05, -0.03;
  out.push_back(x);
  x << -0.7, -0.9, 1.4, 0.1, -0.2, -0.5, -0.9, 0.6,
    -0.5, -0.2, 0.35, 0.3, -0.6, -0.3, -0.12, 0.08;
  out.push_back(x);
  x << 1.2, 0.8, 0.2, 1.6, 0.0, 0.0, 2.0, 0.9,
    0.9, 0.15, 0.2, 0.5, 0.2, 0.4, 0.0, 0.0;
  out.push_back(x);
  x << 0.0, 1.5, 0.4, 0.05, 0.35, 0.9, -1.5, 0.2,
    -0.2, 0.3, 0.45, -0.25, 0.5, -0.35, 0.2, 0.15;
  out.push_back(x);
  x << 0.9, -0.4, 1.1, 1.2, -0.45, 0.7, 0.4, 1.1,
    0.25, -0.3, -0.2, 0.4, -0.7, 0.3, -0.08, -0.2;
  out.push_back(x);
  x.setZero();
  x.tail<8>() = crane_model::Q::Constant(0.3);
  out.push_back(x);
  return out;
}

// Three payloads: none carried, the block, and ten times it grasped off centre.
// `valid == false` is not among them -- contract §4 refuses it, and
// `test_contract` is where that is asserted.
std::vector<crane_model::Payload> payloads()
{
  // Eigen leaves a fixed-size matrix uninitialized, so every payload here says
  // what its inertia is. An empty gripper is a *declared* payload of zero mass
  // (contract §4), not an absent one.
  crane_model::Payload empty;
  empty.valid = true;
  empty.center_of_mass_k8_m.setZero();
  empty.inertia_k8_kg_m2.setZero();

  crane_model::Payload block;
  block.valid = true;
  block.mass_kg = 42.0;
  block.center_of_mass_k8_m = Eigen::Vector3d(0.0, 0.0, -0.35);
  block.inertia_k8_kg_m2 = Eigen::Matrix3d::Identity() * 0.8;

  crane_model::Payload heavy;
  heavy.valid = true;
  heavy.mass_kg = 420.0;
  heavy.center_of_mass_k8_m = Eigen::Vector3d(0.08, -0.05, -0.6);
  heavy.inertia_k8_kg_m2 = Eigen::Matrix3d::Identity() * 12.0;

  return {empty, block, heavy};
}

// The graph and the numeric model run the same Pinocchio algorithm in the same
// double precision and differ only in the order the projection sums are
// accumulated, so what separates them is round-off and nothing else. This is
// that: a relative bound of 1e-9 against the largest entry in play, with a
// floor for quantities that are legitimately near zero.
double round_off(double scale)
{
  return 1.0e-9 * std::max(1.0, std::abs(scale));
}

// What mpc §3.1's smoothing is worth, per axis, at a given piston velocity.
// The graph's Q_i is A^{+-}_smooth(v) sqrt(v^2 + eps^2) and the numeric one is
// A^{+-}_step(v) |v|, so
//
//   |Q_graph - Q_numeric| <= A_max eps + |A^+ - A^-|/2 (1 - tanh(|v|/eps_v)) |v| ,
//
// which is a statement about the two smoothings and not a tolerance chosen to
// make the assertion pass. It collapses to A_max eps -- around 1e-8 m^3/s --
// once an axis is moving at more than a few millimetres per second.
double flow_smoothing_bound(std::size_t axis, double velocity)
{
  const crane_model::cylinder::AxisAreas areas = crane_model::cylinder::axis_areas()[axis];
  const double largest = std::max(areas.a_eff_pos, areas.a_eff_neg);
  const double step = std::abs(areas.a_eff_pos - areas.a_eff_neg);
  return largest * crane_model::symbolic::kEpsAbs +
         0.5 * step * (1.0 - std::tanh(std::abs(velocity) / crane_model::symbolic::kEpsV)) *
         std::abs(velocity);
}

// A description that parses, carries all eight canonical joints and the link a
// payload attaches to, and has no mass anywhere. Every entry of M is zero, so
// M_uu is singular at every configuration and the passive rows do not resolve:
// this is a description that cannot produce a usable graph, which is a
// different thing from a graph that came out the wrong size.
const char * const kMasslessDescription =
  "<robot name=\"fixture\">"
  "<link name=\"K0_mounting_base\"/><link name=\"a\"/><link name=\"b\"/><link name=\"c\"/>"
  "<link name=\"d\"/><link name=\"e\"/><link name=\"f\"/><link name=\"g\"/>"
  "<link name=\"K8_rotator_lower_part\"/>"
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
  "<parent link=\"g\"/><child link=\"K8_rotator_lower_part\"/>"
  "<axis xyz=\"0 0 1\"/><limit lower=\"0\" upper=\"1\" effort=\"1\" velocity=\"1\"/></joint>"
  "</robot>";

}  // namespace

TEST(CraneModelSymbolicGraph, ThePublicHandleReportsTheGraphItWasBuiltFrom)
{
  // Contract §9's frozen dimensions, read back off the built CasADi functions
  // by `build_graph` rather than compiled into the handle. The output map is
  // the spec's to ask for, and asking for it is what `has_output_map` reports.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = crane_model::Model::create(configuration_for(tool));
    ASSERT_TRUE(model.ok()) << model.status().message;
    const auto payload = payloads().at(1);

    const auto with_map = model.value().symbolic_graph({}, payload);
    ASSERT_TRUE(with_map.ok()) << with_map.status().message;
    EXPECT_EQ(with_map.value().state_dimension(), crane_model::kStateDof);
    EXPECT_EQ(with_map.value().input_dimension(), crane_model::kInputDof);
    EXPECT_TRUE(with_map.value().has_output_map());

    crane_model::SymbolicGraphSpec bare;
    bare.include_output_map = false;
    const auto without_map = model.value().symbolic_graph(bare, payload);
    ASSERT_TRUE(without_map.ok()) << without_map.status().message;
    EXPECT_EQ(without_map.value().state_dimension(), crane_model::kStateDof);
    EXPECT_EQ(without_map.value().input_dimension(), crane_model::kInputDof);
    EXPECT_FALSE(without_map.value().has_output_map());
  }
}

TEST(CraneModelSymbolicGraph, AConsumerLinksThePrivateTargetAndEvaluatesTheHandles)
{
  // Contract §9's other half: a consumer that opts in gets CasADi function
  // objects and can evaluate them. That is what this whole file does; this test
  // is the minimal statement of it, including the one shape the frozen
  // `SymbolicGraphSpec` controls.
  const crane_model::ModelConfig config = configuration_for(crane_model::Tool::Pzs100);
  const auto payload = payloads().at(1);

  const auto handle = crane_model::symbolic::casadi_graph(config, {}, payload);
  ASSERT_TRUE(handle.ok()) << handle.status().message;
  const crane_model::symbolic::CasadiGraph& graph = *handle.value();

  EXPECT_EQ(graph.f.n_in(), 2);
  EXPECT_EQ(graph.f.numel_in(0), static_cast<casadi_int>(crane_model::kStateDof));
  EXPECT_EQ(graph.f.numel_in(1), static_cast<casadi_int>(crane_model::kInputDof));
  EXPECT_EQ(graph.f.numel_out(0), static_cast<casadi_int>(crane_model::kStateDof));
  EXPECT_EQ(
    graph.z.numel_out(0), static_cast<casadi_int>(crane_model::symbolic::kOutputDof));
  EXPECT_EQ(graph.passive_rows.n_out(), 3);
  EXPECT_DOUBLE_EQ(graph.sample_time_s, crane_model::SymbolicGraphSpec{}.sample_time_s);

  const std::vector<casadi::DM> arguments{
    column(states().front()), column(Eigen::VectorXd::Zero(crane_model::kInputDof))};
  const casadi::DM derivative = graph.f(arguments).at(0);
  ASSERT_TRUE(derivative.is_regular());
  // The first six rows of xdot are dq_a by construction, whatever the dynamics
  // do -- so this is the cheapest possible proof that the handle evaluates the
  // state layout of contract §2 and not some other one.
  for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
    EXPECT_DOUBLE_EQ(
      entry(derivative, axis), states().front()[8 + static_cast<Eigen::Index>(axis)]);
  }

  crane_model::SymbolicGraphSpec bare;
  bare.include_output_map = false;
  const auto without_map = crane_model::symbolic::casadi_graph(config, bare, payload);
  ASSERT_TRUE(without_map.ok()) << without_map.status().message;
  EXPECT_TRUE(without_map.value()->z.is_null());
  EXPECT_FALSE(without_map.value()->f.is_null());
}

TEST(CraneModelSymbolicGraph, ThePassiveAccelerationIsTheOneTheNumericModelResolves)
{
  // The criterion the "one implementation" claim rests on, first half. The
  // graph's ddq_u is the Schur complement of robot_model §3.1, and the model's
  // own statement of that is the invariant of contract §7: at the consistent
  // acceleration the passive rows of the inverse dynamics vanish. So the graph
  // is asked for ddq_u, the answer is handed to `inverse_dynamics` as part of a
  // full ddq, and its passive rows have to be zero.
  //
  // Nothing in this test forms M_uu, M_ua or h_u. If the graph transcribed a
  // sign, the residual is a torque of order 1e3 N m and this fails at once.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = crane_model::Model::create(configuration_for(tool));
    ASSERT_TRUE(model.ok()) << model.status().message;

    for (const crane_model::Payload& payload : payloads()) {
      const auto handle =
        crane_model::symbolic::casadi_graph(configuration_for(tool), {}, payload);
      ASSERT_TRUE(handle.ok()) << handle.status().message;

      for (const crane_model::State& x : states()) {
        const Canonical state = canonical(x);
        crane_model::Input u;
        u << 0.2, -0.1, 0.05, 0.01, 0.3, -0.02;

        const casadi::DM derivative =
          handle.value()->f(std::vector<casadi::DM>{column(x), column(u)}).at(0);
        ASSERT_TRUE(derivative.is_regular());

        crane_model::DQ ddq = crane_model::DQ::Zero();
        constexpr std::array<Eigen::Index, 6> actuated{{0, 1, 2, 3, 6, 7}};
        for (Eigen::Index axis = 0; axis < 6; ++axis) {
          ddq[actuated[static_cast<std::size_t>(axis)]] = u[axis];
          // The rows of xdot that carry ddq_a are u itself, exactly.
          EXPECT_DOUBLE_EQ(entry(derivative, static_cast<std::size_t>(8 + axis)), u[axis]);
        }
        ddq[4] = entry(derivative, 14);
        ddq[5] = entry(derivative, 15);
        // And the rows that carry dq_u are dq_u.
        EXPECT_DOUBLE_EQ(entry(derivative, 6), state.dq[4]);
        EXPECT_DOUBLE_EQ(entry(derivative, 7), state.dq[5]);

        const auto tau = model.value().inverse_dynamics(state.q, state.dq, ddq, payload);
        ASSERT_TRUE(tau.ok()) << tau.status().message;
        // The bound contract §7's invariant is asserted at, in N m.
        EXPECT_NEAR(tau.value()[4], 0.0, 1.0e-6) << "tip, " << x.transpose();
        EXPECT_NEAR(tau.value()[5], 0.0, 1.0e-6) << "tilt, " << x.transpose();
      }
    }
  }
}

TEST(CraneModelSymbolicGraph, TheActuatedForceIsTheReductionTheNumericModelReturns)
{
  // Second half. `reduced_actuated_dynamics` is robot_model §3.4, and the
  // output map's tau_a block is the same quantity assembled the other way --
  // from the actuated rows of M ddq + h at the graph's own ddq_u. The two are
  // the same double arithmetic in a different order, so they agree to
  // round-off; anything larger is a difference in the equations.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = crane_model::Model::create(configuration_for(tool));
    ASSERT_TRUE(model.ok()) << model.status().message;

    for (const crane_model::Payload& payload : payloads()) {
      const auto handle =
        crane_model::symbolic::casadi_graph(configuration_for(tool), {}, payload);
      ASSERT_TRUE(handle.ok()) << handle.status().message;

      for (const crane_model::State& x : states()) {
        const Canonical state = canonical(x);
        crane_model::Input u;
        u << 0.2, -0.1, 0.05, 0.01, 0.3, -0.02;

        const casadi::DM z =
          handle.value()->z(std::vector<casadi::DM>{column(x), column(u)}).at(0);
        ASSERT_TRUE(z.is_regular());

        const auto reduced =
          model.value().reduced_actuated_dynamics(state.q, state.dq, u, payload);
        ASSERT_TRUE(reduced.ok()) << reduced.status().message;
        const crane_model::Vector6 tau_a =
          reduced.value().mass_eff * u + reduced.value().bias_eff;

        for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
          EXPECT_NEAR(
            entry(z, crane_model::symbolic::kActuatedForceOffset + axis),
            tau_a[static_cast<Eigen::Index>(axis)],
            round_off(tau_a.cwiseAbs().maxCoeff())) << "axis " << axis << ", " << x.transpose();
        }
      }
    }
  }
}

TEST(CraneModelSymbolicGraph, ThePassiveRowsArePartOfTheSameMassMatrix)
{
  // What `wiki/trajectory_planning.md` §5.2 carries out of the graph: M_uu,
  // M_ua and h_u, with h_u already containing D_uu dq_u. The oracle is
  // `full_dynamics`, whose own oracle is the kinetic and potential energy of
  // the description (`test_contract`), so this ties the planner's OCP back to
  // the description through two independent steps and not through a second
  // copy of the algebra.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = crane_model::Model::create(configuration_for(tool));
    ASSERT_TRUE(model.ok()) << model.status().message;

    for (const crane_model::Payload& payload : payloads()) {
      const auto handle =
        crane_model::symbolic::casadi_graph(configuration_for(tool), {}, payload);
      ASSERT_TRUE(handle.ok()) << handle.status().message;

      for (const crane_model::State& x : states()) {
        const Canonical state = canonical(x);
        const std::vector<casadi::DM> rows =
          handle.value()->passive_rows(std::vector<casadi::DM>{column(x)});
        ASSERT_EQ(rows.size(), 3U);

        const auto full = model.value().full_dynamics(state.q, state.dq, payload);
        ASSERT_TRUE(full.ok()) << full.status().message;
        const double mass_scale = full.value().mass.cwiseAbs().maxCoeff();
        const double bias_scale = full.value().bias.cwiseAbs().maxCoeff();

        constexpr std::array<Eigen::Index, 6> actuated{{0, 1, 2, 3, 6, 7}};
        constexpr std::array<Eigen::Index, 2> passive{{4, 5}};
        for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
          for (std::size_t column_index = 0; column_index < crane_model::kPassiveDof;
            ++column_index)
          {
            EXPECT_NEAR(
              entry(rows.at(0), row, column_index),
              full.value().mass(passive[row], passive[column_index]),
              round_off(mass_scale)) << "M_uu " << row << column_index;
          }
          for (std::size_t column_index = 0; column_index < crane_model::kActuatedDof;
            ++column_index)
          {
            EXPECT_NEAR(
              entry(rows.at(1), row, column_index),
              full.value().mass(passive[row], actuated[column_index]),
              round_off(mass_scale)) << "M_ua " << row << column_index;
          }
          EXPECT_NEAR(
            entry(rows.at(2), row), full.value().bias[passive[row]],
            round_off(bias_scale)) << "h_u " << row;
        }
      }
    }
  }
}

TEST(CraneModelSymbolicGraph, TheOutputMapCarriesTheTransmissionTheMpcConstrains)
{
  // Acceptance: a consumer must not have to rebuild the transmission in its own
  // solver. So the output map's v and Q blocks are checked against
  // `Model::transmission` at the same argument, and its F_cyl block against the
  // cylinder force tau_a induces through the same J_cyl.
  //
  // v is bit-exact -- both instantiate one template. Q is not, and must not be:
  // mpc §3.1 requires the model's Q to be smoothed for a gradient-based solver
  // while `transmission` keeps the physical step. `flow_smoothing_bound` is what
  // that difference is worth.
  for (const auto tool : {crane_model::Tool::Pzs100, crane_model::Tool::Epsilon7040}) {
    const auto model = crane_model::Model::create(configuration_for(tool));
    ASSERT_TRUE(model.ok()) << model.status().message;
    const auto payload = payloads().at(1);
    const auto handle =
      crane_model::symbolic::casadi_graph(configuration_for(tool), {}, payload);
    ASSERT_TRUE(handle.ok()) << handle.status().message;

    for (const crane_model::State& x : states()) {
      const Canonical state = canonical(x);
      crane_model::Input u;
      u << 0.2, -0.1, 0.05, 0.01, 0.3, -0.02;
      crane_model::DQA dq_a;
      for (Eigen::Index axis = 0; axis < 6; ++axis) {
        dq_a[axis] = x[8 + axis];
      }

      const casadi::DM z =
        handle.value()->z(std::vector<casadi::DM>{column(x), column(u)}).at(0);
      ASSERT_TRUE(z.is_regular());

      crane_model::ChamberPressure pressure;
      pressure.p_a_pa.setZero();
      pressure.p_b_pa.setZero();
      const auto transmission = model.value().transmission(state.q, dq_a, pressure);
      ASSERT_TRUE(transmission.ok()) << transmission.status().message;
      const auto cylinder = model.value().cylinder_jacobian(state.q);
      ASSERT_TRUE(cylinder.ok()) << cylinder.status().message;

      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        const Eigen::Index index = static_cast<Eigen::Index>(axis);
        const double velocity = transmission.value().cylinder_velocity[index];
        EXPECT_DOUBLE_EQ(
          entry(z, crane_model::symbolic::kPistonVelocityOffset + axis), velocity)
          << "v, axis " << axis;
        EXPECT_NEAR(
          entry(z, crane_model::symbolic::kAxisFlowOffset + axis),
          transmission.value().pump_flow[index],
          flow_smoothing_bound(axis, velocity)) << "Q, axis " << axis;

        // Constraint 6 in the form the graph hands over: F_i = tau_{a,i} /
        // J_{c,ii}, so a consumer bounds it by F_i^max directly.
        const double ratio = cylinder.value()(index, index);
        ASSERT_NE(ratio, 0.0);
        const double tau_a = entry(z, crane_model::symbolic::kActuatedForceOffset + axis);
        EXPECT_NEAR(
          entry(z, crane_model::symbolic::kCylinderForceOffset + axis), tau_a / ratio,
          round_off(std::abs(tau_a / ratio))) << "F_cyl, axis " << axis;
      }
    }
  }
}

TEST(CraneModelSymbolicGraph, SampleTimeIsTheDiscretizationStep)
{
  // `SymbolicGraphSpec::sample_time_s` is honoured, and honoured as the ERK4
  // step mpc §5.2 asks for -- not as a label. The oracle is an ERK4 step this
  // test takes itself through the graph's own `f`, so the assertion is that
  // `F_disc` is that integrator at that step and nothing else.
  const crane_model::ModelConfig config = configuration_for(crane_model::Tool::Epsilon7040);
  const auto payload = payloads().at(1);

  for (const double sample_time : {0.04, 0.02, 0.1}) {
    crane_model::SymbolicGraphSpec spec;
    spec.sample_time_s = sample_time;
    const auto handle = crane_model::symbolic::casadi_graph(config, spec, payload);
    ASSERT_TRUE(handle.ok()) << handle.status().message;
    const crane_model::symbolic::CasadiGraph& graph = *handle.value();
    EXPECT_DOUBLE_EQ(graph.sample_time_s, sample_time);

    for (const crane_model::State& x : states()) {
      crane_model::Input u;
      u << 0.2, -0.1, 0.05, 0.01, 0.3, -0.02;
      const casadi::DM state = column(x);
      const casadi::DM command = column(u);
      const auto derivative = [&graph, &command](const casadi::DM& at) {
          return graph.f(std::vector<casadi::DM>{at, command}).at(0);
        };

      const casadi::DM k1 = derivative(state);
      const casadi::DM k2 = derivative(state + (0.5 * sample_time) * k1);
      const casadi::DM k3 = derivative(state + (0.5 * sample_time) * k2);
      const casadi::DM k4 = derivative(state + sample_time * k3);
      const casadi::DM expected =
        state + (sample_time / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

      const casadi::DM next = graph.F_disc(std::vector<casadi::DM>{state, command}).at(0);
      ASSERT_TRUE(next.is_regular());
      for (std::size_t row = 0; row < crane_model::kStateDof; ++row) {
        EXPECT_NEAR(entry(next, row), entry(expected, row), round_off(entry(expected, row)))
          << "row " << row << " at Ts = " << sample_time;
      }
      // And it really is a step of that length: the position rows advance by
      // the sample time times their own rate, to the order an integrator does.
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        EXPECT_NEAR(
          entry(next, axis) - x[static_cast<Eigen::Index>(axis)],
          sample_time * x[static_cast<Eigen::Index>(8 + axis)],
          0.6 * sample_time * sample_time) << "axis " << axis;
      }
    }
  }
}

TEST(CraneModelSymbolicGraph, ADescriptionThatCannotProduceAGraphIsAFailureNotAGraph)
{
  // Contract §5: a failure is a non-`Ok` status and no value, never a value
  // that happens to be degenerate. The massless fixture parses, carries all
  // eight canonical joints and the payload link, and builds an expression whose
  // passive rows are 0/0 -- so the graph exists as an expression and is not a
  // model, and that is `SymbolicBackendFailure`.
  crane_model::ModelConfig config;
  config.robot_description_xml = kMasslessDescription;
  const auto model = crane_model::Model::create(config);
  ASSERT_TRUE(model.ok()) << model.status().message;

  const crane_model::Payload payload = payloads().front();

  const auto graph = model.value().symbolic_graph({}, payload);
  EXPECT_FALSE(graph.ok());
  EXPECT_EQ(graph.status().code, crane_model::ErrorCode::SymbolicBackendFailure);
  EXPECT_FALSE(graph.status().message.empty());

  const auto handle = crane_model::symbolic::casadi_graph(config, {}, payload);
  EXPECT_FALSE(handle.ok());
  EXPECT_EQ(handle.status().code, crane_model::ErrorCode::SymbolicBackendFailure);

  // A spec that is not a discretization is a different failure, with a
  // different code: a caller can tell "your step is not a step" from "this
  // description does not give a model".
  const crane_model::ModelConfig real = configuration_for(crane_model::Tool::Pzs100);
  const auto working = crane_model::Model::create(real);
  ASSERT_TRUE(working.ok()) << working.status().message;
  for (const double sample_time : {0.0, -0.04, std::numeric_limits<double>::infinity()}) {
    crane_model::SymbolicGraphSpec spec;
    spec.sample_time_s = sample_time;
    const auto refused = working.value().symbolic_graph(spec, payloads().at(1));
    EXPECT_FALSE(refused.ok()) << "Ts = " << sample_time;
    EXPECT_EQ(refused.status().code, crane_model::ErrorCode::InvalidArgument);
  }

  // And an unknown payload is refused before the backend is reached at all,
  // exactly as it is on the numeric calls (contract §4).
  const auto unknown = working.value().symbolic_graph({}, crane_model::Payload{});
  EXPECT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.status().code, crane_model::ErrorCode::InvalidPayload);
}

TEST(CraneModelSymbolicGraph, BuildingAGraphIsNotARealTimeCall)
{
  // Contract §10 lists symbolic-graph creation as non-real-time, and this is
  // the demonstration rather than the assertion: one construction allocates,
  // and it allocates a lot. It is not a call to put on the 100 Hz path, and the
  // README says so.
  const crane_model::ModelConfig config = configuration_for(crane_model::Tool::Pzs100);
  const auto payload = payloads().at(1);
  ASSERT_TRUE(crane_model::symbolic::casadi_graph(config, {}, payload).ok());

  g_allocation_count.store(0, std::memory_order_relaxed);
  g_allocation_guard.store(true, std::memory_order_relaxed);
  const bool ok = crane_model::symbolic::casadi_graph(config, {}, payload).ok();
  g_allocation_guard.store(false, std::memory_order_relaxed);
  EXPECT_TRUE(ok);
  EXPECT_GT(g_allocation_count.load(std::memory_order_relaxed), 1000U);
}
