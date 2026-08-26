// The tool axis in the **numeric model**, after the coordinate left both OCPs.
//
// Issue 068 drops the tool coordinate from `crane_mpc`'s OCP and from
// `crane_planning`'s timing OCP: the gripper is opened and closed by the
// low-level velocity/position controller, so its motion does not need planning.
// What it does **not** drop is the tool: the link keeps its mass and its inertia,
// the cylinder keeps its transmission, and both are still `crane_model`'s. This
// file is where that is asserted, in the package that owns it, so that "the
// transmission stayed behind" is a property something tests rather than a
// sentence in a header.
//
// It carries the model-level half of `crane_planning/test/test_tool_axis.cpp`.
// The other half -- the grip cosine, `drive_tool_axis`, the phase limits -- could
// not come with it: those are `crane_planning`'s own functions and `crane_model`
// may not depend on a planner. They stay where they are and still run.
//
// Offline C++ against both machine descriptions. Nothing here launches, links a
// simulator or reaches a ROS graph.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "crane_model/model.hpp"

namespace
{

/// The tool joint is canonical coordinate #8 of contract §2, i.e. row seven.
constexpr Eigen::Index kToolCoordinate = 7;

/// The actuated projection of contract §2, so an arm row can be named.
constexpr std::array<Eigen::Index, crane_model::kActuatedDof> kActuatedRows{{0, 1, 2, 3, 6, 7}};

struct Machine
{
  const char * name;
  const char * file;
  crane_model::Tool tool;
  const char * tool_joint;
};

const std::array<Machine, 2> & machines()
{
  static const std::array<Machine, 2> all{
    {{"pzs100", "pzs100.urdf", crane_model::Tool::Pzs100, "q9_left_rail_joint"},
      {"epsilon_7040", "epsilon_7040.urdf", crane_model::Tool::Epsilon7040,
        "theta10_outer_jaw_joint"}}};
  return all;
}

std::string read_description(const char * file)
{
  const std::string path = std::string(CRANE_MODEL_TEST_DESCRIPTION_DIR) + "/" + file;
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "cannot read " << path;
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

/// The range the **description** gives a joint, read out of it rather than typed.
/**
 * The tool axis is swept across its own range below, and the two ends of that
 * range are description facts: `0.2 ... 0.7 m` on the PZS100's rail and
 * `0 ... 3.15 rad` on the 7040's jaw. Writing them here as constants would make
 * this file a second place they live, and the reason the crossing at `0.2623 rad`
 * matters at all is that it is *inside* the range the description states.
 */
struct AxisRange
{
  double lower{};
  double upper{};
};

AxisRange joint_range(const std::string & xml, const std::string & joint)
{
  const std::size_t at = xml.find("<joint name=\"" + joint + "\"");
  EXPECT_NE(at, std::string::npos) << "the description carries no joint " << joint;
  if (at == std::string::npos) {
    return AxisRange{};
  }
  const std::size_t end = xml.find("</joint>", at);
  const std::size_t limit = xml.find("<limit", at);
  EXPECT_LT(limit, end) << joint << " has no <limit>, so it has no range to sweep";
  const auto attribute = [&](const char * name) {
      const std::size_t key = xml.find(std::string(name) + "=\"", limit);
      EXPECT_LT(key, end) << joint << " has no " << name;
      const std::size_t value = xml.find('"', key) + 1U;
      return std::stod(xml.substr(value, xml.find('"', value) - value));
    };
  return AxisRange{attribute("lower"), attribute("upper")};
}

crane_model::Model build(const Machine & machine)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = read_description(machine.file);
  config.tool = machine.tool;
  auto model = crane_model::Model::create(config);
  EXPECT_TRUE(model.ok()) << machine.name << ": " << model.status().message;
  return std::move(model).value();
}

crane_model::Payload empty_gripper()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 0.0;
  payload.center_of_mass_k8_m.setZero();
  payload.inertia_k8_kg_m2.setZero();
  return payload;
}

/// A pose inside the working range, hanging at its own equilibrium.
crane_model::Q settled(const crane_model::Model & model, const crane_model::QA & q_a)
{
  crane_model::Q q = crane_model::Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[kActuatedRows[row]] = q_a[static_cast<Eigen::Index>(row)];
  }
  const auto equilibrium = model.passive_equilibrium(q_a, empty_gripper());
  EXPECT_TRUE(equilibrium.ok()) << equilibrium.status().message;
  if (equilibrium.ok()) {
    q[4] = equilibrium.value()[0];
    q[5] = equilibrium.value()[1];
  }
  return q;
}

/// The working pose the tests share, with the tool axis left at `q8`.
crane_model::QA working_pose(const AxisRange & range, double fraction)
{
  crane_model::QA q_a;
  q_a << 0.2, 0.4, 1.2, 0.6, 0.1, range.lower + fraction * (range.upper - range.lower);
  return q_a;
}

/// `J_c,GR(q8)`, the tool row of the cylinder Jacobian at one tool coordinate.
double tool_ratio(const crane_model::Model & model, crane_model::Q q, double q8)
{
  q[kToolCoordinate] = q8;
  const auto jacobian = model.cylinder_jacobian(q);
  EXPECT_TRUE(jacobian.ok()) << jacobian.status().message;
  if (!jacobian.ok()) {
    return 0.0;
  }
  const Eigen::Index row = static_cast<Eigen::Index>(crane_model::kActuatedDof - 1U);
  return jacobian.value()(row, row);
}

}  // namespace

TEST(CraneModelToolAxis, TheToolRowOfTheCylinderJacobianIsAFunctionOfTheToolCoordinateAlone)
{
  // Moved from `crane_planning`, where `probe_tool_transmission` sweeps `q8`
  // across one `Q` and leaves the arm and the passive pair where the caller put
  // them. That is only sound because the tool row of `J_cyl` depends on nothing
  // else, and this is where it is measured rather than assumed -- if a later
  // description coupled the jaw to the arm, the probe would be reading the wrong
  // number and this fails first. It belongs here now that the coordinate has left
  // both OCPs: the transmission is the model's whatever the solvers plan.
  for (const Machine & machine : machines()) {
    const crane_model::Model model = build(machine);
    const AxisRange range = joint_range(read_description(machine.file), machine.tool_joint);

    crane_model::Q left = settled(model, working_pose(range, 0.25));
    crane_model::QA elsewhere = working_pose(range, 0.25);
    elsewhere[0] += 0.7;   // slew
    elsewhere[1] += 0.2;   // boom
    elsewhere[2] -= 0.3;   // arm
    elsewhere[3] += 0.3;   // telescope
    crane_model::Q right = settled(model, elsewhere);
    // ...and a passive pair deliberately away from either equilibrium.
    right[4] += 0.15;
    right[5] -= 0.12;

    for (int step = 0; step <= 8; ++step) {
      const double q8 = range.lower + (range.upper - range.lower) * step / 8.0;
      EXPECT_NEAR(tool_ratio(model, left, q8), tool_ratio(model, right, q8), 1.0e-15)
        << machine.name << " at q8 = " << q8;
    }
  }
}

TEST(CraneModelToolAxis, TheJawReversesSignInsideItsOwnRangeAndTheRailDoesNot)
{
  // Issue 037's notes recorded this reversal on the 7040 and put it near
  // `q8 = 0.29 rad` -- the linear interpolation of the six-point table they
  // tabulate, between `-0.016 m/rad` at `0.20` and `+0.057 m/rad` at `0.50`.
  // Bisected off the model the root is **0.2623 rad**, which is what
  // `crane_planning` refuses a travel across, so that is what is pinned here. The
  // PZS100's rail cylinder is one-to-one with `q8` over the whole of its range, so
  // there is nothing to cross there.
  //
  // The whole point of keeping this after issue 068: an axis with a transmission
  // zero inside its own range is a real property of this machine, and it does not
  // stop being one because no OCP plans that axis any more.
  for (const Machine & machine : machines()) {
    const crane_model::Model model = build(machine);
    const AxisRange range = joint_range(read_description(machine.file), machine.tool_joint);
    const crane_model::Q q = settled(model, working_pose(range, 0.25));

    double smallest = std::numeric_limits<double>::infinity();
    double previous = tool_ratio(model, q, range.lower);
    double crossing = std::numeric_limits<double>::quiet_NaN();
    constexpr int kSamples = 64;
    for (int step = 1; step <= kSamples; ++step) {
      const double q8 = range.lower + (range.upper - range.lower) * step / kSamples;
      const double ratio = tool_ratio(model, q, q8);
      smallest = std::min(smallest, std::abs(ratio));
      if (previous * ratio < 0.0 && std::isnan(crossing)) {
        // Bracketed; bisect for the root the planner reports.
        double low = range.lower + (range.upper - range.lower) * (step - 1) / kSamples;
        double high = q8;
        for (int bisect = 0; bisect < 60; ++bisect) {
          const double middle = 0.5 * (low + high);
          if (tool_ratio(model, q, low) * tool_ratio(model, q, middle) <= 0.0) {
            high = middle;
          } else {
            low = middle;
          }
        }
        crossing = 0.5 * (low + high);
      }
      previous = ratio;
    }

    if (machine.tool == crane_model::Tool::Epsilon7040) {
      ASSERT_FALSE(std::isnan(crossing)) << "the 7040 jaw's four-bar no longer reverses sign";
      EXPECT_NEAR(crossing, 0.2623, 1.0e-3)
        << "the jaw four-bar's own root, which issue 037's table brackets";
      // The table itself, at the two entries that bracket the root: the cylinder
      // retracts as the jaw opens on one side of it and extends on the other.
      EXPECT_NEAR(tool_ratio(model, q, 0.2), -0.016, 1.0e-3);
      EXPECT_NEAR(tool_ratio(model, q, 0.5), 0.057, 1.0e-3);
    } else {
      EXPECT_TRUE(std::isnan(crossing)) << "the PZS100 rail cylinder is one-to-one with q8";
      EXPECT_GT(smallest, 1.0e-3) << machine.name;
    }
  }
}

TEST(CraneModelToolAxis, TheChamberSwitchStillReachesTheToolAxisPumpDraw)
{
  // `Q_i = A_i^{+-}(v_i) |v_i|`: the effective area is a step at `v = 0`, so the
  // draw per unit of piston travel is not the same extending and retracting. The
  // tool axis has that step like every other, and `Model::transmission` is where
  // it lives -- it is the numeric twin of the expression the OCPs read off the
  // symbolic graph, and it stays whole now that the OCPs no longer sum the tool's
  // term into constraint 7.
  for (const Machine & machine : machines()) {
    const crane_model::Model model = build(machine);
    const AxisRange range = joint_range(read_description(machine.file), machine.tool_joint);
    const crane_model::Q q = settled(model, working_pose(range, 0.25));
    const Eigen::Index tool = static_cast<Eigen::Index>(crane_model::kActuatedDof - 1U);

    crane_model::ChamberPressure quiescent;
    quiescent.p_a_pa.setZero();
    quiescent.p_b_pa.setZero();

    crane_model::DQA opening = crane_model::DQA::Zero();
    opening[tool] = 0.1;
    crane_model::DQA closing = crane_model::DQA::Zero();
    closing[tool] = -0.1;

    const auto out = model.transmission(q, opening, quiescent);
    const auto back = model.transmission(q, closing, quiescent);
    ASSERT_TRUE(out.ok() && back.ok()) << machine.name;

    EXPECT_GT(out.value().pump_flow[tool], 0.0) << machine.name;
    EXPECT_GT(back.value().pump_flow[tool], 0.0) << machine.name;
    EXPECT_NE(out.value().pump_flow[tool], back.value().pump_flow[tool])
      << machine.name
      << ": the two directions drew the same flow at the same rate, so the chamber switch is not "
         "reaching the tool axis";
    // ...and the draw is the transmission's, not a fixed number: it is the piston
    // velocity `J_c,GR dq8` that the area multiplies.
    EXPECT_NEAR(
      std::abs(out.value().cylinder_velocity[tool]),
      std::abs(tool_ratio(model, q, q[kToolCoordinate]) * opening[tool]), 1.0e-12) << machine.name;
  }
}

TEST(CraneModelToolAxis, TheToolLinkKeepsItsInertiaAndTheArmCarriesIt)
{
  // Dropping the coordinate from the OCPs is a statement about what is *planned*
  // and never about what has mass. The OCPs pin the tool link at its commanded
  // configuration and evaluate `crane_model`'s own eight-coordinate graph there,
  // so the arm is still planned against the inertia it is really carrying -- and
  // that is only worth anything if the inertia is in `M(q)` in the first place.
  //
  // Two statements, and the second is the one that would break silently: the tool
  // coordinate has inertia of its own, and the **arm** rows of `M` change when the
  // tool moves. If the tool link were massless -- or absent from the description
  // -- both would be zero and every arm row would be blind to `q8`.
  for (const Machine & machine : machines()) {
    const crane_model::Model model = build(machine);
    const AxisRange range = joint_range(read_description(machine.file), machine.tool_joint);

    const crane_model::Q closed = settled(model, working_pose(range, 0.0));
    const crane_model::Q open = settled(model, working_pose(range, 1.0));
    const auto at_closed = model.full_dynamics(closed, crane_model::DQ::Zero(), empty_gripper());
    const auto at_open = model.full_dynamics(open, crane_model::DQ::Zero(), empty_gripper());
    ASSERT_TRUE(at_closed.ok() && at_open.ok()) << machine.name;

    // The tool's own diagonal inertia. Positive definite mass matrix, so this is
    // strictly positive for a link that has mass and exactly zero for one that
    // does not.
    EXPECT_GT(at_closed.value().mass(kToolCoordinate, kToolCoordinate), 0.0) << machine.name;
    EXPECT_GT(at_open.value().mass(kToolCoordinate, kToolCoordinate), 0.0) << machine.name;

    // ...and the arm feels it. The four arm rows of `M` are compared across the
    // whole of the tool's travel: a tool that weighs nothing, or one the arm's
    // rows do not see, would leave this block identical.
    double largest = 0.0;
    for (Eigen::Index row = 0; row < 4; ++row) {
      for (Eigen::Index column = 0; column < 4; ++column) {
        largest = std::max(
          largest,
          std::abs(
            at_closed.value().mass(kActuatedRows[static_cast<std::size_t>(row)],
            kActuatedRows[static_cast<std::size_t>(column)]) -
            at_open.value().mass(kActuatedRows[static_cast<std::size_t>(row)],
            kActuatedRows[static_cast<std::size_t>(column)])));
      }
    }
    EXPECT_GT(largest, 1.0e-6)
      << machine.name
      << ": the arm's own block of the mass matrix did not move when the tool did, so the tool's "
         "inertia is not reaching it";

    // ...and so does the pendulum, which is the half of it the MPC's sway terms
    // are about: `M_uu` is what sets the period of `theta6_tip` and `theta7_tilt`,
    // and the tool is the mass hanging on them. An OCP that planned against a
    // tool-less `M_uu` would be predicting the wrong sway.
    double passive = 0.0;
    for (Eigen::Index row = 4; row <= 5; ++row) {
      for (Eigen::Index column = 4; column <= 5; ++column) {
        passive = std::max(
          passive,
          std::abs(at_closed.value().mass(row, column) - at_open.value().mass(row, column)));
      }
    }
    EXPECT_GT(passive, 1.0e-6)
      << machine.name << ": M_uu did not move when the tool did, so the sway the OCPs plan against "
                         "is a pendulum with no tool on it";

    // What the arm's **static** load does not do, so that nobody adds it later:
    // at a settled pose the tool's weight hangs straight down through the tilt
    // axis whatever `q8` is (`wiki/robot_model.md` 5.1), so the gravity torque on
    // the arm rows is invariant in the tool coordinate to round-off. The inertia
    // moves; the held load does not. Measured, not assumed -- it is under `1e-9`
    // on the 7040 and under `1e-14` on the PZS100.
    const auto held_closed = model.inverse_dynamics(
      closed, crane_model::DQ::Zero(), crane_model::DQ::Zero(), empty_gripper());
    const auto held_open = model.inverse_dynamics(
      open, crane_model::DQ::Zero(), crane_model::DQ::Zero(), empty_gripper());
    ASSERT_TRUE(held_closed.ok() && held_open.ok()) << machine.name;
    EXPECT_LT(
      (held_closed.value().segment(0, 4) - held_open.value().segment(0, 4)).cwiseAbs().maxCoeff(),
      1.0e-8) << machine.name;
  }
}
