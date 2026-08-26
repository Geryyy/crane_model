// The Python symbolic model, against the numeric `crane_model`.
//
// `scripts/crane_symbolic.py` states the crane's dynamics a second time, in
// Python, so that issues 071 and 073 can build their acados problems out of it.
// Two statements of the same equations is exactly the failure the C++ symbolic
// graph exists to rule out, so the Python one arrives with this file beside it:
// the algebra and the OCP assembly are different jobs, and the algebra is the
// one that can silently be wrong.
//
// The oracle is `crane_model`'s **numeric** public API -- `full_dynamics`,
// `inverse_dynamics`, `reduced_actuated_dynamics`, `cylinder_jacobian`,
// `transmission`, `cylinder_force`, `passive_equilibrium` -- and never
// `symbolic_graph.cpp` or any second copy of the algebra. That choice is not
// stylistic: the numeric model survives the port (planner IK, collision and
// `passive_equilibrium` need it), the C++ symbolic graph does not, so comparing
// against the graph would tie this test to something scheduled for deletion.
//
// The Python model reaches this file as C. `scripts/export_model_fixture.py`
// runs CasADi's code generation on it and checks the result into
// `test/generated/`, so the comparison needs no Python interpreter, no acados
// and no solver -- and a disagreement here is about the dynamics rather than
// about a toolchain. That the checked-in C is what the model still generates is
// the separate `export_model_fixture_is_current` test.
//
// Offline C++ against both machine descriptions. Nothing here launches, links a
// simulator or reaches a ROS graph.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "crane_model/model.hpp"

// Private to the implementation; on this test's include path only, so the
// smoothing bound below is written in the same effective areas the model uses
// rather than in a magic epsilon.
#include "cylinder_geometry.hpp"

// The generated declarations. `casadi_real` and `casadi_int` come with them, and
// so do the `_SZ_ARG`/`_SZ_RES` sizes, so nothing here restates the C interface
// of a function CasADi wrote.
#include "crane_symbolic_fixture_api.inc"  // NOLINT(build/include)

namespace
{

// --- the two coordinate sets -------------------------------------------------
//
// `crane_model` still carries the canonical eight of contract §2 and a
// sixteen-state; issue 068's reduction is at the OCP boundary, which is what the
// Python model is written at. So this file holds both and converts once.

/// The actuated projection [0, 1, 2, 3, 6, 7] and the passive one [4, 5].
constexpr std::array<Eigen::Index, crane_model::kActuatedDof> kActuatedRows{{0, 1, 2, 3, 6, 7}};
constexpr std::array<Eigen::Index, crane_model::kPassiveDof> kPassiveRows{{4, 5}};

/// The tool's slot among the actuated six, i.e. `cylinder::kToolAxis`.
constexpr std::size_t kToolAxis = crane_model::kActuatedDof - 1U;

/// What the OCPs plan after issue 068: the actuated six without the tool.
constexpr std::size_t kPlannedDof = crane_model::kActuatedDof - 1U;
constexpr std::size_t kNx = 2U * kPlannedDof + 2U * crane_model::kPassiveDof;
constexpr std::size_t kNu = kPlannedDof;
constexpr std::size_t kNp = 11U;

/// The blocks of `x`, in the order `crane_mpc/src/ocp.cpp` reduces to.
constexpr std::size_t kPlannedPosition = 0U;
constexpr std::size_t kPassivePosition = kPlannedDof;
constexpr std::size_t kPlannedVelocity = kPlannedDof + crane_model::kPassiveDof;
constexpr std::size_t kPassiveVelocity = 2U * kPlannedDof + crane_model::kPassiveDof;

/// The output map's four blocks of six, the offsets of `casadi_graph.hpp`.
constexpr std::size_t kOutputDof = 24U;
constexpr std::size_t kActuatedForceOffset = 0U;
constexpr std::size_t kCylinderForceOffset = 6U;
constexpr std::size_t kPistonVelocityOffset = 12U;
constexpr std::size_t kAxisFlowOffset = 18U;

/// The parameter vector: the pinned tool coordinate, then the payload body.
constexpr std::size_t kToolParameter = 0U;
constexpr std::size_t kPayloadParameter = 1U;

// --- calling the generated C -------------------------------------------------

using Entry = int (*)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
using Work = int (*)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);

/// One call into the fixture.
/**
 * Every function in it reports `SZ_IW == 0` and `SZ_W == 0`: the dynamics are
 * one straight-line expression and there is nothing to carry between calls, so
 * the two workspaces are null here. `TheFixtureNeedsNoWorkspace` asserts that
 * rather than leaving it assumed.
 */
void call(
  Entry entry, std::vector<const casadi_real *> arguments,
  std::vector<casadi_real *> results)
{
  ASSERT_EQ(entry(arguments.data(), results.data(), nullptr, nullptr, 0), 0);
}

/// The seven outputs of `<tool>_evaluate`, in the order it declares them.
/**
 * Matrices are dense and column major, which is what `ca.densify` in the export
 * script is for: a sparse output would make every read here a search through a
 * compressed column pattern, and the pattern is not what this file is testing.
 */
struct Evaluation
{
  std::array<double, kNx> xdot{};
  std::array<double, kOutputDof> z{};
  std::array<double, crane_model::kGeneralizedDof * crane_model::kGeneralizedDof> mass{};
  std::array<double, crane_model::kGeneralizedDof> bias{};
  std::array<double, crane_model::kPassiveDof * crane_model::kPassiveDof> mass_uu{};
  std::array<double, crane_model::kPassiveDof * kPlannedDof> mass_ua{};
  std::array<double, crane_model::kPassiveDof> bias_u{};

  [[nodiscard]] double m(std::size_t row, std::size_t column) const
  {
    return mass[row + crane_model::kGeneralizedDof * column];
  }
  [[nodiscard]] double m_uu(std::size_t row, std::size_t column) const
  {
    return mass_uu[row + crane_model::kPassiveDof * column];
  }
  [[nodiscard]] double m_ua(std::size_t row, std::size_t column) const
  {
    return mass_ua[row + crane_model::kPassiveDof * column];
  }
};

// --- the descriptions and the generated entry points --------------------------

/// The generated symbols of one description, bound by name.
/**
 * `mimic_rail` is null on the 7040, which carries no rail: the export writes one
 * function per `<mimic>` the description declares, named after the joint it
 * reconstructs, so a description that lost a mimic is a missing symbol at link
 * time rather than a plausible mass matrix at run time.
 */
struct Generated
{
  Entry evaluate{nullptr};
  Work evaluate_work{nullptr};
  Entry cylinder_jacobian{nullptr};
  Entry chamber_force{nullptr};
  Entry damping{nullptr};
  Entry mimic_telescope{nullptr};
  Entry mimic_rail{nullptr};
};

struct Machine
{
  const char * name;
  const char * file;
  crane_model::Tool tool;
  /// Where the low-level controller is holding the gripper for these points.
  double tool_position;
  /// The joint `<mimic>` the rail function reconstructs, or null.
  const char * rail_joint;
  const char * rail_source;
  Generated generated;
};

const std::array<Machine, 2> & machines()
{
  static const std::array<Machine, 2> all{
    {{"pzs100", "pzs100.urdf", crane_model::Tool::Pzs100, 0.45,
      "q11_right_rail_joint", "q9_left_rail_joint",
      {pzs100_evaluate, pzs100_evaluate_work, pzs100_cylinder_jacobian,
        pzs100_chamber_force, pzs100_damping, pzs100_mimic_q5_small_telescope,
        pzs100_mimic_q11_right_rail_joint}},
      // 0.26 rad is 0.0023 rad from the jaw transmission's reversal, which
      // `test_tool_axis` locates at 0.2623 rad: the one configuration-dependent
      // cylinder geometry the tool axis has, held near where it degenerates.
      {"epsilon_7040", "epsilon_7040.urdf", crane_model::Tool::Epsilon7040, 0.26,
        nullptr, nullptr,
        {epsilon7040_evaluate, epsilon7040_evaluate_work, epsilon7040_cylinder_jacobian,
          epsilon7040_chamber_force, epsilon7040_damping,
          epsilon7040_mimic_q5_small_telescope, nullptr}}}};
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

const std::string & description_of(const Machine & machine)
{
  static std::string rail = read_description("pzs100.urdf");
  static std::string jaws = read_description("epsilon_7040.urdf");
  return machine.tool == crane_model::Tool::Pzs100 ? rail : jaws;
}

crane_model::ModelConfig configuration_for(const Machine & machine)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description_of(machine);
  config.tool = machine.tool;
  return config;
}

/// One attribute of one element of the description, read out of it rather than typed.
/**
 * The mimic multiplier and the per-joint damping below are *description* facts,
 * and this test compares the Python model against them. Writing them here as
 * constants would make this file a second place they live -- and the whole claim
 * about the damping override is that the number in the description is the one
 * the model must not use, which cannot be asserted without reading it.
 *
 * Deliberately a scan of the XML text and not a parse: `crane_model` is
 * ROS-free and this test links the public library alone, so pulling a URDF
 * front end in here to read two attributes would be a heavier dependency than
 * the assertion is worth. `test_tool_axis` reads the joint ranges the same way.
 */
std::string element_attribute(
  const std::string & xml, const std::string & joint, const std::string & element,
  const std::string & attribute)
{
  const std::size_t at = xml.find("<joint name=\"" + joint + "\"");
  EXPECT_NE(at, std::string::npos) << "the description carries no joint " << joint;
  if (at == std::string::npos) {
    return {};
  }
  const std::size_t end = xml.find("</joint>", at);
  const std::size_t opening = xml.find("<" + element, at);
  if (opening >= end) {
    return {};
  }
  const std::size_t key = xml.find(attribute + "=\"", opening);
  if (key >= end) {
    return {};
  }
  const std::size_t value = xml.find('"', key) + 1U;
  return xml.substr(value, xml.find('"', value) - value);
}

// --- the points the comparison is made at ------------------------------------

/// The fourteen the OCPs plan, and the rates that go with them.
/**
 * Six across the working range -- folded, reaching, slewing fast, the telescope
 * out, the crane's own zero -- then three where a cylinder geometry is close to
 * degenerate, then one at rest.
 *
 * The three singular ones are the point of the spread and are not decoration.
 * `q2 = 2.38` sits 0.018 rad from where the boom four-bar stops closing at
 * 2.398, beyond which `wiki/hydraulics.md` §2.2's `delta` is negative and there
 * is no piston displacement to report. `q3 = 1.84` is where the arm cylinder's
 * ratio passes through zero, which is the only degeneracy that axis has -- its
 * out-of-plane offset keeps the stroke itself away from zero. The tool axis
 * carries its own, at the pinned coordinate: see `machines()`.
 *
 * The last is at `v = 0` on every axis, where `wiki/mpc.md` §3.1's smoothing is
 * doing all the work: the graph's `A^±(v) sqrt(v² + eps²)` against a numeric
 * `A^-(0) |0|` that is exactly zero.
 */
const std::vector<std::array<double, kNx>> & states()
{
  static const std::vector<std::array<double, kNx>> all{
    {{0.4, 0.35, 0.9, 0.8, 0.12, 0.6, 0.3, 0.3, 0.4, -0.3, 0.2, 0.05, 0.4, 0.25}},
    {{-0.7, -0.9, 1.4, 0.1, -0.9, -0.2, -0.5, -0.5, -0.2, 0.35, 0.3, -0.12, -0.6, -0.3}},
    {{1.2, 0.8, 0.2, 1.6, 2.0, 0.0, 0.0, 0.9, 0.15, 0.2, 0.5, 0.0, 0.2, 0.4}},
    {{0.0, 1.5, 0.4, 0.05, -1.5, 0.35, 0.9, -0.2, 0.3, 0.45, -0.25, 0.2, 0.5, -0.35}},
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}},
    // Near the boom four-bar's closure limit.
    {{0.3, 2.38, 0.5, 0.7, 0.4, -0.2, 0.35, 0.2, 0.1, -0.15, 0.25, 0.3, -0.1, 0.05}},
    // Where the arm cylinder's transmission ratio passes through zero.
    {{-0.4, 0.6, 1.84, 1.1, -0.7, 0.5, -0.3, -0.3, 0.25, 0.2, -0.4, 0.15, 0.2, -0.2}},
    // Both at once, with the telescope near the end of its travel.
    {{0.8, -1.1, -1.29, 2.1, 1.2, 0.15, -0.6, 0.4, -0.2, 0.3, 0.1, -0.25, 0.05, 0.3}},
    // Every axis at rest.
    {{0.5, 0.7, 0.8, 0.9, 0.3, 0.2, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};
  return all;
}

/// The input, one per state and never zero, so the reduction is not evaluated trivially.
crane_model::Input actuated_acceleration()
{
  crane_model::Input u;
  // The tool row is zero and is not a decision: the gripper is held still.
  u << 0.2, -0.1, 0.05, 0.01, 0.3, 0.0;
  return u;
}

/// Three payloads: none carried, the block, and ten times it grasped off centre.
/**
 * `valid == false` is not among them -- contract §4 refuses it and `test_contract`
 * is where that is asserted. An empty gripper is a *declared* payload of zero
 * mass, not an absent one.
 */
const std::vector<crane_model::Payload> & payloads()
{
  static const std::vector<crane_model::Payload> all = [] {
      crane_model::Payload empty;
      empty.valid = true;
      empty.mass_kg = 0.0;
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
      return std::vector<crane_model::Payload>{empty, block, heavy};
    }();
  return all;
}

/// The parameter vector the fixture takes: the pinned tool coordinate, then the payload.
std::array<double, kNp> parameters(const Machine & machine, const crane_model::Payload & payload)
{
  std::array<double, kNp> p{};
  p[kToolParameter] = machine.tool_position;
  p[kPayloadParameter] = payload.mass_kg;
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    p[kPayloadParameter + 1U + static_cast<std::size_t>(axis)] =
      payload.center_of_mass_k8_m[axis];
  }
  // The six independent entries of a symmetric 3x3, in the order
  // `crane_symbolic.INERTIA_ENTRIES` packs them.
  static constexpr std::array<std::pair<Eigen::Index, Eigen::Index>, 6> kEntries{
    {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};
  for (std::size_t entry = 0; entry < kEntries.size(); ++entry) {
    p[kPayloadParameter + 4U + entry] =
      payload.inertia_k8_kg_m2(kEntries[entry].first, kEntries[entry].second);
  }
  return p;
}

/// The canonical eight the numeric calls take, out of the fourteen and the parameter.
struct Canonical
{
  crane_model::Q q{crane_model::Q::Zero()};
  crane_model::DQ dq{crane_model::DQ::Zero()};
};

Canonical canonical(const std::array<double, kNx> & x, const Machine & machine)
{
  Canonical out;
  for (std::size_t axis = 0; axis < kPlannedDof; ++axis) {
    out.q[kActuatedRows[axis]] = x[kPlannedPosition + axis];
    out.dq[kActuatedRows[axis]] = x[kPlannedVelocity + axis];
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    out.q[kPassiveRows[row]] = x[kPassivePosition + row];
    out.dq[kPassiveRows[row]] = x[kPassiveVelocity + row];
  }
  // The tool coordinate is pinned and its rate is zero -- that is the whole
  // content of issue 068's reduction, and the numeric model has to be asked at
  // the same configuration or the comparison is against a different machine.
  out.q[kActuatedRows[kToolAxis]] = machine.tool_position;
  return out;
}

Evaluation evaluate(
  const Machine & machine, const std::array<double, kNx> & x, const crane_model::Input & u,
  const std::array<double, kNp> & p)
{
  std::array<double, kNu> input{};
  for (std::size_t axis = 0; axis < kNu; ++axis) {
    input[axis] = u[static_cast<Eigen::Index>(axis)];
  }
  Evaluation out;
  call(
    machine.generated.evaluate, {x.data(), input.data(), p.data()},
    {out.xdot.data(), out.z.data(), out.mass.data(), out.bias.data(), out.mass_uu.data(),
      out.mass_ua.data(), out.bias_u.data()});
  return out;
}

crane_model::Model build(const Machine & machine)
{
  auto model = crane_model::Model::create(configuration_for(machine));
  EXPECT_TRUE(model.ok()) << machine.name << ": " << model.status().message;
  return std::move(model).value();
}

// --- what the two are allowed to differ by ------------------------------------

/// Round-off, and nothing else.
/**
 * The two models run the same two Pinocchio algorithms in the same double
 * precision on the same description, and differ only in the order the projection
 * sums are accumulated and in what CasADi's code generator chose to fold. So the
 * bound is a relative `1e-9` against the largest entry in play, with a floor for
 * quantities that are legitimately near zero. It is `test_symbolic_graph`'s, and
 * for its reason: a disagreement above it is a real difference between two
 * models, which is the whole point of this file.
 */
double round_off(double scale)
{
  return 1.0e-9 * std::max(1.0, std::abs(scale));
}

/// What `wiki/mpc.md` §3.1's smoothing is worth, per axis, at a given piston velocity.
/**
 * The Python model's `Q_i` is `A^±_smooth(v) sqrt(v² + eps²)` and the numeric
 * one is `A^±_step(v) |v|`, so
 *
 *   |Q_python - Q_numeric| <= A_max eps + |A^+ - A^-|/2 (1 - tanh(|v|/eps_v)) |v| ,
 *
 * which is a statement about the two smoothings and not a tolerance chosen to
 * make the assertion pass. The areas and the two epsilons come out of
 * `config/hydraulics.yaml`, the same file the Python model read them from, so
 * this bound moves with the file rather than restating what it says.
 */
double flow_smoothing_bound(std::size_t axis, double velocity)
{
  crane_model::hydraulics::Constants constants;
  EXPECT_TRUE(crane_model::hydraulics::load(constants).ok());
  const crane_model::cylinder::AxisAreas areas =
    crane_model::cylinder::axis_areas(constants)[axis];
  const double largest = std::max(areas.a_eff_pos, areas.a_eff_neg);
  const double step = std::abs(areas.a_eff_pos - areas.a_eff_neg);
  return largest * constants.eps_abs +
         0.5 * step * (1.0 - std::tanh(std::abs(velocity) / constants.eps_v)) *
         std::abs(velocity);
}

}  // namespace

TEST(CraneModelSymbolicParity, TheFixtureNeedsNoWorkspace)
{
  // The whole fixture is called above with two null workspaces, which is only
  // sound because CasADi reported it needs neither. Asserted here rather than
  // assumed: a future model with an `if_else` or a solve in it would need both,
  // and would otherwise write through those nulls.
  for (const Machine & machine : machines()) {
    casadi_int sz_arg = 0;
    casadi_int sz_res = 0;
    casadi_int sz_iw = -1;
    casadi_int sz_w = -1;
    ASSERT_EQ(machine.generated.evaluate_work(&sz_arg, &sz_res, &sz_iw, &sz_w), 0)
      << machine.name;
    EXPECT_EQ(sz_iw, 0) << machine.name;
    EXPECT_EQ(sz_w, 0) << machine.name;
    EXPECT_EQ(sz_arg, 3) << machine.name;
    EXPECT_EQ(sz_res, 7) << machine.name;
  }
}

TEST(CraneModelSymbolicParity, TheMimicJointsAreReconstructed)
{
  // Pinocchio drops `<mimic>`, so `q5_small_telescope` -- the second telescope
  // stage of `wiki/robot_model.md` §0 -- and the PZS100's `q11_right_rail_joint`
  // have to be put back by both models. This is the single most likely place for
  // the two to disagree, and it is asserted **directly** and not only through the
  // dynamics: a projection error can cancel in a mass matrix, and an export that
  // silently left `q5` at its neutral value would produce a telescope half as
  // long and a perfectly plausible `M`.
  //
  // The oracle is the description's own `<mimic>` element, because that is what
  // the numeric model holds: `Model::Impl::write_configuration` writes
  // `multiplier * q_source + offset` into the mimic's slot and nothing else.
  for (const Machine & machine : machines()) {
    const std::string & xml = description_of(machine);
    const auto model = build(machine);

    struct Mimic
    {
      const char * joint;
      const char * source;
      Entry entry;
    };
    std::vector<Mimic> mimics{
      {"q5_small_telescope", "q4_big_telescope", machine.generated.mimic_telescope}};
    if (machine.generated.mimic_rail != nullptr) {
      mimics.push_back(
        {machine.rail_joint, machine.rail_source, machine.generated.mimic_rail});
    }

    for (const Mimic & mimic : mimics) {
      // The description says which joint drives this one and with what law. If it
      // ever stops saying so, this test fails here rather than in a mass matrix.
      EXPECT_EQ(element_attribute(xml, mimic.joint, "mimic", "joint"), mimic.source)
        << machine.name << ", " << mimic.joint;
      const std::string multiplier_text =
        element_attribute(xml, mimic.joint, "mimic", "multiplier");
      const std::string offset_text = element_attribute(xml, mimic.joint, "mimic", "offset");
      ASSERT_FALSE(multiplier_text.empty()) << machine.name << ", " << mimic.joint;
      const double multiplier = std::stod(multiplier_text);
      const double offset = offset_text.empty() ? 0.0 : std::stod(offset_text);

      // Which canonical coordinate that is, out of the public joint map, so the
      // source index is the model's own and not a number typed here.
      const auto & names = model.urdf_joint_names();
      const auto found = std::find(names.begin(), names.end(), mimic.source);
      ASSERT_NE(found, names.end()) << machine.name << ", " << mimic.source;
      const std::size_t source = static_cast<std::size_t>(std::distance(names.begin(), found));

      for (const std::array<double, kNx> & x : states()) {
        const Canonical state = canonical(x, machine);
        std::array<double, crane_model::kGeneralizedDof> q{};
        std::array<double, crane_model::kGeneralizedDof> dq{};
        for (std::size_t index = 0; index < crane_model::kGeneralizedDof; ++index) {
          q[index] = state.q[static_cast<Eigen::Index>(index)];
          dq[index] = state.dq[static_cast<Eigen::Index>(index)];
        }
        std::array<double, 2> reconstructed{};
        call(mimic.entry, {q.data(), dq.data()}, {reconstructed.data()});

        EXPECT_DOUBLE_EQ(reconstructed[0], multiplier * q[source] + offset)
          << machine.name << ", " << mimic.joint << " position";
        EXPECT_DOUBLE_EQ(reconstructed[1], multiplier * dq[source])
          << machine.name << ", " << mimic.joint << " rate";
      }
    }
  }
}

TEST(CraneModelSymbolicParity, TheTelescopeDampingIsTheOverrideAndNotTheDescriptionsNumber)
{
  // `wiki/implementation/parameters.md` §5 calls `q4_big_telescope`'s URDF
  // damping a simulation stability hack an order of magnitude above the
  // identified value and says it must not enter the model, and
  // `config/hydraulics.yaml`'s `damping_overrides` is where that is carried.
  // Grill D2's point is exactly this: a faithful URDF read is the *wrong* model
  // here, so an export that read the description honestly would produce a
  // different system on day one and nothing else in this file would notice.
  for (const Machine & machine : machines()) {
    const std::string & xml = description_of(machine);
    const auto model = build(machine);

    // dq = 1 on every row, so what comes back is D itself.
    std::array<double, crane_model::kGeneralizedDof> ones{};
    ones.fill(1.0);
    std::array<double, crane_model::kGeneralizedDof> damping{};
    call(machine.generated.damping, {ones.data()}, {damping.data()});

    constexpr std::size_t kTelescope = 3U;
    const std::string & telescope = model.urdf_joint_names()[kTelescope];
    EXPECT_EQ(telescope, "q4_big_telescope") << machine.name;

    const std::string described =
      element_attribute(xml, telescope, "dynamics", "damping");
    ASSERT_FALSE(described.empty()) << machine.name << ": " << telescope << " has no damping";
    // The assertion only means something while the description still states a
    // number to override, so it is checked before the override is.
    EXPECT_GT(std::stod(described), 0.0) << machine.name;
    EXPECT_DOUBLE_EQ(damping[kTelescope], 0.0)
      << machine.name << ": the telescope carries the description's " << described
      << " rather than the override";

    // And every other row *is* the description's own number, so this is a test
    // of the damping read and not only of the one entry that is overridden.
    for (std::size_t row = 0; row < crane_model::kGeneralizedDof; ++row) {
      if (row == kTelescope) {
        continue;
      }
      const std::string entry =
        element_attribute(xml, model.urdf_joint_names()[row], "dynamics", "damping");
      EXPECT_DOUBLE_EQ(damping[row], entry.empty() ? 0.0 : std::stod(entry))
        << machine.name << ", " << model.urdf_joint_names()[row];
    }
  }
}

TEST(CraneModelSymbolicParity, TheMassMatrixAndTheBiasAreTheNumericModelsOwn)
{
  // `full_dynamics` is `wiki/robot_model.md` §1 in the canonical eight, and its
  // own oracle is the kinetic and potential energy of the description
  // (`test_contract`) rather than a second torque algorithm. So this ties the
  // Python model back to the description through two independent steps.
  //
  // This is where the mimic projection is checked *through* the dynamics, which
  // is a different claim from `TheMimicJointsAreReconstructed`: `M` counts the
  // second telescope stage and the mirrored rail on the coordinate that drives
  // them, and a P that reconstructed the positions but dropped the velocity rows
  // would pass that test and fail this one.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    for (const crane_model::Payload & payload : payloads()) {
      const std::array<double, kNp> p = parameters(machine, payload);
      for (const std::array<double, kNx> & x : states()) {
        const Canonical state = canonical(x, machine);
        const Evaluation evaluated = evaluate(machine, x, actuated_acceleration(), p);

        const auto full = model.full_dynamics(state.q, state.dq, payload);
        ASSERT_TRUE(full.ok()) << machine.name << ": " << full.status().message;
        const double mass_scale = full.value().mass.cwiseAbs().maxCoeff();
        const double bias_scale = full.value().bias.cwiseAbs().maxCoeff();

        for (std::size_t row = 0; row < crane_model::kGeneralizedDof; ++row) {
          for (std::size_t column = 0; column < crane_model::kGeneralizedDof; ++column) {
            EXPECT_NEAR(
              evaluated.m(row, column),
              full.value().mass(
                static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)),
              round_off(mass_scale))
              << machine.name << " M(" << row << "," << column << ")";
          }
          EXPECT_NEAR(
            evaluated.bias[row], full.value().bias[static_cast<Eigen::Index>(row)],
            round_off(bias_scale)) << machine.name << " h(" << row << ")";
        }
      }
    }
  }
}

TEST(CraneModelSymbolicParity, ThePassiveAccelerationIsTheOneTheNumericModelResolves)
{
  // The Schur complement of `wiki/robot_model.md` §3.1, checked the way the model
  // states it rather than by forming it a second time: at the consistent
  // acceleration the passive rows of the inverse dynamics vanish (contract §7).
  // So the Python model is asked for `ddq_u`, the answer is handed to
  // `inverse_dynamics` as part of a full `ddq`, and its passive rows have to be
  // zero. Nothing here forms `M_uu`, `M_ua` or `h_u`; a transcribed sign is a
  // torque of order 1e3 N m and fails at once.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    for (const crane_model::Payload & payload : payloads()) {
      const std::array<double, kNp> p = parameters(machine, payload);
      for (const std::array<double, kNx> & x : states()) {
        const Canonical state = canonical(x, machine);
        const crane_model::Input u = actuated_acceleration();
        const Evaluation evaluated = evaluate(machine, x, u, p);

        crane_model::DQ ddq = crane_model::DQ::Zero();
        for (std::size_t axis = 0; axis < kPlannedDof; ++axis) {
          ddq[kActuatedRows[axis]] = u[static_cast<Eigen::Index>(axis)];
          // The rows of xdot that carry the planned positions are their own
          // rates, and the rows that carry the planned rates are u itself.
          EXPECT_DOUBLE_EQ(evaluated.xdot[kPlannedPosition + axis], x[kPlannedVelocity + axis])
            << machine.name << " dq_a row " << axis;
          EXPECT_DOUBLE_EQ(
            evaluated.xdot[kPlannedVelocity + axis], u[static_cast<Eigen::Index>(axis)])
            << machine.name << " ddq_a row " << axis;
        }
        // The tool is held still: `ddq_tool` is zero and is not carried at all.
        for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
          EXPECT_DOUBLE_EQ(evaluated.xdot[kPassivePosition + row], x[kPassiveVelocity + row])
            << machine.name << " dq_u row " << row;
          ddq[kPassiveRows[row]] = evaluated.xdot[kPassiveVelocity + row];
        }

        const auto tau = model.inverse_dynamics(state.q, state.dq, ddq, payload);
        ASSERT_TRUE(tau.ok()) << machine.name << ": " << tau.status().message;
        // The bound contract §7's invariant is asserted at, in N m.
        EXPECT_NEAR(tau.value()[kPassiveRows[0]], 0.0, 1.0e-6) << machine.name << " tip";
        EXPECT_NEAR(tau.value()[kPassiveRows[1]], 0.0, 1.0e-6) << machine.name << " tilt";
      }
    }
  }
}

TEST(CraneModelSymbolicParity, ThePassiveRowsArePartOfTheSameMassMatrix)
{
  // What `wiki/trajectory_planning.md` §5.2 carries out of the model: M_uu, M_ua
  // and h_u, with h_u already containing D_uu dq_u. `M_ua` is over the five
  // planned columns and not the machine's six, because the tool contributes no
  // forcing to the sway it is not allowed to accelerate.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    for (const crane_model::Payload & payload : payloads()) {
      const std::array<double, kNp> p = parameters(machine, payload);
      for (const std::array<double, kNx> & x : states()) {
        const Canonical state = canonical(x, machine);
        const Evaluation evaluated = evaluate(machine, x, actuated_acceleration(), p);

        const auto full = model.full_dynamics(state.q, state.dq, payload);
        ASSERT_TRUE(full.ok()) << machine.name << ": " << full.status().message;
        const double mass_scale = full.value().mass.cwiseAbs().maxCoeff();
        const double bias_scale = full.value().bias.cwiseAbs().maxCoeff();

        for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
          for (std::size_t column = 0; column < crane_model::kPassiveDof; ++column) {
            EXPECT_NEAR(
              evaluated.m_uu(row, column),
              full.value().mass(kPassiveRows[row], kPassiveRows[column]),
              round_off(mass_scale)) << machine.name << " M_uu " << row << column;
          }
          for (std::size_t column = 0; column < kPlannedDof; ++column) {
            EXPECT_NEAR(
              evaluated.m_ua(row, column),
              full.value().mass(kPassiveRows[row], kActuatedRows[column]),
              round_off(mass_scale)) << machine.name << " M_ua " << row << column;
          }
          EXPECT_NEAR(
            evaluated.bias_u[row], full.value().bias[kPassiveRows[row]],
            round_off(bias_scale)) << machine.name << " h_u " << row;
        }
      }
    }
  }
}

TEST(CraneModelSymbolicParity, TheActuatedForceIsTheReductionTheNumericModelReturns)
{
  // `reduced_actuated_dynamics` is `wiki/robot_model.md` §3.4, and the output
  // map's tau_a block is the same quantity assembled the other way -- from the
  // actuated rows of M ddq + h at the Python model's own ddq_u. The two are the
  // same double arithmetic in a different order, so they agree to round-off;
  // anything larger is a difference in the equations.
  //
  // All six rows, including the tool's. It is not a decision variable, but the
  // force needed to hold the gripper where the low-level controller has it is
  // still a force this machine applies, and dropping the row would drop the
  // check that the tool link is still in `M`.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    for (const crane_model::Payload & payload : payloads()) {
      const std::array<double, kNp> p = parameters(machine, payload);
      for (const std::array<double, kNx> & x : states()) {
        const Canonical state = canonical(x, machine);
        const crane_model::Input u = actuated_acceleration();
        const Evaluation evaluated = evaluate(machine, x, u, p);

        const auto reduced = model.reduced_actuated_dynamics(state.q, state.dq, u, payload);
        ASSERT_TRUE(reduced.ok()) << machine.name << ": " << reduced.status().message;
        const crane_model::Vector6 tau_a =
          reduced.value().mass_eff * u + reduced.value().bias_eff;

        for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
          EXPECT_NEAR(
            evaluated.z[kActuatedForceOffset + axis],
            tau_a[static_cast<Eigen::Index>(axis)],
            round_off(tau_a.cwiseAbs().maxCoeff())) << machine.name << " tau_a axis " << axis;
        }
      }
    }
  }
}

TEST(CraneModelSymbolicParity, TheOutputMapCarriesTheTransmissionTheMpcConstrains)
{
  // Acceptance: a consumer must not have to rebuild the transmission in its own
  // solver. So the output map's v and Q blocks are checked against
  // `Model::transmission` at the same argument, and its F_cyl block against the
  // cylinder force tau_a induces through the same J_cyl.
  //
  // Q is not exact and must not be: `wiki/mpc.md` §3.1 requires the model's Q to
  // be smoothed for a gradient-based solver while `transmission` keeps the
  // physical step. `flow_smoothing_bound` is what that difference is worth, and
  // the state with every axis at rest is where it is the whole of it.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    const crane_model::Payload & payload = payloads().at(1);
    const std::array<double, kNp> p = parameters(machine, payload);

    for (const std::array<double, kNx> & x : states()) {
      const Canonical state = canonical(x, machine);
      const Evaluation evaluated = evaluate(machine, x, actuated_acceleration(), p);

      crane_model::DQA dq_a;
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        dq_a[static_cast<Eigen::Index>(axis)] = state.dq[kActuatedRows[axis]];
      }
      crane_model::ChamberPressure pressure;
      pressure.p_a_pa.setZero();
      pressure.p_b_pa.setZero();
      const auto transmission = model.transmission(state.q, dq_a, pressure);
      ASSERT_TRUE(transmission.ok()) << machine.name << ": " << transmission.status().message;
      const auto cylinder = model.cylinder_jacobian(state.q);
      ASSERT_TRUE(cylinder.ok()) << machine.name << ": " << cylinder.status().message;

      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        const Eigen::Index index = static_cast<Eigen::Index>(axis);
        const double velocity = transmission.value().cylinder_velocity[index];
        // Relative, not bit-exact, and that is the one tolerance here that is
        // looser than `test_symbolic_graph`'s. There, `v` is asserted with
        // `EXPECT_DOUBLE_EQ` because the graph and the numeric model instantiate
        // *one C++ template*; here they are two transcriptions in two languages
        // and CasADi's code generator reassociates. Measured: they part company
        // by four ULP -- 8e-16 relative -- on the boom axis at `q2 = 2.38`,
        // where `wiki/hydraulics.md` §2.2's ratio comes out of a dot product
        // that is cancelling. Six decades under this bound.
        EXPECT_NEAR(
          evaluated.z[kPistonVelocityOffset + axis], velocity, round_off(velocity))
          << machine.name << " v, axis " << axis;
        EXPECT_NEAR(
          evaluated.z[kAxisFlowOffset + axis], transmission.value().pump_flow[index],
          flow_smoothing_bound(axis, velocity)) << machine.name << " Q, axis " << axis;

        // Constraint 6 in the form the model hands over: F_i = tau_a,i / J_c,ii,
        // so a consumer bounds it by F_i^max directly.
        const double ratio = cylinder.value()(index, index);
        ASSERT_NE(ratio, 0.0) << machine.name << " axis " << axis;
        const double tau_a = evaluated.z[kActuatedForceOffset + axis];
        EXPECT_NEAR(
          evaluated.z[kCylinderForceOffset + axis], tau_a / ratio,
          round_off(std::abs(tau_a / ratio))) << machine.name << " F_cyl, axis " << axis;
      }
    }
  }
}

TEST(CraneModelSymbolicParity, TheCylinderJacobianIsTheNumericModelsOwn)
{
  // `cylinder_jacobian` is public API and survives the port, so the Python
  // transmission is compared against it directly and not only through the output
  // map. This is the test the three near-singular states in `states()` and the
  // pinned tool coordinate of `machines()` exist for: the boom four-bar 0.018 rad
  // from where it stops closing, the arm ratio at its zero, and the 7040 jaw
  // beside its reversal are where two transcriptions of `wiki/hydraulics.md`
  // §2.2, §2.3 and §2.6 stop agreeing if they were ever going to.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    for (const std::array<double, kNx> & x : states()) {
      const Canonical state = canonical(x, machine);
      std::array<double, crane_model::kGeneralizedDof> q{};
      for (std::size_t index = 0; index < crane_model::kGeneralizedDof; ++index) {
        q[index] = state.q[static_cast<Eigen::Index>(index)];
      }
      std::array<double, crane_model::kActuatedDof> diagonal{};
      call(machine.generated.cylinder_jacobian, {q.data()}, {diagonal.data()});

      const auto cylinder = model.cylinder_jacobian(state.q);
      ASSERT_TRUE(cylinder.ok()) << machine.name << ": " << cylinder.status().message;
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        const Eigen::Index index = static_cast<Eigen::Index>(axis);
        EXPECT_NEAR(
          diagonal[axis], cylinder.value()(index, index),
          round_off(cylinder.value()(index, index)))
          << machine.name << " J_cyl axis " << axis << " at q2 = " << q[1] << ", q3 = " << q[2];
        // The geometry does not couple the axes at all, so what the Python model
        // returns as a diagonal really is the whole of J_cyl.
        for (std::size_t column = 0; column < crane_model::kActuatedDof; ++column) {
          if (column != axis) {
            EXPECT_DOUBLE_EQ(cylinder.value()(index, static_cast<Eigen::Index>(column)), 0.0);
          }
        }
      }
    }
  }
}

TEST(CraneModelSymbolicParity, TheChamberForceIsTheNumericModelsOwn)
{
  // `cylinder_force` is the other half of the transmission that stays public:
  // F_i = A_A p_A - A_B p_B (`wiki/hydraulics.md` §4). It is the only place the
  // composed areas are visible on their own -- the doubled slewing and arm
  // cylinders, the regenerative telescope, V_m in both slots of the rotator --
  // so a Python model that composed §6.1's numbers differently is caught here
  // and not through a pump flow that mostly agrees.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    const std::array<std::array<double, crane_model::kActuatedDof>, 3> pressures_a{
      {{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
        {{1.0e7, 2.0e7, 5.0e6, 1.5e7, 3.0e6, 8.0e6}},
        {{2.5e7, 0.0, 1.0e7, 0.0, 1.2e7, 4.0e6}}}};
    const std::array<std::array<double, crane_model::kActuatedDof>, 3> pressures_b{
      {{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
        {{5.0e6, 1.0e7, 2.0e7, 4.0e6, 9.0e6, 1.1e7}},
        {{0.0, 1.8e7, 0.0, 6.0e6, 0.0, 2.0e7}}}};

    for (std::size_t sample = 0; sample < pressures_a.size(); ++sample) {
      std::array<double, crane_model::kActuatedDof> force{};
      call(
        machine.generated.chamber_force,
        {pressures_a[sample].data(), pressures_b[sample].data()}, {force.data()});

      crane_model::ChamberPressure pressure;
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        const Eigen::Index index = static_cast<Eigen::Index>(axis);
        pressure.p_a_pa[index] = pressures_a[sample][axis];
        pressure.p_b_pa[index] = pressures_b[sample][axis];
      }
      const auto expected = model.cylinder_force(pressure);
      ASSERT_TRUE(expected.ok()) << machine.name << ": " << expected.status().message;
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        EXPECT_NEAR(
          force[axis], expected.value()[static_cast<Eigen::Index>(axis)],
          round_off(expected.value()[static_cast<Eigen::Index>(axis)]))
          << machine.name << " F axis " << axis << ", sample " << sample;
      }
    }
  }
}

TEST(CraneModelSymbolicParity, ThePassiveEquilibriumIsAZeroOfThePythonModelsPassiveRows)
{
  // `passive_equilibrium` is the numeric model's own statement of where the tool
  // hangs once it has stopped swinging (`wiki/robot_model.md` §2.3), and it is a
  // solve rather than an evaluation -- so it is an oracle the Python model
  // cannot have been fitted to. At the pose it returns, with the machine at rest,
  // the Python model's h_u must be zero, at the same 1e-6 N m the invariant of
  // contract §7 is asserted at.
  //
  // It also exercises the two passive rows where the payload dominates them: the
  // heavy off-centre grasp moves the hanging pose by a visible amount, and a
  // payload attached at the wrong frame would put the zero somewhere else.
  for (const Machine & machine : machines()) {
    const auto model = build(machine);
    // Points where the solve had an answer. The skip below is legitimate and
    // this is what stops it from quietly emptying the test.
    std::size_t compared = 0;
    for (const crane_model::Payload & payload : payloads()) {
      const std::array<double, kNp> p = parameters(machine, payload);
      for (const std::array<double, kNx> & sample : states()) {
        crane_model::QA q_a;
        for (std::size_t axis = 0; axis < kPlannedDof; ++axis) {
          q_a[static_cast<Eigen::Index>(axis)] = sample[kPlannedPosition + axis];
        }
        q_a[static_cast<Eigen::Index>(kToolAxis)] = machine.tool_position;

        const auto equilibrium = model.passive_equilibrium(q_a, payload);
        if (!equilibrium.ok()) {
          // `wiki/robot_model.md` §2.3 has no answer where the free hanging pose
          // needs more tip travel than the description gives that joint. That is
          // a fact about the machine and not a disagreement between two models,
          // so the point is skipped rather than the tolerance loosened.
          continue;
        }

        std::array<double, kNx> x{};
        for (std::size_t axis = 0; axis < kPlannedDof; ++axis) {
          x[kPlannedPosition + axis] = q_a[static_cast<Eigen::Index>(axis)];
        }
        for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
          x[kPassivePosition + row] = equilibrium.value()[static_cast<Eigen::Index>(row)];
        }
        const Evaluation evaluated = evaluate(machine, x, crane_model::Input::Zero(), p);
        for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
          EXPECT_NEAR(evaluated.bias_u[row], 0.0, 1.0e-6)
            << machine.name << " h_u row " << row << " at the hanging pose";
        }
        ++compared;
      }
    }
    EXPECT_GT(compared, states().size()) << machine.name << ": nowhere to hang anywhere";
  }
}
