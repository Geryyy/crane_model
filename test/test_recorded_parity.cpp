// The recorded-trajectory parity comparison of PRD user story 65.
//
// `crane_model`'s Pinocchio backend and the generated Maple/MATLAB model are
// evaluated at the same recorded configurations and their answers are
// subtracted.  The recorded half is `test/recorded_parity_fixture.txt`: 64
// configurations taken out of the four 2026-08-19 machine recordings, with the
// generated model's answers beside them.  `test/derive_recorded_parity.py`
// produced that file once, offline; this test reads it and nothing else, so the
// comparison runs without ROS, without a bag and without `mp_crane` on any
// dependency surface -- which is what lets slice 4 assert the generated
// packages are absent from the `hardware` closure while still validating
// against them (PRD 12).
//
// What it found, and what this file therefore asserts:
//
//   forward kinematics    identical to 3e-15 m and 1e-15 rad
//   passive equilibrium   identical to 1e-15 rad
//   the passive rows of M and h
//                         *different*, by 50% relative rms -- and the
//                         difference is exactly two defects in the generated
//                         model, both of which the description decides.
//
// The two defects, and how each was established (`wiki/robot_model.md` 6.3):
//
//   1.  The generated model does not carry the PZS100 rail gripper at all.
//       Deleting `K10_left_rail` and `K12_right_rail` from the description, or
//       doubling them, leaves every one of its answers bit-identical, while
//       `crane_model`'s move by 85%.  The description hangs 200 kg of gripper
//       below the tip joint, 48% of the 420 kg there.
//   2.  It hangs the 150 kg tool frame 48.7 mm too low.  It reads `s8` and
//       `I_8` off the URDF link `K8_tool_center_point`, whose inertial origin
//       is in *that link's* frame, and applies them in the DH frame
//       `K8_rotator_lower_part`.  The two are one fixed joint apart --
//       `K8_tool_center_point_mount`, `xyz = 0 0 0.0487` -- and the
//       description says so in a comment on that joint.
//
// So the comparison is a ladder, not a tolerance.  `crane_model` is evaluated
// three times: on the description as checked in, on the description with the
// gripper deleted, and on the description with the gripper deleted *and* the
// tool frame moved down by that fixed-joint offset.  The third one is the
// generated model's own geometry, and there the two agree to 1e-4.  Nothing is
// rounded away and no tolerance is stretched to cover a defect: each rung names
// one, and the residual after both is what parity actually is.

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "crane_model/model.hpp"

namespace
{

// The fixture's column legend, in its own order.  Kept as named widths so the
// row length the reader checks is the legend's own sum.
constexpr int kBagColumns = 1;
constexpr int kStampColumns = 1;
constexpr int kQColumns = 8;
constexpr int kK8PositionColumns = 3;
constexpr int kK8RotationColumns = 9;
constexpr int kTipPositionColumns = 3;
constexpr int kGainColumns = 10;
constexpr int kBiasColumns = 2;
constexpr int kEquilibriumColumns = 2;
constexpr int kRowColumns = kBagColumns + kStampColumns + 2 * kQColumns +
  kK8PositionColumns + kK8RotationColumns + kTipPositionColumns + kGainColumns +
  2 * kBiasColumns + kEquilibriumColumns;
constexpr std::size_t kSampleCount = 64;

// The passive rows of contract 2, and the five actuated coordinates the
// generated model carries.  Its reduced state has no `q8`: the tool coordinate
// is a parameter to it and a coordinate here, which is why the gain it exposes
// has five columns and not six.
constexpr std::array<Eigen::Index, 2> kPassiveRows{4, 5};
constexpr std::array<Eigen::Index, 5> kGeneratedActuated{0, 1, 2, 3, 6};
const std::array<const char *, 2> kPassiveNames{"q5 tip", "q6 tilt"};
const std::array<const char *, 5> kActuatedNames{
  "q1 slewing", "q2 boom", "q3 arm", "q4 telescope", "q7 rotator"};

struct Sample
{
  int bag{};
  double stamp_s{};
  crane_model::Q q{crane_model::Q::Zero()};
  crane_model::DQ dq{crane_model::DQ::Zero()};
  Eigen::Vector3d k8_position_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d k8_rotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d tip_position_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 2, 5> ddq_u_gain{Eigen::Matrix<double, 2, 5>::Zero()};
  Eigen::Vector2d ddq_u_bias{Eigen::Vector2d::Zero()};
  Eigen::Vector2d ddq_u_bias_static{Eigen::Vector2d::Zero()};
  Eigen::Vector2d passive_equilibrium_rad{Eigen::Vector2d::Zero()};
};

std::string read_file(const std::string & path)
{
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "cannot read " << path;
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

const std::string & fixture_text()
{
  static const std::string text =
    read_file(std::string(CRANE_MODEL_TEST_FIXTURE_DIR) + "/recorded_parity_fixture.txt");
  return text;
}

const std::vector<Sample> & samples()
{
  static const std::vector<Sample> parsed = [] {
      std::vector<Sample> rows;
      std::istringstream stream(fixture_text());
      std::string line;
      while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
          continue;
        }
        std::istringstream fields(line);
        std::array<double, kRowColumns> value{};
        for (double & entry : value) {
          fields >> entry;
        }
        EXPECT_FALSE(fields.fail()) << "short row in the fixture: " << line;
        double extra = 0.0;
        EXPECT_TRUE((fields >> extra).fail()) << "long row in the fixture: " << line;

        Sample sample;
        int at = 0;
        sample.bag = static_cast<int>(value[at++]);
        sample.stamp_s = value[at++];
        for (int index = 0; index < kQColumns; ++index) {
          sample.q[index] = value[at++];
        }
        for (int index = 0; index < kQColumns; ++index) {
          sample.dq[index] = value[at++];
        }
        for (int index = 0; index < kK8PositionColumns; ++index) {
          sample.k8_position_m[index] = value[at++];
        }
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            sample.k8_rotation(row, column) = value[at++];
          }
        }
        for (int index = 0; index < kTipPositionColumns; ++index) {
          sample.tip_position_m[index] = value[at++];
        }
        for (int row = 0; row < 2; ++row) {
          for (int column = 0; column < 5; ++column) {
            sample.ddq_u_gain(row, column) = value[at++];
          }
        }
        for (int index = 0; index < 2; ++index) {
          sample.ddq_u_bias[index] = value[at++];
        }
        for (int index = 0; index < 2; ++index) {
          sample.ddq_u_bias_static[index] = value[at++];
        }
        for (int index = 0; index < 2; ++index) {
          sample.passive_equilibrium_rad[index] = value[at++];
        }
        EXPECT_EQ(at, kRowColumns);
        rows.push_back(sample);
      }
      return rows;
    }();
  return parsed;
}

// The three descriptions the ladder is evaluated on.  `AsDescribed` is the
// machine; the other two carry one named defect of the generated model each, so
// that what is left over after them is parity and not tolerance.
enum class Variant
{
  AsDescribed,
  WithoutTheGripper,
  AsTheGeneratedModelHasIt,
};

// The one substitution the variants need.  It asserts how many occurrences it
// replaced, so a description that no longer carries the markup fails here
// rather than quietly producing a model that means something else.
void substitute(
  std::string & xml, const std::string & from, const std::string & to, int expected)
{
  int found = 0;
  for (std::size_t at = xml.find(from); at != std::string::npos;
    at = xml.find(from, at + to.size()))
  {
    xml.replace(at, from.size(), to);
    ++found;
  }
  ASSERT_EQ(found, expected) << "the description no longer carries: " << from;
}

std::string description(Variant variant)
{
  std::string xml = read_file(std::string(CRANE_MODEL_TEST_DESCRIPTION_DIR) + "/pzs100.urdf");
  if (variant == Variant::AsDescribed) {
    return xml;
  }

  // Defect 1: the two 100 kg rail bodies the generated model does not have.
  // Both carry the same inertial block, so one substitution covers the pair.
  substitute(
    xml,
    "<mass value=\"100.0\"/>\n"
    "      <inertia ixx=\"3.3547704\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"9.1819806\" iyz=\"-0.0\" "
    "izz=\"7.4407964\"/>",
    "<mass value=\"1e-9\"/>\n"
    "      <inertia ixx=\"1e-12\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1e-12\" iyz=\"0.0\" "
    "izz=\"1e-12\"/>",
    2);
  if (variant == Variant::WithoutTheGripper) {
    return xml;
  }

  // Defect 2: the tool frame, 48.7 mm lower -- the offset of the fixed joint
  // `K8_tool_center_point_mount`, which is what the generated model drops when
  // it applies `s8` in `K8_rotator_lower_part`. -0.2 - 0.04870000000000002.
  substitute(xml, "xyz=\" 0. 0. -0.2\"", "xyz=\" 0. 0. -0.24870000000000003\"", 1);
  return xml;
}

crane_model::Model production_model(Variant variant)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description(variant);
  config.tool = crane_model::Tool::Pzs100;
  // The fixture header records the generated model's run at 9.81 m/s^2 down;
  // this is the same gravity in the contract's own sign convention.
  config.gravity_m_s2 = Eigen::Vector3d(0.0, 0.0, -9.81);
  auto model = crane_model::Model::create(config);
  EXPECT_TRUE(model.ok()) << model.status().message;
  return std::move(model).value();
}

// The empty gripper the fixture was derived with.  `Payload{}` would be an
// *unknown* payload and is refused; carrying nothing is a declared payload of
// zero mass, which is the statement the generated model's run made.
crane_model::Payload carrying_nothing()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 0.0;
  payload.center_of_mass_k8_m.setZero();
  payload.inertia_k8_kg_m2.setZero();
  return payload;
}

// One reported residual: what it is, how big it got, against what, and where.
class Residual
{
public:
  Residual(std::string name, std::string unit)
  : name_(std::move(name)), unit_(std::move(unit)) {}

  void add(double mine, double theirs, std::size_t sample)
  {
    const double error = std::abs(mine - theirs);
    error_squared_ += error * error;
    reference_squared_ += theirs * theirs;
    reference_worst_ = std::max(reference_worst_, std::abs(theirs));
    ++count_;
    if (error > worst_) {
      worst_ = error;
      worst_sample_ = sample;
    }
  }

  [[nodiscard]] double worst() const { return worst_; }
  [[nodiscard]] double relative() const
  {
    return reference_squared_ > 0.0 ? std::sqrt(error_squared_ / reference_squared_) : 0.0;
  }

  [[nodiscard]] std::string line() const
  {
    std::ostringstream out;
    out << std::left << std::setw(34) << name_ << std::right << std::scientific
        << std::setprecision(3) << " max " << worst_ << " " << std::left << std::setw(7)
        << unit_ << std::right;
    // The orientation residual is already an angle between two rotations, so it
    // is compared against nothing and a relative figure would divide by zero.
    if (reference_squared_ > 0.0) {
      out << " of at most " << reference_worst_ << "   relative rms " << std::fixed
          << std::setprecision(5) << relative();
    } else {
      out << "   (absolute; there is no reference magnitude to divide by)";
    }
    out << "   worst at sample " << worst_sample_ << "/" << count_;
    return out.str();
  }

private:
  std::string name_;
  std::string unit_;
  double worst_{0.0};
  double error_squared_{0.0};
  double reference_squared_{0.0};
  double reference_worst_{0.0};
  std::size_t count_{0};
  std::size_t worst_sample_{0};
};

void report(const std::string & heading, const std::vector<Residual> & residuals)
{
  std::cout << "\n[ parity ] " << heading << "\n";
  for (const Residual & residual : residuals) {
    std::cout << "[ parity ]   " << residual.line() << "\n";
  }
  std::cout << std::flush;
}

// `-M_uu^-1 (M_ua ddq_a + h_u)` of robot_model 3.1, split into the affine map
// the generated model exposes: the gain `d(ddq_u)/d(ddq_a)` over the five
// coordinates it carries, and the bias at `ddq_a = 0`. This is the only route
// to the generated model's mass matrix and bias -- it publishes neither, and
// the timber stack consumed it through exactly this solve.
struct PassiveMap
{
  Eigen::Matrix<double, 2, 5> gain{Eigen::Matrix<double, 2, 5>::Zero()};
  Eigen::Vector2d bias{Eigen::Vector2d::Zero()};
};

testing::AssertionResult passive_map(
  const crane_model::Model & model, const crane_model::Q & q, const crane_model::DQ & dq,
  PassiveMap & out)
{
  auto dynamics = model.full_dynamics(q, dq, carrying_nothing());
  if (!dynamics.ok()) {
    return testing::AssertionFailure() << dynamics.status().message;
  }
  const crane_model::FullMass & mass = dynamics.value().mass;
  const crane_model::DQ & bias = dynamics.value().bias;

  crane_model::PassiveMass mass_uu;
  Eigen::Vector2d bias_u;
  Eigen::Matrix<double, 2, 5> mass_ua;
  for (Eigen::Index row = 0; row < 2; ++row) {
    const Eigen::Index u = kPassiveRows[static_cast<std::size_t>(row)];
    bias_u[row] = bias[u];
    for (Eigen::Index column = 0; column < 2; ++column) {
      mass_uu(row, column) = mass(u, kPassiveRows[static_cast<std::size_t>(column)]);
    }
    for (Eigen::Index column = 0; column < 5; ++column) {
      mass_ua(row, column) = mass(u, kGeneratedActuated[static_cast<std::size_t>(column)]);
    }
  }
  const Eigen::LLT<crane_model::PassiveMass> factorisation(mass_uu);
  if (factorisation.info() != Eigen::Success) {
    return testing::AssertionFailure() << "M_uu is not positive definite at this configuration";
  }
  out.gain = -factorisation.solve(mass_ua);
  out.bias = -factorisation.solve(bias_u);
  return testing::AssertionSuccess();
}

// The whole passive-row comparison for one description, per axis. `overall` is
// the relative rms over the gain and the gravity bias together, which is the
// one number the ladder below is read off.
struct PassiveComparison
{
  std::vector<Residual> per_axis;
  double overall{0.0};
};

testing::AssertionResult compare_passive_rows(Variant variant, PassiveComparison & out)
{
  const crane_model::Model model = production_model(variant);
  std::vector<Residual> gain;
  for (const char * passive : kPassiveNames) {
    for (const char * actuated : kActuatedNames) {
      gain.emplace_back(
        std::string("d(ddq ") + passive + ")/d(ddq " + actuated + ")", "1");
    }
  }
  std::vector<Residual> moving;
  std::vector<Residual> resting;
  for (const char * passive : kPassiveNames) {
    moving.emplace_back(std::string("ddq ") + passive + " at dq recorded", "rad/s^2");
    resting.emplace_back(std::string("ddq ") + passive + " at dq = 0", "rad/s^2");
  }

  double error_squared = 0.0;
  double reference_squared = 0.0;
  for (std::size_t index = 0; index < samples().size(); ++index) {
    const Sample & sample = samples()[index];
    PassiveMap at_speed;
    PassiveMap at_rest;
    const testing::AssertionResult first = passive_map(model, sample.q, sample.dq, at_speed);
    if (!first) {
      return first;
    }
    const testing::AssertionResult second =
      passive_map(model, sample.q, crane_model::DQ::Zero(), at_rest);
    if (!second) {
      return second;
    }
    for (Eigen::Index row = 0; row < 2; ++row) {
      for (Eigen::Index column = 0; column < 5; ++column) {
        const double theirs = sample.ddq_u_gain(row, column);
        const double mine = at_speed.gain(row, column);
        gain[static_cast<std::size_t>(row * 5 + column)].add(mine, theirs, index);
        error_squared += (mine - theirs) * (mine - theirs);
        reference_squared += theirs * theirs;
      }
      moving[static_cast<std::size_t>(row)].add(
        at_speed.bias[row], sample.ddq_u_bias[row], index);
      const double theirs = sample.ddq_u_bias_static[row];
      resting[static_cast<std::size_t>(row)].add(at_rest.bias[row], theirs, index);
      error_squared += (at_rest.bias[row] - theirs) * (at_rest.bias[row] - theirs);
      reference_squared += theirs * theirs;
    }
  }

  out.per_axis = gain;
  out.per_axis.insert(out.per_axis.end(), moving.begin(), moving.end());
  out.per_axis.insert(out.per_axis.end(), resting.begin(), resting.end());
  out.overall = std::sqrt(error_squared / reference_squared);
  return testing::AssertionSuccess();
}

}  // namespace

// --- the fixture says where it came from --------------------------------------

TEST(CraneModelRecordedParity, TheFixtureNamesItsBagsAndItsTopic)
{
  const std::string & text = fixture_text();
  for (const char * bag : {"2026-08-19_12-06-08-record", "2026-08-19_13-35-36-record",
      "2026-08-19_13-46-41-record", "2026-08-19_14-28-24-record"})
  {
    EXPECT_NE(text.find(bag), std::string::npos) << "the fixture does not name " << bag;
  }
  EXPECT_NE(text.find("/joint_states"), std::string::npos);
  EXPECT_NE(text.find("derive_recorded_parity.py"), std::string::npos);
  // The recordings are named as provenance and opened at derivation time only.
  // Nothing at test time reads a path outside this package, which is what makes
  // the comparison runnable in a CI container with no recordings mounted.
  EXPECT_NE(text.find("/home/vscode/Documents/control_recordings"), std::string::npos)
    << "the fixture does not say which recordings it came from";
  EXPECT_EQ(samples().size(), kSampleCount);
}

// --- forward kinematics --------------------------------------------------------

TEST(CraneModelRecordedParity, ForwardKinematicsAgreesWithTheGeneratedModel)
{
  const crane_model::Model model = production_model(Variant::AsDescribed);
  const std::array<const char *, 3> axes{"x", "y", "z"};
  std::vector<Residual> residuals;
  for (const char * axis : axes) {
    residuals.emplace_back(std::string("K8 position ") + axis, "m");
  }
  for (const char * axis : axes) {
    residuals.emplace_back(std::string("tip position ") + axis, "m");
  }
  Residual orientation("K8 orientation", "rad");

  for (std::size_t index = 0; index < samples().size(); ++index) {
    const Sample & sample = samples()[index];
    auto k8 = model.forward_kinematics(
      sample.q, crane_model::Frame::MountingBase, crane_model::Frame::RotatorLowerPart);
    ASSERT_TRUE(k8.ok()) << k8.status().message;
    auto tip = model.forward_kinematics(
      sample.q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
    ASSERT_TRUE(tip.ok()) << tip.status().message;

    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      residuals[static_cast<std::size_t>(axis)].add(
        k8.value().position_m[axis], sample.k8_position_m[axis], index);
      residuals[static_cast<std::size_t>(axis) + 3].add(
        tip.value().position_m[axis], sample.tip_position_m[axis], index);
    }
    const Eigen::AngleAxisd difference(
      k8.value().orientation.toRotationMatrix().transpose() * sample.k8_rotation);
    orientation.add(difference.angle(), 0.0, index);
  }
  residuals.push_back(orientation);
  report("forward kinematics, crane_model - mp_crane, 64 recorded configurations", residuals);

  // Both models place the linkage from the same description, and place it
  // identically. The bound is machine precision with room for a different BLAS,
  // not a fitted tolerance: the observed worst is 2.7e-15 m and 6.1e-16 rad.
  for (const Residual & residual : residuals) {
    EXPECT_LT(residual.worst(), 1.0e-12) << residual.line();
  }
}

// --- the passive equilibrium ---------------------------------------------------

TEST(CraneModelRecordedParity, ThePassiveEquilibriumAgreesWithTheGeneratedModel)
{
  const crane_model::Model model = production_model(Variant::AsDescribed);
  std::vector<Residual> residuals;
  for (const char * passive : kPassiveNames) {
    residuals.emplace_back(std::string("equilibrium ") + passive, "rad");
  }

  for (std::size_t index = 0; index < samples().size(); ++index) {
    const Sample & sample = samples()[index];
    crane_model::QA q_a;
    q_a << sample.q[0], sample.q[1], sample.q[2], sample.q[3], sample.q[6], sample.q[7];
    auto equilibrium = model.passive_equilibrium(q_a, carrying_nothing());
    ASSERT_TRUE(equilibrium.ok()) << equilibrium.status().message;
    for (Eigen::Index row = 0; row < 2; ++row) {
      residuals[static_cast<std::size_t>(row)].add(
        equilibrium.value()[row], sample.passive_equilibrium_rad[row], index);
    }
  }
  report(
    "the passive equilibrium, crane_model - mp_crane, 64 recorded configurations", residuals);

  // Where the tool hangs is a statement about the linkage and the direction of
  // gravity, not about how the mass is distributed along it, so the two models
  // agree here even though the mass matrix below does not. Observed worst is
  // 4.4e-16 rad; the bound leaves room for the Newton solve's own tolerance.
  for (const Residual & residual : residuals) {
    EXPECT_LT(residual.worst(), 1.0e-9) << residual.line();
  }
}

// --- the mass matrix and the bias, through the map the generated model exposes -

TEST(CraneModelRecordedParity, ThePassiveRowsDisagreeAndTheDescriptionSaysSo)
{
  PassiveComparison described;
  ASSERT_TRUE(compare_passive_rows(Variant::AsDescribed, described));
  report(
    "the passive rows of M and h, crane_model - mp_crane, description as checked in",
    described.per_axis);
  std::cout << "[ parity ]   relative rms over the gain and the gravity bias: " << std::fixed
            << std::setprecision(5) << described.overall << "\n"
            << std::flush;

  // Not a tolerance failure and not to be rounded away: the two models really do
  // disagree about the pendulum, by half of it. The next test says why, and the
  // description decides. This bound exists so that closing the gap by bending
  // `crane_model` towards the generated model fails here.
  EXPECT_GT(described.overall, 0.3)
    << "the generated model now agrees with the description about the tool; if that is a "
       "real change, wiki/robot_model.md 6.3 is what has to be rewritten";
}

TEST(CraneModelRecordedParity, TheDisagreementIsTheTwoDefectsAndNothingElse)
{
  PassiveComparison described;
  ASSERT_TRUE(compare_passive_rows(Variant::AsDescribed, described));
  PassiveComparison without_gripper;
  ASSERT_TRUE(compare_passive_rows(Variant::WithoutTheGripper, without_gripper));
  PassiveComparison as_generated;
  ASSERT_TRUE(compare_passive_rows(Variant::AsTheGeneratedModelHasIt, as_generated));

  report(
    "the passive rows of M and h, crane_model - mp_crane, on the generated model's own geometry",
    as_generated.per_axis);
  std::cout << "[ parity ]   relative rms over the gain and the gravity bias:\n"
            << "[ parity ]     description as checked in               " << std::fixed
            << std::setprecision(5) << described.overall << "\n"
            << "[ parity ]     without the 200 kg rail gripper         "
            << without_gripper.overall << "\n"
            << "[ parity ]     and the tool frame 48.7 mm lower        "
            << as_generated.overall << "\n"
            << std::flush;

  // Rung one: the gripper is most of it. Observed 0.504 -> 0.069.
  EXPECT_LT(without_gripper.overall, described.overall / 4.0)
    << "deleting the rail bodies no longer accounts for most of the disagreement";
  // Rung two: the tool-frame offset is the rest. Observed 0.069 -> 1.0e-4.
  EXPECT_LT(as_generated.overall, 1.0e-3)
    << "something beyond the two defects of wiki/robot_model.md 6.3 now separates the two "
       "models: the parity result no longer holds as written";

  // Per axis, so no single axis can hide behind the aggregate. The bias at the
  // *recorded* dq is deliberately not bounded: the generated model carries a
  // hardcoded linear friction on the passive rows (`fricLinear[4] = 60`,
  // `[5] = 40`, from `add_hardcoded_parameters`) where the description carries
  // joint damping, so those two entries report and do not assert.
  for (std::size_t index = 0; index < as_generated.per_axis.size(); ++index) {
    const bool at_recorded_speed = index >= static_cast<std::size_t>(kGainColumns) &&
      index < static_cast<std::size_t>(kGainColumns + kBiasColumns);
    if (at_recorded_speed) {
      continue;
    }
    EXPECT_LT(as_generated.per_axis[index].relative(), 5.0e-3)
      << as_generated.per_axis[index].line();
  }
}
