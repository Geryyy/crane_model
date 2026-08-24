// Offline reference evaluator for the recorded-trajectory parity fixture.
//
// This file is NOT part of the build.  It is the other half of
// `derive_recorded_parity.py`, which compiles and runs it once, offline, to
// produce the expected columns of `test/recorded_parity_fixture.txt`.  Nothing
// in `CMakeLists.txt` or `package.xml` names `mp_crane`, and nothing in the CI
// suite links it: the generated model is reachable from this directory only
// through a checked-in table of numbers, which is what lets slice 4 assert the
// package is absent from the `hardware` closure while still validating against
// it (PRD 12).
//
// Build and run it through the Python driver:
//
//     python3 test/derive_recorded_parity.py --help
//
// It reads one sample per stdin line -- `q[8] dq[8]` in the canonical order of
// `wiki/implementation/model_api_contract.md` 2 -- and writes one line per
// sample to the file named by its second argument, with the generated model's
// answers in the column order the fixture's legend documents.  The answers do
// not go to stdout because the retained parameter parser writes its own
// commentary there.
//
// What the generated model is asked, and why each one:
//
//   `calcDirKinCrane`         forward kinematics of `K8_rotator_lower_part`,
//                             with the tool transform set to the identity so
//                             that the answer is the same frame the new
//                             backend's `Frame::RotatorLowerPart` names.
//   `calcDirKinCraneTipJoint` forward kinematics of the tip joint, a second
//                             point earlier in the chain, so that a
//                             disagreement can be placed before or after the
//                             passive pair.
//   `integrateDynamicsSemiImplicitEuler`
//                             the mass matrix and the bias.  The generated
//                             model has no public call that returns either:
//                             it exposes the passive-row solve
//                             `ddq_u = -M_uu^-1 (M_ua ddq_a + h_u)`, which is
//                             how the timber stack used it, and that map is
//                             affine in `ddq_a`, so one evaluation at
//                             `ddq_a = 0` and five at the unit vectors recover
//                             it exactly.
//   `get_equilibrium_point_singleRBJaw_analytical`
//                             the passive equilibrium, which is the gravity
//                             part of the bias read through a different route.

#include "epsilon_crane_parameter/parameter_def.hpp"
#include "mp_crane/mp_crane_lib.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// The five actuated coordinates of the generated model's reduced 7-vector.
// Its own `b_iv` uses exactly this list, and it is the canonical actuated
// projection of contract 2 without `q8`.
constexpr std::array<int, 5> kActuated{0, 1, 2, 3, 6};
constexpr int kTip = 4;
constexpr int kTilt = 5;

// A unit step, so `ddq_u` comes back as `(f[i + 7] - xk[i + 7])` with no
// division: the integrator is a plain Euler step and `Ts` only scales.
constexpr double kStep = 1.0;

std::FILE * g_out = nullptr;

void print(double value) { std::fprintf(g_out, " %.17g", value); }

// Return the two passive accelerations the generated model produces for one
// `(q, dq, ddq_a)`.
std::array<double, 2> passive_acceleration(
  const parStruct & par, const double q[7], const double dq[7], const double ddq_a[5])
{
  double xk[14];
  for (int i = 0; i < 7; ++i) {
    xk[i] = q[i];
    xk[i + 7] = dq[i];
  }
  double uk[5];
  for (int i = 0; i < 5; ++i) {
    uk[i] = ddq_a[i];
  }
  double f[14];
  double a[196];
  double b[70];
  bool ok = false;
  mp_crane::integrateDynamicsSemiImplicitEuler(xk, uk, kStep, &par, f, a, b, &ok);
  if (!ok) {
    std::fprintf(stderr, "mp_crane: the passive rows did not solve at this sample\n");
    std::exit(1);
  }
  return {(f[kTip + 7] - xk[kTip + 7]) / kStep, (f[kTilt + 7] - xk[kTilt + 7]) / kStep};
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 3) {
    std::fprintf(stderr, "usage: mp_crane_reference <robot_description.urdf> <output>\n");
    return 2;
  }
  std::ifstream description(argv[1]);
  if (!description) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }
  g_out = std::fopen(argv[2], "w");
  if (g_out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[2]);
    return 2;
  }
  std::stringstream buffer;
  buffer << description.rdbuf();

  // Value-initialised: `parStruct` has no constructor and
  // `add_hardcoded_parameters` leaves `fricLinear[7]` untouched, so a plain
  // declaration would carry one uninitialised double into the comparison.
  parStruct par{};
  if (!par.set_urdf_model(buffer.str())) {
    std::fprintf(stderr, "mp_crane: the description did not parse\n");
    return 1;
  }
  par.add_controlled_joint_names(
    {"theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope",
      "theta8_rotator_joint", "q9_left_rail_joint"});
  par.parse_crane_model();
  par.add_hardcoded_parameters();
  par.fill_arrays();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    double q[8];
    double dq[8];
    for (double & value : q) {
      fields >> value;
    }
    for (double & value : dq) {
      fields >> value;
    }
    if (!fields) {
      std::fprintf(stderr, "expected 16 numbers per line, got: %s\n", line.c_str());
      return 1;
    }

    // The tool transform is the identity, so `H0Tcp` is the pose of the frame
    // the generated model calls `K8` and the contract calls
    // `Frame::RotatorLowerPart`.  MATLAB Coder writes matrices column-major.
    const double identity[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    double h0_tcp[16];
    double position[3];
    double orientation[9];
    mp_crane::calcDirKinCrane(q, identity, &par, h0_tcp, position, orientation);

    double tip[3];
    mp_crane::calcDirKinCraneTipJoint(q, &par, tip);

    double q7[7];
    double dq7[7];
    for (int i = 0; i < 7; ++i) {
      q7[i] = q[i];
      dq7[i] = dq[i];
    }
    double zero_input[5]{0, 0, 0, 0, 0};
    const auto bias = passive_acceleration(par, q7, dq7, zero_input);

    double rest[7]{0, 0, 0, 0, 0, 0, 0};
    const auto bias_static = passive_acceleration(par, q7, rest, zero_input);

    std::array<std::array<double, 2>, 5> gain{};
    for (std::size_t column = 0; column < kActuated.size(); ++column) {
      double unit[5]{0, 0, 0, 0, 0};
      unit[column] = 1.0;
      const auto response = passive_acceleration(par, q7, dq7, unit);
      gain[column] = {response[0] - bias[0], response[1] - bias[1]};
    }

    double equilibrium[8];
    mp_crane::get_equilibrium_point_singleRBJaw_analytical(q, &par, equilibrium);

    for (double value : position) {
      print(value);
    }
    // Row-major, so the fixture reads the way a rotation matrix is written.
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        print(h0_tcp[row + 4 * column]);
      }
    }
    for (double value : tip) {
      print(value);
    }
    for (int row = 0; row < 2; ++row) {
      for (const auto & column : gain) {
        print(column[row]);
      }
    }
    print(bias[0]);
    print(bias[1]);
    print(bias_static[0]);
    print(bias_static[1]);
    print(equilibrium[kTip]);
    print(equilibrium[kTilt]);
    std::fprintf(g_out, "\n");
  }
  std::fclose(g_out);
  return 0;
}
