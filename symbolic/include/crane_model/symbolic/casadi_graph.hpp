#ifndef CRANE_MODEL__SYMBOLIC__CASADI_GRAPH_HPP_
#define CRANE_MODEL__SYMBOLIC__CASADI_GRAPH_HPP_

#include <casadi/casadi.hpp>

#include <cstddef>
#include <memory>

#include "crane_model/model.hpp"

// The backend-specific half of contract §9. `Model::symbolic_graph` returns a
// backend-neutral handle; this header hands the same graph over as CasADi
// function objects, for the two consumers that need them:
//
//   * `crane_mpc` (PRD §2, slice 6) exports its acados model from `f` and `z`;
//   * `crane_planning` (slice 5) carries the passive rows of
//     `wiki/trajectory_planning.md` §5.2 out of `passive_rows`.
//
// It is a *separate target*, `crane_model::casadi_graph`, and a consumer opts
// in by linking it. Contract §9 forbids `casadi::SX` and `casadi::MX` in the
// public header and in the compile fixtures; this is not the public header,
// and `test/consumer_public_only.cpp` is what enforces the difference.
//
// Everything here is non-real-time (contract §10). Building a graph parses the
// description, runs Pinocchio's `crba` and `nonLinearEffects` symbolically and
// hands the result to CasADi -- tens of milliseconds and a few thousand
// expression nodes, all of it allocating. Build it once, off the control cycle.

namespace crane_model
{
namespace symbolic
{

// The algebraic output z of wiki/nomenclature.md §10, in four blocks of six.
// It carries what wiki/mpc.md §2 costs and §3 constrains, so a consumer does
// not rebuild the transmission inside its own solver:
//
//   tau_a   the actuated generalized force, the effort term of §2 and the left
//           side of constraint 6;
//   F_cyl   the cylinder force tau_a induces, F_i = tau_{a,i} / J_{c,ii}, so
//           constraint 6 is the plain box |F_i| <= F_i^max;
//   v       piston velocity, v = J_cyl dq_a;
//   Q       per-axis pump draw, Q_i = A_i^{+-}(v_i) |v_i|, so constraint 7 is
//           sum_i Q_i <= Q_P^max.
//
// F_i^max and Q_P^max are limits and belong to the consumer, not here.
inline constexpr std::size_t kOutputDof = 24;
inline constexpr std::size_t kActuatedForceOffset = 0;
inline constexpr std::size_t kCylinderForceOffset = 6;
inline constexpr std::size_t kPistonVelocityOffset = 12;
inline constexpr std::size_t kAxisFlowOffset = 18;

// The smoothing wiki/mpc.md §3.1 requires of constraint 7, applied to the
// *model* and never to the inner loop's dead-zone compensator. Both are
// compiled in, because `SymbolicGraphSpec` is frozen and carries no place for
// them; both are far below the resolution any axis is commanded at, so the
// smoothed Q agrees with `Model::transmission`'s to well under a part in 10^6
// once |v| is a millimetre per second or more.
inline constexpr double kEpsAbs = 1.0e-6;  // eps,   |v| ~= sqrt(v^2 + eps^2), m/s
inline constexpr double kEpsV = 1.0e-3;    // eps_v, the A^{+-} switch scale,  m/s

// The same equations as the numeric model, as CasADi functions.
//
// x is the 16-state [q_a, q_u, dq_a, dq_u] of contract §2 and u the six
// actuated accelerations ddq_a. `f`, `z` and `passive_rows` are `SXFunction`s,
// which is what acados wants to export from; `F_disc` is the one `MXFunction`,
// so a rollout calls `f` four times instead of carrying four copies of its
// expression graph.
struct CasadiGraph
{
  // f(x, u) -> xdot, the continuous dynamics. Input-affine in u by
  // construction (wiki/mpc.md §5.2), because ddq_u depends on u through
  // -M_uu^{-1} M_ua u alone.
  casadi::Function f;

  // F_disc(x, u) -> x_next, one ERK4 step of `sample_time_s`. wiki/mpc.md §5.2:
  // with the ideal inner loop the model is not stiff, so an explicit integrator
  // is enough.
  casadi::Function F_disc;

  // z(x, u) -> z, the output map above. Null when
  // `SymbolicGraphSpec::include_output_map` was false.
  casadi::Function z;

  // passive_rows(x) -> M_uu, M_ua, h_u. The three blocks
  // wiki/trajectory_planning.md §5.2 builds its path-constrained OCP out of;
  // h_u already carries D_uu dq_u, exactly as `full_dynamics`'s bias does.
  casadi::Function passive_rows;

  double sample_time_s{};
};

// The graph is handed over by shared handle rather than by value. Two reasons,
// and neither is a preference: `casadi::Function` has no `noexcept` move, so a
// `Result<CasadiGraph>` would have a deleted one and would not compile against
// the frozen `Result` of contract §5; and the public `SymbolicGraph` holds the
// very same object, so a handle is what it already is.
using CasadiGraphHandle = std::shared_ptr<const CasadiGraph>;

// Builds the graph of the description in `config`, with `payload` baked in as
// a constant.
//
// It takes a `ModelConfig` and not a `Model` because it has to: `Model` keeps
// its parse behind a private pimpl and the public header is frozen, so there is
// no way in from the outside and no member to add one to. Pass the same
// `ModelConfig` that built the `Model`. The description is parsed a second
// time, by the same parser, through the same joint map, the same mimic
// projection and the same damping rule -- one implementation, two invocations,
// both off the control cycle.
//
// Fails with `SymbolicBackendFailure` when CasADi cannot build a usable graph,
// and with the ordinary codes of contract §5 -- `InvalidRobotDescription`,
// `MissingJoint`, `InvalidPayload`, `FrameUnavailable`, `InvalidArgument` --
// before it gets that far.
Result<CasadiGraphHandle> casadi_graph(
  const ModelConfig& config, const SymbolicGraphSpec& spec, const Payload& payload);

}  // namespace symbolic
}  // namespace crane_model

#endif  // CRANE_MODEL__SYMBOLIC__CASADI_GRAPH_HPP_
