#ifndef SYMBOLIC_GRAPH_HPP_
#define SYMBOLIC_GRAPH_HPP_

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <pinocchio/multibody/model.hpp>

#include "crane_model/model.hpp"

// The seam between the numeric model and the symbolic one. `model.cpp` parses
// the description once and hands what it parsed across this header;
// `symbolic_graph.cpp` turns it into the CasADi graph of contract §9. Nothing
// here mentions a CasADi type, which is what lets `model.cpp` stay free of
// them.
//
// This header is private to the implementation. It is not installed and
// nothing on the public API mentions it (contract §6).

namespace crane_model
{

namespace symbolic
{
struct CasadiGraph;
}  // namespace symbolic

namespace detail
{

// The actuated projection [0, 1, 2, 3, 6, 7] and the passive projection [4, 5]
// of the model API contract §2, i.e. I_a and I_u of wiki/robot_model.md §0.1.
// Every block of M and every row of h is gathered through these, on both sides
// of this header; nothing assumes the two classes are contiguous, because they
// are not.
constexpr std::array<Eigen::Index, kActuatedDof> kActuatedRows{{0, 1, 2, 3, 6, 7}};
constexpr std::array<Eigen::Index, kPassiveDof> kPassiveRows{{4, 5}};

// One canonical coordinate's place in the parsed model. `unbounded` is the
// `continuous` rotator of ROS 2 Interfaces §3.1: Pinocchio stores such a joint
// as (cos, sin), so it occupies two configuration entries and one velocity
// entry.
struct JointSlot
{
  Eigen::Index config_index{0};
  Eigen::Index velocity_index{0};
  bool unbounded{false};
};

// A joint the description declares as a `<mimic>` of a canonical one. Two of
// them matter: `q5_small_telescope` mimics `q4_big_telescope`, which is the
// second telescope stage of robot_model §0 and the doubling of hydraulics §2.4,
// and the PZS100's `q11_right_rail_joint` mimics `q9_left_rail_joint`, the
// simulation-only coupled joint of contract §2. The multiplier and offset are
// read out of the description, not assumed.
struct CoupledJoint
{
  JointSlot slot;
  std::size_t source{0};  // canonical index driving this joint
  double multiplier{1.0};
  double offset{0.0};
};

// One canonical coordinate's footprint in the parsed model's velocity vector:
// its own row, weight one, plus one row per `<mimic>` that follows it, weighted
// by that mimic's multiplier. dq_full = P dq and ddq_full = P ddq for the
// constant P these rows describe, so M = P^T M_full P and h = P^T h_full --
// which is how the second telescope stage and the mirrored rail get their
// inertia counted on the coordinate that drives them.
struct Drive
{
  Eigen::Index velocity_index{0};
  double weight{1.0};
};

constexpr std::size_t kMaxDrives = 4;

// Where a payload attaches (wiki/robot_model.md §5): the joint that carries
// K8_rotator_lower_part, the fixed placement of that link within it, and the
// inertia that joint has without a payload.
struct PayloadMount
{
  pinocchio::JointIndex joint{0};
  pinocchio::SE3 placement{pinocchio::SE3::Identity()};
  pinocchio::Inertia bare_inertia{pinocchio::Inertia::Zero()};
  bool attachable{false};
};

// The description as parsed. `Model::create` builds its implementation state
// out of one of these and `symbolic::casadi_graph` builds a graph out of
// another, so the joint map, the mimic projection P, the damping and the
// payload mount are stated once whichever way a caller arrives.
struct SymbolicSource
{
  pinocchio::Model model;
  Eigen::VectorXd neutral;
  std::array<JointSlot, kGeneralizedDof> joints{};
  std::vector<CoupledJoint> coupled;
  std::array<std::array<Drive, kMaxDrives>, kGeneralizedDof> drives{};
  std::array<std::size_t, kGeneralizedDof> drive_count{};
  Q damping{Q::Zero()};
  PayloadMount mount;
  Tool tool{Tool::Pzs100};
};

// The built graph, held by the public `SymbolicGraph` so that handle owns a
// real graph rather than a claim about one. The type is opaque here: only
// `symbolic_graph.cpp` and a consumer of the `crane_model::casadi_graph` target
// ever see a CasADi function.
using GraphHandle = std::shared_ptr<const symbolic::CasadiGraph>;

// What the public handle reports, read back off the built functions rather
// than compiled in, so a graph that came out the wrong size is caught here and
// not by the consumer that evaluates it.
struct GraphDimensions
{
  std::size_t state{0};
  std::size_t input{0};
  bool output_map{false};
};

// Parses `config`. Defined in `model.cpp`, which owns the URDF front end.
Status parse(const ModelConfig& config, SymbolicSource& out);

// Writes the payload body of wiki/robot_model.md §5 into the joint inertia of
// `mount`, or says why it cannot. Defined in `model.cpp`; the one statement of
// where a payload goes, shared by every numeric call and by the graph.
Status attach_payload(
  pinocchio::Model& model, const PayloadMount& mount, const Payload& payload);

// The graph of contract §9. Defined in `symbolic_graph.cpp`; the only place a
// CasADi expression is built.
Status build_graph(
  const SymbolicSource& source, const SymbolicGraphSpec& spec, GraphHandle& handle,
  GraphDimensions& dimensions);

}  // namespace detail
}  // namespace crane_model

#endif  // SYMBOLIC_GRAPH_HPP_
