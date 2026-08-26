#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// pinocchio/autodiff/casadi.hpp teaches Eigen about casadi::SX and has to be
// seen before any other Pinocchio header, or the algorithms below instantiate
// against the default NumTraits and fail deep inside Eigen. The standard
// headers above it are the include order cpplint asks for and pull in neither.
#include <pinocchio/autodiff/casadi.hpp>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <casadi/casadi.hpp>

#include "crane_model/symbolic/casadi_graph.hpp"

#include "cylinder_geometry.hpp"
#include "symbolic_graph.hpp"

// wiki/mpc.md §5.2 and wiki/trajectory_planning.md §5.2 both want one dynamics
// implementation in the system. This file is how that is kept true: it does not
// restate the equations of motion, it runs Pinocchio's own `crba` and
// `nonLinearEffects` over `casadi::SX` on the same parsed description, with the
// same projection P, the same damping and the same payload the numeric calls
// use. The transmission is the same template `Model::cylinder_jacobian`
// instantiates (`cylinder_geometry.hpp`). What is written out here is only the
// assembly: which rows go where, and the Schur complement of robot_model §3.1.

namespace crane_model
{
namespace
{

using SX = casadi::SX;
using CasadiModel = pinocchio::ModelTpl<SX>;
using CasadiData = pinocchio::DataTpl<SX>;
using SymbolicVector = Eigen::Matrix<SX, Eigen::Dynamic, 1>;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

casadi_int index_of(std::size_t value)
{
  return static_cast<casadi_int>(value);
}

// The canonical eight coordinates and their rates, read out of the 16-state of
// contract §2: x = [q_a, q_u, dq_a, dq_u], with q_a in the actuated projection
// order and q_u in the passive one.
struct Coordinates
{
  std::array<SX, kGeneralizedDof> q{};
  std::array<SX, kGeneralizedDof> dq{};
  std::array<SX, kActuatedDof> q_a{};
  std::array<SX, kActuatedDof> dq_a{};
  std::array<SX, kPassiveDof> dq_u{};
};

Coordinates unpack(const SX& x)
{
  Coordinates out;
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    out.q_a[axis] = x(index_of(axis));
    out.dq_a[axis] = x(index_of(kActuatedDof + kPassiveDof + axis));
    out.q[static_cast<std::size_t>(detail::kActuatedRows[axis])] = out.q_a[axis];
    out.dq[static_cast<std::size_t>(detail::kActuatedRows[axis])] = out.dq_a[axis];
  }
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    out.dq_u[row] = x(index_of(2 * kActuatedDof + kPassiveDof + row));
    out.q[static_cast<std::size_t>(detail::kPassiveRows[row])] = x(index_of(kActuatedDof + row));
    out.dq[static_cast<std::size_t>(detail::kPassiveRows[row])] = out.dq_u[row];
  }
  return out;
}

// The parsed model's configuration and velocity vectors, written from the
// canonical eight exactly as Model::Impl::write_configuration and ::expand do:
// the mimics follow their source, and every joint outside both sets keeps its
// neutral value and zero velocity.
void expand(
  const detail::SymbolicSource& source, const Coordinates& coordinates,
  SymbolicVector& configuration, SymbolicVector& velocity)
{
  configuration.resize(source.model.nq);
  for (Eigen::Index index = 0; index < source.model.nq; ++index) {
    configuration[index] = SX(source.neutral[index]);
  }
  velocity.setZero(source.model.nv);

  const auto write_slot = [&configuration](const detail::JointSlot& slot, const SX& value) {
      if (slot.unbounded) {
        configuration[slot.config_index] = cos(value);
        configuration[slot.config_index + 1] = sin(value);
      } else {
        configuration[slot.config_index] = value;
      }
    };

  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    write_slot(source.joints[index], coordinates.q[index]);
  }
  for (const detail::CoupledJoint& coupled : source.coupled) {
    write_slot(
      coupled.slot, coupled.multiplier * coordinates.q[coupled.source] + coupled.offset);
  }
  for (std::size_t index = 0; index < kGeneralizedDof; ++index) {
    for (std::size_t entry = 0; entry < source.drive_count[index]; ++entry) {
      const detail::Drive& drive = source.drives[index][entry];
      velocity[drive.velocity_index] = drive.weight * coordinates.dq[index];
    }
  }
}

// M and h + D dq of robot_model §1 in the canonical eight coordinates, from the
// same two Pinocchio algorithms the numeric path calls.
struct Equations
{
  std::array<std::array<SX, kGeneralizedDof>, kGeneralizedDof> mass{};
  std::array<SX, kGeneralizedDof> bias{};
};

Equations evaluate(
  const CasadiModel& model, CasadiData& data, const detail::SymbolicSource& source,
  const Coordinates& coordinates)
{
  SymbolicVector configuration;
  SymbolicVector velocity;
  expand(source, coordinates, configuration, velocity);

  pinocchio::crba(model, data, configuration);
  pinocchio::nonLinearEffects(model, data, configuration, velocity);

  // `crba` fills the upper triangle of data.M only, so the lower half is read
  // back transposed rather than mirrored into place -- the same reading
  // Model::Impl::mass_entry does.
  const auto mass_entry = [&data](Eigen::Index row, Eigen::Index column) {
      return row <= column ? data.M(row, column) : data.M(column, row);
    };

  Equations out;
  for (std::size_t row = 0; row < kGeneralizedDof; ++row) {
    for (std::size_t column = row; column < kGeneralizedDof; ++column) {
      SX sum = SX::zeros();
      for (std::size_t left = 0; left < source.drive_count[row]; ++left) {
        for (std::size_t right = 0; right < source.drive_count[column]; ++right) {
          sum += source.drives[row][left].weight * source.drives[column][right].weight *
            mass_entry(
            source.drives[row][left].velocity_index,
            source.drives[column][right].velocity_index);
        }
      }
      out.mass[row][column] = sum;
      out.mass[column][row] = sum;
    }
  }
  for (std::size_t row = 0; row < kGeneralizedDof; ++row) {
    SX sum = SX::zeros();
    for (std::size_t entry = 0; entry < source.drive_count[row]; ++entry) {
      const detail::Drive& drive = source.drives[row][entry];
      sum += drive.weight * data.nle[drive.velocity_index];
    }
    out.bias[row] =
      sum + source.damping[static_cast<Eigen::Index>(row)] * coordinates.dq[row];
  }
  return out;
}

// M and h split by the actuated/passive partition of contract §2, gathered by
// index because I_a is not contiguous.
struct Partition
{
  std::array<std::array<SX, kActuatedDof>, kActuatedDof> M_aa{};
  std::array<std::array<SX, kPassiveDof>, kActuatedDof> M_au{};
  std::array<std::array<SX, kActuatedDof>, kPassiveDof> M_ua{};
  std::array<std::array<SX, kPassiveDof>, kPassiveDof> M_uu{};
  std::array<SX, kActuatedDof> h_a{};
  std::array<SX, kPassiveDof> h_u{};
};

Partition partition(const Equations& equations)
{
  const auto actuated = [](std::size_t index) {
      return static_cast<std::size_t>(detail::kActuatedRows[index]);
    };
  const auto passive = [](std::size_t index) {
      return static_cast<std::size_t>(detail::kPassiveRows[index]);
    };

  Partition split;
  for (std::size_t row = 0; row < kActuatedDof; ++row) {
    split.h_a[row] = equations.bias[actuated(row)];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      split.M_aa[row][column] = equations.mass[actuated(row)][actuated(column)];
    }
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      split.M_au[row][column] = equations.mass[actuated(row)][passive(column)];
    }
  }
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    split.h_u[row] = equations.bias[passive(row)];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      split.M_ua[row][column] = equations.mass[passive(row)][actuated(column)];
    }
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      split.M_uu[row][column] = equations.mass[passive(row)][passive(column)];
    }
  }
  return split;
}

// robot_model §3.1: ddq_u = -M_uu^{-1} (M_ua ddq_a + h_u), with h_u already
// carrying D_uu dq_u. Two by two, so the inverse is the adjugate and there is
// no factorisation to branch on -- a configuration where M_uu is singular
// leaves a non-finite expression, which is what the numeric self-check below
// looks for.
std::array<SX, kPassiveDof> passive_acceleration(
  const Partition& split, const std::array<SX, kInputDof>& u)
{
  std::array<SX, kPassiveDof> right_hand{};
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    right_hand[row] = split.h_u[row];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      right_hand[row] += split.M_ua[row][column] * u[column];
    }
  }
  const SX determinant =
    split.M_uu[0][0] * split.M_uu[1][1] - split.M_uu[0][1] * split.M_uu[1][0];
  return {{
    -(split.M_uu[1][1] * right_hand[0] - split.M_uu[0][1] * right_hand[1]) / determinant,
    -(split.M_uu[0][0] * right_hand[1] - split.M_uu[1][0] * right_hand[0]) / determinant}};
}

// tau_a of robot_model §3.3, the actuated rows of M ddq + h at the consistent
// passive acceleration. It equals M_eff u + h_eff of §3.4 and is the quantity
// mpc §2's effort term and §3's constraint 6 are written in.
std::array<SX, kActuatedDof> actuated_force(
  const Partition& split, const std::array<SX, kInputDof>& u,
  const std::array<SX, kPassiveDof>& ddq_u)
{
  std::array<SX, kActuatedDof> tau_a{};
  for (std::size_t row = 0; row < kActuatedDof; ++row) {
    tau_a[row] = split.h_a[row];
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      tau_a[row] += split.M_aa[row][column] * u[column];
    }
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      tau_a[row] += split.M_au[row][column] * ddq_u[column];
    }
  }
  return tau_a;
}

// The output map z. wiki/mpc.md §3.1 requires both non-smooth pieces of
// constraint 7 to be smoothed for a gradient-based solver: |v| by
// sqrt(v^2 + eps^2), and the step in A^{+-} at v = 0 by a tanh of width eps_v.
SX output_map(
  const detail::SymbolicSource& source, const Coordinates& coordinates,
  const std::array<SX, kActuatedDof>& tau_a)
{
  const std::array<SX, kActuatedDof> j_cyl = cylinder::jacobian_diagonal<SX>(
    source.constants, source.tool, coordinates.q[1], coordinates.q[2], coordinates.q[7]);
  const std::array<cylinder::AxisAreas, kActuatedDof> areas =
    cylinder::axis_areas(source.constants);

  std::vector<SX> blocks;
  blocks.reserve(4 * kActuatedDof);
  std::vector<SX> cylinder_force;
  std::vector<SX> piston_velocity;
  std::vector<SX> axis_flow;
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    const SX velocity = j_cyl[axis] * coordinates.dq_a[axis];
    const SX magnitude =
      sqrt(velocity * velocity + source.constants.eps_abs * source.constants.eps_abs);
    const double mean_area = 0.5 * (areas[axis].a_eff_pos + areas[axis].a_eff_neg);
    const double half_step = 0.5 * (areas[axis].a_eff_pos - areas[axis].a_eff_neg);
    const SX effective_area =
      mean_area + half_step * tanh(velocity / source.constants.eps_v);

    cylinder_force.push_back(tau_a[axis] / j_cyl[axis]);
    piston_velocity.push_back(velocity);
    axis_flow.push_back(effective_area * magnitude);
  }
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    blocks.push_back(tau_a[axis]);
  }
  for (const SX& entry : cylinder_force) {
    blocks.push_back(entry);
  }
  for (const SX& entry : piston_velocity) {
    blocks.push_back(entry);
  }
  for (const SX& entry : axis_flow) {
    blocks.push_back(entry);
  }
  return SX::vertcat(blocks);
}

// One numeric evaluation at the state the description's own neutral
// configuration gives, with zero velocity and zero input. It is not a physical
// check -- issue 036's parity campaign is -- but it is what separates a graph
// from an expression: a description whose passive mass is singular there, a
// massless fixture for instance, builds an expression that evaluates to NaN,
// and a consumer must be told that here rather than by its first solve.
Status evaluate_once(const symbolic::CasadiGraph& graph)
{
  const std::vector<casadi::DM> arguments{
    casadi::DM::zeros(index_of(kStateDof)), casadi::DM::zeros(index_of(kInputDof))};
  const auto regular = [](const std::vector<casadi::DM>& values) {
      for (const casadi::DM& value : values) {
        if (!value.is_regular()) {
          return false;
        }
      }
      return true;
    };

  if (!regular(graph.f(arguments)) || !regular(graph.F_disc(arguments))) {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      "the graph does not evaluate to finite numbers at the neutral configuration, so the "
      "description cannot produce a usable symbolic model");
  }
  if (!graph.z.is_null() && !regular(graph.z(arguments))) {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      "the output map does not evaluate to finite numbers at the neutral configuration");
  }
  if (!regular(graph.passive_rows(std::vector<casadi::DM>{arguments.front()}))) {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      "the passive rows do not evaluate to finite numbers at the neutral configuration");
  }
  return Status{};
}

Status assemble(
  const detail::SymbolicSource& source, const SymbolicGraphSpec& spec,
  symbolic::CasadiGraph& graph)
{
  const CasadiModel model = source.model.cast<SX>();
  CasadiData data(model);

  const SX x = SX::sym("x", index_of(kStateDof));
  const SX u = SX::sym("u", index_of(kInputDof));
  const Coordinates coordinates = unpack(x);

  std::array<SX, kInputDof> input{};
  for (std::size_t axis = 0; axis < kInputDof; ++axis) {
    input[axis] = u(index_of(axis));
  }

  const Equations equations = evaluate(model, data, source, coordinates);
  const Partition split = partition(equations);
  const std::array<SX, kPassiveDof> ddq_u = passive_acceleration(split, input);

  std::vector<SX> derivative;
  derivative.reserve(kStateDof);
  for (std::size_t axis = 0; axis < kActuatedDof; ++axis) {
    derivative.push_back(coordinates.dq_a[axis]);
  }
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    derivative.push_back(coordinates.dq_u[row]);
  }
  for (std::size_t axis = 0; axis < kInputDof; ++axis) {
    derivative.push_back(input[axis]);
  }
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    derivative.push_back(ddq_u[row]);
  }
  graph.f = casadi::Function(
    "f", {x, u}, {SX::vertcat(derivative)}, {"x", "u"}, {"xdot"});

  if (spec.include_output_map) {
    graph.z = casadi::Function(
      "z", {x, u},
      {output_map(source, coordinates, actuated_force(split, input, ddq_u))},
      {"x", "u"}, {"z"});
  }

  SX mass_uu = SX::zeros(index_of(kPassiveDof), index_of(kPassiveDof));
  SX mass_ua = SX::zeros(index_of(kPassiveDof), index_of(kActuatedDof));
  SX bias_u = SX::zeros(index_of(kPassiveDof), 1);
  for (std::size_t row = 0; row < kPassiveDof; ++row) {
    bias_u(index_of(row)) = split.h_u[row];
    for (std::size_t column = 0; column < kPassiveDof; ++column) {
      mass_uu(index_of(row), index_of(column)) = split.M_uu[row][column];
    }
    for (std::size_t column = 0; column < kActuatedDof; ++column) {
      mass_ua(index_of(row), index_of(column)) = split.M_ua[row][column];
    }
  }
  graph.passive_rows = casadi::Function(
    "passive_rows", {x}, {mass_uu, mass_ua, bias_u}, {"x"}, {"M_uu", "M_ua", "h_u"});

  // ERK4 over one sample time (mpc §5.2). Built on MX so the four stages call
  // `f` rather than each carrying a copy of its expression graph.
  const casadi::MX state = casadi::MX::sym("x", index_of(kStateDof));
  const casadi::MX command = casadi::MX::sym("u", index_of(kInputDof));
  const double step = spec.sample_time_s;
  const casadi::MX k1 = graph.f(std::vector<casadi::MX>{state, command}).at(0);
  const casadi::MX k2 =
    graph.f(std::vector<casadi::MX>{state + (0.5 * step) * k1, command}).at(0);
  const casadi::MX k3 =
    graph.f(std::vector<casadi::MX>{state + (0.5 * step) * k2, command}).at(0);
  const casadi::MX k4 = graph.f(std::vector<casadi::MX>{state + step * k3, command}).at(0);
  graph.F_disc = casadi::Function(
    "F_disc", {state, command},
    {state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)}, {"x", "u"}, {"x_next"});
  graph.sample_time_s = step;
  return Status{};
}

}  // namespace

namespace detail
{

Status build_graph(
  const SymbolicSource& source, const SymbolicGraphSpec& spec, GraphHandle& handle,
  GraphDimensions& dimensions)
{
  if (!std::isfinite(spec.sample_time_s) || spec.sample_time_s <= 0.0) {
    return failure(
      ErrorCode::InvalidArgument, "sample_time_s must be finite and strictly positive");
  }

  auto graph = std::make_shared<symbolic::CasadiGraph>();
  Status status;
  try {
    status = assemble(source, spec, *graph);
    if (status.ok()) {
      status = evaluate_once(*graph);
    }
  } catch (const std::exception& error) {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      std::string("CasADi could not build the symbolic graph: ") + error.what());
  }
  if (!status.ok()) {
    return status;
  }

  // Read back rather than assumed: a graph that came out the wrong size is a
  // backend failure and not a graph, so no consumer ever sees a handle whose
  // dimensions do not match contract §9.
  dimensions.state = static_cast<std::size_t>(graph->f.size1_in(0));
  dimensions.input = static_cast<std::size_t>(graph->f.size1_in(1));
  dimensions.output_map = !graph->z.is_null();
  if (dimensions.state != kStateDof || dimensions.input != kInputDof) {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      "the built graph has state dimension " + std::to_string(dimensions.state) +
      " and input dimension " + std::to_string(dimensions.input) +
      ", which is not the frozen 16 by 6");
  }
  if (dimensions.output_map &&
    static_cast<std::size_t>(graph->z.size1_out(0)) != symbolic::kOutputDof)
  {
    return failure(
      ErrorCode::SymbolicBackendFailure,
      "the built output map has dimension " + std::to_string(graph->z.size1_out(0)) +
      " rather than " + std::to_string(symbolic::kOutputDof));
  }

  handle = std::move(graph);
  return Status{};
}

}  // namespace detail

namespace symbolic
{

Result<CasadiGraphHandle> casadi_graph(
  const ModelConfig& config, const SymbolicGraphSpec& spec, const Payload& payload)
{
  detail::SymbolicSource source;
  Status status = detail::parse(config, source);
  if (status.ok()) {
    status = detail::attach_payload(source.model, source.mount, payload);
  }
  if (!status.ok()) {
    return Result<CasadiGraphHandle>::failure(std::move(status));
  }
  detail::GraphHandle handle;
  detail::GraphDimensions dimensions;
  status = detail::build_graph(source, spec, handle, dimensions);
  if (!status.ok()) {
    return Result<CasadiGraphHandle>::failure(std::move(status));
  }
  return Result<CasadiGraphHandle>::success(std::move(handle));
}

}  // namespace symbolic
}  // namespace crane_model
