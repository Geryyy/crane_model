#ifndef CRANE_MODEL__MODEL_DETAIL_HPP_
#define CRANE_MODEL__MODEL_DETAIL_HPP_

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <vector>

#include <pinocchio/multibody/model.hpp>

#include "crane_model/model.hpp"

namespace crane_model::detail
{

constexpr std::array<Eigen::Index, kActuatedDof> kActuatedRows{{0, 1, 2, 3, 6, 7}};
constexpr std::array<Eigen::Index, kPassiveDof> kPassiveRows{{4, 5}};

struct JointSlot
{
  Eigen::Index config_index{0};
  Eigen::Index velocity_index{0};
  bool unbounded{false};
};

struct CoupledJoint
{
  JointSlot slot;
  std::size_t source{0};
  double multiplier{1.0};
  double offset{0.0};
};

struct Drive
{
  Eigen::Index velocity_index{0};
  double weight{1.0};
};

constexpr std::size_t kMaxDrives = 4;

struct PayloadMount
{
  pinocchio::JointIndex joint{0};
  pinocchio::SE3 placement{pinocchio::SE3::Identity()};
  pinocchio::Inertia bare_inertia{pinocchio::Inertia::Zero()};
  bool attachable{false};
};

struct ParsedModel
{
  hydraulics::Constants constants;
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

Status parse(const ModelConfig& config, ParsedModel& out);
Status attach_payload(
  pinocchio::Model& model, const PayloadMount& mount, const Payload& payload);

}  // namespace crane_model::detail

#endif  // CRANE_MODEL__MODEL_DETAIL_HPP_
