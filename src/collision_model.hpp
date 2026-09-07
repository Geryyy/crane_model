#ifndef COLLISION_MODEL_HPP_
#define COLLISION_MODEL_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

#include "crane_model/model.hpp"

// The collision geometry of the machine, and the pairs of it that are not worth
// checking. Both are derived once by `scripts/derive_collision_model.py` from
// the descriptions under `test/description/` and the meshes those descriptions
// name; the script is the derivation the issue asks to be checked in beside the
// list, and `config/allowed_collisions.srdf` is the same list in the SRDF
// spelling `wiki/implementation/libraries.md` keeps MoveIt's format for.
//
// It is compiled in rather than read, because the descriptions carry their link
// collision geometry as `package://` STL meshes and this library is ROS-free:
// `ModelConfig` carries the description XML and nothing else, so there is no
// package index to resolve a URI against and no path to read a mesh from. Every
// *placement* still comes from the description at runtime -- each primitive
// below is attached to the link frame Pinocchio parsed -- and the `<collision>`
// element it was fitted to is recorded beside it, so
// `CraneModelCollision.PrimitivesWereFittedToThisDescription` fails when a
// description moves that element instead of silently keeping this fit.
//
// This header is private to the implementation and to the contract test. It is
// not installed and nothing on the public API mentions it (contract 6).

namespace crane_model
{
namespace collision_model
{

enum class LinkShape : std::uint8_t { Capsule, Box };

// One convex primitive, in the frame of the link it belongs to.
//
// `wiki/trajectory_planning.md` 4.2 asks for capsules. The fit takes the smaller
// of an enclosing capsule and an enclosing box per link, which gives a capsule
// where the link really is a beam and a box where it is not: the enclosing
// capsule around `K0_mounting_base` has a 0.98 m radius, and a model that
// refuses motions clearing the base by a metre is not conservative, it is
// unusable. Both candidates contain the mesh, so the smaller one is safe in the
// direction that matters and only ever tighter.
struct LinkPrimitive
{
  Tool tool;
  const char * link;
  LinkShape shape;
  // capsule: {radius, half length, unused}; box: the three half sides.
  std::array<double, 3> extents_m;
  std::array<double, 3> origin_m;      // the primitive's centre in the link frame
  std::array<double, 4> orientation;   // x, y, z, w; a capsule lies along its own z
  const char * source;                 // the <collision> geometry it was fitted to
  std::array<double, 6> source_pose;   // that element's origin, xyz then rpy
};

inline constexpr std::size_t kLinkPrimitiveCount = 15;

inline constexpr std::array<LinkPrimitive, kLinkPrimitiveCount> kLinkPrimitives{{
  {Tool::Pzs100, "K0_mounting_base",
    LinkShape::Box,
    {2.266948, 0.797056, 0.408198},
    {-0.143463, 0.349672, -0.223394},
    {0.453228552, -0.459227828, -0.538705081, 0.541747651},
    "mesh:package://epsilon_crane_description/meshes/collision_mounting_base.stl@0.001,0.001,0.001",
    {-0.315000, 0.350000, 0.148500, 1.570796327, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K10_left_rail",
    LinkShape::Box,
    {0.684869, 0.586269, 0.600045},
    {-0.751178, 0.116359, -0.020920},
    {0.391727478, -0.190780996, -0.364216992, 0.823102774},
    "mesh:package://pzs100_description/meshes/rail_collision.stl@0.001,0.001,0.001",
    {0.000000, 0.000000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K12_right_rail",
    LinkShape::Box,
    {0.684869, 0.586269, 0.600045},
    {-0.751178, 0.116359, -0.020920},
    {0.391727478, -0.190780996, -0.364216992, 0.823102774},
    "mesh:package://pzs100_description/meshes/rail_collision.stl@0.001,0.001,0.001",
    {0.000000, 0.000000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K1_slewing_column",
    LinkShape::Box,
    {0.999934, 0.136000, 0.135579},
    {0.068339, -0.914728, -0.000000},
    {0.527243253, -0.471184202, -0.471184202, 0.527243253},
    "mesh:package://epsilon_crane_description/meshes/collision_slewing_column.stl@"
    "0.001,0.001,0.001",
    {0.180000, -2.365200, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K2_boom",
    LinkShape::Box,
    {1.322342, 0.192616, 0.204940},
    {-2.014769, 0.065974, 0.000094},
    {0.616467561, -0.032669239, -0.041567978, 0.785603316},
    "mesh:package://epsilon_crane_description/meshes/collision_boom.stl@0.001,0.001,0.001",
    {-3.492883, 0.000000, 0.000000, 0.000000000, 0.000000000, -0.101235000}},
  {Tool::Pzs100, "K3_arm",
    LinkShape::Box,
    {1.519035, 0.162261, 0.133446},
    {0.319070, 0.000940, 1.443636},
    {0.516646139, 0.486030229, 0.520980542, -0.474795385},
    "mesh:package://epsilon_crane_description/meshes/collision_arm.stl@0.001,0.001,0.001",
    {0.392500, 0.000000, 0.000000, -1.570796327, -1.570796327, 0.000000000}},
  {Tool::Pzs100, "K4_outer_telescope",
    LinkShape::Box,
    {1.510013, 0.093703, 0.078000},
    {0.290500, 0.000000, -1.581200},
    {0.499966374, 0.500033623, 0.500033623, -0.499966374},
    "mesh:package://epsilon_crane_description/meshes/collision_outer_telescope.stl@"
    "0.001,0.001,0.001",
    {0.290500, 0.000000, -0.071200, -1.570796327, -1.570796327, 0.000000000}},
  {Tool::Pzs100, "K5_inner_telescope",
    LinkShape::Box,
    {1.504409, 0.137035, 0.066000},
    {0.264529, 1.447720, 0.000000},
    {-0.695082668, 0.718929819, 0.000000000, 0.000000000},
    "mesh:package://epsilon_crane_description/meshes/collision_inner_telescope.stl@"
    "0.001,0.001,0.001",
    {0.281000, 0.000000, 0.000000, 0.000000000, 0.000000000, -1.570796327}},
  {Tool::Pzs100, "K6_double_joint_link",
    LinkShape::Box,
    {0.185387, 0.103836, 0.069430},
    {0.118307, -0.003869, -0.000305},
    {0.010059754, 0.003772131, 0.999771462, -0.018482327},
    "mesh:package://epsilon_crane_description/meshes/collision_double_joint_link.stl@"
    "0.001,0.001,0.001",
    {0.000000, 0.000000, 0.000000, 1.570796327, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K7_rotator_upper_part",
    LinkShape::Capsule,
    {0.163801, 0.044935, 0.000000},
    {-0.000424, -0.007699, 0.118874},
    {0.706073133, 0.706272194, 0.034345133, -0.038219505},
    "mesh:package://epsilon_crane_description/meshes/collision_rotator_upper_part_clamp.stl@"
    "0.001,0.00109,0.001",
    {0.000000, 0.000000, 0.000000, -1.570796327, 0.000000000, -1.570796327}},
  {Tool::Pzs100, "K8_rotator_lower_part",
    LinkShape::Box,
    {0.189872, 0.106714, 0.071277},
    {0.001910, 0.000083, -0.363300},
    {0.713453863, 0.700605936, -0.006891080, -0.009349894},
    "mesh:package://epsilon_crane_description/meshes/collision_rotator_lower_part.stl@"
    "0.001,0.001,0.001",
    {0.000000, 0.000000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "K8_tool_center_point",
    LinkShape::Box,
    {0.469156, 0.370046, 0.210251},
    {0.000521, 0.026838, -0.207673},
    {0.717403524, 0.696646249, -0.002235602, -0.003345096},
    "mesh:package://pzs100_description/meshes/frame_collision.stl@0.001,0.001,0.001",
    {0.000000, 0.000000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "control_box",
    LinkShape::Box,
    {0.550000, 0.400000, 0.225000},
    {-0.000000, -0.040000, 0.000000},
    {0.500000000, 0.500000000, 0.500000000, -0.500000000},
    "box:0.8,0.45,1.1",
    {0.000000, -0.040000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "control_box_sensor_mount",
    LinkShape::Box,
    {0.550000, 0.400000, 0.225000},
    {-0.000000, -0.040000, 0.000000},
    {0.500000000, 0.500000000, 0.500000000, -0.500000000},
    "box:0.8,0.45,1.1",
    {0.000000, -0.040000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
  {Tool::Pzs100, "hydraulic_block",
    LinkShape::Box,
    {0.340000, 0.310000, 0.200000},
    {-0.000000, 0.000000, 0.000000},
    {0.500000000, 0.500000000, 0.500000000, 0.500000000},
    "box:0.4,0.68,0.62",
    {0.000000, 0.000000, 0.000000, 0.000000000, 0.000000000, 0.000000000}},
}};

// A pair of links self-collision does not check. The reason on each line is the
// derivation's, in the categories the MoveIt SRDF format uses; the same pairs,
// with the same reasons, are checked in as `config/allowed_collisions.srdf`.
struct LinkPair
{
  const char * first;
  const char * second;
};

inline constexpr std::size_t kAllowedSelfPairCount = 56;

inline constexpr std::array<LinkPair, kAllowedSelfPairCount> kAllowedSelfPairs{{
  {"K0_mounting_base", "K1_slewing_column"},  // Adjacent
  {"K0_mounting_base", "control_box"},  // Never
  {"K0_mounting_base", "control_box_sensor_mount"},  // Never
  {"K0_mounting_base", "hydraulic_block"},  // Never
  {"K10_left_rail", "K12_right_rail"},  // Always
  {"K10_left_rail", "K6_double_joint_link"},  // Default
  {"K10_left_rail", "K7_rotator_upper_part"},  // Default
  {"K10_left_rail", "K8_rotator_lower_part"},  // Always
  {"K10_left_rail", "K8_tool_center_point"},  // Adjacent
  {"K12_right_rail", "K6_double_joint_link"},  // Default
  {"K12_right_rail", "K7_rotator_upper_part"},  // Default
  {"K12_right_rail", "K8_rotator_lower_part"},  // Always
  {"K12_right_rail", "K8_tool_center_point"},  // Adjacent
  {"K1_slewing_column", "K2_boom"},  // Adjacent
  {"K1_slewing_column", "K4_outer_telescope"},  // Never
  {"K1_slewing_column", "control_box"},  // Adjacent
  {"K1_slewing_column", "control_box_sensor_mount"},  // Always
  {"K1_slewing_column", "hydraulic_block"},  // Adjacent
  {"K2_boom", "K3_arm"},  // Adjacent
  {"K2_boom", "K6_double_joint_link"},  // Never
  {"K2_boom", "K7_rotator_upper_part"},  // Never
  {"K2_boom", "K8_rotator_lower_part"},  // Never
  {"K2_boom", "control_box"},  // Never
  {"K2_boom", "control_box_sensor_mount"},  // Default
  {"K2_boom", "hydraulic_block"},  // Never
  {"K3_arm", "K4_outer_telescope"},  // Adjacent
  {"K3_arm", "K5_inner_telescope"},  // Default
  {"K3_arm", "K6_double_joint_link"},  // Never
  {"K3_arm", "control_box"},  // Never
  {"K3_arm", "control_box_sensor_mount"},  // Never
  {"K3_arm", "hydraulic_block"},  // Never
  {"K4_outer_telescope", "K5_inner_telescope"},  // Adjacent
  {"K4_outer_telescope", "K6_double_joint_link"},  // Never
  {"K4_outer_telescope", "K7_rotator_upper_part"},  // Never
  {"K4_outer_telescope", "control_box"},  // Never
  {"K4_outer_telescope", "control_box_sensor_mount"},  // Never
  {"K5_inner_telescope", "K6_double_joint_link"},  // Adjacent
  {"K5_inner_telescope", "control_box"},  // Never
  {"K5_inner_telescope", "control_box_sensor_mount"},  // Never
  {"K6_double_joint_link", "K7_rotator_upper_part"},  // Adjacent
  {"K6_double_joint_link", "K8_rotator_lower_part"},  // Never
  {"K6_double_joint_link", "K8_tool_center_point"},  // Never
  {"K6_double_joint_link", "control_box"},  // Never
  {"K6_double_joint_link", "control_box_sensor_mount"},  // Never
  {"K6_double_joint_link", "hydraulic_block"},  // Never
  {"K7_rotator_upper_part", "K8_rotator_lower_part"},  // Adjacent
  {"K7_rotator_upper_part", "K8_tool_center_point"},  // Never
  {"K7_rotator_upper_part", "control_box"},  // Never
  {"K7_rotator_upper_part", "control_box_sensor_mount"},  // Never
  {"K7_rotator_upper_part", "hydraulic_block"},  // Never
  {"K8_rotator_lower_part", "K8_tool_center_point"},  // Adjacent
  {"K8_rotator_lower_part", "control_box"},  // Never
  {"K8_rotator_lower_part", "control_box_sensor_mount"},  // Never
  {"control_box", "control_box_sensor_mount"},  // Adjacent
  {"control_box", "hydraulic_block"},  // Never
  {"control_box_sensor_mount", "hydraulic_block"},  // Never
}};

}  // namespace collision_model
}  // namespace crane_model

#endif  // COLLISION_MODEL_HPP_
