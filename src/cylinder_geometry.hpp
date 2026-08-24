#ifndef CYLINDER_GEOMETRY_HPP_
#define CYLINDER_GEOMETRY_HPP_

#include <array>
#include <cstddef>

#include "crane_model/model.hpp"

// The cylinder transmission of `wiki/hydraulics.md` §2--§4, written once and
// evaluated twice: with `double` by `Model::cylinder_jacobian` and
// `Model::transmission`, and with `casadi::SX` by the symbolic graph of
// contract §9. That is the whole reason this is a header and a template rather
// than three functions in `model.cpp` -- the transmission is in the MPC's
// constraints 6 and 7 (`wiki/mpc.md` §3) and in the planner's §5.1, and a
// second transcription of the four-bars is exactly the failure the symbolic
// graph exists to rule out.
//
// Nothing here allocates and nothing branches on the value of a coordinate, so
// the same expression is valid for every scalar type. A configuration at which
// a linkage does not close produces a non-finite ratio rather than a flag: the
// `double` caller tests `std::isfinite` and reports `SingularConfiguration`,
// and the symbolic caller has no configuration to test at build time.
//
// This header is private to the implementation and to the contract test. It is
// not installed and nothing on the public API mentions it (contract §6).

namespace crane_model
{
namespace cylinder
{

// The six actuated axes in the canonical projection [0, 1, 2, 3, 6, 7] of the
// model API contract §2, i.e. the hardware axis codes SW, HA, KA, SA, RO, GR.
enum AxisIndex : std::size_t
{
  kSlewingAxis = 0,
  kBoomAxis = 1,
  kArmAxis = 2,
  kTelescopeAxis = 3,
  kRotatorAxis = 4,
  kToolAxis = 5,
};

// Machine constants, wiki/hydraulics.md §6. They are verified there against
// parameter_def.cpp and the URDF; symbols follow wiki/nomenclature.md §7.
constexpr double kGearRadius = 0.1;               // r_gear, m
constexpr double kMotorDisplacement = 1.4324e-4;  // V_m, m^3/rad

constexpr double kSlewingPistonArea = 6.362e-3;      // A_A, q1
constexpr double kBoomPistonArea = 1.5394e-2;        // A_A, q2
constexpr double kBoomAnnulusArea = 9.032e-3;        // A_B, q2
constexpr double kArmPistonArea = 6.362e-3;          // A_A, q3
constexpr double kArmAnnulusArea = 2.5133e-3;        // A_B, q3
constexpr double kTelescopePistonArea = 3.8485e-3;   // A_A, q4
constexpr double kTelescopeAnnulusArea = 2.592e-3;   // A_B, q4
constexpr double kToolPistonArea = 7.854e-3;         // A_A, q8
constexpr double kToolAnnulusArea = 4.737e-3;        // A_B, q8

// Boom four-bar, wiki/hydraulics.md §2.2 with the values of §6.2. Every length
// here except the two bar lengths is also a placement in the real description,
// and CraneModelDescription.LinkageConstantsAgreeWithTheDescription asserts
// that the two agree; the split of a_2 from p_S2x exists so each side of that
// sum can be checked against the frame that carries it.
constexpr double kBoomFootX = 0.433;             // p_S0
constexpr double kBoomFootY = -1.7682;
constexpr double kBoomPivotX = -0.12;            // p_S1
constexpr double kBoomPivotY = -0.07;
constexpr double kBoomJointToLinkX = 3.49288;    // a_2, theta2 joint to K2_boom
constexpr double kBoomAttachmentX = -3.039;      // p_S2x, in K2_boom
constexpr double kBoomLinkX = kBoomJointToLinkX + kBoomAttachmentX;
constexpr double kBoomLinkY = -0.036034;         // p_S2y
// r_13 and r_23. The description closes this loop only in Gazebo, so neither
// bar length is a placement Pinocchio can read back out of it.
constexpr double kDrawbarLength = 0.57;          // r_13, Zugstange
constexpr double kPushbarLength = 0.124;         // r_23, Druckstange

// Arm cylinder, wiki/hydraulics.md §2.3 with the values of §6.2. The lateral
// term is p_S4y - p_S3z and not p_S4z - p_S3z because the two attachment
// points live in links with different CAD-to-model rotations; see the callout
// in §2.3 before "normalising" it.
constexpr double kArmFootX = -1.6802;              // p_S3x
constexpr double kArmFootY = -0.0485;              // p_S3y
constexpr double kArmFootZ = 0.224;                // p_S3z
constexpr double kArmJointToLinkX = -0.3925;       // a_3, theta3 joint to K3_arm
constexpr double kArmAttachmentX = 0.274489;       // p_S4x
constexpr double kArmAttachmentY = 0.224;          // p_S4y
constexpr double kArmLinkX = kArmJointToLinkX + kArmAttachmentX;
constexpr double kArmLinkY = -0.468;               // -p_S4z
constexpr double kArmLateralOffset = kArmAttachmentY - kArmFootZ;

// 7040 jaw four-bar, wiki/hydraulics.md §2.6. §6.2 carries none of its numbers,
// so every constant below is ported from the deployed model under src/ and
// names the file it came from. None of them is a §6.2 value.
//
// The fitted mirror law phi_sim,9(q8), which §2.6 says is the variant the
// deployed model uses, in the Horner form of
// timber_crane_model_cpp/include/timber_crane_model_cpp/maple/
// comp_eq_four_bar.hpp:7, with jaw_linkage_p of
// crane_tools_description/7040/config/gripper_parameter.yaml:13.
constexpr double kJawMirrorQuadratic = -0.121220;  // p_0
constexpr double kJawMirrorLinear = 1.400122;      // p_1
constexpr double kJawMirrorConstant = -0.227867;   // p_2

// The two jaw pivots on the pincer frame and the two jaw arm lengths. The
// deployed model reads them from the URDF DH transforms in
// timber_crane_parameter/src/parameter_def.cpp:196-206; the joint origins are
// in crane_tools_description/7040/urdf/links/. Note a_9 and a_11 are the y and
// a_10 and a_12 the x coordinate of their origin, per the comment there.
constexpr double kOuterJawPivotOffset = 0.328;  // a_9, -dh_trans9 y
constexpr double kOuterJawArmLength = 0.8126;   // a_10, dh_trans10 x
constexpr double kInnerJawPivotOffset = 0.336;  // a_11, dh_trans11 y
constexpr double kInnerJawArmLength = 0.8172;   // a_12, dh_trans12 x

// The jaw cylinder's two attachment points. The barrel end sits on the outer
// jaw, joint pincer_cylinder_mounting_outer_jaw_joint of crane_tools_description
// /7040/urdf/joints/hydraulic/gripper_cylinder_joints.urdf.xacro:15. The rod end
// sits on the inner jaw and is hard-coded in gripper_parameter.yaml:5-7 because
// the URDF closes that loop only in Gazebo.
constexpr double kJawCylinderOuterX = -0.8971;    // p_S7x
constexpr double kJawCylinderOuterY = 0.01754;    // p_S7y
constexpr double kJawCylinderInnerX = -0.908397;  // p_S8x
constexpr double kJawCylinderInnerY = 0.015289;   // p_S8y

// Effective areas per axis. a_a and a_b are the force-producing areas of §4,
// a_eff_pos and a_eff_neg the direction-dependent pump draw of §3. The rotator
// carries V_m in all four, in m^3/rad.
struct AxisAreas
{
  double a_a{};
  double a_b{};
  double a_eff_pos{};
  double a_eff_neg{};
};

inline std::array<AxisAreas, kActuatedDof> axis_areas()
{
  std::array<AxisAreas, kActuatedDof> areas{};
  // q1 slewing: two cylinders, symmetric circuit, so both chambers see 2 A_A.
  areas[kSlewingAxis] = {
    2.0 * kSlewingPistonArea, 2.0 * kSlewingPistonArea,
    2.0 * kSlewingPistonArea, 2.0 * kSlewingPistonArea};
  // q2 boom: a single differential cylinder.
  areas[kBoomAxis] = {
    kBoomPistonArea, kBoomAnnulusArea, kBoomPistonArea, kBoomAnnulusArea};
  // q3 arm: two differential cylinders, one expression (§2.3).
  areas[kArmAxis] = {
    2.0 * kArmPistonArea, 2.0 * kArmAnnulusArea,
    2.0 * kArmPistonArea, 2.0 * kArmAnnulusArea};
  // q4 telescope: regenerative on extension. Rod-side oil is fed back to the
  // piston side, so only the annulus difference is drawn from the pump. The
  // force still follows the physical areas, which is why a_a and a_eff_pos
  // differ on this axis alone.
  areas[kTelescopeAxis] = {
    kTelescopePistonArea, kTelescopeAnnulusArea,
    kTelescopePistonArea - kTelescopeAnnulusArea, kTelescopeAnnulusArea};
  // q7 rotator: a motor, where V_m takes the role the piston area plays
  // elsewhere (§2.5). Symmetric in both directions.
  areas[kRotatorAxis] = {
    kMotorDisplacement, kMotorDisplacement, kMotorDisplacement, kMotorDisplacement};
  // q8 tool: a cylinder axis on the same pump, areas per §6.1.
  areas[kToolAxis] = {
    kToolPistonArea, kToolAnnulusArea, kToolPistonArea, kToolAnnulusArea};
  return areas;
}

// --- planar helpers ----------------------------------------------------------
//
// Every linkage below is planar, so the arithmetic is two components and a
// handful of products. They are written out rather than taken from Eigen
// because `Eigen::Matrix<casadi::SX, 2, 1>` cannot be scaled by a `double`
// without a ScalarBinaryOpTraits specialisation, and the mixed expressions
// that would need are exactly where a transcription error hides. The operand
// order is the one Eigen uses, so the `double` instantiation is unchanged.

template<typename Scalar>
struct Planar
{
  Scalar x{};
  Scalar y{};
};

template<typename Scalar>
Planar<Scalar> rotate(const Scalar& angle, const Planar<Scalar>& value)
{
  using std::cos;
  using std::sin;
  const Scalar cosine = cos(angle);
  const Scalar sine = sin(angle);
  return Planar<Scalar>{cosine * value.x - sine * value.y, sine * value.x + cosine * value.y};
}

// S_perp v, the planar 90 degree rotation of wiki/nomenclature.md §7.
template<typename Scalar>
Planar<Scalar> perpendicular(const Planar<Scalar>& value)
{
  return Planar<Scalar>{-value.y, value.x};
}

// diag(-1, 1) v, the reflection the 7040's outer jaw arm carries (§2.6).
template<typename Scalar>
Planar<Scalar> mirrored(const Planar<Scalar>& value)
{
  return Planar<Scalar>{-value.x, value.y};
}

template<typename Scalar>
Planar<Scalar> operator-(const Planar<Scalar>& left, const Planar<Scalar>& right)
{
  return Planar<Scalar>{left.x - right.x, left.y - right.y};
}

template<typename Scalar>
Planar<Scalar> operator+(const Planar<Scalar>& left, const Planar<Scalar>& right)
{
  return Planar<Scalar>{left.x + right.x, left.y + right.y};
}

template<typename Scalar>
Planar<Scalar> operator*(const Scalar& factor, const Planar<Scalar>& value)
{
  return Planar<Scalar>{factor * value.x, factor * value.y};
}

template<typename Scalar>
Scalar dot(const Planar<Scalar>& left, const Planar<Scalar>& right)
{
  return left.x * right.x + left.y * right.y;
}

template<typename Scalar>
Scalar squared_norm(const Planar<Scalar>& value)
{
  return dot(value, value);
}

template<typename Scalar>
Scalar norm(const Planar<Scalar>& value)
{
  using std::sqrt;
  return sqrt(squared_norm(value));
}

// Piston displacement s_i of one axis together with its transmission ratio
// ds_i/dq_i, which is the diagonal entry of J_cyl. The frozen API exposes only
// the ratio today; s_i is carried because the stroke limits of §6.3 are an
// independent constraint on the two nonlinear axes and will need it.
template<typename Scalar>
struct CylinderStroke
{
  Scalar stroke{};
  Scalar ratio{};
};

// wiki/hydraulics.md §2.2. The cylinder drives the coupler point p_J of a
// four-bar, not the boom directly, so s_2 is the distance from the cylinder
// foot p_S0 to that coupler point. The derivative is the analytic chain rule
// through p_J(d^2(q2)); it is exact, not a difference quotient.
//
// A q2 at which the triangle inequality on d, r_13 and r_23 fails leaves
// `delta` negative and its square root, and therefore the ratio, not finite:
// the linkage cannot close there and there is no piston displacement to report.
template<typename Scalar>
CylinderStroke<Scalar> boom_stroke(const Scalar& q2)
{
  using std::sqrt;
  const Planar<Scalar> p_s0{Scalar(kBoomFootX), Scalar(kBoomFootY)};
  const Planar<Scalar> p_s1{Scalar(kBoomPivotX), Scalar(kBoomPivotY)};

  const Planar<Scalar> p_s2 =
    rotate(q2, Planar<Scalar>{Scalar(kBoomLinkX), Scalar(kBoomLinkY)});
  const Planar<Scalar> d_pivot = p_s2 - p_s1;
  const Planar<Scalar> d_pivot_rate = perpendicular(p_s2);  // d(p_S2)/dq2

  const Scalar d_squared = squared_norm(d_pivot);
  const Scalar d_squared_rate = 2.0 * dot(d_pivot, d_pivot_rate);

  const double reach_sum = kDrawbarLength + kPushbarLength;
  const double reach_difference = kDrawbarLength - kPushbarLength;
  const Scalar delta = (reach_sum * reach_sum - d_squared) *
    (d_squared - reach_difference * reach_difference);

  const double link_difference =
    kDrawbarLength * kDrawbarLength - kPushbarLength * kPushbarLength;
  const Scalar alpha = (link_difference + d_squared) / (2.0 * d_squared);
  const Scalar alpha_rate = -link_difference / (2.0 * d_squared * d_squared) * d_squared_rate;

  const Scalar root = sqrt(delta);
  const Scalar delta_gradient =
    reach_sum * reach_sum + reach_difference * reach_difference - 2.0 * d_squared;
  // The branch is fixed by the derivation: the intersection whose local y is
  // negative in the frame with p_S1 at the origin and p_S2 on +x (§2.2). Since
  // S_perp d points along local +y, that is the negative root, and it is the
  // branch whose stroke spans the s_2 limits of §6.3.
  const Scalar beta = -root / (2.0 * d_squared);
  const Scalar beta_rate =
    -(delta_gradient * d_squared / root - 2.0 * root) /
    (4.0 * d_squared * d_squared) * d_squared_rate;

  const Planar<Scalar> p_j = p_s1 + alpha * d_pivot + beta * perpendicular(d_pivot);
  const Planar<Scalar> p_j_rate = alpha_rate * d_pivot + alpha * d_pivot_rate +
    beta_rate * perpendicular(d_pivot) + beta * perpendicular(d_pivot_rate);

  const Planar<Scalar> c_cyl = p_j - p_s0;
  const Scalar stroke = norm(c_cyl);
  return CylinderStroke<Scalar>{stroke, dot(c_cyl, p_j_rate) / stroke};
}

// wiki/hydraulics.md §2.3. A direct cylinder: the in-plane part rotates with
// q3 while the out-of-plane offset stays constant. The lateral offset keeps the
// stroke away from zero, so this axis has no degenerate configuration.
template<typename Scalar>
CylinderStroke<Scalar> arm_stroke(const Scalar& q3)
{
  using std::sqrt;
  const Planar<Scalar> moving =
    rotate(q3, Planar<Scalar>{Scalar(kArmLinkX), Scalar(kArmLinkY)});
  const Planar<Scalar> in_plane =
    moving - Planar<Scalar>{Scalar(kArmFootX), Scalar(kArmFootY)};
  const Planar<Scalar> in_plane_rate = perpendicular(moving);

  const Scalar stroke =
    sqrt(squared_norm(in_plane) + kArmLateralOffset * kArmLateralOffset);
  return CylinderStroke<Scalar>{stroke, dot(in_plane, in_plane_rate) / stroke};
}

// wiki/hydraulics.md §2.6 for the 7040, composed the way the deployed model
// composes it. The commanded outer-jaw angle q8 drives the inner jaw through
// the quadratic fit phi_sim,9(q8), and the cylinder spans the two jaws, so its
// length depends on both angles. Frames 19 and 20 of the deployed
// comp_transform_8_19.hpp and comp_transform_8_20.hpp place the two pins: the
// outer jaw hangs off -a_9 with its arm mirrored, the inner jaw off +a_11.
// Planar, exactly as the deployed comp_transform_8_24.hpp is -- the two pins
// are p_S8z + p_S7z = 24.25 mm apart out of plane, which the deployed model
// drops and which is worth at most 0.6 mm of length over the jaw range.
template<typename Scalar>
CylinderStroke<Scalar> jaw_stroke(const Scalar& q8)
{
  const Scalar mirror_angle =
    (kJawMirrorQuadratic * q8 + kJawMirrorLinear) * q8 + kJawMirrorConstant;
  const Scalar mirror_rate = 2.0 * kJawMirrorQuadratic * q8 + kJawMirrorLinear;

  const Planar<Scalar> outer_arm = rotate(
    q8,
    Planar<Scalar>{
      Scalar(kOuterJawArmLength + kJawCylinderOuterX), Scalar(kJawCylinderOuterY)});
  const Planar<Scalar> inner_arm = rotate(
    mirror_angle,
    Planar<Scalar>{
      Scalar(kInnerJawArmLength + kJawCylinderInnerX), Scalar(kJawCylinderInnerY)});

  // The reflection applies to the outer arm and to its rate alike, and it is
  // applied *after* S_perp in both -- diag(-1, 1) and S_perp do not commute.
  const Planar<Scalar> c_cyl = inner_arm - mirrored(outer_arm) +
    Planar<Scalar>{Scalar(kInnerJawPivotOffset + kOuterJawPivotOffset), Scalar(0.0)};
  const Planar<Scalar> c_cyl_rate =
    mirror_rate * perpendicular(inner_arm) - mirrored(perpendicular(outer_arm));

  const Scalar stroke = norm(c_cyl);
  return CylinderStroke<Scalar>{stroke, dot(c_cyl, c_cyl_rate) / stroke};
}

// The diagonal of J_cyl (wiki/nomenclature.md §7), which is the whole of it:
// the geometry does not couple the axes at all (§5.7), so the off-diagonal
// entries are structurally zero and are never formed.
//
// Only q2, q3 and -- for the 7040 -- q8 enter. The other three axes are a
// constant radius or one-to-one, so they carry no configuration dependence for
// either scalar type.
template<typename Scalar>
std::array<Scalar, kActuatedDof> jacobian_diagonal(
  Tool tool, const Scalar& q2, const Scalar& q3, const Scalar& q8)
{
  std::array<Scalar, kActuatedDof> diagonal{};
  // q1 slewing: rack and pinion, the only constant transmission (§2.1).
  diagonal[kSlewingAxis] = Scalar(kGearRadius);
  diagonal[kBoomAxis] = boom_stroke(q2).ratio;
  diagonal[kArmAxis] = arm_stroke(q3).ratio;
  // q4 telescope: the cylinder is one-to-one with the joint coordinate; the
  // factor of two sits between q4 and the tip travel, not here (§2.4).
  diagonal[kTelescopeAxis] = Scalar(1.0);
  // q7 rotator: a motor, so the motor angle is the joint coordinate (§2.5).
  diagonal[kRotatorAxis] = Scalar(1.0);
  // q8 tool (§2.6). The PZS100 rail cylinder is one-to-one with q8 and there is
  // no linkage to solve; the 7040 jaw is a four-bar and its cylinder spans both
  // jaws, so its ratio is configuration-dependent like the boom's.
  diagonal[kToolAxis] =
    tool == Tool::Pzs100 ? Scalar(1.0) : jaw_stroke(q8).ratio;
  return diagonal;
}

}  // namespace cylinder
}  // namespace crane_model

#endif  // CYLINDER_GEOMETRY_HPP_
