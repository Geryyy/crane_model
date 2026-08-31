#ifndef CYLINDER_GEOMETRY_HPP_
#define CYLINDER_GEOMETRY_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "crane_model/model.hpp"

// The numeric cylinder transmission of `wiki/hydraulics.md` §2--§4.
//
// The numbers it is evaluated with are not here. They are
// `config/hydraulics.yaml`, read once into the `hydraulics::Constants` below
// and threaded through every function as an argument. The Python code generator
// reads the same file independently.
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
namespace hydraulics
{

// `config/hydraulics.yaml`, as read. Everything in it is a number the robot
// description does not carry and a Python export of this model would otherwise
// have to be told a second time, which is the whole reason it is a file: a
// constant compiled in here and typed in there is drift nothing would catch.
// The symbols are the `Code` column of wiki/nomenclature.md §7 and §10; the
// file names, per entry, the wiki/hydraulics.md section each number came from.
//
// It is declared in this header rather than in one of its own because this is
// the header both sides of the model already include: `model.cpp` evaluates the
// transmission below with the numeric model; the Python generator reads the
// same YAML constants independently.

// One joint whose damping the description states and the model does not use.
// Carried as data with its reason rather than as a branch in `parse`, so the
// model applies the table and knows nothing about which joint is on it.
struct DampingOverride
{
  std::string joint;
  double d{};          // the entry of D, wiki/nomenclature.md §6
  std::string reason;  // why the description's own entry is not used
};

struct Constants
{
  double r_gear{};  // §6.1, slewing rack-and-pinion radius, m
  double v_m{};     // §6.1, rotator motor volumetric displacement, m^3/rad

  // §6.1. The force-producing areas of §4, per hydraulic axis. Slewing carries
  // no A_B because §6.1 states none: its circuit is symmetric.
  double slewing_a_a{};
  double boom_a_a{};
  double boom_a_b{};
  double arm_a_a{};
  double arm_a_b{};
  double telescope_a_a{};
  double telescope_a_b{};
  double tool_a_a{};
  double tool_a_b{};

  // Boom four-bar, §2.2 with the values of §6.2. a_2 is split from p_S2x so
  // each side of that sum can be checked against the frame that carries it;
  // `LinkagePlacementsAgreeWithTheCompiledConstants` does exactly that.
  double boom_ps0_x{};
  double boom_ps0_y{};
  double boom_ps1_x{};
  double boom_ps1_y{};
  double boom_a2{};      // theta2 joint to K2_boom
  double boom_ps2_x{};   // in K2_boom
  double boom_ps2_y{};
  double r13{};          // Zugstange
  double r23{};          // Druckstange

  // Arm cylinder, §2.3 with the values of §6.2.
  double arm_ps3_x{};
  double arm_ps3_y{};
  double arm_ps3_z{};
  double arm_a3{};       // theta3 joint to K3_arm
  double arm_ps4_x{};
  double arm_ps4_y{};
  double arm_ps4_z{};

  // 7040 jaw four-bar, §2.6, whose numbers §6.2 does not carry.
  double jaw_p0{};       // phi_sim,9(q8), quadratic coefficient
  double jaw_p1{};
  double jaw_p2{};
  double jaw_a9{};       // outer jaw pivot on the pincer frame
  double jaw_a10{};      // outer jaw arm
  double jaw_a11{};      // inner jaw pivot
  double jaw_a12{};      // inner jaw arm
  double jaw_ps7_x{};    // barrel end, on the outer jaw
  double jaw_ps7_y{};
  double jaw_ps8_x{};    // rod end, on the inner jaw
  double jaw_ps8_y{};

  // The smoothing wiki/mpc.md §3.1 requires of constraint 7, applied to the
  // *model* and never to the inner loop's dead-zone compensator.
  double eps_abs{};      // |v| ~= sqrt(v^2 + eps^2), m/s
  double eps_v{};        // the A^{+-} switch scale, m/s

  // The canonical eight of the model API contract §2, per tool. Only the
  // eighth entry differs between the two descriptions.
  std::array<std::string, kGeneralizedDof> pzs100_joints{};
  std::array<std::string, kGeneralizedDof> epsilon7040_joints{};

  std::vector<DampingOverride> damping_overrides;

  // The sums the linkages actually rotate. Written here rather than in the
  // file because they are not measurements: each is a statement about which
  // frame the other two numbers are expressed in.
  [[nodiscard]] double boom_link_x() const { return boom_a2 + boom_ps2_x; }
  [[nodiscard]] double arm_link_x() const { return arm_a3 + arm_ps4_x; }
  [[nodiscard]] double arm_link_y() const { return -arm_ps4_z; }
  [[nodiscard]] double arm_lateral_offset() const { return arm_ps4_y - arm_ps3_z; }

  [[nodiscard]] const std::array<std::string, kGeneralizedDof>& joints(Tool tool) const
  {
    return tool == Tool::Pzs100 ? pzs100_joints : epsilon7040_joints;
  }
};

// Where the build put `config/hydraulics.yaml`: the installed copy if it is
// there, the source tree otherwise. `crane_model` is ROS-free (contract §6), so
// there is no ament index to ask and the two candidates are compiled in.
std::string default_path();

// Reads `path`, or `default_path()`. Defined in `model.cpp`, which owns the
// file front ends. A missing file, a missing key and a key that is not a number
// are all `InvalidArgument` naming the key -- never a default, because a wrong
// hydraulic constant is a wrong force limit.
Status load(const std::string& path, Constants& out);
Status load(Constants& out);

}  // namespace hydraulics

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

// What each axis does with the areas of §6.1 -- two cylinders, a differential
// one, a regenerative circuit, a motor -- is the circuit topology of §2 and §3
// and stays here. Only the numbers it composes come out of the file.
inline std::array<AxisAreas, kActuatedDof> axis_areas(const hydraulics::Constants& constants)
{
  std::array<AxisAreas, kActuatedDof> areas{};
  // q1 slewing: two cylinders, symmetric circuit, so both chambers see 2 A_A.
  areas[kSlewingAxis] = {
    2.0 * constants.slewing_a_a, 2.0 * constants.slewing_a_a,
    2.0 * constants.slewing_a_a, 2.0 * constants.slewing_a_a};
  // q2 boom: a single differential cylinder.
  areas[kBoomAxis] = {
    constants.boom_a_a, constants.boom_a_b, constants.boom_a_a, constants.boom_a_b};
  // q3 arm: two differential cylinders, one expression (§2.3).
  areas[kArmAxis] = {
    2.0 * constants.arm_a_a, 2.0 * constants.arm_a_b,
    2.0 * constants.arm_a_a, 2.0 * constants.arm_a_b};
  // q4 telescope: regenerative on extension. Rod-side oil is fed back to the
  // piston side, so only the annulus difference is drawn from the pump. The
  // force still follows the physical areas, which is why a_a and a_eff_pos
  // differ on this axis alone.
  areas[kTelescopeAxis] = {
    constants.telescope_a_a, constants.telescope_a_b,
    constants.telescope_a_a - constants.telescope_a_b, constants.telescope_a_b};
  // q7 rotator: a motor, where V_m takes the role the piston area plays
  // elsewhere (§2.5). Symmetric in both directions.
  areas[kRotatorAxis] = {constants.v_m, constants.v_m, constants.v_m, constants.v_m};
  // q8 tool: a cylinder axis on the same pump, areas per §6.1.
  areas[kToolAxis] = {
    constants.tool_a_a, constants.tool_a_b, constants.tool_a_a, constants.tool_a_b};
  return areas;
}

// --- planar helpers ----------------------------------------------------------
//
// Every linkage below is planar, so the arithmetic is two components and a
// handful of products. They are written out rather than taken from Eigen
// because the small mixed expressions are easier to audit than generic Eigen
// expressions here.

struct Planar
{
  double x{};
  double y{};
};

inline Planar rotate(double angle, const Planar& value)
{
  using std::cos;
  using std::sin;
  const double cosine = cos(angle);
  const double sine = sin(angle);
  return Planar{cosine * value.x - sine * value.y, sine * value.x + cosine * value.y};
}

// S_perp v, the planar 90 degree rotation of wiki/nomenclature.md §7.
inline Planar perpendicular(const Planar& value)
{
  return Planar{-value.y, value.x};
}

// diag(-1, 1) v, the reflection the 7040's outer jaw arm carries (§2.6).
inline Planar mirrored(const Planar& value)
{
  return Planar{-value.x, value.y};
}

inline Planar operator-(const Planar& left, const Planar& right)
{
  return Planar{left.x - right.x, left.y - right.y};
}

inline Planar operator+(const Planar& left, const Planar& right)
{
  return Planar{left.x + right.x, left.y + right.y};
}

inline Planar operator*(double factor, const Planar& value)
{
  return Planar{factor * value.x, factor * value.y};
}

inline double dot(const Planar& left, const Planar& right)
{
  return left.x * right.x + left.y * right.y;
}

inline double squared_norm(const Planar& value)
{
  return dot(value, value);
}

inline double norm(const Planar& value)
{
  using std::sqrt;
  return sqrt(squared_norm(value));
}

// Piston displacement s_i of one axis together with its transmission ratio
// ds_i/dq_i, which is the diagonal entry of J_cyl. The frozen API exposes only
// the ratio today; s_i is carried because the stroke limits of §6.3 are an
// independent constraint on the two nonlinear axes and will need it.
struct CylinderStroke
{
  double stroke{};
  double ratio{};
};

// wiki/hydraulics.md §2.2. The cylinder drives the coupler point p_J of a
// four-bar, not the boom directly, so s_2 is the distance from the cylinder
// foot p_S0 to that coupler point. The derivative is the analytic chain rule
// through p_J(d^2(q2)); it is exact, not a difference quotient.
//
// A q2 at which the triangle inequality on d, r_13 and r_23 fails leaves
// `delta` negative and its square root, and therefore the ratio, not finite:
// the linkage cannot close there and there is no piston displacement to report.
inline CylinderStroke boom_stroke(const hydraulics::Constants& constants, double q2)
{
  using std::sqrt;
  const Planar p_s0{double(constants.boom_ps0_x), double(constants.boom_ps0_y)};
  const Planar p_s1{double(constants.boom_ps1_x), double(constants.boom_ps1_y)};

  const Planar p_s2 =
    rotate(q2, Planar{double(constants.boom_link_x()), double(constants.boom_ps2_y)});
  const Planar d_pivot = p_s2 - p_s1;
  const Planar d_pivot_rate = perpendicular(p_s2);  // d(p_S2)/dq2

  const double d_squared = squared_norm(d_pivot);
  const double d_squared_rate = 2.0 * dot(d_pivot, d_pivot_rate);

  const double reach_sum = constants.r13 + constants.r23;
  const double reach_difference = constants.r13 - constants.r23;
  const double delta = (reach_sum * reach_sum - d_squared) *
    (d_squared - reach_difference * reach_difference);

  const double link_difference =
    constants.r13 * constants.r13 - constants.r23 * constants.r23;
  const double alpha = (link_difference + d_squared) / (2.0 * d_squared);
  const double alpha_rate = -link_difference / (2.0 * d_squared * d_squared) * d_squared_rate;

  const double root = sqrt(delta);
  const double delta_gradient =
    reach_sum * reach_sum + reach_difference * reach_difference - 2.0 * d_squared;
  // The branch is fixed by the derivation: the intersection whose local y is
  // negative in the frame with p_S1 at the origin and p_S2 on +x (§2.2). Since
  // S_perp d points along local +y, that is the negative root, and it is the
  // branch whose stroke spans the s_2 limits of §6.3.
  const double beta = -root / (2.0 * d_squared);
  const double beta_rate =
    -(delta_gradient * d_squared / root - 2.0 * root) /
    (4.0 * d_squared * d_squared) * d_squared_rate;

  const Planar p_j = p_s1 + alpha * d_pivot + beta * perpendicular(d_pivot);
  const Planar p_j_rate = alpha_rate * d_pivot + alpha * d_pivot_rate +
    beta_rate * perpendicular(d_pivot) + beta * perpendicular(d_pivot_rate);

  const Planar c_cyl = p_j - p_s0;
  const double stroke = norm(c_cyl);
  return CylinderStroke{stroke, dot(c_cyl, p_j_rate) / stroke};
}

// wiki/hydraulics.md §2.3. A direct cylinder: the in-plane part rotates with
// q3 while the out-of-plane offset stays constant. The lateral offset keeps the
// stroke away from zero, so this axis has no degenerate configuration.
inline CylinderStroke arm_stroke(const hydraulics::Constants& constants, double q3)
{
  using std::sqrt;
  const Planar moving =
    rotate(q3, Planar{double(constants.arm_link_x()), double(constants.arm_link_y())});
  const Planar in_plane =
    moving - Planar{double(constants.arm_ps3_x), double(constants.arm_ps3_y)};
  const Planar in_plane_rate = perpendicular(moving);

  const double lateral = constants.arm_lateral_offset();
  const double stroke = sqrt(squared_norm(in_plane) + lateral * lateral);
  return CylinderStroke{stroke, dot(in_plane, in_plane_rate) / stroke};
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
inline CylinderStroke jaw_stroke(const hydraulics::Constants& constants, double q8)
{
  const double mirror_angle =
    (constants.jaw_p0 * q8 + constants.jaw_p1) * q8 + constants.jaw_p2;
  const double mirror_rate = 2.0 * constants.jaw_p0 * q8 + constants.jaw_p1;

  const Planar outer_arm = rotate(
    q8,
    Planar{
      double(constants.jaw_a10 + constants.jaw_ps7_x), double(constants.jaw_ps7_y)});
  const Planar inner_arm = rotate(
    mirror_angle,
    Planar{
      double(constants.jaw_a12 + constants.jaw_ps8_x), double(constants.jaw_ps8_y)});

  // The reflection applies to the outer arm and to its rate alike, and it is
  // applied *after* S_perp in both -- diag(-1, 1) and S_perp do not commute.
  const Planar c_cyl = inner_arm - mirrored(outer_arm) +
    Planar{double(constants.jaw_a11 + constants.jaw_a9), double(0.0)};
  const Planar c_cyl_rate =
    mirror_rate * perpendicular(inner_arm) - mirrored(perpendicular(outer_arm));

  const double stroke = norm(c_cyl);
  return CylinderStroke{stroke, dot(c_cyl, c_cyl_rate) / stroke};
}

// The diagonal of J_cyl (wiki/nomenclature.md §7), which is the whole of it:
// the geometry does not couple the axes at all (§5.7), so the off-diagonal
// entries are structurally zero and are never formed.
//
// Only q2, q3 and -- for the 7040 -- q8 enter. The other three axes are a
// constant radius or one-to-one, so they carry no configuration dependence for
// either scalar type.
inline std::array<double, kActuatedDof> jacobian_diagonal(
  const hydraulics::Constants& constants, Tool tool,
  double q2, double q3, double q8)
{
  std::array<double, kActuatedDof> diagonal{};
  // q1 slewing: rack and pinion, the only constant transmission (§2.1).
  diagonal[kSlewingAxis] = double(constants.r_gear);
  diagonal[kBoomAxis] = boom_stroke(constants, q2).ratio;
  diagonal[kArmAxis] = arm_stroke(constants, q3).ratio;
  // q4 telescope: the cylinder is one-to-one with the joint coordinate; the
  // factor of two sits between q4 and the tip travel, not here (§2.4).
  diagonal[kTelescopeAxis] = double(1.0);
  // q7 rotator: a motor, so the motor angle is the joint coordinate (§2.5).
  diagonal[kRotatorAxis] = double(1.0);
  // q8 tool (§2.6). The PZS100 rail cylinder is one-to-one with q8 and there is
  // no linkage to solve; the 7040 jaw is a four-bar and its cylinder spans both
  // jaws, so its ratio is configuration-dependent like the boom's.
  diagonal[kToolAxis] =
    tool == Tool::Pzs100 ? double(1.0) : jaw_stroke(constants, q8).ratio;
  return diagonal;
}

}  // namespace cylinder
}  // namespace crane_model

#endif  // CYLINDER_GEOMETRY_HPP_
