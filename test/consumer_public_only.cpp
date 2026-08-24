// Contract §9: `casadi::MX`, `casadi::SX` and generated-function types must not
// appear in the public header. This fixture links `crane_model` and nothing
// else, includes `crane_model/model.hpp` and nothing else, and asserts the
// restriction at compile time rather than trusting that nobody added an
// include.
//
// It does *not* claim that no CasADi header is reachable from here in the
// absolute: CasADi is installed under `/usr/local/include`, which is a default
// system include directory, so no target in this image can be denied it. What
// the build can and does deny is `crane_model/symbolic/casadi_graph.hpp` --
// that header has its own include root, carried only by the
// `crane_model::casadi_graph` target -- and what this file denies is CasADi
// arriving *through the public header*, which is what §9 restricts.

#include "crane_model/model.hpp"

#ifdef CASADI_CASADI_HPP
#error "crane_model/model.hpp pulled in <casadi/casadi.hpp> (contract §9)"
#endif

#ifdef PINOCCHIO_WITH_CASADI_SUPPORT
#error "crane_model/model.hpp pulled in <pinocchio/autodiff/casadi.hpp> (contract §9)"
#endif

// The compiler evaluates this; `ament_cppcheck`'s preprocessor does not know
// `__has_include` and reports the directive itself as an error, so it is
// suppressed here rather than the assertion being weakened.
// cppcheck-suppress preprocessorErrorDirective
#if __has_include(<crane_model/symbolic/casadi_graph.hpp>)
#error "the CasADi graph header is on a public-only consumer's include path"
#endif

// `casadi::SX` is a typedef for `casadi::Matrix<SXElem>`, so if CasADi had
// arrived by any route at all this redeclaration would be a redeclaration as a
// different kind of symbol and would not compile. The guards above catch the
// headers by name; this catches the type however it got here.
namespace casadi
{
class SX;
}  // namespace casadi

int main()
{
  // An empty description is not a crane, so this must fail -- and it must fail
  // through the public status enum, with no backend type in sight.
  const auto model = crane_model::Model::create(crane_model::ModelConfig{});
  return model.status().code == crane_model::ErrorCode::InvalidRobotDescription ? 0 : 1;
}
