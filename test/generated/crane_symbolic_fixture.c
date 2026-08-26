// The one hand-written file in `test/generated/`.
//
// `crane_symbolic_fixture.inc` is CasADi's own output for the Python model of
// `scripts/crane_symbolic.py`, and `crane_symbolic_fixture_api.inc` is its
// declarations. Neither ends in `.c` or `.h` on purpose: every C and C++ file in
// this workspace goes through `ament_cpplint` and `ament_cppcheck`, and machine
// output fails them by thousands -- while reformatting it to pass would make
// `export_model_fixture.py --check` compare against whatever the formatter last
// did rather than against the model. So the generated text is *included*, and
// this three-line translation unit is what the build compiles.
//
// Do not edit either `.inc`. Regenerate them:
//
//     ./scripts/export_model_fixture.py

#include "crane_symbolic_fixture.inc"  // NOLINT(build/include)
