cmake_minimum_required(VERSION 3.8)
project(crane_model)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
# Coal is used directly, not through Pinocchio: the Pinocchio in this image is
# built without its collision module (no `pinocchio/collision`, no
# PINOCCHIO_WITH_COLLISION), so `pinocchio::GeometryModel` carries a fake
# geometry. Pinocchio still places the links; Coal answers the queries.
find_package(coal REQUIRED)
# CasADi is used through `pinocchio::pinocchio_casadi`, which is Pinocchio's own
# algorithms instantiated for `casadi::SX`: contract §9 asks the symbolic graph
# to be the same equations as the numeric model, and running the same `crba` and
# `nonLinearEffects` over a symbolic scalar is the only way to mean that
# literally. `find_package(pinocchio)` already brings `casadi::casadi` in.
find_package(casadi REQUIRED)
# config/hydraulics.yaml is read in `Model::create`. yaml-cpp is the parser
# wiki/implementation/libraries.md lists for configuration outside the parameter
# system; it brings no ROS with it, so contract §6 still holds.
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} src/model.cpp src/symbolic_graph.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  # PRIVATE, so the CasADi-facing header is visible to the builder that defines
  # its one function and to nobody else: a consumer reaches it only through the
  # `crane_model::casadi_graph` target below.
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/symbolic/include)
# Pinocchio and Coal are PRIVATE: contract §6 keeps them private implementation
# details, so a consumer links the shared libraries but never sees the headers.
# PIC because a consumer of this static archive may itself be a shared object --
# pluginlib loads `crane_control`'s controller as one -- and Pinocchio's and
# Coal's own libraries are shared.
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME}
  PUBLIC Eigen3::Eigen
  PRIVATE pinocchio::pinocchio pinocchio::pinocchio_casadi coal::coal yaml-cpp)

# Where `hydraulics::default_path` looks for config/hydraulics.yaml. Two
# candidates rather than one because the library is ROS-free (contract §6) and
# has no ament index to ask, and `ModelConfig` is frozen so a caller cannot
# hand it a third: the installed copy is what a deployed build reads, and the
# source tree is what a build tree reads before anything has been installed.
target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CRANE_MODEL_HYDRAULICS_YAML_INSTALLED="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/hydraulics.yaml"
    CRANE_MODEL_HYDRAULICS_YAML_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/config/hydraulics.yaml")

# The CasADi-facing half of contract §9, in a target of its own. It carries the
# header and the CasADi usage requirements and nothing else: the builder itself
# lives in `crane_model`, because `Model::symbolic_graph` has to be able to fail
# with `SymbolicBackendFailure` and therefore has to be able to build a graph.
# What linking this adds is the *visibility* of the handles -- a consumer that
# links only `crane_model` never sees a CasADi header through it.
add_library(${PROJECT_NAME}_casadi_graph INTERFACE)
add_library(${PROJECT_NAME}::casadi_graph ALIAS ${PROJECT_NAME}_casadi_graph)
target_compile_features(${PROJECT_NAME}_casadi_graph INTERFACE cxx_std_17)
# Its own include root, in the install tree as well as in the build tree. That
# is the enforcement behind `test/consumer_public_only.cpp`: a consumer that
# links `crane_model` alone does not have `crane_model/symbolic/` on its include
# path at all, so contract §9's restriction is a property of the build and not
# a convention.
target_include_directories(${PROJECT_NAME}_casadi_graph
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/symbolic/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}_casadi_graph>)
target_link_libraries(${PROJECT_NAME}_casadi_graph
  INTERFACE ${PROJECT_NAME} casadi::casadi)
set_target_properties(${PROJECT_NAME}_casadi_graph
  PROPERTIES EXPORT_NAME casadi_graph)

install(
  DIRECTORY include/
  DESTINATION include)
# The shared constants themselves, not just the code that reads them: this is
# the copy `hydraulics::default_path` prefers and the copy the Python export
# will read out of an installed workspace.
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(
  DIRECTORY symbolic/include/
  DESTINATION include/${PROJECT_NAME}_casadi_graph)
install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_casadi_graph
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  EXPORT export_${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 pinocchio coal casadi yaml-cpp)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # This target is installed only with BUILD_TESTING=ON, so normal deployment
  # builds omit the test fixture and its target from the prefix.
  add_library(${PROJECT_NAME}_mock_model
    test_support/src/mock_model.cpp)
  add_library(${PROJECT_NAME}::mock_model ALIAS ${PROJECT_NAME}_mock_model)
  target_compile_features(${PROJECT_NAME}_mock_model PUBLIC cxx_std_17)
  target_include_directories(${PROJECT_NAME}_mock_model
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test_support/include>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(${PROJECT_NAME}_mock_model
    PUBLIC ${PROJECT_NAME} Eigen3::Eigen)
  set_target_properties(${PROJECT_NAME}_mock_model
    PROPERTIES EXPORT_NAME mock_model)
  install(
    DIRECTORY test_support/include/
    DESTINATION include)
  install(
    TARGETS ${PROJECT_NAME}_mock_model
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  # The contract test loads the real machine description, so it needs the
  # checked-in fixtures, and it reads the linkage placements back out of them
  # with Pinocchio to cross-check the compiled-in cylinder geometry. That second
  # use is the only reason a test here links Pinocchio directly; nothing that
  # merely *uses* the model API may.
  ament_add_gtest(test_contract
    test/test_contract.cpp)
  target_link_libraries(test_contract ${PROJECT_NAME}::mock_model pinocchio::pinocchio)
  # src/ is on the test's include path for `collision_model.hpp` alone: the
  # generated geometry and the allowed-collision list are compiled into the
  # library, and the test is what ties them back to the description and to
  # config/allowed_collisions.srdf. Nothing installed exposes that header.
  target_include_directories(test_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(test_contract
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description"
      CRANE_MODEL_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")

  # The symbolic graph, and the consumer that opts into the CasADi handles.
  # This is the only test that links `crane_model::casadi_graph`, and it is what
  # says the private target is usable: it obtains the function handles and
  # evaluates them against the numeric model.
  ament_add_gtest(test_symbolic_graph
    test/test_symbolic_graph.cpp)
  target_link_libraries(test_symbolic_graph
    ${PROJECT_NAME}::casadi_graph)
  # src/ is on this test's include path for `cylinder_geometry.hpp` alone, so
  # the smoothing bound it asserts is written in the model's own effective
  # areas rather than in a magic epsilon. Nothing installed exposes that header.
  target_include_directories(test_symbolic_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(test_symbolic_graph
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description")

  # The Python symbolic model of `scripts/crane_symbolic.py`, against the numeric
  # API. It reaches the build as **C**: `scripts/export_model_fixture.py` runs
  # CasADi's code generation on the Python model and checks the result into
  # `test/generated/`, so this comparison needs no Python interpreter, no acados
  # and no CasADi C++ library -- it links the public library and libm and nothing
  # else. `test/generated/crane_symbolic_fixture.c` is the one hand-written file
  # in that directory; it includes the generated text, which is spelled `.inc` so
  # that the C and C++ linters do not try to review machine output.
  ament_add_gtest(test_symbolic_parity
    test/test_symbolic_parity.cpp
    test/generated/crane_symbolic_fixture.c)
  target_link_libraries(test_symbolic_parity ${PROJECT_NAME} m)
  # src/ is on this test's include path for `cylinder_geometry.hpp` alone, so the
  # smoothing bound it asserts is written in the model's own effective areas;
  # test/generated/ is there for the generated declarations. Nothing installed
  # exposes either.
  target_include_directories(test_symbolic_parity
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/test/generated)
  target_compile_definitions(test_symbolic_parity
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description")

  # ...and the other half of that claim: the checked-in C is still what the
  # Python model generates. It is a separate test because it is a different
  # failure -- `test_symbolic_parity` says the artefact agrees with the numeric
  # model, this says the artefact is the current model -- and because it is the
  # only test in the package that needs a Python interpreter. That interpreter is
  # required rather than optional: a check that silently disappears when a
  # dependency is missing is worse than no check, and this is the one that makes
  # issue 067's CasADi pin mean something.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME export_model_fixture_is_current
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_model_fixture.py --check)

  # The tool axis in the numeric model, after issue 068 took its coordinate out
  # of both OCPs. It links the public library alone -- the claim is about
  # `cylinder_jacobian`, `transmission` and `full_dynamics`, none of which is
  # symbolic -- and reads the tool joint's own range out of the descriptions.
  ament_add_gtest(test_tool_axis
    test/test_tool_axis.cpp)
  target_link_libraries(test_tool_axis ${PROJECT_NAME})
  target_compile_definitions(test_tool_axis
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description")

  # The recorded-trajectory parity comparison of PRD user story 65. It links
  # the public library and nothing else: the generated model reaches it only
  # through `test/recorded_parity_fixture.txt`, a checked-in table of numbers,
  # so no target here names `mp_crane` and the comparison needs neither a bag
  # nor ROS.
  ament_add_gtest(test_recorded_parity
    test/test_recorded_parity.cpp)
  target_link_libraries(test_recorded_parity ${PROJECT_NAME})
  target_compile_definitions(test_recorded_parity
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description"
      CRANE_MODEL_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  # The CasADi pin of `.devcontainer/Dockerfile.vscode`, checked against the
  # environment the tests are actually running in. It lives in this package
  # because this is where CasADi enters the CBS stack -- `crane_mpc` and
  # `crane_planning` reach it through `crane_model::casadi_graph` -- and it
  # links `casadi::casadi` rather than that target because it asks about the
  # library and not about the graph.
  #
  # Keep this string equal to the `CASADI_VERSION` build arg of the Dockerfile.
  # The two cannot be derived from one another: the Dockerfile is in the outer
  # repository and this package has to build standalone.
  set(CRANE_MODEL_CASADI_PINNED_VERSION "3.7.2")
  ament_add_gtest(test_casadi_version test/test_casadi_version.cpp)
  target_link_libraries(test_casadi_version casadi::casadi)
  target_compile_definitions(test_casadi_version
    PRIVATE
      CRANE_MODEL_CASADI_PINNED_VERSION="${CRANE_MODEL_CASADI_PINNED_VERSION}")

  add_executable(test_consumer_w04 test/consumer_w04.cpp)
  add_executable(test_consumer_w05 test/consumer_w05.cpp)
  add_executable(test_consumer_w06 test/consumer_w06.cpp)
  add_executable(test_consumer_w07 test/consumer_w07.cpp)
  add_executable(test_consumer_w08 test/consumer_w08.cpp)
  foreach(consumer IN ITEMS
      test_consumer_w04 test_consumer_w05 test_consumer_w06
      test_consumer_w07 test_consumer_w08)
    target_link_libraries(${consumer} PRIVATE ${PROJECT_NAME}::mock_model)
    add_test(NAME ${consumer} COMMAND ${consumer})
  endforeach()

  # Contract §9's restriction on the public header, enforced rather than
  # documented: this fixture links the public library alone, so a CasADi header
  # reaching it could only come through `crane_model/model.hpp`.
  add_executable(test_consumer_public_only test/consumer_public_only.cpp)
  target_link_libraries(test_consumer_public_only PRIVATE ${PROJECT_NAME})
  add_test(NAME test_consumer_public_only COMMAND test_consumer_public_only)
endif()

ament_package()
