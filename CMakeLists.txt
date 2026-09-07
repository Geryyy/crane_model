cmake_minimum_required(VERSION 3.8)
project(crane_model)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
include(cmake/boost_python310_headers.cmake)
find_package(pinocchio REQUIRED)
# Coal is used directly, not through Pinocchio: the Pinocchio in this image is
# built without its collision module (no `pinocchio/collision`, no
# PINOCCHIO_WITH_COLLISION), so `pinocchio::GeometryModel` carries a fake
# geometry. Pinocchio still places the links; Coal answers the queries.
find_package(coal REQUIRED)
# config/hydraulics.yaml is read in `Model::create`. yaml-cpp is the parser
# wiki/implementation/libraries.md lists for configuration outside the parameter
# system; it brings no ROS with it, so contract §6 still holds.
find_package(yaml-cpp REQUIRED)
# The planning-side model is Python on Pinocchio's and Coal's own bindings; it
# reads the same URDF and the same config/hydraulics.yaml as the C++ library.
find_package(ament_cmake_python REQUIRED)

add_library(${PROJECT_NAME} src/model.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
# Pinocchio and Coal are PRIVATE: contract §6 keeps them private implementation
# details, so a consumer links the shared libraries but never sees the headers.
# PIC because a consumer of this static archive may itself be a shared object --
# pluginlib loads `crane_control`'s controller as one -- and Pinocchio's and
# Coal's own libraries are shared.
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME}
  PUBLIC Eigen3::Eigen
  PRIVATE pinocchio::pinocchio coal::coal yaml-cpp)

# Where `hydraulics::default_path` looks for config/hydraulics.yaml. Two
# candidates rather than one because the library is ROS-free (contract §6) and
# has no ament index to ask, and `ModelConfig` is frozen so a caller cannot
# hand it a third: the installed copy is what a deployed build reads, and the
# source tree is what a build tree reads before anything has been installed.
target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CRANE_MODEL_HYDRAULICS_YAML_INSTALLED="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/hydraulics.yaml"
    CRANE_MODEL_HYDRAULICS_YAML_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/config/hydraulics.yaml")


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
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  EXPORT export_${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake)

ament_python_install_package(${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 pinocchio coal yaml-cpp)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # The recorded-trajectory parity comparison of PRD user story 65, and the only
  # gtest left in this package: it is the one statement in the workspace that
  # ties a chamber area or a damping number to the real machine. It links the
  # public library and nothing else -- the fixture reaches it as
  # `test/recorded_parity_fixture.txt`, a checked-in table of numbers derived
  # from the 2026-08-19 recordings by `test/derive_recorded_parity.py`, so the
  # comparison needs neither a bag nor ROS.
  ament_add_gtest(test_recorded_parity
    test/test_recorded_parity.cpp)
  target_link_libraries(test_recorded_parity ${PROJECT_NAME})
  target_compile_definitions(test_recorded_parity
    PRIVATE
      CRANE_MODEL_TEST_DESCRIPTION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/description"
      CRANE_MODEL_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  # The Python model against the same recorded fixture the gtest above uses.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_python_parity test/test_python_parity.py)
  ament_add_pytest_test(test_python_collision test/test_python_collision.py)

  # The fixture against the description it came from. `export_ocp.py --check`
  # regenerates from the fixture, so only this catches the fixture itself going
  # stale -- which is how the pre-fit damping stayed baked into the shipped
  # solver.
  ament_add_pytest_test(test_description_snapshot test/test_description_snapshot.py)
endif()

ament_package(CONFIG_EXTRAS "cmake/boost_python310_headers.cmake")
