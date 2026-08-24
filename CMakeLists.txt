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
  PRIVATE pinocchio::pinocchio coal::coal)

install(
  DIRECTORY include/
  DESTINATION include)
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

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 pinocchio coal)

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
endif()

ament_package()
