cmake_minimum_required(VERSION 3.8)
project(crane_model)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} src/model.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

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
ament_export_dependencies(Eigen3)

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

  ament_add_gtest(test_contract
    test/test_contract.cpp)
  target_link_libraries(test_contract ${PROJECT_NAME}::mock_model)

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
