# Exported to every consumer of crane_model, and included before
# ament_cmake_export_dependencies-extras.cmake finds pinocchio again.
#
# CMake 3.22's FindBoost only strips 1-2 digit version suffixes from a python
# component name (_Boost_COMPONENT_HEADERS, FindBoost.cmake:1395), so the
# "python310" component eigenpy asks for gets no header mapping and FindBoost
# emits a stderr warning for every pinocchio consumer. Supplying the mapping is
# what a newer FindBoost does on its own; the library was always found.
set(_Boost_PYTHON310_HEADERS "boost/python.hpp")
