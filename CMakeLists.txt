project(teb)
cmake_minimum_required(VERSION 3.16)
find_package(Eigen3 REQUIRED)
add_subdirectory(third_party/g2o)
add_executable(teb src/teb.cpp)
target_include_directories(teb SYSTEM
                           PRIVATE third_party/g2o)
target_link_libraries(teb
                      solver_structure_only
                      solver_csparse
                      types_slam2d)
