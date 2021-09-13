cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(inno_ogtc_nn)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
