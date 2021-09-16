cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(inno_ogtc_nn)

find_package(Torch REQUIRED)

include_directories(include)

set(srcFiles src/DatasetModule.cpp src/NetProcessing.cpp)

add_library(cppLib STATIC ${srcFiles})
target_compile_definitions(cppLib PUBLIC cppLib)
target_link_libraries(cppLib "${TORCH_LIBRARIES}")

add_executable(gen_datasets src/gen_datasets.cpp)

add_executable(train src/train.cpp)

target_link_libraries(train "${TORCH_LIBRARIES}")
target_link_libraries(train cppLib)
set_property(TARGET train PROPERTY CXX_STANDARD 14)

add_executable(man_test src/man_test.cpp)

target_link_libraries(man_test "${TORCH_LIBRARIES}")
target_link_libraries(man_test cppLib)
set_property(TARGET man_test PROPERTY CXX_STANDARD 14)

add_executable(test src/test.cpp)

target_link_libraries(test "${TORCH_LIBRARIES}")
target_link_libraries(test cppLib)
set_property(TARGET test PROPERTY CXX_STANDARD 14)


