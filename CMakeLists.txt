cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(inno_ogtc_nn)

find_package(Torch REQUIRED)

include_directories(include)

set(srcFiles src/DatasetModule.cpp src/NetProcessing.cpp)

add_executable(gen_datasets src/gen_datasets.cpp)

add_executable(train src/train.cpp ${srcFiles})

target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 14)

add_executable(test src/test.cpp ${srcFiles})

target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)
