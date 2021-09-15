cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(inno_ogtc_nn)

find_package(Torch REQUIRED)

#add_executable(${PROJECT_NAME} main.cpp 
#    DatasetModule.h DatasetModule.cpp
#    NetProcessing.h NetProcessing.cpp
#    Net.h)

add_executable(gen_datasets gen_datasets.cpp)

add_executable(train train.cpp 
    DatasetModule.h DatasetModule.cpp
    NetProcessing.h NetProcessing.cpp
    Net.h)

target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 14)
