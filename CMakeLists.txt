cmake_minimum_required(VERSION 3.22)
project(dbscan LANGUAGES CXX CUDA)

#setting flags
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2")
# change to your architecture for best performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -rdc=true -g -G")
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_executable(dbscan main.cu)
target_link_libraries(dbscan PRIVATE ${Python3_LIBRARIES})
set_target_properties(dbscan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
